#!/usr/bin/env python3
"""Q-Former based reranking pipeline for Encyclopedic-VQA.

This script replaces the existing reranker stage with the EchoSight Q-Former
model (``reranker.pth``) and evaluates recall for both image retrieval and
section reranking.  It performs the following steps for each dataset sample:

1.  Retrieve top-k candidate documents via EVA-CLIP + FAISS (existing code).
2.  Split the documents into sections and compute Q-Former text embeddings.
3.  Encode the query image + question into multimodal Q-Former embeddings.
4.  Score each section using (a) the original CLS-based max similarity and
    (b) late-interaction (MaxSim and LogSumExp) over 32 token pairs.
5.  Report Recall@{1,3,5,10} for image retrieval and each reranker variant,
    writing the summary to JSON.

Usage::

    python qformer_reranker_pipeline.py --config config.yaml \
        --qformer-ckpt datasets/reranker.pth --output qformer_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Repository-relative imports
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ECHOSIGHT_ROOT = ROOT.parent / "EchoSight"
if str(ECHOSIGHT_ROOT) not in sys.path:
    sys.path.insert(0, str(ECHOSIGHT_ROOT))

from omegaconf import OmegaConf  # type: ignore

from lavis.models import load_preprocess  # type: ignore
from lavis.models.blip2_models.blip2_qformer_reranker import (  # type: ignore
    Blip2QformerReranker,
)

from src.config import Config
from src.dataloader import VQADataset
from src.embedding import encode_image
from src.models import load_image_model
from src.segmenter import SectionSegmenter
from src.utils import (
    load_faiss_and_ids,
    load_image,
    load_kb,
    normalize_title,
    normalize_url_to_title,
)

# -----------------------------------------------------------------------------
# Helper data structures
# -----------------------------------------------------------------------------

@dataclass
class SectionEntry:
    doc_title: str
    section_idx: int
    section_text: str


@dataclass
class ScoringResults:
    cls_max: torch.Tensor
    maxsim: torch.Tensor
    logsumexp: torch.Tensor


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    default_ckpt = ROOT / "datasets" / "reranker.pth"
    parser = argparse.ArgumentParser(
        description="Evaluate Q-Former reranker on Encyclopedic-VQA dataset"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--output",
        default="qformer_metrics.json",
        help="Path to JSON file where metrics will be written",
    )
    parser.add_argument(
        "--qformer-ckpt",
        default=str(default_ckpt),
        help="Path to EchoSight Q-Former reranker checkpoint",
    )
    parser.add_argument(
        "--qformer-device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for Q-Former inference (e.g., cuda:0 or cpu)",
    )
    parser.add_argument(
        "--section-batch-size",
        type=int,
        default=32,
        help="Batch size for encoding section texts",
    )
    parser.add_argument(
        "--text-token-count",
        type=int,
        default=32,
        help="Number of text tokens kept for late interaction",
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=256,
        help="Maximum tokenized length for section texts (<= tokenizer limit)",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Q-Former utilities
# -----------------------------------------------------------------------------


def load_qformer_model(
    ckpt_path: Path,
    device: str,
    max_text_length: int,
) -> Tuple[torch.nn.Module, callable, callable]:
    """Load the EchoSight Q-Former reranker and preprocessing utilities."""

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Q-Former checkpoint not found: {ckpt_path}")

    if "cuda" in device and not torch.cuda.is_available():
        device = "cpu"
    device_obj = torch.device(device)

    cfg_path = (
        ECHOSIGHT_ROOT / "lavis" / "configs" / "models" / "blip2" / "blip2_pretrain.yaml"
    )
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_readonly(cfg, False)
    model_cfg = cfg.model
    OmegaConf.set_readonly(model_cfg, False)
    model_cfg.load_finetuned = False
    model_cfg.load_pretrained = False
    model_cfg.pretrained = ""

    qformer = Blip2QformerReranker.from_config(model_cfg)
    qformer.eval()
    qformer.use_vanilla_qformer = True

    vis_processors, txt_processors = load_preprocess(cfg.preprocess)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    filtered = [k for k in list(state_dict.keys()) if k.startswith("Qformer.cls.predictions")]
    for key in filtered:
        state_dict.pop(key, None)
    msg = qformer.load_state_dict(state_dict, strict=False)
    meta_params = [n for n, p in qformer.named_parameters() if getattr(p, "is_meta", False) or p.device.type == "meta"]
    if meta_params:
        print(f"[WARN] Found {len(meta_params)} meta parameters; reinitializing on CPU")
        qformer = qformer.to_empty(device=torch.device("cpu"))
        qformer.load_state_dict(state_dict, strict=False)
        meta_params = [n for n, p in qformer.named_parameters() if getattr(p, "is_meta", False) or p.device.type == "meta"]
        if meta_params:
            raise RuntimeError(f"Q-Former still has meta parameters after reload: {meta_params[:5]}")
    if msg.missing_keys:
        missing = [k for k in msg.missing_keys if not k.startswith("Qformer.cls.predictions")]
        if missing:
            print(f"[WARN] Missing keys when loading Q-Former: {missing[:5]}...")
    if msg.unexpected_keys:
        unexpected = [k for k in msg.unexpected_keys if not k.startswith("Qformer.cls.predictions")]
        if unexpected:
            print(f"[WARN] Unexpected keys: {unexpected[:5]}...")

    qformer = qformer.to(device_obj)

    tokenizer = qformer.tokenizer
    if tokenizer is not None and max_text_length is not None and max_text_length > 0:
        tokenizer.model_max_length = min(max_text_length, tokenizer.model_max_length)

    vis_proc = vis_processors.get("eval") if vis_processors else None
    txt_proc = txt_processors.get("eval") if txt_processors else None
    return qformer, vis_proc, txt_proc


@torch.no_grad()
def encode_query_multimodal(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    question: str,
) -> torch.Tensor:
    """Return normalized multimodal query embeddings (num_query_token, dim)."""

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    sample = {
        "image": image_tensor.to(device=device, dtype=dtype),
        "text_input": [question],
    }
    multimodal = model.extract_features(sample, mode="multimodal").multimodal_embeds
    return multimodal[0].to(torch.float32).cpu()


@torch.no_grad()
def encode_sections_text(
    model: torch.nn.Module,
    texts: List[str],
    batch_size: int,
    token_keep: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode section texts and return CLS + token embeddings with masks.

    Returns
    -------
    cls_embs: torch.Tensor shape (N, D)
        Q-Former text embeddings for the [CLS] token.
    token_embs: torch.Tensor shape (N, token_keep, D)
        Normalised projections for the first ``token_keep`` tokens (excluding CLS).
    token_masks: torch.Tensor shape (N, token_keep)
        Boolean mask indicating valid tokens (False denotes padding).
    """

    device = next(model.parameters()).device
    tokenizer = model.tokenizer
    text_proj = model.text_proj

    cls_chunks: List[torch.Tensor] = []
    token_chunks: List[torch.Tensor] = []
    mask_chunks: List[torch.Tensor] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)

        output = model.Qformer.bert(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = output.last_hidden_state
        hidden = hidden.to(text_proj.weight.dtype)
        proj = text_proj(hidden)
        proj = F.normalize(proj, dim=-1).to(torch.float32).cpu()

        cls_chunks.append(proj[:, 0, :])
        token_part = proj[:, 1:, :]
        mask = tokens.attention_mask[:, 1:].to(torch.float32)

        if token_part.shape[1] < token_keep:
            pad_len = token_keep - token_part.shape[1]
            token_part = F.pad(token_part, (0, 0, 0, pad_len))
            mask = F.pad(mask, (0, pad_len))
        else:
            token_part = token_part[:, :token_keep, :]
            mask = mask[:, :token_keep]

        token_chunks.append(token_part)
        mask_chunks.append(mask)

    cls_embs = torch.cat(cls_chunks, dim=0)
    token_embs = torch.cat(token_chunks, dim=0)
    token_masks = torch.cat(mask_chunks, dim=0).bool()
    return cls_embs, token_embs, token_masks


# -----------------------------------------------------------------------------
# Scoring utilities
# -----------------------------------------------------------------------------


def compute_cls_max_scores(
    fusion_tokens: torch.Tensor, cls_embs: torch.Tensor
) -> torch.Tensor:
    """Original Q-Former score using CLS embeddings and max similarity."""

    sim = cls_embs @ fusion_tokens.T  # (N, num_tokens)
    scores, _ = torch.max(sim, dim=1)
    return scores


def _pairwise_similarity(
    section_tokens: torch.Tensor,
    fusion_tokens: torch.Tensor,
) -> torch.Tensor:
    # section_tokens: (N, L, D), fusion_tokens: (Q, D)
    return torch.einsum("nld,qd->nlq", section_tokens, fusion_tokens)


def compute_maxsim_scores(
    fusion_tokens: torch.Tensor,
    section_tokens: torch.Tensor,
    token_masks: torch.Tensor,
) -> torch.Tensor:
    sim = _pairwise_similarity(section_tokens, fusion_tokens)
    mask = token_masks.unsqueeze(-1)
    sim = sim.masked_fill(~mask, float("-inf"))
    per_query_max = torch.max(sim, dim=1).values
    per_query_max = torch.nan_to_num(per_query_max, nan=-1e4, neginf=-1e4)
    return per_query_max.mean(dim=1)


def compute_logsumexp_scores(
    fusion_tokens: torch.Tensor,
    section_tokens: torch.Tensor,
    token_masks: torch.Tensor,
) -> torch.Tensor:
    sim = _pairwise_similarity(section_tokens, fusion_tokens)
    mask = token_masks.unsqueeze(-1)
    sim = sim.masked_fill(~mask, float("-inf"))
    logsum = torch.logsumexp(sim, dim=1)
    token_counts = mask.sum(dim=1).clamp(min=1).to(logsum.dtype)
    logsum = logsum - torch.log(token_counts)
    logsum = torch.nan_to_num(logsum, nan=-1e4, neginf=-1e4)
    return logsum.mean(dim=1)


def score_sections(
    fusion_tokens: torch.Tensor,
    section_tokens: torch.Tensor,
    cls_embs: torch.Tensor,
    token_masks: torch.Tensor,
) -> ScoringResults:
    scores_cls = compute_cls_max_scores(fusion_tokens, cls_embs)
    scores_maxsim = compute_maxsim_scores(fusion_tokens, section_tokens, token_masks)
    scores_logsumexp = compute_logsumexp_scores(
        fusion_tokens, section_tokens, token_masks
    )
    return ScoringResults(scores_cls, scores_maxsim, scores_logsumexp)


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------

K_VALUES = [1, 3, 5, 10]


def update_hits(
    indices: Iterable[int],
    sections: List[SectionEntry],
    gt_title_set: set,
    gt_pairs: set,
    doc_hits: Dict[int, int],
    section_hits: Dict[int, int],
) -> None:
    idx_list = list(indices)
    for k in K_VALUES:
        top_idx = idx_list[:k]
        if any(
            normalize_title(sections[i].doc_title) in gt_title_set for i in top_idx
        ):
            doc_hits[k] += 1
        if any(
            (
                normalize_title(sections[i].doc_title),
                sections[i].section_idx,
            )
            in gt_pairs
            for i in top_idx
        ):
            section_hits[k] += 1


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    cfg = Config.from_yaml(args.config)
    dataset = VQADataset(
        csv_path=cfg.dataset_csv,
        id2name_paths=cfg.id2name_paths,
        image_root=cfg.dataset_image_root,
        googlelandmark_root=cfg.dataset_google_root,
        start=cfg.dataset_start,
        end=cfg.dataset_end,
    )

    kb_list, url_to_idx = load_kb(cfg.kb_json_path)
    faiss_index, kb_ids = load_faiss_and_ids(cfg.base_path, kb_list, url_to_idx)
    image_model, image_processor = load_image_model(device_map=cfg.image_device)
    segmenter = SectionSegmenter()

    qformer, vis_processor, text_processor = load_qformer_model(
        Path(args.qformer_ckpt), args.qformer_device, args.max_text_length
    )

    doc_cache: Dict[int, List[SectionEntry]] = {}

    image_hits = {k: 0 for k in K_VALUES}
    cls_doc_hits = {k: 0 for k in K_VALUES}
    cls_section_hits = {k: 0 for k in K_VALUES}
    maxsim_doc_hits = {k: 0 for k in K_VALUES}
    maxsim_section_hits = {k: 0 for k in K_VALUES}
    logsum_doc_hits = {k: 0 for k in K_VALUES}
    logsum_section_hits = {k: 0 for k in K_VALUES}

    total = 0

    search_expand = cfg.search_expand or cfg.k_value * 20

    for sample in tqdm(dataset, desc="Evaluating Q-Former Reranker"):
        if not sample.image_paths:
            continue

        gt_titles_raw = str(sample.wikipedia_title or "").split("|")
        gt_urls_raw = str(sample.wikipedia_url or "").split("|")

        gt_titles: List[str] = []
        for title in gt_titles_raw:
            if title.strip():
                gt_titles.append(normalize_title(title))
        for url in gt_urls_raw:
            if url.strip():
                gt_titles.append(normalize_url_to_title(url))

        if not gt_titles:
            continue

        sec_ids_raw = sample.metadata.get("evidence_section_id")
        section_ids: List[int] = []
        if isinstance(sec_ids_raw, str):
            for val in sec_ids_raw.split("|"):
                if val.strip():
                    try:
                        section_ids.append(int(val))
                    except ValueError:
                        continue
        elif sec_ids_raw is not None:
            try:
                section_ids.append(int(sec_ids_raw))
            except ValueError:
                pass

        gt_pairs = set(zip(gt_titles, section_ids)) if section_ids else set()
        gt_title_set = set(gt_titles)

        total += 1

        # ------------------------------------------------------------------
        # Image retrieval (EVA-CLIP + FAISS)
        # ------------------------------------------------------------------
        try:
            pil_image = load_image(sample.image_paths[0])
        except Exception as err:
            print(f"[Row {sample.row_idx}] Failed to load image: {err}")
            continue

        try:
            if vis_processor is not None:
                image_tensor = vis_processor(pil_image)
            else:
                raise TypeError("vis_processor is None")
        except Exception:
            # Fall back to simple resize if EchoSight processor fails or missing
            image_tensor = image_processor(pil_image, return_tensors="pt").pixel_values[0]

        img_emb = encode_image(pil_image, image_model, image_processor)
        img_emb_np = img_emb.cpu().numpy().astype("float32")
        search_k = min(search_expand, faiss_index.ntotal)
        distances, indices = faiss_index.search(img_emb_np, search_k)

        # Collect unique documents up to k_value
        unique_doc_indices: List[int] = []
        seen_docs = set()
        for idx in indices[0]:
            doc_idx = int(kb_ids[int(idx)])
            if doc_idx in seen_docs:
                continue
            if doc_idx < 0 or doc_idx >= len(kb_list):
                continue
            title_norm = normalize_title(kb_list[doc_idx].get("title", ""))
            if any(phrase in title_norm for phrase in ("list of", "outline of", "index of")):
                continue
            unique_doc_indices.append(doc_idx)
            seen_docs.add(doc_idx)
            if len(unique_doc_indices) >= cfg.k_value:
                break

        # Image recall
        doc_rank = None
        ordered_titles = []
        for rank, doc_idx in enumerate(unique_doc_indices, start=1):
            title_norm = normalize_title(kb_list[doc_idx].get("title", ""))
            ordered_titles.append(title_norm)
            if title_norm in gt_title_set and doc_rank is None:
                doc_rank = rank
        for k in K_VALUES:
            if doc_rank is not None and doc_rank <= k:
                image_hits[k] += 1

        if not unique_doc_indices:
            continue

        # ------------------------------------------------------------------
        # Section preparation
        # ------------------------------------------------------------------
        sections: List[SectionEntry] = []
        for doc_idx in unique_doc_indices:
            if doc_idx not in doc_cache:
                doc = kb_list[doc_idx]
                segment_entries: List[SectionEntry] = []
                for seg in segmenter.get_segments(doc):
                    text = seg.get("section_text", "").strip()
                    if not text:
                        continue
                    processed_text = text_processor(text) if text_processor else text
                    segment_entries.append(
                        SectionEntry(
                            doc_title=doc.get("title", ""),
                            section_idx=seg.get("section_idx", -1),
                            section_text=processed_text,
                        )
                    )
                doc_cache[doc_idx] = segment_entries
            sections.extend(doc_cache[doc_idx])

        if not sections:
            continue

        # ------------------------------------------------------------------
        # Q-Former encoding & scoring
        # ------------------------------------------------------------------
        question_text = (
            text_processor(sample.question) if text_processor else sample.question
        )
        fusion_tokens = encode_query_multimodal(
            qformer, image_tensor.unsqueeze(0), question_text
        )

        section_texts = [entry.section_text for entry in sections]
        cls_embs, token_embs, token_masks = encode_sections_text(
            qformer,
            section_texts,
            batch_size=args.section_batch_size,
            token_keep=args.text_token_count,
        )

        scores = score_sections(fusion_tokens, token_embs, cls_embs, token_masks)

        idx_cls = torch.argsort(scores.cls_max, descending=True)
        idx_maxsim = torch.argsort(scores.maxsim, descending=True)
        idx_logsum = torch.argsort(scores.logsumexp, descending=True)

        update_hits(idx_cls.tolist(), sections, gt_title_set, gt_pairs, cls_doc_hits, cls_section_hits)
        update_hits(idx_maxsim.tolist(), sections, gt_title_set, gt_pairs, maxsim_doc_hits, maxsim_section_hits)
        update_hits(idx_logsum.tolist(), sections, gt_title_set, gt_pairs, logsum_doc_hits, logsum_section_hits)

    # ----------------------------------------------------------------------
    # Metrics summary
    # ----------------------------------------------------------------------
    if total == 0:
        print("No valid samples were evaluated. Aborting.")
        return

    def to_ratio(hit_dict: Dict[int, int]) -> Dict[str, float]:
        return {f"R@{k}": hit_dict[k] / total for k in K_VALUES}

    summary = {
        "total_samples": total,
        "image_recall": to_ratio(image_hits),
        "reranker": {
            "cls_max": {
                "doc": to_ratio(cls_doc_hits),
                "section": to_ratio(cls_section_hits),
            },
            "maxsim": {
                "doc": to_ratio(maxsim_doc_hits),
                "section": to_ratio(maxsim_section_hits),
            },
            "logsumexp": {
                "doc": to_ratio(logsum_doc_hits),
                "section": to_ratio(logsum_section_hits),
            },
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
