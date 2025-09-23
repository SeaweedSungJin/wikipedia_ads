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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Sequence

import torch
import numpy as np
import faiss
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Repository-relative imports
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

from src.config import Config
from src.dataloader import VQADataset
from src.embedding import encode_image
from src.models import load_image_model
from src.segmenter import SectionSegmenter
from src.evaluation_utils import (
    build_ground_truth,
    compute_k_values,
    init_recall_dict,
    update_recall_from_rank,
    update_section_hits,
)
from src.utils import load_faiss_and_ids, load_image, load_kb, normalize_title
from src.qformer_utils import (
    load_qformer_resources,
    encode_query_multimodal,
    encode_sections_text,
    score_sections,
)

# -----------------------------------------------------------------------------
# Helper data structures
# -----------------------------------------------------------------------------

@dataclass
class SectionEntry:
    doc_title: str
    section_idx: int
    section_text: str
    doc_idx: int


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
    parser.add_argument(
        "--clip-cache-dir",
        default="datasets/clip_cache",
        help="Directory to cache EVA-CLIP retrieval results",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Ranking helpers
# -----------------------------------------------------------------------------


def _dedup_indices_by_doc(indices: Iterable[int], sections: Sequence[SectionEntry]) -> List[int]:
    """Return indices with at most one entry per document."""

    seen: set[int] = set()
    deduped: List[int] = []
    for idx in indices:
        doc_id = sections[idx].doc_idx
        if doc_id in seen:
            continue
        seen.add(doc_id)
        deduped.append(idx)
    return deduped


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
    try:
        norm = faiss.NormalizationTransform(faiss_index.d, 2)
        faiss_index = faiss.IndexPreTransform(norm, faiss_index)
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            res = faiss.StandardGpuResources()
            faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
        except Exception:
            pass
    image_model, image_processor = load_image_model(device_map=cfg.image_device)
    segmenter = SectionSegmenter()

    cache_data = None
    cache_file: Path | None = None
    cache_dirty = False
    if args.clip_cache_dir:
        cache_root = Path(args.clip_cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        model_name = getattr(getattr(image_model, "config", None), "_name_or_path", "eva_clip")
        safe_name = model_name.replace('/', '__')
        cache_file = cache_root / f"{safe_name}.json"
        expected_meta = {
            "dataset_csv": cfg.dataset_csv,
            "dataset_start": cfg.dataset_start,
            "dataset_end": cfg.dataset_end,
            "k_value": cfg.k_value,
        }
        if cache_file.exists():
            try:
                cache_data = json.loads(cache_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cache_data = None
            if not isinstance(cache_data, dict) or "data" not in cache_data:
                cache_data = None
            if cache_data is not None and cache_data.get("meta") != expected_meta:
                cache_data = None
        if cache_data is None:
            cache_data = {"meta": expected_meta, "data": {}}

    resources = load_qformer_resources(
        Path(args.qformer_ckpt), device=args.qformer_device, max_text_length=args.max_text_length
    )
    qformer = resources.model
    vis_processor = resources.vis_processor
    text_processor = resources.text_processor

    doc_cache: Dict[int, List[SectionEntry]] = {}

    k_values = compute_k_values(cfg.k_value)
    image_hits = init_recall_dict(k_values)
    cls_doc_hits = init_recall_dict(k_values)
    cls_section_hits = init_recall_dict(k_values)
    maxsim_doc_hits = init_recall_dict(k_values)
    maxsim_section_hits = init_recall_dict(k_values)
    logsum_doc_hits = init_recall_dict(k_values)
    logsum_section_hits = init_recall_dict(k_values)
    hybrid_doc_hits = init_recall_dict(k_values)
    hybrid_section_hits = init_recall_dict(k_values)
    hybrid_cls_doc_hits = init_recall_dict(k_values)
    hybrid_cls_section_hits = init_recall_dict(k_values)

    total = 0

    search_expand = cfg.search_expand or cfg.k_value * 20
    max_unique_docs = max(cfg.k_value, search_expand)

    def make_empty_hits() -> Dict[int, int]:
        """Return a zero-filled recall dictionary that matches ``k_values``."""

        return {k: 0 for k in k_values}

    for sample in tqdm(dataset, desc="Evaluating Q-Former Reranker"):
        if not sample.image_paths:
            continue

        ground_truth = build_ground_truth(sample)
        if ground_truth is None:
            continue

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

        row_key = str(sample.row_idx)
        cache_entry = None
        if cache_data is not None:
            cache_entry = cache_data.get("data", {}).get(row_key)

        unique_doc_indices: List[int] = []
        doc_scores: Dict[int, float] = {}
        if cache_entry is not None:
            unique_doc_indices = [int(v) for v in cache_entry.get("doc_indices", [])]
            scores = cache_entry.get("doc_scores", [])
            doc_scores = {unique_doc_indices[i]: float(scores[i]) for i in range(min(len(unique_doc_indices), len(scores)))}
        else:
            img_emb = encode_image(pil_image, image_model, image_processor)
            img_emb_np = img_emb.cpu().numpy().astype("float32")
            search_k = min(search_expand, faiss_index.ntotal)
            # FAISS returns inner-product scores that depend on index scaling.
            norm = np.linalg.norm(img_emb_np, ord=2)
            search_query = img_emb_np if norm == 0 else img_emb_np / norm
            distances, indices = faiss_index.search(search_query, search_k)

            seen_docs: set[int] = set()
            for pos, idx in enumerate(indices[0]):
                doc_idx = int(kb_ids[int(idx)])
                if doc_idx < 0 or doc_idx >= len(kb_list):
                    continue
                title_norm = normalize_title(kb_list[doc_idx].get("title", ""))
                if any(phrase in title_norm for phrase in ("list of", "outline of", "index of")):
                    continue
                if doc_idx not in seen_docs:
                    unique_doc_indices.append(doc_idx)
                    seen_docs.add(doc_idx)
                if doc_idx not in doc_scores:
                    doc_scores[doc_idx] = float(distances[0][pos])
                else:
                    doc_scores[doc_idx] = max(doc_scores[doc_idx], float(distances[0][pos]))
                if len(unique_doc_indices) >= max_unique_docs and pos >= search_expand - 1:
                    break

            if cache_data is not None:
                cache_data.setdefault("data", {})[row_key] = {
                    "doc_indices": unique_doc_indices,
                    "doc_scores": [doc_scores[idx] for idx in unique_doc_indices],
                }
                cache_dirty = True

        # Image recall
        doc_rank = None
        for rank, doc_idx in enumerate(unique_doc_indices, start=1):
            title_norm = normalize_title(kb_list[doc_idx].get("title", ""))
            if title_norm in ground_truth.title_set and doc_rank is None:
                doc_rank = rank
        update_recall_from_rank(image_hits, doc_rank, k_values)

        if not unique_doc_indices:
            continue

        # ------------------------------------------------------------------
        # Section preparation
        # ------------------------------------------------------------------
        sections: List[SectionEntry] = []
        for doc_idx in unique_doc_indices:
            if doc_idx not in doc_cache:
                doc = kb_list[doc_idx]
                doc_title = (doc.get("title", "") or "").strip()
                segment_entries: List[SectionEntry] = []
                for seg in segmenter.get_segments(doc):
                    raw_text = seg.get("section_text", "").strip()
                    if not raw_text:
                        continue
                    section_title = (seg.get("section_title", "") or "").strip()
                    header = (
                        f"# Wiki Article: {doc_title}\n"
                        f"## Section Title: {section_title}\n"
                    )
                    combined_text = f"{header}{raw_text}"
                    processed_text = (
                        text_processor(combined_text) if text_processor else combined_text
                    )
                    segment_entries.append(
                        SectionEntry(
                            doc_title=doc_title,
                            section_idx=seg.get("section_idx", -1),
                            section_text=processed_text,
                            doc_idx=doc_idx,
                        )
                    )
                doc_cache[doc_idx] = segment_entries
            sections.extend(doc_cache[doc_idx])

        if not sections:
            continue

        vision_scores = torch.tensor(
            [doc_scores.get(entry.doc_idx, 0.0) for entry in sections], dtype=torch.float32
        )

        # ------------------------------------------------------------------
        # Q-Former encoding & scoring
        # ------------------------------------------------------------------
        question_text = sample.question
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

        cls_scores = scores.cls_max.cpu()
        maxsim_scores = scores.maxsim.cpu()
        logsum_scores = scores.logsumexp.cpu()
        idx_cls = torch.argsort(cls_scores, descending=True)
        idx_maxsim = torch.argsort(maxsim_scores, descending=True)
        idx_logsum = torch.argsort(logsum_scores, descending=True)
        update_section_hits(
            idx_cls.tolist(),
            sections,
            ground_truth,
            cls_doc_hits,
            cls_section_hits,
            k_values,
            lambda sec: sec.doc_title,
            lambda sec: sec.section_idx,
        )
        combo_scores = 0.5 * vision_scores + 0.5 * maxsim_scores
        idx_combo = torch.argsort(combo_scores, descending=True)
        dedup_idx_combo = _dedup_indices_by_doc(idx_combo.tolist(), sections)
        # Deduped rankings are used for document recall, while the full index list
        # is preserved for section recall so we do not discard valid sections from
        # the same article.
        update_section_hits(
            dedup_idx_combo,
            sections,
            ground_truth,
            hybrid_doc_hits,
            make_empty_hits(),
            k_values,
            lambda sec: sec.doc_title,
            lambda sec: sec.section_idx,
        )
        update_section_hits(
            idx_combo.tolist(),
            sections,
            ground_truth,
            make_empty_hits(),
            hybrid_section_hits,
            k_values,
            lambda sec: sec.doc_title,
            lambda sec: sec.section_idx,
        )
        update_section_hits(
            idx_maxsim.tolist(),
            sections,
            ground_truth,
            maxsim_doc_hits,
            maxsim_section_hits,
            k_values,
            lambda sec: sec.doc_title,
            lambda sec: sec.section_idx,
        )
        cls_combo_scores = 0.5 * vision_scores + 0.5 * cls_scores
        idx_cls_combo = torch.argsort(cls_combo_scores, descending=True)
        dedup_idx_cls_combo = _dedup_indices_by_doc(idx_cls_combo.tolist(), sections)
        update_section_hits(
            dedup_idx_cls_combo,
            sections,
            ground_truth,
            hybrid_cls_doc_hits,
            make_empty_hits(),
            k_values,
            lambda sec: sec.doc_title,
            lambda sec: sec.section_idx,
        )
        update_section_hits(
            idx_cls_combo.tolist(),
            sections,
            ground_truth,
            make_empty_hits(),
            hybrid_cls_section_hits,
            k_values,
            lambda sec: sec.doc_title,
            lambda sec: sec.section_idx,
        )
        update_section_hits(
            idx_logsum.tolist(),
            sections,
            ground_truth,
            logsum_doc_hits,
            logsum_section_hits,
            k_values,
            lambda sec: sec.doc_title,
            lambda sec: sec.section_idx,
        )

    # ----------------------------------------------------------------------
    # Metrics summary
    # ----------------------------------------------------------------------
    if total == 0:
        print("No valid samples were evaluated. Aborting.")
        return

    def to_ratio(hit_dict: Dict[int, int]) -> Dict[str, float]:
        return {f"R@{k}": hit_dict[k] / total for k in k_values}

    if cache_file is not None and cache_data is not None and cache_dirty:
        cache_file.write_text(json.dumps(cache_data, ensure_ascii=False), encoding="utf-8")

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
            "vision_maxsim": {
                "doc": to_ratio(hybrid_doc_hits),
                "section": to_ratio(hybrid_section_hits),
            },
            "vision_cls_max": {
                "doc": to_ratio(hybrid_cls_doc_hits),
                "section": to_ratio(hybrid_cls_section_hits),
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
