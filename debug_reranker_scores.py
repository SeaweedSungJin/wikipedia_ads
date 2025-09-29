#!/usr/bin/env python3
"""Dump raw image and reranker scores for inspection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.config import Config
from src.dataloader import VQADataset
from src.embedding import encode_image
from src.models import load_image_model, load_mpnet_biencoder
from src.segmenter import SectionSegmenter
from src.utils import load_faiss_and_ids, load_image, load_kb, normalize_title
from src.qformer_utils import (
    load_qformer_resources,
    encode_query_multimodal,
    encode_sections_text,
    score_sections,
)
from src.rerankers import BGEReranker, ElectraReranker, JinaReranker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect raw reranker scores")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--samples", type=int, default=10, help="Number of dataset rows to process")
    parser.add_argument("--doc-topk", type=int, default=20, help="Top-N documents to keep")
    parser.add_argument("--section-topk", type=int, default=20, help="Top-N sections per reranker")
    parser.add_argument("--output", default="debug_reranker_scores.json", help="Where to write the results")
    return parser.parse_args()


def _build_document_list(
    distances: np.ndarray,
    indices: np.ndarray,
    kb_ids: np.ndarray,
    kb_list: List[Dict],
    doc_topk: int,
) -> Tuple[List[int], Dict[int, float]]:
    doc_scores: Dict[int, float] = {}
    unique_doc_indices: List[int] = []
    for pos in range(indices.shape[1]):
        faiss_idx = int(indices[0, pos])
        doc_idx = int(kb_ids[faiss_idx])
        if doc_idx in doc_scores:
            continue
        if not (0 <= doc_idx < len(kb_list)):
            continue
        title_norm = normalize_title(kb_list[doc_idx].get("title", ""))
        if any(pat in title_norm for pat in ("list of", "outline of", "index of")):
            continue
        doc_scores[doc_idx] = float(distances[0, pos])
        unique_doc_indices.append(doc_idx)
        if len(unique_doc_indices) >= doc_topk:
            break
    return unique_doc_indices, doc_scores


def _build_sections(doc_indices: List[int], kb_list: List[Dict], segmenter: SectionSegmenter) -> List[Dict]:
    sections: List[Dict] = []
    for doc_idx in doc_indices:
        doc = kb_list[doc_idx]
        doc_title = (doc.get("title", "") or "").strip()
        for seg in segmenter.get_segments(doc):
            sections.append(
                {
                    "doc_idx": doc_idx,
                    "doc_title": doc_title,
                    "section_idx": seg.get("section_idx", -1),
                    "section_title": seg.get("section_title", ""),
                    "section_text": seg.get("section_text", ""),
                }
            )
    return sections


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

    use_jina_tiny = cfg.rerankers.get("jina_tiny", False)
    use_jina_turbo = cfg.rerankers.get("jina_turbo", False)
    use_bge = cfg.rerankers.get("bge", False)
    use_electra = cfg.rerankers.get("electra", False)
    use_mpnet = cfg.rerankers.get("mpnet", False)
    use_qformer = cfg.rerankers.get("q-former", False) or cfg.rerankers.get("qformer", False)

    jina_model = None
    if use_jina_tiny or use_jina_turbo:
        jina_model = JinaReranker(
            model_name="jinaai/jina-reranker-v1-turbo-en" if use_jina_turbo else "jinaai/jina-reranker-v1-tiny-en",
            device=cfg.bge_device,
        )

    bge_model = None
    if use_bge:
        bge_model = BGEReranker(
            cfg.bge_model,
            device=cfg.bge_device,
            max_length=cfg.bge_max_length,
            batch_size=cfg.bge_batch_size,
        )

    electra_model = None
    if use_electra:
        electra_model = ElectraReranker(
            cfg.electra_model,
            device=cfg.bge_device,
            max_length=cfg.bge_max_length,
            batch_size=cfg.electra_batch_size,
        )

    mpnet_model = None
    if use_mpnet:
        mpnet_device = f"cuda:{cfg.bge_device}" if isinstance(cfg.bge_device, int) else cfg.bge_device
        mpnet_model = load_mpnet_biencoder(cfg.mpnet_model, mpnet_device)

    qformer_resources = None
    if use_qformer:
        ckpt_cfg = getattr(cfg, "qformer_ckpt", "datasets/reranker.pth")
        ckpt_path = Path(ckpt_cfg)
        if not ckpt_path.is_absolute():
            ckpt_path = Path(__file__).resolve().parent / ckpt_cfg
        qformer_resources = load_qformer_resources(
            ckpt_path,
            device=getattr(cfg, "qformer_device", 0),
            max_text_length=getattr(cfg, "qformer_max_text_length", 256),
        )

    outputs: List[Dict] = []
    search_expand = cfg.search_expand or cfg.k_value * 20

    for sample in tqdm(dataset, total=args.samples, desc="Collecting scores"):
        if len(outputs) >= args.samples:
            break
        if not sample.image_paths:
            continue

        pil_image = load_image(sample.image_paths[0])
        img_tensor = image_processor(pil_image, return_tensors="pt").pixel_values
        img_emb = encode_image(pil_image, image_model, image_processor)
        img_emb_np = img_emb.cpu().numpy().astype("float32")
        img_emb_np /= np.linalg.norm(img_emb_np, axis=1, keepdims=True)

        search_k = min(search_expand, faiss_index.ntotal)
        distances, indices = faiss_index.search(img_emb_np, search_k)
        doc_indices, doc_scores = _build_document_list(distances, indices, kb_ids, kb_list, args.doc_topk)
        if not doc_indices:
            continue

        sections = _build_sections(doc_indices, kb_list, segmenter)
        if not sections:
            outputs.append(
                {
                    "sample_id": sample.row_idx,
                    "question": sample.question,
                    "docs_raw": [
                        {
                            "title": kb_list[idx].get("title", ""),
                            "raw_score": doc_scores[idx],
                        }
                        for idx in doc_indices
                    ],
                    "sections": {},
                }
            )
            continue

        section_texts = [sec["section_text"] for sec in sections]

        section_scores: Dict[str, List[float]] = {}

        if jina_model is not None:
            section_scores["jina"] = [float(s) for s in jina_model.score(sample.question, section_texts)]

        if bge_model is not None:
            section_scores["bge"] = [float(s) for s in bge_model.score(sample.question, section_texts)]

        if electra_model is not None:
            section_scores["electra"] = [float(s) for s in electra_model.score(sample.question, section_texts)]

        if mpnet_model is not None:
            mpnet_device = next(mpnet_model.parameters()).device
            query_emb = mpnet_model.encode(
                sample.question,
                convert_to_tensor=True,
                device=mpnet_device,
                normalize_embeddings=True,
            )
            sec_embs = mpnet_model.encode(
                section_texts,
                batch_size=256,
                convert_to_tensor=True,
                device=mpnet_device,
                normalize_embeddings=True,
            )
            scores = torch.nn.functional.cosine_similarity(sec_embs, query_emb.unsqueeze(0)).cpu().tolist()
            section_scores["mpnet"] = [float(s) for s in scores]

        if qformer_resources is not None:
            try:
                image_tensor = qformer_resources.vis_processor(pil_image)
            except Exception:
                image_tensor = img_tensor[0]
            fusion_tokens = encode_query_multimodal(qformer_resources.model, image_tensor.unsqueeze(0), sample.question)
            processed_texts: List[str] = []
            for sec in sections:
                header = (
                    f"# Wiki Article: {sec['doc_title']}\n"
                    f"## Section Title: {sec['section_title']}\n"
                )
                combined = header + sec["section_text"]
                if qformer_resources.text_processor:
                    try:
                        processed_texts.append(qformer_resources.text_processor(combined))
                    except Exception:
                        processed_texts.append(combined)
                else:
                    processed_texts.append(combined)
            cls_embs, token_embs, token_masks = encode_sections_text(
                qformer_resources.model,
                processed_texts,
                batch_size=getattr(cfg, "qformer_section_batch_size", 32),
                token_keep=getattr(cfg, "qformer_text_token_count", 32),
            )
            scores = score_sections(fusion_tokens, token_embs, cls_embs, token_masks)
            section_scores["qformer_cls"] = scores.cls_max.cpu().tolist()
            section_scores["qformer_maxsim"] = scores.maxsim.cpu().tolist()
            section_scores["qformer_logsumexp"] = scores.logsumexp.cpu().tolist()

        doc_results = [
            {
                "title": kb_list[idx].get("title", ""),
                "raw_score": doc_scores[idx],
            }
            for idx in doc_indices
        ]

        section_outputs: Dict[str, List[Dict]] = {}
        doc_score_map = {idx: doc_scores[idx] for idx in doc_indices}

        for name, scores in section_scores.items():
            ranked = sorted(
                zip(sections, scores),
                key=lambda item: item[1],
                reverse=True,
            )[: args.section_topk]
            section_outputs[name] = [
                {
                    "doc_title": sec["doc_title"],
                    "section_idx": sec["section_idx"],
                    "section_title": sec["section_title"],
                    "image_score_raw": doc_score_map.get(sec["doc_idx"], None),
                    "reranker_score_raw": float(score),
                }
                for sec, score in ranked
            ]

        outputs.append(
            {
                "sample_id": sample.row_idx,
                "question": sample.question,
                "image_path": sample.image_paths[0],
                "docs_raw": doc_results,
                "sections": section_outputs,
            }
        )

    Path(args.output).write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()


"""
python debug_reranker_scores.py \
    --config config.yaml \
    --samples 10 \
    --doc-topk 20 \
    --section-topk 20 \
    --output debug_reranker_scores.json
    """