from __future__ import annotations

import re
from typing import Optional

import torch
from tqdm import tqdm

from src.config import Config
from src.dataloader import VQADataset
from src.pipeline import search_rag_pipeline


def normalize_title(title: Optional[str]) -> str:
    """Normalize a Wikipedia title for comparison."""
    if not isinstance(title, str):
        return ""
    title = title.lower()
    title = re.sub(r"\([^)]*\)", "", title)
    title = re.sub(r"[^a-z0-9\s]", "", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def evaluate(cfg: Config) -> None:
    """Compute retrieval accuracy for a dataset."""
    if cfg.segment_level == "sentence":
        print("Sentence-level evaluation is not supported. Use section or paragraph segmentation.")
        return

    dataset = VQADataset(
        csv_path=cfg.dataset_csv,
        id2name_path=cfg.id2name_json,
        image_root=cfg.dataset_image_root,
        googlelandmark_root=cfg.dataset_google_root,
        start=cfg.dataset_start,
        end=cfg.dataset_end,
    )
    print(
        f"데이터셋 평가 범위: start={cfg.dataset_start}, end={cfg.dataset_end}."
    )

    doc_match_top1 = doc_match_top3 = doc_match_top5 = 0
    sec_match_top1 = sec_match_top3 = sec_match_top5 = 0
    total = 0

    for sample in tqdm(dataset, desc="Evaluating"):
        if not sample.image_paths:
            continue
        cfg.image_path = sample.image_paths[0]
        cfg.text_query = sample.question
        image_results, sections = search_rag_pipeline(cfg)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            correct_idx = int(sample.metadata.get("evidence_section_id"))
        except (TypeError, ValueError):
            continue

        gt_title_norm = normalize_title(sample.wikipedia_title)
        doc_rank = None
        sec_rank = None
        # Determine document rank from image search results only
        for i, res in enumerate(image_results, 1):
            if normalize_title(res["doc"].get("title")) == gt_title_norm:
                doc_rank = i
                break

        for i, sec in enumerate(sections, 1):
            sec_title_norm = normalize_title(sec.get("source_title"))
            if sec_title_norm != gt_title_norm:
                continue

            # When using paragraph-level segmentation the paragraph is
            # considered correct if it originates from the correct section.
            if sec.get("section_idx") == correct_idx:
                sec_rank = i
                break

        if doc_rank == 1:
            doc_match_top1 += 1
            doc_match_top3 += 1
            doc_match_top5 += 1
        elif doc_rank is not None and doc_rank <= 3:
            doc_match_top3 += 1
            doc_match_top5 += 1
        elif doc_rank is not None and doc_rank <= 5:
            doc_match_top5 += 1

        if sec_rank == 1:
            sec_match_top1 += 1
            sec_match_top3 += 1
            sec_match_top5 += 1
        elif sec_rank is not None and sec_rank <= 3:
            sec_match_top3 += 1
            sec_match_top5 += 1
        elif sec_rank is not None and sec_rank <= 5:
            sec_match_top5 += 1

        total += 1

        top1_title = sections[0].get("source_title") if sections else "N/A"
        top1_sec = sections[0].get("section_title") if sections else "N/A"
        top1_idx = sections[0].get("section_idx") if sections else -1
        print(
            f"[Row {sample.row_idx}] Top1: {top1_title} / {top1_sec} (#{top1_idx}) | "
            f"GT: {sample.wikipedia_title} (#{correct_idx}) | "
            f"Doc rank: {doc_rank} | Section rank: {sec_rank}"
        )

    if total == 0:
        print("No samples evaluated.")
        return

    print("\n=== Document & Section Match Accuracy ===")
    print(f"Total samples: {total}")
    print(f"Top1 doc match: {doc_match_top1}/{total}")
    print(f"Top3 doc match: {doc_match_top3}/{total}")
    print(f"Top5 doc match: {doc_match_top5}/{total}")
    print(f"Top1 doc+section match: {sec_match_top1}/{total}")
    print(f"Top3 doc+section match: {sec_match_top3}/{total}")
    print(f"Top5 doc+section match: {sec_match_top5}/{total}")


if __name__ == "__main__":
    cfg = Config.from_yaml("config.yaml")
    if not cfg.dataset_csv or not cfg.id2name_json:
        raise ValueError("dataset_csv and id2name_json must be set in config")
    evaluate(cfg)