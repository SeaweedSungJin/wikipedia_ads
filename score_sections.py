"""Evaluate RAG section retrieval accuracy using ``evidence_section_id``.

This script runs the full search pipeline for a slice of the EVQA
dataset and reports how often the ground truth section appears within
the top ranked results.  It prints detailed debugging information for
each sample so we can inspect which document and section were selected
by the system versus the annotated answer.
"""
from __future__ import annotations

from src.config import Config
from src import pipeline
from src.pipeline import search_rag_pipeline
from src.dataloader import VQADataset
import torch

def evaluate(cfg: Config) -> None:
    """Compute retrieval accuracy for a dataset."""
    # Load dataset subset for evaluation
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

    rank1 = 0
    rank3 = 0
    rank5 = 0
    total = 0

    # Iterate over each question in the dataset.  For each entry we run the
    # full RAG pipeline and compare the top ranked sections against the
    # ``evidence_section_id`` provided in the CSV.
    for sample in dataset:
        if not sample.image_paths:
            continue
        cfg.image_path = sample.image_paths[0]
        cfg.text_query = sample.question
        _, sections = search_rag_pipeline(cfg)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
            
        correct_idx = sample.metadata.get("evidence_section_id")
        if correct_idx is None:
            continue
        try:
            correct_idx = int(correct_idx)
        except ValueError:
            continue

        matched_rank = None
        for i, sec in enumerate(sections, 1):
            if (
                sec.get("source_title") == sample.wikipedia_title
                and sec.get("section_idx") == correct_idx
            ):
                matched_rank = i
                break
                
        # Show debug information about the chosen and correct sections
        top1_title = sections[0].get("source_title") if sections else "N/A"
        top1_idx = sections[0].get("section_idx") if sections else -1
        top1_sec = sections[0].get("section_title") if sections else "N/A"

        correct_sec_title = None
        kb_list = pipeline._KB_LIST
        if kb_list:
            for doc in kb_list:
                if doc.get("title") == sample.wikipedia_title:
                    titles = doc.get("section_titles", [])
                    if 0 <= correct_idx < len(titles):
                        correct_sec_title = titles[correct_idx]
                    break
        print(
            f"[Row {sample.row_idx}] Predicted: {top1_title} / {top1_sec} (#{top1_idx}) | "
            f"Actual: {sample.wikipedia_title} / {correct_sec_title} (#{correct_idx}) | "
            f"Match rank: {matched_rank}"
        )

        # Count accuracies based on the matched section rank
        if matched_rank == 1:
            rank1 += 1
            rank3 += 1
            rank5 += 1

        elif matched_rank is not None and matched_rank <= 3:
            rank3 += 1
            rank5 += 1
        elif matched_rank is not None and matched_rank <= 5:
            rank5 += 1
        total += 1
        
    # Avoid division by zero
    if total == 0:
        print("No samples evaluated")
        return

    # Report aggregated metrics
    print(f"Total samples: {total}")
    print(f"Top1 correct: {rank1}/{total}")
    print(f"Top3 correct: {rank3}/{total}")
    print(f"Top5 correct: {rank5}/{total}")



if __name__ == "__main__":
    cfg = Config.from_yaml("config.yaml")
    if not cfg.dataset_csv or not cfg.id2name_json:
        raise ValueError("dataset_csv and id2name_json must be set in config")
    evaluate(cfg) 