"""Evaluate RAG section retrieval accuracy using ``evidence_section_id``.

This script runs the full search pipeline for a slice of the EVQA
dataset and reports how often the ground truth section appears within
the top ranked results.  It prints detailed debugging information for
each sample so we can inspect which document and section were selected
by the system versus the annotated answer.
"""
from __future__ import annotations
import torch
import re
from tqdm import tqdm
from src import pipeline
from src.pipeline import search_rag_pipeline
from src.dataloader import VQADataset
from src.config import Config

def normalize_title(title: str) -> str:
    """문서 제목 정규화: 괄호, 특수문자 제거, 소문자화, 공백 정리"""
    if not isinstance(title, str):
        return ""
    title = title.lower()
    title = re.sub(r"\([^)]*\)", "", title)
    title = re.sub(r"[^a-z0-9\s]", "", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()

def evaluate(cfg: Config) -> None:
    dataset = VQADataset(
        csv_path=cfg.dataset_csv,
        id2name_path=cfg.id2name_json,
        image_root=cfg.dataset_image_root,
        googlelandmark_root=cfg.dataset_google_root,
        start=cfg.dataset_start,
        end=cfg.dataset_end,
    )
    print(f"데이터셋 평가 범위: start={cfg.dataset_start}, end={cfg.dataset_end}.")

    total = 0
    doc_match_top1 = doc_match_top3 = doc_match_top5 = 0
    section_match_top1 = section_match_top3 = section_match_top5 = 0

    for sample in tqdm(dataset, desc="Evaluating"):
        if not sample.image_paths:
            continue

        cfg.image_path = sample.image_paths[0]
        cfg.text_query = sample.question
        _, sections = search_rag_pipeline(cfg)

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            correct_section_idx = int(sample.metadata.get("evidence_section_id"))
        except (TypeError, ValueError):
            continue

        gt_title_norm = normalize_title(sample.wikipedia_title)

        doc_matched_rank = None
        section_matched_rank = None

        for i, sec in enumerate(sections, 1):
            sec_title_norm = normalize_title(sec.get("source_title", ""))
            if sec_title_norm != gt_title_norm:
                continue  # 문서 불일치 시 넘어감

            if doc_matched_rank is None:
                doc_matched_rank = i  # 문서 일치 최초 위치 기록

            if sec.get("section_idx") == correct_section_idx:
                section_matched_rank = i
                break  # 문서+섹션 모두 일치 시 중단

        # 문서만 매치된 통계
        if doc_matched_rank == 1:
            doc_match_top1 += 1
            doc_match_top3 += 1
            doc_match_top5 += 1
        elif doc_matched_rank is not None and doc_matched_rank <= 3:
            doc_match_top3 += 1
            doc_match_top5 += 1
        elif doc_matched_rank is not None and doc_matched_rank <= 5:
            doc_match_top5 += 1

        # 문서+섹션 모두 매치된 통계
        if section_matched_rank == 1:
            section_match_top1 += 1
            section_match_top3 += 1
            section_match_top5 += 1
        elif section_matched_rank is not None and section_matched_rank <= 3:
            section_match_top3 += 1
            section_match_top5 += 1
        elif section_matched_rank is not None and section_matched_rank <= 5:
            section_match_top5 += 1

        total += 1

        # 디버깅 출력
        top1_title = sections[0].get("source_title") if sections else "N/A"
        top1_section = sections[0].get("section_title") if sections else "N/A"
        top1_idx = sections[0].get("section_idx") if sections else -1

        print(
            f"[Row {sample.row_idx}] Top1: {top1_title} / {top1_section} (#{top1_idx}) | "
            f"GT: {sample.wikipedia_title} (#{correct_section_idx}) | "
            f"Doc rank: {doc_matched_rank} | Section rank: {section_matched_rank}"
        )

    if total == 0:
        print("No samples evaluated.")
        return

    print("\n=== Document & Section Match Accuracy ===")
    print(f"Total samples: {total}")
    print(f"Top1 doc match: {doc_match_top1}/{total}")
    print(f"Top3 doc match: {doc_match_top3}/{total}")
    print(f"Top5 doc match: {doc_match_top5}/{total}")
    print(f"Top1 doc+section match: {section_match_top1}/{total}")
    print(f"Top3 doc+section match: {section_match_top3}/{total}")
    print(f"Top5 doc+section match: {section_match_top5}/{total}")



if __name__ == "__main__":
    cfg = Config.from_yaml("config.yaml")
    if not cfg.dataset_csv or not cfg.id2name_json:
        raise ValueError("dataset_csv and id2name_json must be set in config")
    evaluate(cfg) 