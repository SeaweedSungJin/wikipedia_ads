from __future__ import annotations

import math
from typing import List

import torch
from src.config import Config
from src.dataloader import VQADataset
from src.pipeline import search_rag_pipeline


def softmax(xs: List[float]) -> List[float]:
    t = torch.tensor(xs)
    return torch.softmax(t, dim=0).tolist()


def run_bge(cfg: Config) -> None:
    dataset = VQADataset(
        csv_path=cfg.dataset_csv,
        id2name_path=cfg.id2name_json,
        image_root=cfg.dataset_image_root,
        googlelandmark_root=cfg.dataset_google_root,
        start=cfg.dataset_start,
        end=cfg.dataset_end,
    )
    print(f"데이터셋 평가 범위: start={cfg.dataset_start}, end={cfg.dataset_end}.")

    doc_match_total = 0
    sec_match_total = 0
    sample_total = 0
    total_elapsed = 0.0

    for sample in dataset:
        if not sample.image_paths:
            continue
        cfg.image_path = sample.image_paths[0]
        cfg.text_query = sample.question
        img_results, top_sections, sections, elapsed = search_rag_pipeline(
            cfg, return_time=True, return_candidates=True
        )
        total_elapsed += elapsed
        sample_total += 1

        print(f"\n=== Row {sample.row_idx} ===")
        print(f"질문: {sample.question}")
        print("-- Top-k 문서 후보 (이미지 검색 기반) --")
        for i, res in enumerate(img_results, 1):
            title = res["doc"].get("title", "")
            print(f"  {i}. {title}")

        if top_sections:
            probs = [s.get("prob", 0.0) for s in top_sections]
            scores = [s.get("similarity", 0.0) for s in top_sections]
            entropy = -sum(p * math.log(p + 1e-12) for p in probs)
            norm = math.log(len(probs)) if len(probs) > 1 else 1.0
            confidence = 1 - entropy / norm

            print("-- BGE 섹션 점수 (Top-M) --")
            for i, (sec, prob) in enumerate(zip(top_sections, probs), 1):
                stitle = sec.get("source_title", "")
                sectitle = sec.get("section_title", "")
                sidx = sec.get("section_idx")
                score = sec.get("similarity", 0.0)
                print(
                    f"  {i}. {stitle} / {sectitle} (#{sidx}) "
                    f"score={score:.3f}, prob={prob:.3f}"
                )
            print(f"Entropy: {entropy:.3f} | Confidence: {confidence:.3f}")

            print("-- 최종 선택된 섹션 --")
            for i, sec in enumerate(sections, 1):
                stitle = sec.get("source_title", "")
                sectitle = sec.get("section_title", "")
                sidx = sec.get("section_idx")
                print(f"  {i}. {stitle} / {sectitle} (#{sidx})")

            gt_idx_raw = sample.metadata.get("evidence_section_id")
            try:
                gt_idx = int(gt_idx_raw) if gt_idx_raw is not None else None
            except (TypeError, ValueError):
                gt_idx = None

            doc_match = any(
                sec.get("source_title") == sample.wikipedia_title for sec in sections
            )
            sec_match = any(
                sec.get("source_title") == sample.wikipedia_title
                and sec.get("section_idx") == gt_idx
                for sec in sections
            )
            print(
                f"정답 문서: {sample.wikipedia_title} | 정답 섹션 idx: {gt_idx} | 정답: {sample.answer}"
            )
            print(
                f"선택된 섹션에 정답 문서 포함: {doc_match} | 정답 섹션 포함: {sec_match}"
            )

            print(f"총 검색 시간: {elapsed:.2f}s")

            if doc_match:
                doc_match_total += 1
            if sec_match:
                sec_match_total += 1
        else:
            print("섹션 결과가 없습니다.")

    print("\n-- 평가 요약 --")
    print(f"문서 일치 수: {doc_match_total}/{sample_total}")
    print(f"문서+섹션 일치 수: {sec_match_total}/{sample_total}")
    print(f"검색 시간 합계: {total_elapsed:.2f}s")


if __name__ == "__main__":
    cfg = Config.from_yaml("config.yaml")
    if not cfg.dataset_csv or not cfg.id2name_json:
        raise ValueError("dataset_csv and id2name_json must be set in config")
    run_bge(cfg)