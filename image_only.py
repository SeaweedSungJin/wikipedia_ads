from __future__ import annotations

from typing import Optional

from src.config import Config
from src.dataloader import VQADataset
from src.pipeline import search_rag_pipeline

def normalize_title(title: Optional[str]) -> str:
    if not isinstance(title, str):
        return ""
    import re
    title = title.lower()
    title = re.sub(r"\([^)]*\)", "", title)
    title = re.sub(r"[^a-z0-9\s]", "", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()

def evaluate_image_search(cfg: Config) -> None:
    dataset = VQADataset(
        csv_path=cfg.dataset_csv,
        id2name_path=cfg.id2name_json,
        image_root=cfg.dataset_image_root,
        googlelandmark_root=cfg.dataset_google_root,
        start=cfg.dataset_start,
        end=cfg.dataset_end,
    )
    print(f"데이터셋 평가 범위: start={cfg.dataset_start}, end={cfg.dataset_end}.")

    top1 = top3 = top5 = topk = 0
    total = 0
    total_elapsed = 0.0

    for sample in dataset:
        if not sample.image_paths:
            continue
        cfg.image_path = sample.image_paths[0]
        cfg.text_query = sample.question
        image_results, elapsed = search_rag_pipeline(
            cfg, return_time=True, image_only=True
        )
        total_elapsed += elapsed
        total += 1

        print(f"\n=== Row {sample.row_idx} ===")
        print(f"질문: {sample.question}")
        print("-- Top-k 문서 후보 (이미지 검색 기반) --")
        for i, res in enumerate(image_results, 1):
            title = res["doc"].get("title", "")
            print(f"  {i}. {title}")

        gt_norm = normalize_title(sample.wikipedia_title)
        doc_rank = None
        for i, res in enumerate(image_results, 1):
            if normalize_title(res["doc"].get("title")) == gt_norm:
                doc_rank = i
                break
        print(
            f"정답 문서: {sample.wikipedia_title} | rank: {doc_rank}"
        )

        if doc_rank == 1:
            top1 += 1
            top3 += 1
            top5 += 1
            topk += 1
        elif doc_rank is not None:
            if doc_rank <= 3:
                top3 += 1
                top5 += 1
                topk += 1 if doc_rank <= cfg.k_value else 0
            elif doc_rank <= 5:
                top5 += 1
                topk += 1 if doc_rank <= cfg.k_value else 0
            elif doc_rank <= cfg.k_value:
                topk += 1

    if total == 0:
        print("No samples evaluated.")
        return

    print("\n-- 평가 요약 --")
    print(f"총 샘플 수: {total}")
    print(f"Top1 문서 일치 수: {top1}/{total}")
    print(f"Top3 문서 일치 수: {top3}/{total}")
    print(f"Top5 문서 일치 수: {top5}/{total}")
    print(f"Top{cfg.k_value} 문서 일치 수: {topk}/{total}")
    print(f"검색 시간 합계: {total_elapsed:.2f}s")

if __name__ == "__main__":
    cfg = Config.from_yaml("config.yaml")
    if not cfg.dataset_csv or not cfg.id2name_json:
        raise ValueError("dataset_csv and id2name_json must be set in config")
    evaluate_image_search(cfg)