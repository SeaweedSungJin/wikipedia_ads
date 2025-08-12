from __future__ import annotations

from typing import Optional

import logging
import time
import torch
from src.config import Config
from src.dataloader import VQADataset
from src.pipeline import search_rag_pipeline
from src.nli_cluster import load_nli_model, cluster_sections_clique


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_title(title: Optional[str]) -> str:
    if not isinstance(title, str):
        return ""
    import re
    title = title.lower()
    title = re.sub(r"\([^)]*\)", "", title)
    title = re.sub(r"[^a-z0-9\s]", "", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def run_bge_nli_graph_dataset(cfg: Config) -> None:
    dataset = VQADataset(
        csv_path=cfg.dataset_csv,
        id2name_path=cfg.id2name_json,
        image_root=cfg.dataset_image_root,
        googlelandmark_root=cfg.dataset_google_root,
        start=cfg.dataset_start,
        end=cfg.dataset_end,
    )
    print(f"데이터셋 평가 범위: start={cfg.dataset_start}, end={cfg.dataset_end}.")

    def _resolve(dev):
        if isinstance(dev, int):
            dev = f"cuda:{dev}"
        if not torch.cuda.is_available() and isinstance(dev, str) and "cuda" in dev:
            dev = "cpu"
        return torch.device(dev)

    device = _resolve(cfg.nli_device)
    model, tokenizer = load_nli_model(cfg.nli_model, device)
    logger.info(
        "NLI clustering mode: clique-weighted (ent-contr), nli_threshold is ignored; using e_min/margin/tau"
    )
    print(
        "이미지 검색 모드:",
        "대표 이미지 1개만 사용" if cfg.first_image_only else "문서의 모든 이미지 사용",
    )

    total_bge_elapsed = 0.0
    total_nli_elapsed = 0.0
    sample_total = 0

    k_values = [1, 3, 5, 10]
    img_doc_hits = {k: 0 for k in k_values}
    bge_doc_hits = {k: 0 for k in k_values}
    bge_sec_hits = {k: 0 for k in k_values}
    nli_doc_hits = {k: 0 for k in k_values}
    nli_sec_hits = {k: 0 for k in k_values}

    for sample in dataset:
        if not sample.image_paths:
            continue
        cfg.image_path = sample.image_paths[0]
        cfg.text_query = sample.question

        img_results, top_sections, _, bge_elapsed = search_rag_pipeline(
            cfg, return_time=True, return_candidates=True
        )
        total_bge_elapsed += bge_elapsed
        sample_total += 1

        print(f"\n=== Row {sample.row_idx} ===")
        print(f"질문: {sample.question}")
        print("-- Top-k 문서 후보 (이미지 검색 기반) --")
        for i, res in enumerate(img_results, 1):
            title = res["doc"].get("title", "")
            print(f"  {i}. {title}")

        gt_norm = normalize_title(sample.wikipedia_title)
        doc_rank = None
        for i, res in enumerate(img_results, 1):
            if normalize_title(res["doc"].get("title")) == gt_norm:
                doc_rank = i
                break
        print(f"정답 문서: {sample.wikipedia_title} | rank: {doc_rank}")
        for k in k_values:
            if doc_rank is not None and doc_rank <= k:
                img_doc_hits[k] += 1

        if top_sections:
            print("-- BGE 섹션 점수 (Top-M) --")
            for i, sec in enumerate(top_sections, 1):
                stitle = sec.get("source_title", "")
                sectitle = sec.get("section_title", "")
                sidx = sec.get("section_idx")
                score = sec.get("similarity", 0.0)
                print(f"  {i}. {stitle} / {sectitle} (#{sidx}) score={score:.3f}")

            gt_idx_raw = sample.metadata.get("evidence_section_id")
            try:
                gt_idx = int(gt_idx_raw) if gt_idx_raw is not None else None
            except (TypeError, ValueError):
                gt_idx = None

            for k in k_values:
                subset = top_sections[:k]
                if any(sec.get("source_title") == sample.wikipedia_title for sec in subset):
                    bge_doc_hits[k] += 1
                if any(
                    sec.get("source_title") == sample.wikipedia_title
                    and sec.get("section_idx") == gt_idx
                    for sec in subset
                ):
                    bge_sec_hits[k] += 1

            nli_start = time.time()
            e_min = getattr(cfg, "nli_e_min", None) or getattr(cfg, "nli_threshold", 0.5)
            clusters, stats = cluster_sections_clique(
                top_sections,
                model=model,
                tokenizer=tokenizer,
                max_length=cfg.nli_max_length,
                device=device,
                max_cluster_size=cfg.nli_max_cluster,
                lambda_score=cfg.nli_lambda,
                e_min=e_min,
                margin=cfg.nli_margin,
                tau=cfg.nli_tau,
                batch_size=cfg.nli_batch_size,
            )
            nli_elapsed = time.time() - nli_start
            total_nli_elapsed += nli_elapsed
            print("-- NLI 관계 그래프 --")
            print(
                f"  entailment edges: {stats['entailment']}, "
                f"neutral edges: {stats['neutral']}, contradictions: {stats['contradiction']}"
            )

            raw = stats.get("raw_cliques")
            if raw:
                raw = [[i + 1 for i in cl] for cl in raw]
            print("Clusters (indices):", raw)
            print("-- 최종 NLI 클러스터 (그래프 기반) --")
            for c_idx, cl in enumerate(clusters, 1):
                idx_disp = [i + 1 for i in cl["indices"]]
                print(
                    f"Cluster {c_idx} indices={idx_disp}: avg_score={cl['avg_score']:.3f}"
                )
                for sec in cl["sections"]:
                    stitle = sec.get("source_title", "")
                    sectitle = sec.get("section_title", "")
                    sidx = sec.get("section_idx")
                    score = sec.get("similarity", 0.0)
                    print(f"    - {stitle} / {sectitle} (#{sidx}) score={score:.3f}")

            top_cluster = clusters[0]["sections"] if clusters else []
            doc_match = any(
                sec.get("source_title") == sample.wikipedia_title for sec in top_cluster
            )
            sec_match = any(
                sec.get("source_title") == sample.wikipedia_title
                and sec.get("section_idx") == gt_idx
                for sec in top_cluster
            )
            print(
                f"정답 문서: {sample.wikipedia_title} | 정답 섹션 idx: {gt_idx} | 정답: {sample.answer}"
            )
            print(
                f"선택된 클러스터에 정답 문서 포함: {doc_match} | 정답 섹션 포함: {sec_match}"
            )

            for k in k_values:
                cl_subset = clusters[:k]
                secs = [s for cl in cl_subset for s in cl["sections"]]
                if any(sec.get("source_title") == sample.wikipedia_title for sec in secs):
                    nli_doc_hits[k] += 1
                if any(
                    sec.get("source_title") == sample.wikipedia_title
                    and sec.get("section_idx") == gt_idx
                    for sec in secs
                ):
                    nli_sec_hits[k] += 1
        else:
            print("섹션 결과가 없습니다.")
            nli_elapsed = 0.0

        print(f"BGE 검색 시간: {bge_elapsed:.2f}s")
        print(f"NLI 클러스터링 시간: {nli_elapsed:.2f}s")
        print(f"총 검색 시간: {bge_elapsed + nli_elapsed:.2f}s")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n-- 평가 요약 --")
    print("Image search:")
    for k in k_values:
        print(f"  Recall@{k} 문서 일치: {img_doc_hits[k]}/{sample_total}")
    print("BGE reranker:")
    for k in k_values:
        print(f"  Recall@{k} 문서 일치: {bge_doc_hits[k]}/{sample_total}")
        print(f"  Recall@{k} 문서+섹션 일치: {bge_sec_hits[k]}/{sample_total}")
    print("NLI clustering (graph):")
    for k in k_values:
        print(f"  Recall@{k} 문서 일치: {nli_doc_hits[k]}/{sample_total}")
        print(f"  Recall@{k} 문서+섹션 일치: {nli_sec_hits[k]}/{sample_total}")
    print(f"BGE 검색 시간 합계: {total_bge_elapsed:.2f}s")
    print(f"NLI 클러스터링 시간 합계: {total_nli_elapsed:.2f}s")
    print(f"총 검색 시간 합계: {total_bge_elapsed + total_nli_elapsed:.2f}s")


if __name__ == "__main__":
    cfg = Config.from_yaml("config.yaml")
    if not cfg.dataset_csv or not cfg.id2name_json:
        raise ValueError("dataset_csv and id2name_json must be set in config")
    run_bge_nli_graph_dataset(cfg)