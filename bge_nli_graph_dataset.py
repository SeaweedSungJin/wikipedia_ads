from __future__ import annotations

import torch
from src.config import Config
from src.dataloader import VQADataset
from src.pipeline import search_rag_pipeline
from src.nli_cluster import load_nli_model, build_relation_graph, maximal_cliques


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_nli_model(cfg.nli_model, device)

    total_elapsed = 0.0
    sample_total = 0

    k_values = [1, 3, 5, 10]
    bge_doc_hits = {k: 0 for k in k_values}
    bge_sec_hits = {k: 0 for k in k_values}
    nli_doc_hits = {k: 0 for k in k_values}
    nli_sec_hits = {k: 0 for k in k_values}

    for sample in dataset:
        if not sample.image_paths:
            continue
        cfg.image_path = sample.image_paths[0]
        cfg.text_query = sample.question

        img_results, top_sections, _, elapsed = search_rag_pipeline(
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

            adj, stats = build_relation_graph(
                top_sections,
                model=model,
                tokenizer=tokenizer,
                max_length=cfg.nli_max_length,
                device=device,
            )
            print("-- NLI 관계 그래프 --")
            print(
                f"  entailment edges: {stats['entailment']}, "
                f"neutral edges: {stats['neutral']}, contradictions: {stats['contradiction']}"
            )
            cliques = maximal_cliques(adj)
            clusters = []
            for cl in cliques:
                members = [top_sections[i] for i in cl]
                avg_score = sum(m.get("similarity", 0.0) for m in members) / len(members)
                members.sort(key=lambda s: s.get("similarity", 0.0), reverse=True)
                if cfg.nli_max_cluster and len(members) > cfg.nli_max_cluster:
                    members = members[: cfg.nli_max_cluster]
                clusters.append({"avg_score": avg_score, "sections": members})

            clusters.sort(key=lambda c: c["avg_score"], reverse=True)

            print("-- 최종 NLI 클러스터 (그래프 기반) --")
            for c_idx, cl in enumerate(clusters, 1):
                print(f"Cluster {c_idx}: avg_score={cl['avg_score']:.3f}")
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

        print(f"총 검색 시간: {elapsed:.2f}s")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n-- 평가 요약 --")
    print("BGE reranker:")
    for k in k_values:
        print(f"  Recall@{k} 문서 일치: {bge_doc_hits[k]}/{sample_total}")
        print(f"  Recall@{k} 문서+섹션 일치: {bge_sec_hits[k]}/{sample_total}")
    print("NLI clustering (graph):")
    for k in k_values:
        print(f"  Recall@{k} 문서 일치: {nli_doc_hits[k]}/{sample_total}")
        print(f"  Recall@{k} 문서+섹션 일치: {nli_sec_hits[k]}/{sample_total}")
    print(f"총 검색 시간 합계: {total_elapsed:.2f}s")


if __name__ == "__main__":
    cfg = Config.from_yaml("config.yaml")
    if not cfg.dataset_csv or not cfg.id2name_json:
        raise ValueError("dataset_csv and id2name_json must be set in config")
    run_bge_nli_graph_dataset(cfg)