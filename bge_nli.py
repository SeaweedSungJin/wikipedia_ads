from __future__ import annotations

import torch
from src.config import Config
from src.pipeline import search_rag_pipeline
from src.nli_cluster import load_nli_model, cluster_sections


def run_bge_nli(cfg: Config) -> None:
    img_results, top_sections, _, elapsed = search_rag_pipeline(
        cfg, return_time=True, return_candidates=True
    )

    print("-- Top-k 문서 후보 (이미지 검색 기반) --")
    for i, res in enumerate(img_results, 1):
        title = res["doc"].get("title", "")
        print(f"  {i}. {title}")

    print("-- BGE 섹션 점수 (Top-M) --")
    for i, sec in enumerate(top_sections, 1):
        stitle = sec.get("source_title", "")
        sectitle = sec.get("section_title", "")
        sidx = sec.get("section_idx")
        score = sec.get("similarity", 0.0)
        print(f"  {i}. {stitle} / {sectitle} (#{sidx}) score={score:.3f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_nli_model(cfg.nli_model, device)
    clusters = cluster_sections(
        top_sections,
        model=model,
        tokenizer=tokenizer,
        threshold=cfg.nli_threshold,
        max_length=cfg.nli_max_length,
        device=device,
        max_cluster_size=cfg.nli_max_cluster,
    )

    print("-- 최종 NLI 클러스터 --")
    for idx, cl in enumerate(clusters, 1):
        print(f"Cluster {idx}: avg_score={cl['avg_score']:.3f}")
        for sec in cl["sections"]:
            stitle = sec.get("source_title", "")
            sectitle = sec.get("section_title", "")
            sidx = sec.get("section_idx")
            score = sec.get("similarity", 0.0)
            print(f"    - {stitle} / {sectitle} (#{sidx}) score={score:.3f}")

    print(f"총 검색 시간: {elapsed:.2f}s")


if __name__ == "__main__":
    cfg = Config.from_yaml("config.yaml")
    run_bge_nli(cfg)