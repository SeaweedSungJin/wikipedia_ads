from __future__ import annotations

import time
import torch
from src.config import Config
from src.dataloader import VQADataset
from src.pipeline import search_rag_pipeline
from src.nli_cluster import cluster_sections_clique, cluster_sections_consistency
from src.models import load_vlm_model, generate_vlm_answer, load_nli_model, resolve_device
from src.eval import evaluate_example
from src.utils import normalize_title, normalize_url_to_title


def run_bge_nli_graph_dataset(cfg: Config) -> None:
    dataset = VQADataset(
        csv_path=cfg.dataset_csv,
        id2name_paths=cfg.id2name_paths,
        image_root=cfg.dataset_image_root,
        googlelandmark_root=cfg.dataset_google_root,
        start=cfg.dataset_start,
        end=cfg.dataset_end,
    )
    print(f"데이터셋 평가 범위: start={cfg.dataset_start}, end={cfg.dataset_end}.")

    device = resolve_device(cfg.nli_device)
    nli_choice = cfg.nli_models.get
    if nli_choice("deberta", False):
        model_name = cfg.deberta_nli_model
    elif nli_choice("roberta", False):
        model_name = cfg.roberta_nli_model
    elif nli_choice("deberta_v3", False):
        model_name = cfg.deberta_v3_nli_model
    else:
        model_name = cfg.deberta_nli_model
    model, tokenizer = load_nli_model(model_name, device)
    vlm_model, vlm_processor = load_vlm_model(device_map=cfg.vlm_device)
    print(
        "NLI clustering mode: clique-weighted (ent-contr) using e_min/margin/tau"
    )
    print("이미지 검색 모드: 문서의 모든 이미지 사용")

    total_bge_elapsed = 0.0
    total_nli_elapsed = 0.0
    sample_total = 0
    vlm_correct = 0
    vlm_total = 0

    max_k = getattr(cfg, "k_value", 10)
    base_k = [1, 3, 5, 10]
    k_values = sorted(set([k for k in base_k if k <= max_k] + [max_k]))
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

        try:
            img_results, top_sections, _, bge_elapsed = search_rag_pipeline(cfg)
        except Exception as e:
            print(f"[Row {sample.row_idx}] 이미지 검색 실패: {e}")
            continue
        total_bge_elapsed += bge_elapsed
        sample_total += 1

        candidate_sections = [
            {
                "doc_title": s.get("source_title", ""),
                "section_id": s.get("section_idx"),
                "section_text": s.get("section_text", ""),
                "similarity": s.get("similarity", 0.0),
            }
            for s in top_sections
        ]

        # --- Ground Truth(정답) 파싱 로직 수정 ---
        gt_titles_raw = str(sample.wikipedia_title or '').split('|')
        gt_urls_raw = str(sample.wikipedia_url or '').split('|')

        gt_titles: list[str] = []
        for title in gt_titles_raw:
            if title.strip():
                gt_titles.append(normalize_title(title))
        for url in gt_urls_raw:
            if url.strip():
                gt_titles.append(normalize_url_to_title(url))
        gt_title_set = set(gt_titles)
        # --- 수정 완료 ---

        sec_ids_raw = sample.metadata.get("evidence_section_id")
        raw_sections: list[str] = []
        if isinstance(sec_ids_raw, str):
            raw_sections = sec_ids_raw.split("|")
        elif sec_ids_raw is not None:
            raw_sections = [str(sec_ids_raw)]
        gt_section_ids = []
        for s in raw_sections:
            try:
                gt_section_ids.append(int(s))
            except ValueError:
                continue
        gt_pairs = set(zip(gt_titles, gt_section_ids))

        # Image search evaluation
        doc_rank = None
        for i, res in enumerate(img_results, 1):
            title_norm = normalize_title(res["doc"].get("title"))
            if title_norm in gt_title_set:
                doc_rank = i
                break
        for k in k_values:
            if doc_rank is not None and doc_rank <= k:
                img_doc_hits[k] += 1

        if candidate_sections:
            for k in k_values:
                subset = candidate_sections[:k]
                if any(
                    normalize_title(sec.get("doc_title")) in gt_title_set
                    for sec in subset
                ):
                    bge_doc_hits[k] += 1
                if any(
                    (
                        normalize_title(sec.get("doc_title")),
                        sec.get("section_id"),
                    )
                    in gt_pairs
                    for sec in subset
                ):
                    bge_sec_hits[k] += 1

            nli_start = time.time()
            if getattr(cfg, "nli_selection", "consistency") == "consistency":
                clusters, _ = cluster_sections_consistency(
                    candidate_sections,
                    model=model,
                    tokenizer=tokenizer,
                    max_length=cfg.nli_max_length,
                    device=device,
                    alpha=getattr(cfg, "nli_alpha", 1.0),
                    beta=getattr(cfg, "nli_beta", 1.0),
                    batch_size=cfg.nli_batch_size,
                    tau_edge=max(0.0, min(1.0, getattr(cfg, "nli_tau", 0.0))),
                    hybrid_lambda=getattr(cfg, "nli_hybrid_lambda", 0.5),
                    autocast=getattr(cfg, "nli_autocast", True),
                    autocast_dtype=getattr(cfg, "nli_autocast_dtype", "fp16"),
                )
            else:
                clusters, _ = cluster_sections_clique(
                    candidate_sections,
                    model=model,
                    tokenizer=tokenizer,
                    max_length=cfg.nli_max_length,
                    device=device,
                    max_cluster_size=cfg.nli_max_cluster,
                    alpha=getattr(cfg, "nli_alpha", 1.0),
                    beta=getattr(cfg, "nli_beta", 1.0),
                    lambda_score=cfg.nli_lambda,
                    e_min=cfg.nli_e_min,
                    margin=cfg.nli_margin,
                    tau=cfg.nli_tau,
                    batch_size=cfg.nli_batch_size,
                    edge_rule=getattr(cfg, "nli_edge_rule", "avg"),
                    dir_margin=getattr(cfg, "nli_dir_margin", 0.0),
                    autocast=getattr(cfg, "nli_autocast", True),
                    autocast_dtype=getattr(cfg, "nli_autocast_dtype", "fp16"),
                )
            nli_elapsed = time.time() - nli_start
            total_nli_elapsed += nli_elapsed
            top_cluster = clusters[0]["sections"] if clusters else []

            # VLM inference using top cluster sections
            sections_text = [sec.get("section_text", "") for sec in top_cluster]
            if sections_text:
                pred = generate_vlm_answer(
                    vlm_model, vlm_processor, sample.question, cfg.image_path, sections_text
                )
                q_type = sample.metadata.get("question_type", "automatic")
                correct = bool(
                    evaluate_example(
                        sample.question, [sample.answer], pred, q_type
                    )
                )
                vlm_total += 1
                vlm_correct += int(correct)
                print(
                    f"Row {sample.row_idx} | 정답: {sample.answer} | 예측: {pred} | 일치: {correct}"
                )

            for k in k_values:
                cl_subset = clusters[:k]
                secs = [s for cl in cl_subset for s in cl["sections"]]
                if any(
                    normalize_title(sec.get("doc_title")) in gt_title_set
                    for sec in secs
                ):
                    nli_doc_hits[k] += 1
                if any(
                    (
                        normalize_title(sec.get("doc_title")),
                        sec.get("section_id"),
                    )
                    in gt_pairs
                    for sec in secs
                ):
                    nli_sec_hits[k] += 1
        else:
            print("섹션 결과가 없습니다.")
            nli_elapsed = 0.0

        print(f"BGE 검색 시간: {bge_elapsed:.2f}s | NLI 시간: {nli_elapsed:.2f}s")
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
    if vlm_total:
        print(f"VLM 정답률: {vlm_correct}/{vlm_total}")


if __name__ == "__main__":
    cfg = Config.from_yaml("config.yaml")
    if not cfg.dataset_csv or not cfg.id2name_paths:
        raise ValueError("dataset_csv and id2name_paths must be set in config")
    run_bge_nli_graph_dataset(cfg)
