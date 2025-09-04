from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch

from src.config import Config
from src.dataloader import VQADataset
from src.pipeline import search_rag_pipeline
from src.nli_cluster import cluster_sections_clique
from src.models import load_vlm_model, generate_vlm_answer, load_nli_model, resolve_device
from src.eval import evaluate_example
from src.utils import normalize_title, normalize_url_to_title
from src.metrics_utils import PowerSampler, MetricsSink, stage_meter, Percentiles, env_info


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure module metrics on EVQA slice")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--dataset", default=None, help="Override dataset CSV path")
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--pctl", nargs="*", type=int, default=[50, 90, 99])
    p.add_argument("--nvml_poll_ms", type=int, default=50)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="Primary device, e.g., cuda:0")
    p.add_argument("--dtype", default=None, choices=["fp16", "bf16", "fp32"], help="Preferred dtype (best effort)")
    p.add_argument("--no_vlm", action="store_true", help="Skip VLM scoring; end-to-end left as null")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg = Config.from_yaml(args.config)
    if args.dataset:
        cfg.dataset_csv = args.dataset

    ds = VQADataset(
        csv_path=cfg.dataset_csv,
        id2name_paths=cfg.id2name_paths,
        image_root=cfg.dataset_image_root,
        googlelandmark_root=cfg.dataset_google_root,
        start=cfg.dataset_start,
        end=cfg.dataset_end,
    )

    out_dir = args.out_dir or os.path.join(
        os.getcwd(), f"metrics_run_{datetime.now().strftime('%Y%m%d_%H%M')}"
    )
    os.makedirs(out_dir, exist_ok=True)
    samples_csv = os.path.join(out_dir, "metrics_samples.csv")
    summary_json = os.path.join(out_dir, "metrics_summary.json")

    # Devices
    device_img = resolve_device(cfg.image_device)
    device_rerank = resolve_device(cfg.bge_device)
    device_nli = resolve_device(cfg.nli_device)

    # Dtype note (best-effort informative; not enforced globally)
    dtype_note = args.dtype or "native"

    # Prepare NLI/VLM
    nli_choice = cfg.nli_models.get
    if nli_choice("deberta", False):
        nli_model_name = cfg.deberta_nli_model
    elif nli_choice("roberta", False):
        nli_model_name = cfg.roberta_nli_model
    elif nli_choice("deberta_v3", False):
        nli_model_name = cfg.deberta_v3_nli_model
    else:
        nli_model_name = cfg.deberta_nli_model
    nli_model, nli_tokenizer = load_nli_model(nli_model_name, device_nli)

    # Optional VLM (hook kept for future; skipped by default)
    vlm_model = vlm_processor = None
    if not args.no_vlm:
        try:
            vlm_model, vlm_processor = load_vlm_model(device_map=cfg.bge_device)
        except Exception:
            vlm_model = vlm_processor = None

    sink = MetricsSink()

    # Accuracy trackers
    max_k = getattr(cfg, "k_value", 10)
    base_k = [1, 3, 5, 10]
    k_values = sorted(set([k for k in base_k if k <= max_k] + [max_k]))
    img_doc_hits = {k: 0 for k in k_values}
    bge_doc_hits = {k: 0 for k in k_values}
    bge_sec_hits = {k: 0 for k in k_values}
    nli_doc_hits = {k: 0 for k in k_values}
    nli_sec_hits = {k: 0 for k in k_values}
    sample_total = 0
    vlm_total = 0
    vlm_correct = 0

    # NVML samplers per stage
    ps_img = PowerSampler(device=device_img, poll_ms=args.nvml_poll_ms)
    ps_rerank = PowerSampler(device=device_rerank, poll_ms=args.nvml_poll_ms)
    ps_nli = PowerSampler(device=device_nli, poll_ms=args.nvml_poll_ms)

    def stage_factory(sample_id: int, warmup: bool):
        def _factory(name: str):
            if name == "image_search":
                return stage_meter("image_search", sink, sample_id, device=device_img, power=ps_img, is_warmup=warmup)
            if name == "reranker":
                return stage_meter("reranker", sink, sample_id, device=device_rerank, power=ps_rerank, is_warmup=warmup)
            return stage_meter(name, sink, sample_id, device=0, power=None, is_warmup=warmup)
        return _factory

    for sample in ds:
        if not sample.image_paths:
            continue
        cfg.image_path = sample.image_paths[0]
        cfg.text_query = sample.question
        warmup = sample_total < args.warmup_steps
        sid = sample.row_idx

        try:
            img_results, top_sections, _, _ = search_rag_pipeline(
                cfg,
                stage_meter_factory=stage_factory(sid, warmup),
            )
        except Exception as e:
            print(f"[Row {sid}] 파이프라인 실패: {e}")
            continue
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

        # Ground truth titles/sections
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

        # Image search recall
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

            # NLI stage timing/energy
            with stage_meter("nli", sink, sid, device=device_nli, power=ps_nli, is_warmup=warmup):
                clusters, stats = cluster_sections_clique(
                    candidate_sections,
                    model=nli_model,
                    tokenizer=nli_tokenizer,
                    max_length=cfg.nli_max_length,
                    device=device_nli,
                    max_cluster_size=cfg.nli_max_cluster,
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

            top_cluster = clusters[0]["sections"] if clusters else []

            # Optional VLM for E2E (hook)
            if not args.no_vlm and vlm_model and vlm_processor:
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

    # Save per-sample CSV
    sink.to_csv(samples_csv)

    # Build summary JSON
    summary = {
        "stages": sink.summarize(args.pctl),
        "accuracy": {
            "image_search": {f"recall@{k}": img_doc_hits[k] for k in k_values},
            "reranker": {
                **{f"doc_recall@{k}": bge_doc_hits[k] for k in k_values},
                **{f"sec_recall@{k}": bge_sec_hits[k] for k in k_values},
                "MRR@10": None,
            },
            "nli": {
                **{f"doc_recall@{k}": nli_doc_hits[k] for k in k_values},
                **{f"sec_recall@{k}": nli_sec_hits[k] for k in k_values},
                "edge_precision": None,
                "edge_recall": None,
                "edge_f1": None,
            },
            "e2e_accuracy": (vlm_correct / vlm_total) if (vlm_total and not args.no_vlm) else None,
            "samples_total": sample_total,
        },
        "environment": {
            **env_info(),
            "dtype": dtype_note,
            "seed": args.seed,
            "pctl": args.pctl,
            "warmup_steps": args.warmup_steps,
            "nvml_poll_ms": args.nvml_poll_ms,
        },
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Console summary
    print("Saved sample metrics to:", samples_csv)
    print("Saved summary to:", summary_json)
    for st, s in summary["stages"].items():
        l = s["latency"]
        print(f"{st}: mean={l['mean']:.3f}s p50={l['p50']:.3f}s p90={l['p90']:.3f}s p99={l['p99']:.3f}s | energy_mean={s['energy_J']['mean']:.1f}J | vram_max={s['peak_vram_gb']['max']:.2f}GB")


if __name__ == "__main__":
    main()
