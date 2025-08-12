from __future__ import annotations

import time
from typing import List, Tuple, Dict

import torch
import faiss
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config import Config
from src.dataloader import VQADataset
from src.utils import (
    load_faiss_and_ids,
    load_kb_list,
    download_nltk_data,
)
from src.pipeline import search_rag_pipeline
from src.models import load_text_encoder, load_bge_reranker


def generate_hypothesis(question: str, tokenizer, model, device) -> str:
    """Generate a hypothetical answer for HyDE retrieval."""
    inputs = tokenizer(question, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=64)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def encode_kb_sections(kb_list: List[dict], text_encoder) -> Tuple[faiss.IndexFlatIP, List[Tuple[int, int]]]:
    """Build a FAISS index for all KB sections on the fly.

    This is used when a precomputed text index is not available. All sections are
    encoded once at start using the same text encoder as HyDE."""

    texts: List[str] = []
    ids: List[Tuple[int, int]] = []
    for doc_idx, doc in enumerate(kb_list):
        for sec_idx, sec in enumerate(doc.get("sections", [])):
            texts.append(sec.get("text", ""))
            ids.append((doc_idx, sec_idx))

    if not texts:
        return faiss.IndexFlatIP(0), ids

    emb = text_encoder.encode(texts).cpu().numpy()
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index, ids


def search_text_sections(
    query: str,
    k: int,
    text_encoder,
    text_index,
    text_ids: List[Tuple[int, int]],
    kb_list: List[dict],
) -> List[dict]:
    """Retrieve top-k sections from a FAISS index built at runtime."""

    q_emb = text_encoder.encode([query]).cpu().numpy()
    k = min(k, text_index.ntotal)
    distances, indices = text_index.search(q_emb, k)
    results: List[dict] = []
    for idx in indices[0]:
        doc_idx, sec_idx = text_ids[idx]
        if 0 <= doc_idx < len(kb_list):
            doc = kb_list[doc_idx]
            sections = doc.get("sections", [])
            if 0 <= sec_idx < len(sections):
                sec = sections[sec_idx]
                results.append(
                    {
                        "source_title": doc.get("title", ""),
                        "section_title": sec.get("title", ""),
                        "section_text": sec.get("text", ""),
                        "section_idx": sec_idx,
                    }
                )
    return results


def rerank_with_bge(
    query: str, sections: List[dict], bge_model_name: str, device
) -> List[dict]:
    """Score sections with the BGE reranker and sort by similarity."""

    model, tokenizer = load_bge_reranker(bge_model_name, device)
    pairs = [[query, s["section_text"]] for s in sections]
    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        scores = model(**inputs, return_dict=True).logits.view(-1).cpu().tolist()
    for sec, score in zip(sections, scores):
        sec["similarity"] = float(score)
    return sorted(sections, key=lambda x: x.get("similarity", -1), reverse=True)


def run_hyde_dataset(cfg: Config) -> None:
    download_nltk_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preload resources
    _, _ = load_faiss_and_ids(cfg.base_path)  # warm image index if needed
    kb_list = load_kb_list(cfg.kb_json_path)
    text_encoder = load_text_encoder(cfg.text_encoder_model, device_map="auto")
    print("텍스트 인덱스 없음: KB 섹션들을 즉석에서 인코딩합니다...")
    text_index, text_ids = encode_kb_sections(kb_list, text_encoder)
    hyde_tokenizer = AutoTokenizer.from_pretrained(cfg.hyde_model)
    hyde_model = AutoModelForSeq2SeqLM.from_pretrained(cfg.hyde_model).to(device)

    dataset = VQADataset(
        csv_path=cfg.dataset_csv,
        id2name_path=cfg.id2name_json,
        image_root=cfg.dataset_image_root,
        googlelandmark_root=cfg.dataset_google_root,
        start=cfg.dataset_start,
        end=cfg.dataset_end,
    )
    print(f"데이터셋 평가 범위: start={cfg.dataset_start}, end={cfg.dataset_end}.")

    doc_hits = {1: 0, 3: 0, 5: 0, 10: 0}
    sec_hits = {1: 0, 3: 0, 5: 0, 10: 0}
    sample_total = 0
    image_time_total = 0.0
    hyde_time_total = 0.0
    rerank_time_total = 0.0

    k_half = cfg.k_value // 2

    for sample in dataset:
        if not sample.image_paths:
            continue
        sample_total += 1
        cfg.image_path = sample.image_paths[0]
        cfg.text_query = sample.question

        print(f"\n=== Row {sample.row_idx} ===")
        print(f"질문: {sample.question}")

        # Path A: image retrieval
        t0 = time.time()
        img_results, top_sections, _ = search_rag_pipeline(
            cfg, return_time=False, return_candidates=True
        )
        image_time = time.time() - t0
        image_time_total += image_time

        print("-- Top-k 문서 후보 (이미지 검색 기반) --")
        for i, res in enumerate(img_results, 1):
            title = res["doc"].get("title", "")
            print(f"  {i}. {title}")

        img_sections = top_sections[:k_half]

        # Path B: HyDE text retrieval
        t1 = time.time()
        hypo = generate_hypothesis(sample.question, hyde_tokenizer, hyde_model, device)
        hyde_sections = search_text_sections(
            hypo, k_half, text_encoder, text_index, text_ids, kb_list
        )
        hyde_time = time.time() - t1
        hyde_time_total += hyde_time

        hyde_docs = []
        for sec in hyde_sections:
            title = sec.get("source_title", "")
            if title not in hyde_docs:
                hyde_docs.append(title)
        print("-- HyDE 기반 문서 후보 --")
        for i, title in enumerate(hyde_docs, 1):
            print(f"  {i}. {title}")

        # Fusion
        fused: Dict[Tuple[str, int], dict] = {}
        for sec in img_sections + hyde_sections:
            key = (sec.get("source_title"), sec.get("section_idx"))
            fused[key] = sec
        candidates = list(fused.values())

        # BGE rerank
        t2 = time.time()
        ranked = rerank_with_bge(sample.question, candidates, cfg.bge_model, device)
        rerank_time = time.time() - t2
        rerank_time_total += rerank_time

        print("-- 최종 섹션 랭킹 --")
        for i, sec in enumerate(ranked[: cfg.m_value], 1):
            stitle = sec.get("source_title", "")
            sectitle = sec.get("section_title", "")
            sidx = sec.get("section_idx")
            score = sec.get("similarity", 0.0)
            print(f"  {i}. {stitle} / {sectitle} (#{sidx}) score={score:.3f}")

        # Evaluation
        gt_idx_raw = sample.metadata.get("evidence_section_id")
        try:
            gt_idx = int(gt_idx_raw) if gt_idx_raw is not None else None
        except (TypeError, ValueError):
            gt_idx = None

        for k in (1, 3, 5, 10):
            topk = ranked[:k]
            if any(sec.get("source_title") == sample.wikipedia_title for sec in topk):
                doc_hits[k] += 1
            if any(
                sec.get("source_title") == sample.wikipedia_title
                and sec.get("section_idx") == gt_idx
                for sec in topk
            ):
                sec_hits[k] += 1

        print(
            f"정답 문서: {sample.wikipedia_title} | 정답 섹션 idx: {gt_idx} | 정답: {sample.answer}"
        )
        doc_match = any(
            sec.get("source_title") == sample.wikipedia_title
            for sec in ranked[: cfg.m_value]
        )
        sec_match = any(
            sec.get("source_title") == sample.wikipedia_title
            and sec.get("section_idx") == gt_idx
            for sec in ranked[: cfg.m_value]
        )
        print(
            f"선택된 섹션에 정답 문서 포함: {doc_match} | 정답 섹션 포함: {sec_match}"
        )
        print(
            f"경로A시간: {image_time:.2f}s | 경로B시간: {hyde_time:.2f}s | 재랭크시간: {rerank_time:.2f}s"
        )

    # Summary
    print("\n-- 평가 요약 --")
    for k in (1, 3, 5, 10):
        print(
            f"BGE+HyDE Recall@{k}: 문서 {doc_hits[k]}/{sample_total}, 섹션 {sec_hits[k]}/{sample_total}"
        )
    print(
        f"시간 합계: 이미지 {image_time_total:.2f}s | HyDE {hyde_time_total:.2f}s | BGE {rerank_time_total:.2f}s"
    )


if __name__ == "__main__":
    cfg = Config.from_yaml("config.yaml")
    if not cfg.dataset_csv or not cfg.id2name_json:
        raise ValueError("dataset_csv and id2name_json must be set in config")
    run_hyde_dataset(cfg)