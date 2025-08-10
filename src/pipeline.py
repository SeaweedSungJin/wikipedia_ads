"""Core search pipeline implementation.

This function orchestrates the image retrieval and text ranking steps
for the RAG system.  The heavy models and indices are cached at the
module level so that repeated calls do not incur additional load time.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import math
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .utils import download_nltk_data, load_image, load_faiss_and_ids, load_kb_list

import torch

from .config import Config
from .embedding import encode_image
from .models import (
    get_device,
    load_image_model,
    load_text_encoder,
    load_jina_reranker,
    setup_cuda,
)
from .encoders import TextEncoder
from .segmenter import (
    Segmenter,
    SectionSegmenter,
    SentenceSegmenter,
    ParagraphSegmenter,
)
# Cached resources
_FAISS_INDEX = None
_KB_IDS = None
_KB_LIST = None

def _get_image_offset(
    faiss_vidx: int, doc_idx: int, doc_idx_starts: Dict[int, int]
) -> int:
    """Return the per-document offset for an image result."""

    # ``faiss_vidx`` indexes into the flat FAISS result list, while
    # ``doc_idx_starts`` gives the starting position of each document's
    # images inside that list.  The difference yields the offset of the
    # matched image within its source document.

    start_pos = doc_idx_starts.get(doc_idx)
    if start_pos is None:
        return -1
    return faiss_vidx - start_pos

def search_rag_pipeline(
    cfg: Config,
    text_encoder: TextEncoder | None = None,
    segmenter: Segmenter | None = None,
    return_time: bool = False,
    return_candidates: bool = False,
    use_all_sections: bool = False,
) -> (
    Tuple[List[dict], List[dict]]
    | Tuple[List[dict], List[dict], float]
    | Tuple[List[dict], List[dict], List[dict]]
    | Tuple[List[dict], List[dict], List[dict], float]
):
    """Execute the RAG search pipeline and return image and section results.

    If ``return_time`` is True, the elapsed time (in seconds) for the search
    after loading the knowledge base is also returned.
    """

    # Ensure required NLTK data is present
    download_nltk_data()

    # Initialise device and load heavy models once
    device = get_device()
    setup_cuda()

    print(f"Using device: {device}")
    image_model, image_processor = load_image_model(device_map=cfg.image_device)

    use_contriever = cfg.rerankers.get("contriever", False)
    use_jina = cfg.rerankers.get("jina_m0", False)
    use_colbert = cfg.rerankers.get("colbert", False)
    use_bge = cfg.rerankers.get("bge", False)

    colbert = None
    bge = None

    if text_encoder is None and use_contriever:
        text_encoder = load_text_encoder(cfg.text_encoder_model, device_map="auto")
    if use_colbert:
        from .models import load_colbert
        from .encoders import ColBERTEncoder

        c_model, c_tok = load_colbert(cfg.colbert_model, device_map="auto")
        colbert = ColBERTEncoder(c_model, c_tok)
    if use_bge:
        from .models import load_bge_reranker
        bge_model, bge_tok = load_bge_reranker(cfg.bge_model, device)
    if segmenter is None:
        if cfg.segment_level == "section":
            segmenter = SectionSegmenter()
        elif cfg.segment_level == "sentence":
            segmenter = SentenceSegmenter()
        else:  # paragraph level
            segmenter = ParagraphSegmenter(cfg.chunk_size)
            
    # Load FAISS index and KB data once and cache globally
    global _FAISS_INDEX, _KB_IDS, _KB_LIST
    if _FAISS_INDEX is None or _KB_IDS is None:
        _FAISS_INDEX, _KB_IDS = load_faiss_and_ids(cfg.base_path)
    if _KB_LIST is None:
        _KB_LIST = load_kb_list(cfg.kb_json_path)

    faiss_index, kb_ids = _FAISS_INDEX, _KB_IDS
    kb_list = _KB_LIST

    start_time = time.time()

    # Build a lookup table so we can convert a FAISS index position back
    # into a per-document offset later when scoring sections.    
    doc_idx_starts = {}
    for i, doc_idx in enumerate(kb_ids):
        if doc_idx not in doc_idx_starts:
            doc_idx_starts[doc_idx] = i

    # Encode the query image and search the FAISS index
    img = load_image(cfg.image_path)
    img_emb_np = encode_image(img, image_model, image_processor).numpy()
    # Optionally build a lookup of the first image index for each document
    doc_to_first_faiss_idx: Dict[int, int] = {}
    if cfg.first_image_only:
        for i, d_idx in enumerate(kb_ids):
            if d_idx not in doc_to_first_faiss_idx:
                doc_to_first_faiss_idx[d_idx] = i

    # Retrieve more candidates than needed so that filtered pages
    # (e.g., "list of" or "outline of" entries) can be skipped
    search_k = min(cfg.k_value * 20, faiss_index.ntotal)
    distances, indices = faiss_index.search(img_emb_np, k=search_k)

    # Collect top-K image search results
    top_k_image_results: List[dict] = []
    unique_doc_indices = set()
    doc_image_map: Dict[int, str] = {}

    limit = indices.shape[1]
    for i in range(limit):
        if len(unique_doc_indices) >= cfg.k_value:
            break
        faiss_vidx = indices[0][i]
        doc_idx = kb_ids[faiss_vidx]

        if doc_idx in unique_doc_indices:
            continue

        if cfg.first_image_only:
            first_idx = doc_to_first_faiss_idx.get(doc_idx)
            if first_idx is None or first_idx != faiss_vidx:
                continue

        if 0 <= doc_idx < len(kb_list):
            doc = kb_list[doc_idx]

            title = doc.get("title", "")
            if any(
                phrase in title.lower() for phrase in ["list of", "outline of", "index of"]
            ):
                continue

            offset = _get_image_offset(faiss_vidx, doc_idx, doc_idx_starts)
            if offset != -1 and offset < len(doc.get("image_reference_descriptions", [])):
                img_url = doc["image_urls"][offset]
                top_k_image_results.append(
                    {
                        "doc": doc,
                        "similarity": distances[0][i],
                        "description": doc["image_reference_descriptions"][offset],
                        "image_url": img_url,
                    }
                )
                unique_doc_indices.add(doc_idx)
                if doc_idx not in doc_image_map:
                    doc_image_map[doc_idx] = img_url

    fused_embeddings = []
    filtered_sections: List[dict] = []  # ensure defined for all code paths
    if use_contriever:
        if text_encoder is None:
            text_encoder = load_text_encoder(cfg.text_encoder_model, device_map="auto")

        # Encode the question text and any image descriptions
        query_emb = text_encoder.encode([cfg.text_query]).to(device)
        valid_descriptions = [
            r["description"]
            for r in top_k_image_results
            if r["description"] and not r["description"].isspace()
        ]
        if valid_descriptions:
            desc_embeddings = text_encoder.encode(valid_descriptions).to(device)
        else:
            desc_embeddings = torch.empty(0, query_emb.shape[1], device=device)

        # Combine query and description embeddings for each image result
        desc_idx = 0
        for res in top_k_image_results:
            if res["description"] and not res["description"].isspace():
                fused_emb = cfg.alpha * query_emb[0] + (1.0 - cfg.alpha) * desc_embeddings[desc_idx]
                desc_idx += 1
            else:
                fused_emb = query_emb[0]
            fused_embeddings.append(fused_emb.unsqueeze(0))

    # Collect candidate sections from documents appearing in image search
    all_sections_data: List[dict] = []
    unique_docs = [(i, kb_list[i]) for i in unique_doc_indices]
    for doc_idx, doc in unique_docs:
        segments = segmenter.get_segments(doc)
        img_url = doc_image_map.get(
            doc_idx,
            doc.get("image_urls", [None])[0] if doc.get("image_urls") else None,
        )
        for seg in segments:
            seg["image_url"] = img_url
        all_sections_data.extend(segments)

    if not all_sections_data:
        elapsed = time.time() - start_time
        if return_time and return_candidates:
            return top_k_image_results, [], [], elapsed
        if return_time:
            return top_k_image_results, [], elapsed
        if return_candidates:
            return top_k_image_results, [], []
        return top_k_image_results, []

    # Optional TF-IDF filtering before expensive embedding comparison
    filtered_sections = all_sections_data
    if cfg.use_tfidf_filter:
        print("TF-IDF 필터링 적용중...")
        texts = [d["section_text"] for d in all_sections_data]
        vectorizer = TfidfVectorizer().fit(texts + [cfg.text_query])
        sec_mat = vectorizer.transform(texts)
        qry_vec = vectorizer.transform([cfg.text_query])
        scores = cosine_similarity(qry_vec, sec_mat).flatten()
        keep_n = max(1, int(len(texts) * cfg.tfidf_ratio))
        top_idx = np.argsort(scores)[-keep_n:]
        filtered_sections = [all_sections_data[i] for i in top_idx]
    else:
        filtered_sections = all_sections_data

    if use_colbert:
        from .models import compute_late_interaction_similarity

        query_tokens = colbert.encode_tokens(cfg.text_query)
        if query_tokens.dim() == 2:
            query_tokens = query_tokens.unsqueeze(0)

        for sec in filtered_sections:
            cand_tokens = colbert.encode_tokens(sec["section_text"])
            if cand_tokens.dim() == 2:
                cand_tokens = cand_tokens.unsqueeze(0)
            score = compute_late_interaction_similarity(query_tokens, cand_tokens)[0].item()
            sec["similarity"] = score
        
    if use_contriever:
        # Embed all candidate section texts
        section_texts_list = [d["section_text"] for d in filtered_sections]
        section_embeddings = text_encoder.encode(section_texts_list).to(device)

        # Compare each section to the fused image/text embeddings
        for i, section_emb in enumerate(section_embeddings):
            title = filtered_sections[i]["source_title"]
            best_score = -1.0
            for j, img_result in enumerate(top_k_image_results):
                if img_result["doc"]["title"] == title:
                    fused_emb = fused_embeddings[j]
                    score = torch.nn.functional.cosine_similarity(section_emb.unsqueeze(0), fused_emb)[0].item()
                    if score > best_score:
                        best_score = score
            filtered_sections[i]["similarity"] = best_score

    if use_jina:
        # Load the cross-encoder reranker on the target device
        reranker = load_jina_reranker(device)
        # Prepare query/document pairs for cross-encoder scoring
        jina_inputs = [
            [cfg.text_query, s["section_text"]] for s in filtered_sections
        ]

        with torch.no_grad():
            scores = reranker.compute_score(
                jina_inputs, max_length=8192, doc_type="text"
            )

        for sec, score in zip(filtered_sections, scores):
            sec["similarity"] = float(score)

    if use_bge:
        from .models import load_bge_reranker

        model, tokenizer = load_bge_reranker(cfg.bge_model, device)
        pairs = [[cfg.text_query, s["section_text"]] for s in filtered_sections]
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            scores = model(**inputs, return_dict=True).logits.view(-1).cpu().float()
        for sec, score in zip(filtered_sections, scores.tolist()):
            sec["similarity"] = float(score)
            
    # Sort sections by similarity score
    sorted_sections = sorted(
        filtered_sections, key=lambda x: x.get("similarity", -1), reverse=True
    )

    if use_all_sections:
        top_m_sections = sorted_sections
        final_sections = sorted_sections
    else:
        # Only consider the top-M sections when computing confidence
        top_m_sections = sorted_sections[: cfg.m_value]

        if use_bge and top_m_sections:
            score_tensor = torch.tensor([s["similarity"] for s in top_m_sections])
            probs = torch.softmax(score_tensor, dim=0).tolist()
            for sec, prob in zip(top_m_sections, probs):
                sec["prob"] = float(prob)
            entropy = -sum(p * math.log(p + 1e-12) for p in probs)
            norm = math.log(len(top_m_sections)) if len(top_m_sections) > 1 else 1.0
            confidence = 1 - entropy / norm
            if confidence >= cfg.bge_conf_threshold:
                keep_n = 1
            else:
                keep_n = max(1, math.ceil((1 - confidence) * len(top_m_sections)))
            final_sections = top_m_sections[:keep_n]
        else:
            final_sections = top_m_sections

    elapsed = time.time() - start_time

    # Release any cached CUDA memory to avoid OOM when processing many queries
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    if return_time and return_candidates:
        return top_k_image_results, top_m_sections, final_sections, elapsed
    if return_time:
        return top_k_image_results, final_sections, elapsed
    if return_candidates:
        return top_k_image_results, top_m_sections, final_sections
    return top_k_image_results, final_sections