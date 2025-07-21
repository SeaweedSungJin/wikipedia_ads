"""Core search pipeline implementation.

This function orchestrates the image retrieval and text ranking steps
for the RAG system.  The heavy models and indices are cached at the
module level so that repeated calls do not incur additional load time.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

EXCLUDED_SECTIONS = {
    "references",
    "external links",
    "see also",
    "notes",
    "bibliography",
    "further reading",
}
# These sections typically contain meta information rather than
# descriptive text, so we ignore them during ranking.


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
) -> Tuple[List[dict], List[dict]]:
    """Execute the RAG search pipeline and return image and section results."""

    # Ensure required NLTK data is present
    download_nltk_data()

    # Initialise device and load heavy models once
    device = get_device()
    setup_cuda()

    print(f"Using device: {device}")
    image_model, image_processor = load_image_model(device_map=cfg.image_device)

    if text_encoder is None:
        text_encoder = load_text_encoder(cfg.text_encoder_model, device_map="auto")
    if segmenter is None:
        if cfg.segment_level == "section":
            segmenter = SectionSegmenter()
        elif cfg.segment_level == "sentence":
            segmenter = SentenceSegmenter()
        else:
            segmenter = ParagraphSegmenter()

    # Load FAISS index and KB data once and cache globally
    global _FAISS_INDEX, _KB_IDS, _KB_LIST
    if _FAISS_INDEX is None or _KB_IDS is None:
        _FAISS_INDEX, _KB_IDS = load_faiss_and_ids(cfg.base_path)
    if _KB_LIST is None:
        _KB_LIST = load_kb_list(cfg.kb_json_path)

    faiss_index, kb_ids = _FAISS_INDEX, _KB_IDS
    kb_list = _KB_LIST

    # Build a lookup table so we can convert a FAISS index position back
    # into a per-document offset later when scoring sections.    
    doc_idx_starts = {}
    for i, doc_idx in enumerate(kb_ids):
        if doc_idx not in doc_idx_starts:
            doc_idx_starts[doc_idx] = i

    # Encode the query image and search the FAISS index
    img = load_image(cfg.image_path)
    img_emb_np = encode_image(img, image_model, image_processor).numpy()
    distances, indices = faiss_index.search(img_emb_np, k=cfg.k_value)

    # Collect top-K image search results
    top_k_image_results: List[dict] = []
    unique_doc_indices = set()
    for i in range(cfg.k_value):
        faiss_vidx = indices[0][i]
        doc_idx = kb_ids[faiss_vidx]
        if 0 <= doc_idx < len(kb_list):
            doc = kb_list[doc_idx]
            offset = _get_image_offset(faiss_vidx, doc_idx, doc_idx_starts)
            if offset != -1 and offset < len(doc.get("image_reference_descriptions", [])):
                top_k_image_results.append(
                    {
                        "doc": doc,
                        "similarity": distances[0][i],
                        "description": doc["image_reference_descriptions"][offset],
                        "image_url": doc["image_urls"][offset],
                    }
                )
                unique_doc_indices.add(doc_idx)

    # Encode the question text and any image descriptions
    query_emb = text_encoder.encode([cfg.text_query]).to(device)
    valid_descriptions = [r["description"] for r in top_k_image_results if r["description"] and not r["description"].isspace()]
    if valid_descriptions:
        desc_embeddings = text_encoder.encode(valid_descriptions).to(device)
    else:
        desc_embeddings = torch.empty(0, query_emb.shape[1], device=device)

    # Combine query and description embeddings for each image result
    fused_embeddings = []
    desc_idx = 0
    for res in top_k_image_results:
        if res["description"] and not res["description"].isspace():
            fused_emb = cfg.alpha * query_emb[0] + (1.0 - cfg.alpha) * desc_embeddings[desc_idx]
            desc_idx += 1
        else:
            fused_emb = query_emb[0]
        fused_embeddings.append(fused_emb.unsqueeze(0))

    # Collect candidate sections from documents appearing in image search
    all_sections_data = []
    unique_docs = [kb_list[i] for i in unique_doc_indices]
    for doc in unique_docs:
        title = doc.get("title", "N/A")
        if any(phrase in title.lower() for phrase in ["list of", "outline of", "index of"]):
            continue
        segments = segmenter.get_segments(doc)
        all_sections_data.extend(segments)

    if not all_sections_data:
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
        
    # If a cross-encoder reranker is specified we bypass the text encoder
    # similarity scoring and instead compute scores directly with the
    # reranker model (e.g. "jinaai/jina-reranker-m0").
    if cfg.ranker_model == "jinaai/jina-reranker-m0":
        reranker = load_jina_reranker(device_map="auto")
        inputs = [[cfg.text_query, s["section_text"]] for s in filtered_sections]
        with torch.no_grad():
            scores = reranker.compute_score(inputs, max_length=8192, doc_type="text")
        for sec, score in zip(filtered_sections, scores):
            sec["similarity"] = float(score)
    else:
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

    # Sort sections by similarity score and keep the top m results
    sorted_sections = sorted(
        filtered_sections, key=lambda x: x.get("similarity", -1), reverse=True
    )
    result = top_k_image_results, sorted_sections[: cfg.m_value]

    # Release any cached CUDA memory to avoid OOM when processing many queries
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return result
