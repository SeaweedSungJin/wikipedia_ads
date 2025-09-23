"""Core search pipeline implementation.

This function orchestrates the image retrieval and text ranking steps
for the RAG system.  The heavy models and indices are cached at the
module level so that repeated calls do not incur additional load time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Callable, ContextManager, Optional
import math
import time
import numpy as np
from .utils import download_nltk_data, load_image, load_faiss_and_ids, normalize_title, load_kb
from tqdm import tqdm
import torch

from .config import Config
from .embedding import encode_image
from .models import (
    get_device,
    load_image_model,
    load_text_encoder,
    setup_cuda,
    load_mpnet_biencoder,
)
from .rerankers import BGEReranker, ElectraReranker, JinaReranker
from .qformer_utils import (
    load_qformer_resources,
    encode_query_multimodal,
    encode_sections_text,
    score_sections,
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
_URL_TO_IDX = None
_BGE_CACHE = None

_SRC_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_ROOT.parent

def _get_bge(model_name: str, device):
    """Load the BGE reranker once and cache it globally."""
    global _BGE_CACHE
    if _BGE_CACHE is None:
        _BGE_CACHE = load_bge_reranker(model_name, device)
    return _BGE_CACHE

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

@torch.no_grad()
def search_rag_pipeline(
    cfg: Config,
    text_encoder: TextEncoder | None = None,
    segmenter: Segmenter | None = None,
    use_all_sections: bool = False,
    stage_meter_factory: Optional[Callable[[str], ContextManager]] = None,
) -> Tuple[List[dict], List[dict], List[dict], float]:
    """Execute the RAG search pipeline.

    Returns a tuple of (image_results, top_sections, final_sections, elapsed_time).
    """

    # Ensure required NLTK data is present
    download_nltk_data()

    # Initialise device and load heavy models once
    device = get_device()
    setup_cuda()

    print(f"Using device: {device}")
    image_model, image_processor = load_image_model(device_map=cfg.image_device)

    use_contriever = cfg.rerankers.get("contriever", False)
    use_jina_tiny = cfg.rerankers.get("jina_tiny", False)
    use_jina_turbo = cfg.rerankers.get("jina_turbo", False)
    use_jina = use_jina_tiny or use_jina_turbo
    use_bge = cfg.rerankers.get("bge", False)
    use_electra = cfg.rerankers.get("electra", False)
    use_mpnet = cfg.rerankers.get("mpnet", False)
    use_qformer = cfg.rerankers.get("q-former", False) or cfg.rerankers.get("qformer", False)

    if text_encoder is None and use_contriever:
        text_encoder = load_text_encoder(cfg.text_encoder_model, device_map="auto")
    if segmenter is None:
        if cfg.segment_level == "section":
            segmenter = SectionSegmenter()
        elif cfg.segment_level == "sentence":
            segmenter = SentenceSegmenter()
        else:  # paragraph level
            segmenter = ParagraphSegmenter(cfg.chunk_size)
            
    # Load FAISS index, mapping IDs and KB list once and cache globally
    global _FAISS_INDEX, _KB_IDS, _KB_LIST, _URL_TO_IDX
    if _KB_LIST is None or _URL_TO_IDX is None:
        _KB_LIST, _URL_TO_IDX = load_kb(cfg.kb_json_path)

    if _FAISS_INDEX is None or _KB_IDS is None:
        _FAISS_INDEX, _KB_IDS = load_faiss_and_ids(
            cfg.base_path, _KB_LIST, _URL_TO_IDX
        )

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
    if stage_meter_factory:
        img_stage = stage_meter_factory("image_search")
    else:
        from contextlib import nullcontext
        img_stage = nullcontext()
    with img_stage:
        img = load_image(cfg.image_path)
        img_emb = encode_image(img, image_model, image_processor)
        img_emb_np = img_emb.numpy().astype("float32")
        # Re-normalise in case of numeric drift after conversion
        img_emb_np /= np.linalg.norm(img_emb_np, axis=1, keepdims=True)

        # Retrieve more candidates than needed so that filtered pages
        # (e.g., "list of" or "outline of" entries) can be skipped
        expand = cfg.search_expand if cfg.search_expand is not None else cfg.k_value * 20
        search_k = min(expand, faiss_index.ntotal)
        distances, indices = faiss_index.search(img_emb_np, k=search_k)

    # Collect top-K image search results using the evaluate_pipeline logic
    top_k_image_results: List[dict] = []
    unique_doc_indices = set()

    limit = indices.shape[1]
    for i in range(limit):
        if len(unique_doc_indices) >= cfg.k_value:
            break
        faiss_vidx = int(indices[0][i])
        doc_idx = int(kb_ids[faiss_vidx])

        if doc_idx in unique_doc_indices:
            continue

        if not (0 <= doc_idx < len(kb_list)):
            continue

        doc = kb_list[doc_idx]
        title_norm = normalize_title(doc.get("title", ""))
        if any(
            phrase in title_norm for phrase in ["list of", "outline of", "index of"]
        ):
            continue

        offset = _get_image_offset(faiss_vidx, doc_idx, doc_idx_starts)
        img_urls = doc.get("image_urls", [])
        img_url = img_urls[offset] if 0 <= offset < len(img_urls) else None
        descriptions = doc.get("image_reference_descriptions", [])
        description = (
            descriptions[offset] if 0 <= offset < len(descriptions) else ""
        )

        top_k_image_results.append(
            {
                "doc": doc,
                "similarity": float(distances[0][i]),
                "description": description,
                "image_url": img_url,
            }
        )
        unique_doc_indices.add(doc_idx)

    fused_embeddings = []
    filtered_sections: List[dict] = []  # ensure defined for all code paths

    # Precompute per-document image score normalization using softmax across retrieved docs
    if top_k_image_results:
        img_vals = np.array([r.get("similarity", 0.0) for r in top_k_image_results], dtype=np.float64)
        # Softmax with temperature
        t_img = float(getattr(cfg, "rank_img_softmax_temp", 1.0))
        if t_img <= 0:
            t_img = 1.0
        x = img_vals / t_img
        x = x - np.max(x)  # for numerical stability
        exps = np.exp(x)
        denom = exps.sum()
        if denom <= 0 or not np.isfinite(denom):
            probs = np.full_like(img_vals, 1.0 / max(1, len(img_vals)))
        else:
            probs = exps / denom
        for res, p in zip(top_k_image_results, probs.tolist()):
            res["img_norm"] = float(p)
        # Map doc title -> img_norm for quick lookup when scoring sections
        doc_img_norm = {res["doc"].get("title", ""): res["img_norm"] for res in top_k_image_results}
    else:
        doc_img_norm = {}

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
        img_url = doc.get("image_urls", [None])[0] if doc.get("image_urls") else None
        for seg in segments:
            seg["image_url"] = img_url
        all_sections_data.extend(segments)

    if not all_sections_data:
        elapsed = time.time() - start_time
        return top_k_image_results, [], [], elapsed

    filtered_sections = all_sections_data

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

    # Reranker stage wrapper (encompasses any enabled reranker)
    def _apply_rerankers() -> None:
        nonlocal filtered_sections
        if not filtered_sections:
            return
        if use_qformer and (use_jina or use_bge or use_electra or use_mpnet):
            raise ValueError(
                "Q-Former reranker currently cannot be combined with other rerankers."
            )

        if use_bge:
            bge_dev = (
                f"cuda:{cfg.bge_device}" if isinstance(cfg.bge_device, int) else cfg.bge_device
            )
            if not torch.cuda.is_available() and isinstance(bge_dev, str) and "cuda" in bge_dev:
                bge_dev = "cpu"
            reranker = BGEReranker(
                cfg.bge_model,
                device=cfg.bge_device,
                max_length=cfg.bge_max_length,
                batch_size=cfg.bge_batch_size,
            )
            print(
                f"BGE Reranking {len(filtered_sections)} sections in batches of {cfg.bge_batch_size}..."
            )
            scores = reranker.score(
                cfg.text_query, [s["section_text"] for s in filtered_sections]
            )
            for sec, score in zip(filtered_sections, scores):
                sec["similarity"] = float(score)
                sec["rerank_score"] = float(score)

        if use_electra:
            ce_dev = (
                f"cuda:{cfg.bge_device}" if isinstance(cfg.bge_device, int) else cfg.bge_device
            )
            if not torch.cuda.is_available() and isinstance(ce_dev, str) and "cuda" in ce_dev:
                ce_dev = "cpu"
            reranker = ElectraReranker(
                cfg.electra_model,
                device=cfg.bge_device,
                max_length=cfg.bge_max_length,
                batch_size=cfg.electra_batch_size,
            )
            print(
                f"Electra Reranking {len(filtered_sections)} sections in batches of {cfg.electra_batch_size}..."
            )
            scores = reranker.score(
                cfg.text_query, [s["section_text"] for s in filtered_sections]
            )
            for sec, score in zip(filtered_sections, scores):
                sec["similarity"] = float(score)
                sec["rerank_score"] = float(score)

        if use_mpnet:
            mpnet_dev = (
                f"cuda:{cfg.bge_device}" if isinstance(cfg.bge_device, int) else cfg.bge_device
            )
            if not torch.cuda.is_available() and isinstance(mpnet_dev, str) and "cuda" in mpnet_dev:
                mpnet_dev = "cpu"
            model = load_mpnet_biencoder(cfg.mpnet_model, mpnet_dev)
            section_texts_list = [d["section_text"] for d in filtered_sections]
            query_emb = model.encode(
                cfg.text_query,
                convert_to_tensor=True,
                device=mpnet_dev,
                normalize_embeddings=True,
            )
            sec_embs = model.encode(
                section_texts_list,
                batch_size=256,
                convert_to_tensor=True,
                device=mpnet_dev,
                normalize_embeddings=True,
            )
            scores = torch.nn.functional.cosine_similarity(
                sec_embs, query_emb.unsqueeze(0)
            ).cpu().tolist()
            for sec, score in zip(filtered_sections, scores):
                sec["similarity"] = float(score)
                sec["rerank_score"] = float(score)

        if use_jina:
            model_name = (
                "jinaai/jina-reranker-v1-turbo-en"
                if use_jina_turbo
                else "jinaai/jina-reranker-v1-tiny-en"
            )
            reranker = JinaReranker(model_name=model_name, device=cfg.bge_device)
            scores = reranker.score(
                cfg.text_query, [s["section_text"] for s in filtered_sections]
            )
            for sec, score in zip(filtered_sections, scores):
                sec["similarity"] = float(score)
                sec["rerank_score"] = float(score)

        if use_qformer:
            batch_size = int(getattr(cfg, "qformer_section_batch_size", 32))
            token_keep = int(getattr(cfg, "qformer_text_token_count", 32))
            max_txt_len = int(getattr(cfg, "qformer_max_text_length", 256))
            ckpt_cfg = getattr(cfg, "qformer_ckpt", "datasets/reranker.pth")
            ckpt_path = Path(ckpt_cfg)
            if not ckpt_path.is_absolute():
                ckpt_path = _REPO_ROOT / ckpt_cfg
            resources = load_qformer_resources(
                ckpt_path,
                device=getattr(cfg, "qformer_device", 0),
                max_text_length=max_txt_len,
            )
            try:
                image_tensor = resources.vis_processor(img)
            except Exception:
                image_tensor = image_processor(img, return_tensors="pt").pixel_values[0]
            fusion_tokens = encode_query_multimodal(
                resources.model,
                image_tensor.unsqueeze(0),
                cfg.text_query,
            )
            processed_texts: List[str] = []
            for sec in filtered_sections:
                doc_title = sec.get("source_title", "") or ""
                sec_title = sec.get("section_title", "") or ""
                header = (
                    f"# Wiki Article: {doc_title}\n"
                    f"## Section Title: {sec_title}\n"
                )
                combined_text = f"{header}{sec.get('section_text', '')}"
                if resources.text_processor is not None:
                    try:
                        processed = resources.text_processor(combined_text)
                    except Exception:
                        processed = combined_text
                else:
                    processed = combined_text
                processed_texts.append(processed)

            if processed_texts:
                cls_embs, token_embs, token_masks = encode_sections_text(
                    resources.model,
                    processed_texts,
                    batch_size=batch_size,
                    token_keep=token_keep,
                )
                scores = score_sections(fusion_tokens, token_embs, cls_embs, token_masks)
                cls_scores = scores.cls_max.cpu().tolist()
                maxsim_scores = scores.maxsim.cpu().tolist()
                logsumexp_scores = scores.logsumexp.cpu().tolist()
                for sec, cls_score, maxsim_score, logsum_score in zip(
                    filtered_sections, cls_scores, maxsim_scores, logsumexp_scores
                ):
                    doc_title = sec.get("source_title", "")
                    sec["rerank_score"] = float(cls_score)
                    sec["qformer_cls_score"] = float(cls_score)
                    sec["qformer_maxsim_score"] = float(maxsim_score)
                    sec["qformer_logsumexp_score"] = float(logsum_score)
                    sec["vision_score"] = float(doc_img_norm.get(doc_title, 0.0))
                    sec["similarity"] = float(cls_score)

    use_any_reranker = use_jina or use_bge or use_electra or use_mpnet or use_qformer
    if stage_meter_factory and use_any_reranker:
        with stage_meter_factory("reranker"):
            _apply_rerankers()
    else:
        _apply_rerankers()
    # Fuse reranker and image scores for section ranking
    # Only applies if a reranker provided similarity scores
    if filtered_sections:
        if any("rerank_score" in s for s in filtered_sections):
            rer_vals = np.array([s.get("rerank_score", 0.0) for s in filtered_sections], dtype=np.float64)
            # Sigmoid normalization with temperature for reranker scores
            t_txt = float(getattr(cfg, "rank_text_temp", 2.0))
            if t_txt <= 0:
                t_txt = 1.0
            rer_norms = (1.0 / (1.0 + np.exp(-(rer_vals / t_txt)))).tolist()
            if use_qformer:
                w_img = float(getattr(cfg, "qformer_doc_weight", 0.5))
                w_rer = float(getattr(cfg, "qformer_text_weight", 0.5))
            else:
                w_img = getattr(cfg, "rank_img_weight", 0.3)
                w_rer = getattr(cfg, "rank_rerank_weight", 0.7)
            total_w = w_img + w_rer
            if total_w <= 0:
                w_img = w_rer = 0.5
            else:
                w_img /= total_w
                w_rer /= total_w
            if use_qformer:
                print(
                    f"Q-Former reranking {len(filtered_sections)} sections "
                    f"(doc_weight={w_img:.2f}, text_weight={w_rer:.2f})"
                )
            for sec, rn in zip(filtered_sections, rer_norms):
                sec["rerank_norm"] = float(rn)
                img_norm = float(doc_img_norm.get(sec.get("source_title", ""), 0.0))
                sec["img_norm"] = img_norm
                combined = w_img * img_norm + w_rer * rn
                sec["combined_score"] = float(combined)
                if use_qformer:
                    sec["vision_score"] = img_norm
                    sec["qformer_cls_norm"] = float(rn)
                    sec["similarity"] = float(combined)
            sort_key = lambda x: x.get("combined_score", x.get("similarity", -1))
        else:
            sort_key = lambda x: x.get("similarity", -1)
    else:
        sort_key = lambda x: x.get("similarity", -1)

    # Sort sections by fused score (if available), else by similarity
    sorted_sections = sorted(filtered_sections, key=sort_key, reverse=True)

    if use_all_sections:
        top_m_sections = sorted_sections
        final_sections = sorted_sections
    else:
        # Only consider the top-M sections when computing confidence
        top_m_sections = sorted_sections[: cfg.m_value]

        if (use_bge or use_electra or use_mpnet or use_qformer) and top_m_sections:
            score_tensor = torch.tensor([s.get("similarity", 0.0) for s in top_m_sections])
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

    return top_k_image_results, top_m_sections, final_sections, elapsed
