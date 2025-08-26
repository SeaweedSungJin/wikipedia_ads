"""Helper utilities for loading data and resources."""
import json
import os
import pickle
from io import BytesIO
from typing import Iterable, List, Tuple, Optional
import re
from urllib.parse import unquote
import faiss
import nltk
import requests
import torch
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
import numpy as np

# Simple User-Agent compliant with Wikimedia policy
USER_AGENT = "wikipedia_ads/1.0 (https://example.com/contact)"

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def download_nltk_data() -> None:
    """Ensure NLTK punkt tokenizer data is available."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def load_image(path: str) -> Image.Image:
    """Load an image from a local path or an HTTP URL.

    Any failures result in a blank RGB image so the pipeline can
    continue processing without crashing.
    """

    try:
        if path.startswith("http"):
            headers = {"User-Agent": USER_AGENT}
            resp = requests.get(path, stream=True, timeout=10, headers=headers)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content))
        else:
            img = Image.open(path)
        return img.convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, requests.RequestException) as e:
        print(f"이미지 로딩 실패 ({path}): {e}")
        return Image.new("RGB", (224, 224), color=0)


def load_faiss_and_ids(base_path: str, kb_json_path: str) -> Tuple[faiss.Index, np.ndarray, List[dict]]:
    """Load FAISS index, kb_ids mapping and the KB list.

    This mirrors the robust loader used by ``evaluate_pipeline.py`` so that
    the main RAG pipeline and evaluation scripts share identical logic.
    """

    index_path = os.path.join(base_path, "kb_index.faiss")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    print("FAISS 인덱스 로딩중...")
    index = faiss.read_index(index_path)
    ntotal = index.ntotal

    # Load KB list once so that URL based mappings can be resolved.
    kb_list = load_kb_list(kb_json_path)
    url_to_idx = {}
    for i, doc in enumerate(kb_list):
        u = doc.get("url") or doc.get("wikipedia_url") or doc.get("source_url") or ""
        if u and u not in url_to_idx:
            url_to_idx[u] = i

    kb_ids: Optional[np.ndarray] = None
    tried: list[str] = []

    # Attempt to load from common sidecar NPY names first
    for cand in ["kb_ids.npy", "faiss_ids.npy", "doc_ids.npy"]:
        path = os.path.join(base_path, cand)
        tried.append(path)
        if os.path.exists(path):
            arr = np.load(path)
            if arr.ndim != 1 or len(arr) != ntotal:
                raise ValueError(f"{path} shape {arr.shape} incompatible with FAISS ntotal {ntotal}")
            kb_ids = arr.astype(np.int64)
            print(f"[INFO] Loaded kb_ids from {path} (len={len(kb_ids):,})")
            break

    # Fallback to legacy PKL mapping
    if kb_ids is None:
        pkl_path = os.path.join(base_path, "kb_index_ids.pkl")
        tried.append(pkl_path)
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                mapping = pickle.load(f)
            if isinstance(mapping, (list, np.ndarray)):
                mapping = np.array(mapping)
                if mapping.ndim != 1 or len(mapping) != ntotal:
                    raise ValueError(f"{pkl_path} length {len(mapping)} incompatible with FAISS ntotal {ntotal}")
                kb_ids = mapping.astype(np.int64)
                print(f"[INFO] Loaded kb_ids from array-like PKL (len={len(kb_ids):,})")
            elif isinstance(mapping, dict):
                kb_ids = np.empty(ntotal, dtype=np.int64)
                kb_ids.fill(-1)
                bad = 0
                for k, v in mapping.items():
                    try:
                        k = int(k)
                    except Exception:
                        continue
                    if isinstance(v, (int, np.integer)):
                        kb_ids[k] = int(v)
                    else:
                        idx = url_to_idx.get(str(v), -1)
                        if idx == -1:
                            norm_v = normalize_url_to_title(str(v))
                            candidates = [i for i, d in enumerate(kb_list) if normalize_title(d.get("title")) == norm_v]
                            idx = candidates[0] if candidates else -1
                        kb_ids[k] = idx
                    if kb_ids[k] == -1:
                        bad += 1
                if bad:
                    print(f"[WARN] {bad} entries in mapping PKL could not be matched to KB docs")
                if (kb_ids == -1).any():
                    raise ValueError("kb_ids contains -1 (unmatched). Provide a consistent mapping.")
                print(f"[INFO] Built kb_ids from dict PKL (len={len(kb_ids):,})")
            else:
                raise ValueError(f"Unsupported mapping PKL type: {type(mapping)}")

    if kb_ids is None:
        raise FileNotFoundError("Could not load kb_ids. Tried: " + ", ".join(tried))
    if len(kb_ids) != ntotal:
        raise AssertionError(f"kb_ids length {len(kb_ids)} != index.ntotal {ntotal}")
    if kb_ids.min() < 0 or kb_ids.max() >= len(kb_list):
        raise ValueError("kb_ids contains out-of-range doc indices for the given KB list.")

    return index, kb_ids, kb_list

def load_kb_list(json_path: str) -> List[dict]:
    """Load a knowledge base stored as JSON or JSONL.

    The file may contain either a list of documents, a mapping of URL to
    document dictionaries, or be line-delimited JSON (JSONL).  Every document
    is guaranteed to have a ``title`` field after loading.
    """

    print("지식베이스 JSON 로딩중...")
    with open(json_path, "r", encoding="utf-8") as f:
        # Peek at the first non-whitespace character to decide how to parse
        first = f.read(1)
        while first.isspace():
            first = f.read(1)
        f.seek(0)

        kb_list: List[dict] = []

        if first in "{[":
            data = json.load(f)
            if isinstance(data, list):
                kb_list = data
            elif isinstance(data, dict):
                for url, doc in data.items():
                    doc = dict(doc)
                    doc.setdefault("url", url)
                    if not doc.get("title"):
                        doc["title"] = doc.get("wikipedia_title") or normalize_url_to_title(url)
                    kb_list.append(doc)
            else:
                raise ValueError("Unsupported KB JSON structure")
        else:
            # Fallback: treat as JSONL
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                kb_list.append(doc)

    # Ensure every document has a title
    for doc in kb_list:
        if not doc.get("title"):
            doc["title"] = doc.get("wikipedia_title") or normalize_url_to_title(
                doc.get("url")
                or doc.get("wikipedia_url")
                or doc.get("source_url")
                or ""
            )

    return kb_list


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def normalize_title(title: Optional[str]) -> str:
    """Standardise a Wikipedia title for comparison."""
    if not isinstance(title, str):
        return ""
    title = title.lower()
    title = re.sub(r"\([^)]*\)", "", title)
    title = re.sub(r"[^a-z0-9\s]", "", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def normalize_url_to_title(url: Optional[str]) -> str:
    """Extract and normalise a title from a Wikipedia URL."""
    if not isinstance(url, str):
        return ""
    match = re.search(r"/wiki/([^#?]*)", url)
    if not match:
        return ""
    slug = unquote(match.group(1))
    slug = slug.replace("_", " ")
    return normalize_title(slug)