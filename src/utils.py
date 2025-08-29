"""Helper utilities for loading data and resources."""
import json
import os
import pickle
from io import BytesIO
from typing import Optional, Tuple, List, Dict
import re
from urllib.parse import unquote
import faiss
import nltk
import requests
from PIL import Image, UnidentifiedImageError
import numpy as np
from tqdm import tqdm

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


def load_kb(knowledge_json_path: str) -> Tuple[List[dict], Dict[str, int]]:
    """Load KB JSON supporting list and dict formats.

    Returns a list of documents and a mapping from URL to document index.
    Each document is expected to contain at least ``title``,
    ``section_titles`` and ``section_texts`` keys.  If the input is a dict the
    keys are treated as URLs.
    """

    with open(knowledge_json_path, "r", encoding="utf-8") as f:
        kb = json.load(f)

    kb_list: List[dict] = []
    url_to_idx: Dict[str, int] = {}

    if isinstance(kb, list):
        kb_list = kb
        for i, doc in enumerate(kb_list):
            u = (
                doc.get("url")
                or doc.get("wikipedia_url")
                or doc.get("source_url")
                or ""
            )
            if u and u not in url_to_idx:
                url_to_idx[u] = i
    elif isinstance(kb, dict):
        for i, (u, doc) in enumerate(kb.items()):
            doc = dict(doc)
            if not doc.get("title"):
                doc["title"] = normalize_url_to_title(u)
            kb_list.append(doc)
            url_to_idx[u] = i
    else:
        raise ValueError("Unsupported KB JSON format")

    for doc in kb_list:
        if not doc.get("title"):
            doc["title"] = (
                doc.get("wikipedia_title") or doc.get("title") or "unknown"
            )

    return kb_list, url_to_idx


def load_faiss_and_ids(
    base_path: str, kb_json_path: str
) -> Tuple[faiss.Index, np.ndarray, List[dict]]:
    """Load FAISS index, accompanying IDs and the KB list.

    ``kb_index.faiss`` is read from ``base_path`` and several sidecar files are
    probed for the FAISS→document ID mapping.  If only a mapping pickle is
    available it may contain either a list/ndarray of indices or a
    ``{faiss_id: url}`` dictionary which is resolved against the KB.
    """

    index_path = os.path.join(base_path, "kb_index.faiss")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    index = faiss.read_index(index_path)
    ntotal = index.ntotal

    kb_list, url_to_idx = load_kb(kb_json_path)

    kb_ids: Optional[np.ndarray] = None
    side_tried: List[str] = []

    side_candidates = [os.path.join(base_path, name) for name in [
        "kb_ids.npy",
        "faiss_ids.npy",
        "doc_ids.npy",
    ]]

    tried = set()
    for path in side_candidates:
        if path in tried:
            continue
        tried.add(path)
        side_tried.append(path)
        if os.path.exists(path):
            arr = np.load(path)
            if arr.ndim != 1:
                raise ValueError(f"{path} must be 1-D array, got shape {arr.shape}")
            if len(arr) != ntotal:
                raise ValueError(
                    f"{path} length {len(arr)} != index.ntotal {ntotal}"
                )
            kb_ids = arr.astype(np.int64)
            print(f"[INFO] Loaded kb_ids from {path} (len={len(kb_ids):,})")
            break

    mapping_pkl_path = os.path.join(base_path, "kb_index_ids.pkl")
    if kb_ids is None and mapping_pkl_path and os.path.exists(mapping_pkl_path):
        side_tried.append(mapping_pkl_path)
        with open(mapping_pkl_path, "rb") as f:
            mapping = pickle.load(f)
        if isinstance(mapping, (list, np.ndarray)):
            mapping = np.array(mapping)
            if mapping.ndim != 1:
                raise ValueError(f"{mapping_pkl_path} must be 1-D array-like")
            if len(mapping) != ntotal:
                raise ValueError(
                    f"{mapping_pkl_path} length {len(mapping)} != index.ntotal {ntotal}"
                )
            kb_ids = mapping.astype(np.int64)
            print(
                f"[INFO] Loaded kb_ids from array-like PKL (len={len(kb_ids):,})"
            )
        elif isinstance(mapping, dict):
            kb_ids = np.empty(ntotal, dtype=np.int64)
            kb_ids.fill(-1)
            bad = 0
            for k, v in mapping.items():
                try:
                    k_int = int(k)
                except Exception:
                    continue
                if isinstance(v, (int, np.integer)):
                    kb_ids[k_int] = int(v)
                else:
                    v_str = str(v)
                    idx = url_to_idx.get(v_str, -1)
                    if idx == -1:
                        norm_v = normalize_url_to_title(v_str)
                        candidates = [
                            i
                            for i, d in enumerate(kb_list)
                            if normalize_title(d.get("title")) == norm_v
                        ]
                        idx = candidates[0] if candidates else -1
                    kb_ids[k_int] = idx
                if kb_ids[k_int] == -1:
                    bad += 1
            if bad:
                print(
                    f"[WARN] {bad} entries in mapping PKL could not be matched to KB docs"
                )
            if (kb_ids == -1).any():
                raise ValueError(
                    "kb_ids contains -1 (unmatched). Please provide kb_ids.npy or a consistent mapping."
                )
            print(f"[INFO] Built kb_ids from dict PKL (len={len(kb_ids):,})")
        else:
            raise ValueError(f"Unsupported mapping PKL type: {type(mapping)}")

    if kb_ids is None:
        raise FileNotFoundError(
            "Could not load kb_ids. Tried: " + ", ".join(side_tried) +
            ". Provide kb_ids.npy (or faiss_ids.npy), or a PKL that can be resolved."
        )

    if len(kb_ids) != ntotal:
        raise AssertionError(
            f"kb_ids length {len(kb_ids)} != index.ntotal {ntotal}"
        )

    if kb_ids.min() < 0 or kb_ids.max() >= len(kb_list):
        raise ValueError(
            "kb_ids contains out-of-range doc indices for the given KB list."
        )

    return index, kb_ids, kb_list

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
