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

# in src/utils.py

def load_faiss_and_ids(base_path: str, kb_list: List[dict], url_to_idx: Dict[str, int]) -> Tuple[faiss.Index, np.ndarray]:
    """
    FAISS 인덱스를 로드하고, pkl 파일의 타입에 따라 ID 매핑을 검증 및 재정렬합니다.
    """
    index_path = os.path.join(base_path, "kb_index.faiss")
    ids_path = os.path.join(base_path, "kb_index_ids.pkl")

    index = faiss.read_index(index_path)
    
    with open(ids_path, "rb") as f:
        mapping = pickle.load(f)

    if isinstance(mapping, dict):
        print(f"[INFO] Dict 타입의 pkl 로드. {len(mapping)}개의 ID를 재매핑합니다...")
        new_ids = np.zeros(len(mapping), dtype=np.int32)
        for faiss_idx, doc_url in tqdm(mapping.items(), desc="Re-mapping FAISS IDs"):
            if doc_url in url_to_idx:
                doc_idx = url_to_idx[doc_url]
                new_ids[faiss_idx] = doc_idx
            else:
                pass  # URL을 찾을 수 없는 경우 무시
        ids = new_ids
        print("[INFO] ID 재매핑 완료.")
    else:
        print(f"[INFO] Array 타입의 pkl 로드 (길이={len(mapping)})")
        ids = np.array(mapping, dtype=np.int32)
    
    # 최종 검증
    assert index.ntotal == len(ids), (
        f"FAISS 인덱스({index.ntotal})와 ID 목록({len(ids)})의 길이가 일치하지 않습니다."
    )
    
    # 이 함수는 이제 kb_list를 반환하지 않습니다.
    return index, ids

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
