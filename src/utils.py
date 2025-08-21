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


def load_faiss_and_ids(base_path: str) -> Tuple[faiss.Index, List[int]]:
    """Load FAISS index and the mapping IDs."""

    print("FAISS 인덱스 로딩중...")
    index = faiss.read_index(os.path.join(base_path, "kb_index.faiss"))
    with open(os.path.join(base_path, "kb_index_ids.pkl"), "rb") as f:
        ids = pickle.load(f)
    return index, ids

def load_kb_list(json_path: str) -> List[dict]:
    """Load the KB JSONL file."""

    print("지식베이스 JSON 로딩중...")
    kb_list = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                kb_list.append(json.loads(line))
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