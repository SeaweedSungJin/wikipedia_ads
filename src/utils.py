"""Helper utilities for loading data and resources."""
import json
import os
import pickle
from typing import Iterable, List, Tuple

import faiss
import nltk
import requests
import torch
from PIL import Image


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
    """Load an image from a local path or an HTTP URL."""

    if path.startswith("http"):
        # Download image if a URL is provided
        img = Image.open(requests.get(path, stream=True).raw)
    else:
        img = Image.open(path)
    return img.convert("RGB")


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