from __future__ import annotations

import logging
import torch
from src.config import Config
from src.dataloader import VQADataset
from src.models import load_image_model
from src.embedding import encode_image
from src.utils import (
    load_faiss_and_ids,
    load_image,
    normalize_title,
    normalize_url_to_title,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_image_search_dataset(cfg: Config) -> None:
    dataset = VQADataset(
        csv_path=cfg.dataset_csv,
        id2name_path=cfg.id2name_json,
        image_root=cfg.dataset_image_root,
        googlelandmark_root=cfg.dataset_google_root,
        start=cfg.dataset_start,
        end=cfg.dataset_end,
    )
    print(f"데이터셋 평가 범위: start={cfg.dataset_start}, end={cfg.dataset_end}.")

    # Resolve device
    dev = cfg.image_device
    if isinstance(dev, int):
        dev = f"cuda:{dev}" if torch.cuda.is_available() else "cpu"
    elif isinstance(dev, str) and "cuda" in dev and not torch.cuda.is_available():
        dev = "cpu"

    image_model, image_processor = load_image_model(device_map=dev)
    faiss_index, kb_ids, kb_list = load_faiss_and_ids(cfg.base_path, cfg.kb_json_path)

    # Precompute first image index per document when enforcing first_image_only
    doc_to_first: dict[int, int] = {}
    if cfg.first_image_only:
        for i, d_idx in enumerate(kb_ids):
            if d_idx not in doc_to_first:
                doc_to_first[d_idx] = i

    base_k = [1, 3, 5, 10]
    k_values = sorted(set([k for k in base_k if k <= cfg.k_value] + [cfg.k_value]))
    doc_hits = {k: 0 for k in k_values}
    total = 0

    search_k = min(cfg.k_value * 20, faiss_index.ntotal)

    for sample in dataset:
        if not sample.image_paths:
            continue
        try:
            img = load_image(sample.image_paths[0])
        except Exception as e:
            print(f"[Row {sample.row_idx}] 이미지 로딩 실패: {e}")
            continue
        img_emb = encode_image(img, image_model, image_processor).numpy()
        distances, indices = faiss_index.search(img_emb, search_k)

        top_docs = []
        seen = set()
        limit = indices.shape[1]
        for i in range(limit):
            if len(seen) >= cfg.k_value:
                break
            faiss_vidx = int(indices[0][i])
            doc_idx = int(kb_ids[faiss_vidx])
            if doc_idx in seen:
                continue
            if cfg.first_image_only:
                first_idx = doc_to_first.get(doc_idx)
                if first_idx is None or first_idx != faiss_vidx:
                    continue
            if 0 <= doc_idx < len(kb_list):
                doc = kb_list[doc_idx]
                title_norm = normalize_title(doc.get("title", ""))
                if any(p in title_norm for p in ["list of", "outline of", "index of"]):
                    continue
                top_docs.append(doc)
                seen.add(doc_idx)

        if sample.wikipedia_title:
            raw_titles = str(sample.wikipedia_title).split("|")
            gt_titles = [normalize_title(t) for t in raw_titles]
        else:
            raw_urls = str(sample.wikipedia_url).split("|") if sample.wikipedia_url else []
            gt_titles = [normalize_url_to_title(u) for u in raw_urls]
        gt_title_set = set(gt_titles)

        if not gt_title_set:
            continue
        total += 1

        for k in k_values:
            subset = top_docs[:k]
            if any(normalize_title(d.get("title")) in gt_title_set for d in subset):
                doc_hits[k] += 1

    if total == 0:
        print("평가할 유효한 샘플이 없습니다.")
        return

    print(f"\n총 {total}개 질문 평가")
    for k in k_values:
        print(f"Image Retrieval Recall@{k}: {doc_hits[k] / total:.4f}")


if __name__ == "__main__":
    cfg = Config.from_yaml("config.yaml")
    if not cfg.dataset_csv or not cfg.id2name_json:
        raise ValueError("dataset_csv and id2name_json must be set in config")
    run_image_search_dataset(cfg)