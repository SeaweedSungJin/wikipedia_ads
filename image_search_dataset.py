from __future__ import annotations

import torch
from tqdm import tqdm
from src.config import Config
from src.dataloader import VQADataset
from src.models import load_image_model
from src.embedding import encode_image
from src.utils import (
    load_faiss_and_ids,
    load_kb,
    load_image,
    normalize_title,
    normalize_url_to_title,
)

def run_image_search_dataset(cfg: Config) -> None:
    # --- 1. 데이터 및 모델 로딩 ---

    print(f"데이터셋 평가 범위: start={cfg.dataset_start}, end={cfg.dataset_end}.")

    # Resolve device
    dev = cfg.image_device
    if isinstance(dev, int):
        dev = f"cuda:{dev}" if torch.cuda.is_available() else "cpu"
    elif isinstance(dev, str) and "cuda" in dev and not torch.cuda.is_available():
        dev = "cpu"

    # KB와 url_to_idx 맵을 먼저 로드 (핵심 수정 ①)
    kb_list, url_to_idx = load_kb(cfg.kb_json_path)

    # 로드한 kb_list와 url_to_idx를 인자로 전달하여 FAISS 로드 (핵심 수정 ②)
    faiss_index, kb_ids = load_faiss_and_ids(cfg.base_path, kb_list, url_to_idx)
    
    # 데이터셋 로더 초기화 (한 번만 수행)
    dataset = VQADataset(
        csv_path=cfg.dataset_csv,
        id2name_paths=cfg.id2name_paths,
        image_root=cfg.dataset_image_root,
        googlelandmark_root=cfg.dataset_google_root,
        start=cfg.dataset_start,
        end=cfg.dataset_end,
    )

    # 이미지 모델 로딩
    image_model, image_processor = load_image_model(device_map=dev)

    # --- 2. 평가 준비 ---

    base_k = [1, 3, 5, 10]
    k_values = sorted(set([k for k in base_k if k <= cfg.k_value] + [cfg.k_value]))
    doc_hits = {k: 0 for k in k_values}
    total_questions = 0

    search_k = min(cfg.k_value * 20, faiss_index.ntotal)

    # --- 3. 평가 루프 ---

    for sample in tqdm(dataset, desc="Evaluating Samples"):
        if not sample.image_paths:
            continue
        
        # --- Ground Truth(정답) 파싱 로직 수정 ---
        gt_titles_raw = str(sample.wikipedia_title or '').split('|')
        gt_urls_raw = str(sample.wikipedia_url or '').split('|')

        gt_titles = set()
        for title in gt_titles_raw:
            if title.strip():
                gt_titles.add(normalize_title(title))
        for url in gt_urls_raw:
            if url.strip():
                gt_titles.add(normalize_url_to_title(url))

        if not gt_titles:
            continue
        
        total_questions += 1

        try:
            img = load_image(sample.image_paths[0])
        except Exception as e:
            print(f"[Row {sample.row_idx}] 이미지 로딩 실패: {e}")
            continue
        
        img_emb = encode_image(img, image_model, image_processor)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        img_emb_np = img_emb.cpu().numpy().astype("float32")
        distances, indices = faiss_index.search(img_emb_np, search_k)

        # --- 후보군 필터링 및 Recall 계산 ---
        top_docs = []
        seen_docs = set()
        limit = indices.shape[1]
        for i in range(limit):
            if len(seen_docs) >= cfg.k_value:
                break
            
            faiss_vidx = int(indices[0][i])
            doc_idx = int(kb_ids[faiss_vidx])

            if doc_idx in seen_docs:
                continue
            
            if 0 <= doc_idx < len(kb_list):
                doc = kb_list[doc_idx]
                title_norm = normalize_title(doc.get("title", ""))
                if any(p in title_norm for p in ["list of", "outline of", "index of"]):
                    continue
                
                top_docs.append(doc)
                seen_docs.add(doc_idx)

        for k in k_values:
            top_k_titles = {normalize_title(doc.get("title")) for doc in top_docs[:k]}
            if not top_k_titles.isdisjoint(gt_titles):
                doc_hits[k] += 1

    # --- 4. 결과 출력 ---
    if total_questions == 0:
        print("평가할 유효한 샘플이 없습니다.")
        return

    print(f"\n총 {total_questions}개 질문 평가")
    for k in k_values:
        recall = doc_hits[k] / total_questions if total_questions > 0 else 0
        print(f"Image Retrieval Recall@{k}: {recall:.4f}")


if __name__ == "__main__":
    cfg = Config.from_yaml("config.yaml")
    if not cfg.dataset_csv or not cfg.id2name_paths:
        raise ValueError("dataset_csv and id2name_paths must be set in config")
    run_image_search_dataset(cfg)