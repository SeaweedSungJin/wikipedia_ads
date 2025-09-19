from __future__ import annotations

import torch
from tqdm import tqdm
from src.config import Config
from src.dataloader import VQADataset
from src.models import load_image_model
from src.embedding import encode_image
from src.evaluation_utils import (
    build_ground_truth,
    compute_k_values,
    init_recall_dict,
    update_recall_from_rank,
)
from src.utils import load_faiss_and_ids, load_kb, load_image, normalize_title


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

    k_values = compute_k_values(cfg.k_value)
    doc_hits = init_recall_dict(k_values)
    total_questions = 0

    search_k = min(cfg.k_value * 20, faiss_index.ntotal)

    # --- 3. 평가 루프 ---

    for sample in tqdm(dataset, desc="Evaluating Samples"):
        if not sample.image_paths:
            continue
        ground_truth = build_ground_truth(sample)
        if ground_truth is None:
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
        unique_doc_indices = []
        seen_docs = set()
        for idx in indices[0]:
            if len(unique_doc_indices) >= cfg.k_value:
                break

            faiss_vidx = int(idx)
            doc_idx = int(kb_ids[faiss_vidx])
            if doc_idx in seen_docs:
                continue

            if 0 <= doc_idx < len(kb_list):
                title_norm = normalize_title(kb_list[doc_idx].get("title", ""))
                if any(p in title_norm for p in ["list of", "outline of", "index of"]):
                    continue

                unique_doc_indices.append(doc_idx)
                seen_docs.add(doc_idx)

        doc_rank = None
        for rank, doc_idx in enumerate(unique_doc_indices, start=1):
            title_norm = normalize_title(kb_list[doc_idx].get("title", ""))
            if title_norm in ground_truth.title_set:
                doc_rank = rank
                break

        update_recall_from_rank(doc_hits, doc_rank, k_values)

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
