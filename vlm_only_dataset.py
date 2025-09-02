from __future__ import annotations

import logging
import torch
from src.config import Config
from src.dataloader import VQADataset
from src.models import load_vlm_model, generate_vlm_answer
from src.eval import evaluate_example

logging.basicConfig(level=logging.INFO)

def run_vlm_only_dataset(cfg: Config) -> None:
    dataset = VQADataset(
        csv_path=cfg.dataset_csv,
        id2name_paths=cfg.id2name_paths,
        image_root=cfg.dataset_image_root,
        googlelandmark_root=cfg.dataset_google_root,
        start=cfg.dataset_start,
        end=cfg.dataset_end,
    )
    print(f"데이터셋 평가 범위: start={cfg.dataset_start}, end={cfg.dataset_end}.")

    device = cfg.bge_device
    if isinstance(device, int):
        device = f"cuda:{device}"
    if not torch.cuda.is_available() and isinstance(device, str) and "cuda" in device:
        device = "cpu"
    vlm_model, vlm_processor = load_vlm_model(device_map=device)

    total = 0
    correct = 0
    for sample in dataset:
        if not sample.image_paths:
            continue
        image_path = sample.image_paths[0]
        try:
            pred = generate_vlm_answer(
                vlm_model, vlm_processor, sample.question, image_path, []
            )
        except Exception as e:
            print(f"[Row {sample.row_idx}] VLM 처리 실패: {e}")
            continue
        q_type = sample.metadata.get("question_type", "automatic")
        match = bool(
            evaluate_example(sample.question, [sample.answer], pred, q_type)
        )
        total += 1
        correct += int(match)
        print(
            f"Row {sample.row_idx} | 정답: {sample.answer} | 예측: {pred} | 일치: {match}"
        )
    if total:
        acc = correct / total
    else:
        acc = 0.0
    print(f"\n총 {total}개 중 {correct}개 정답, 정확도 {acc:.3f}")

if __name__ == "__main__":
    cfg = Config.from_yaml("config.yaml")
    run_vlm_only_dataset(cfg)