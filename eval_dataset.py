"""Evaluate the RAG pipeline on a portion of the EVQA dataset."""
from __future__ import annotations
"""Utility script to evaluate the pipeline on the EVQA dataset."""
from src.config import Config
from src.pipeline import search_rag_pipeline
from src.dataloader import VQADataset
from src.models import load_image_model, load_text_model



def print_results(cfg: Config, img_results, text_results) -> None:
    print(
        f"\n{'='*60}\n📄 1차 검색 결과: 이미지와 관련된 Top-{cfg.k_value} 이미지\n{'='*60}"
    )
    for i, res in enumerate(img_results, 1):
        title = res["doc"].get("title", "제목 없음")
        desc = res["description"][:100] if res["description"] else "(설명 없음)"
        url = res["image_url"]
        sim = res["similarity"]
        print(
            f"\n--- [ 이미지 순위: {i} ] ---"
            f"\n  📘 출처 문서: {title}"
            f"\n  🖼️ 이미지 URL: {url}"
            f"\n  📝 이미지 설명: {desc}..."
            f"\n  📈 이미지 유사도: {sim:.10f}"
        )
    print("=" * 60)

    print(
        f"\n{'='*60}\n🎯 최종 RAG 결과: 텍스트와 가장 관련 높은 Top-{cfg.m_value} 섹션\n{'='*60}"
    )
    for i, data in enumerate(text_results, 1):
        preview = data["section_text"][:500]
        title = data["source_title"]
        sec_title = data.get("section_title", "")
        sim = data.get("similarity", 0.0)
        print(
            f"\n--- [ 최종 섹션 순위: {i} ] ---"
            f"\n  (출처: '{title}')"
            f"\n  (섹션 제목: '{sec_title}')"
            f"\n  (텍스트 유사도: {sim:.10f})"
            f"\n  ▶ \"{preview}...\""
        )
    print("\n" + "=" * 60)


def main() -> None:
    """Run the pipeline for each VQA sample in the configured dataset."""
    cfg = Config.from_yaml("config.yaml")
    if not cfg.dataset_csv or not cfg.id2name_json:
        raise ValueError("dataset_csv and id2name_json must be set in config")

    dataset = VQADataset(
        csv_path=cfg.dataset_csv,
        id2name_path=cfg.id2name_json,
        image_root=cfg.dataset_image_root,
        googlelandmark_root=cfg.dataset_google_root,
        start=cfg.dataset_start,
        end=cfg.dataset_end,
    )

    for sample in dataset:
        print(
            f"\n{'#'*60}\n[Row {sample.row_idx}] Question: {sample.question}\nAnswer: {sample.answer}\n"
            f"Wiki title: {sample.wikipedia_title}\nWiki URL: {sample.wikipedia_url}\n{'#'*60}"
        )
        for img_path in sample.image_paths:
            cfg.image_path = img_path
            cfg.text_query = sample.question
            img_res, txt_res = search_rag_pipeline(cfg)
            print_results(cfg, img_res, txt_res)


if __name__ == "__main__":
    main()