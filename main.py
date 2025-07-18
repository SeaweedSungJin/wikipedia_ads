# Run the RAG search pipeline for a single image and query
"""Entry point for the RAG search pipeline."""
from __future__ import annotations

from src.config import Config
from src.pipeline import search_rag_pipeline


def main() -> None:
    """Run the RAG pipeline using configuration from ``config.yaml``."""
    cfg = Config.from_yaml("config.yaml")
    
    # Execute the core pipeline with the provided configuration
    img_results, text_results = search_rag_pipeline(cfg)

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


if __name__ == "__main__":
    main()