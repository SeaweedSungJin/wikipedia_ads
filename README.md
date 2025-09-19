# Encyclopedic VQA RAG Pipeline

Encyclopedic-VQA 질문을 Wikipedia 기반 지식과 결합해 푸는 RAG(Retrieval-Augmented Generation) 파이프라인입니다. 질문 이미지로 유사한 문서를 검색하고, 문서 섹션을 여러 모델로 재순위 및 검증한 뒤, LLaVA-1.6-Mistral-7B가 최종 답변을 생성합니다. 평가 단계에서는 Exact Match와 BERT 기반 Answer Equivalence Model(BEM)을 함께 사용합니다.

## Pipeline At A Glance
1. **이미지 검색** – EVA-CLIP 임베딩으로 질의 이미지를 인코딩하고, 사전 구축된 FAISS 인덱스(`src/utils.load_faiss_and_ids`)에서 top-K 문서를 찾습니다. `list of`/`outline of`/`index of` 등 목록형 문서는 자동으로 제외됩니다.
2. **후보 섹션 생성** – `src/segmenter` 가 문서를 섹션·단락·문장 단위로 분할합니다. iNaturalist 경로 캐시는 `VQADataset` 초기화 시 `cache/` 디렉터리에 생성됩니다.
3. **(옵션) Contriever 융합** – 텍스트 질문/이미지 설명 임베딩을 융합해 섹션과 코사인 유사도를 구합니다 (`alpha` 가중치).
4. **재순위 및 점수 결합** – `src/rerankers` 를 통해 BGE, Electra, Jina v1(tiny/turbo), MPNet을 선택적으로 적용하고, 이미지 softmax 점수와 시그모이드된 재순위 점수를 가중합(`rank_img_weight`, `rank_rerank_weight`).
5. **엔트로피 기반 섹션 필터링** – 상위 `m_value` 섹션의 softmax 분포 엔트로피로 신뢰도를 계산해 최종 전달 섹션 개수를 동적으로 결정(`bge_conf_threshold`).
6. **NLI 그래프 클러스터링** – `src/nli_cluster` 가 entailment/contradiction 확률로 가중 그래프를 만든 뒤 `consistency` 또는 `clique` 모드로 최대 `nli_max_cluster` 개 섹션을 묶습니다. α·β, edge rule, 양방향 margin 등은 `config.yaml` 로 제어합니다.
7. **VLM 추론 및 평가** – 선택된 섹션과 이미지를 LLaVA에 결합해 답을 생성(`src.models.generate_vlm_answer`), `src.eval.evaluate_example` 로 Exact Match → BEM 순서로 채점합니다.

모든 대형 모델과 인덱스는 모듈 전역 캐시에 저장돼 반복 호출 시 재사용됩니다. CUDA 미사용 환경에서는 자동으로 CPU 폴백이 적용됩니다.

## Entry Points
- **`bge_nli_graph_dataset.py`** – 데이터셋 구간(`dataset_start`, `dataset_end`)마다 전체 파이프라인을 수행합니다. 이미지/BGE/NLI 단계별 Recall@K, 재순위·NLI 소요 시간, VLM 정확도를 출력합니다.
- **`image_search_dataset.py`** – 이미지 임베딩 + FAISS 검색만 실행해 Recall@K를 측정합니다. KB/FAISS는 최초 한 번만 로드하며, 제목·URL 정규화를 통해 정답을 비교합니다.
- **`vlm_only_dataset.py`** – 검색 단계를 건너뛰고 질문과 이미지를 직접 LLaVA에 넣어 기본 VLM 정확도를 계산합니다.
- **`metric_dataset.py`** – 동일 파이프라인을 실행하면서 단계별(latency/VRAM/에너지) 지표를 수집합니다. Warmup 샘플(`--warmup_steps`)은 자원 통계에서 제외되지만 정확도 집계에는 포함됩니다.

## Configuration Highlights (`config.yaml`)
- **경로**: `base_path`, `kb_json_name`, `dataset_csv`, `id2name_paths`, `dataset_image_root`, `dataset_google_root`
- **검색 파라미터**: `k_value`, `m_value`, `search_expand`, `alpha`
- **세그먼트 옵션**: `segment_level` (section/paragraph/sentence), `chunk_size`
- **재순위 설정**: `rerankers` 플래그(`contriever`, `jina_tiny`, `jina_turbo`, `bge`, `electra`, `mpnet`), `bge_model`, `bge_batch_size`, `electra_model`, `electra_batch_size`, `mpnet_model`
- **점수 결합/필터링**: `rank_img_weight`, `rank_rerank_weight`, `rank_img_softmax_temp`, `rank_text_temp`, `bge_conf_threshold`
- **NLI**: `nli_models` 토글(deberta/roberta/deberta_v3), `nli_max_length`, `nli_batch_size`, `nli_e_min`, `nli_margin`, `nli_tau`, `nli_lambda`, `nli_selection`, `nli_hybrid_lambda`, `nli_edge_rule`, `nli_dir_margin`, `nli_autocast`, `nli_autocast_dtype`
- **디바이스**: `image_device`, `bge_device`, `nli_device`, `vlm_device` (정수 또는 `cuda:0` 형태). CUDA 사용 불가 시 자동으로 CPU로 대체합니다.

`Config.from_yaml` 은 예상치 못한 키를 무시하며, 이전 버전 필드(`googlelandmark_root`, `id2name_json`)도 역호환 처리합니다.

## Module Guide (`src/`)
- **`config.py`** – YAML을 `Config` dataclass로 로드하고 경로 유틸(`kb_json_path`)을 제공합니다.
- **`dataloader.py`** – `VQADataset` iterable을 구현하며, iNaturalist ID 캐시를 생성/로드하고 데이터셋 메타데이터를 관리합니다.
- **`pipeline.py`** – 이미지 검색→세그먼트 생성→재순위→엔트로피 필터→(옵션) Contriever/MPNet 처리까지 한 번에 실행하는 `search_rag_pipeline`. NLTK 리소스 준비, stage meter 훅, CUDA 메모리 관리 등을 포함합니다.
- **`rerankers.py`** – BGE/Electra/Jina cross-encoder와 MPNet bi-encoder 래퍼. 배치 프리패치, max_length 자동 클램핑, half precision 지원.
- **`encoders.py`** – HuggingFace 텍스트 인코더(`HFTextEncoder`)와 Jina M0 멀티모달 인코더 래퍼.
- **`embedding.py`** – EVA-CLIP 이미지 임베딩과 일반 텍스트 임베딩(평균 풀링 + 정규화) 헬퍼.
- **`evaluation_utils.py`** – Ground truth 추출과 Recall@K 집계를 위한 공용 헬퍼. 실행 스크립트(`image_search_dataset.py`, `bge_nli_graph_dataset.py`, `metric_dataset.py`, `qformer_reranker_pipeline.py`)에서 공유합니다.
- **`segmenter.py`** – 섹션/문장/단락 분리 로직과 불필요한 섹션(References 등) 필터링.
- **`models.py`** – 이미지·텍스트·재순위·VLM·NLI 모델 로더, 장치 헬퍼(`resolve_device`, `setup_cuda`, `get_device`) 및 Jina 임베딩 유틸.
- **`nli_cluster.py`** – entailment/contradiction 확률로 그래프를 구축하고 `cluster_sections_consistency` 또는 `cluster_sections_clique` 로 상위 클러스터를 선택합니다. 토크나이저/모델 길이 제한을 자동으로 적용합니다.
- **`utils.py`** – KB JSON/FAISS 로드(dict 재매핑, 길이 검증 포함), 이미지 로딩·프리페치, 타이틀/URL 정규화, NLTK 데이터 다운로드.
- **`eval.py`** – 답변 전처리, Exact Match, TensorFlow Hub 기반 BEM 평가 로직.
- **`metrics_utils.py`** – NVML 전력 샘플러, stage meter 컨텍스트, metrics CSV/요약 생성을 담당합니다.
- **`logging_utils.py`** – 통일된 스트림 로거 제공.

## Reranking & NLI Options
- **BGE**: `rerankers.bge: true`, `bge_model`, `bge_max_length`, `bge_batch_size`, `bge_device`
- **Electra**: `rerankers.electra: true`, `electra_model`, `electra_batch_size`
- **Jina v1**: `rerankers.jina_tiny` 또는 `rerankers.jina_turbo`
- **MPNet**: `rerankers.mpnet: true` (SentenceTransformer 기반 bi-encoder)

여러 재순위기를 동시에 켤 수 있지만, 마지막으로 실행된 모듈의 점수가 `combined_score` 계산에 사용됩니다. NLI는 `nli_models` 토글로 선택하며, 두 모드 모두 `nli_autocast`/`nli_autocast_dtype`로 half/bfloat16 추론을 제어합니다.

## Metrics Outputs (`metric_dataset.py`)
실행 시 기본적으로 `metrics/metrics_run_YYYYMMDD_HHMM/` 폴더에 다음 파일을 생성합니다.
- `metrics_samples.csv` – 샘플·스테이지별 latency/VRAM/에너지 (warmup 여부 포함)
- `metrics_summary.json` – 선택한 퍼센타일(`--pctl`)을 포함한 통계와 환경 정보(`env_info`)
- `candidate_docs.csv`, `candidate_sections.csv`, `candidate_clusters.csv` – 각 단계 후보에 대한 점수/클러스터 구성 기록

NVML이 없거나 접근이 막힌 경우 전력/에너지 값은 0으로 채워집니다.

## Running The Tools
```bash
pip install -r requirements.txt

# 전체 파이프라인 + VLM 평가
python bge_nli_graph_dataset.py

# 이미지 검색 성능만 측정
python image_search_dataset.py

# VLM 단독 성능 확인
python vlm_only_dataset.py

# 단계별 지표 수집 (Warmup 10 샘플, NVML 50ms 폴링, VLM 생략), 
python metric_dataset.py --warmup_steps 10 --nvml_poll_ms 50 --no_vlm
#전체 실행 (VLM포함)
python metric_dataset.py 

# Q-Former reranker (EVA-CLIP 검색 캐시 사용)
python qformer_reranker_pipeline.py --clip-cache-dir datasets/clip_cache
```

`qformer_reranker_pipeline.py`는 `--clip-cache-dir`(기본 `datasets/clip_cache`) 경로에 EVA-CLIP
검색 결과를 저장해두고, 다음 실행에서는 캐시를 읽어 재검색을 생략한 뒤 Q-Former reranker만
실행합니다.

`dataset_start`/`dataset_end` 를 조정해 평가 범위를 제한하고, `id2name_paths` + 이미지 루트를 올바르게 지정하면 iNaturalist/Google Landmarks 이미지가 자동으로 매핑됩니다. 검색 파라미터(`k_value`, `m_value`, `search_expand`)와 재순위·NLI 옵션을 바꿔 실험을 진행하세요.
