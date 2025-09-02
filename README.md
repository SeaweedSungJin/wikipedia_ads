# Encyclopedic VQA RAG Pipeline

This project implements a retrieval‑augmented pipeline to answer visual questions using Wikipedia. Given a question and its accompanying image, the system retrieves relevant articles via image similarity, segments them into sections, reranks those sections with text models, and selects high‑confidence sections using entropy of BGE cross‑encoder scores.

## Recent Updates
- Introduced a unified `Reranker` interface (`src/rerankers.py`) with pluggable implementations for BGE, Electra and Jina M0. The pipeline now calls a single `score(query, sections)` method.
- Moved NLI model loading into `src/models.py` (`load_nli_model`) to be consistent with other loaders and removed the loader from `src/nli_cluster.py`.
- Added `resolve_device` to standardize device selection with CPU fallback.
- Capped NLI tokenization to each model's supported context length to prevent overflow on RoBERTa/DeBERTa MNLI checkpoints.
- Fixed Electra reranker to adapt to different head shapes ([B,1]/[B,2]/[B,3]).
- Clarified docstrings (e.g., text embedding uses the provided HF encoder).

## Pipeline Overview
1. Precompute EVA‑CLIP embeddings for Wikipedia images and build a FAISS index.
2. For each query:
   - Embed the question image and search FAISS for top‑k similar images, yielding candidate articles. Non‑article pages (e.g., "list of", "outline of") are filtered and replacements fetched.
   - Split the articles into sections (or paragraphs/sentences depending on the configuration) to form candidate passages.
   - Rerank candidates using a pluggable reranker (`BGEReranker`, `ElectraReranker`, or `JinaReranker`). Reranker scores of the top‑M sections are transformed with softmax, entropy is computed, and only high‑confidence sections are returned.
   - Cluster the BGE top‑M sections with an NLI model using a weighted graph approach. Every section pair is scored for entailment and contradiction. Edges are added only when entailment ≥ ``e_min`` and ``entailment - contradiction`` ≥ ``margin`` with a final cutoff ``tau`` on the weight ``max(0, α·P_e - β·P_c)``. The resulting undirected graph's maximal cliques are truncated to the best three‑section subclique and ranked by a weighted blend of normalized BGE similarity and average edge weight.
   - The top cluster's sections (up to three) are fed to the **LLaVA‑1.6‑Mistral‑7B** vision‑language model along with the original question and image. The model is prompted to output only the target entity.
   - Predictions are scored by `evaluate_example`, which first checks for an exact string match and falls back to a BERT‑based answer equivalence model (threshold 0.5).

## Configuration & Files
- **config.yaml** – paths, model options, search parameters, and thresholds such as the BGE confidence, NLI edge gating parameters (`nli_e_min`, `nli_margin`, `nli_tau`), blend weight `nli_lambda`, max length, and cluster size. The `k_value` field sets how many image candidates are retrieved; evaluation reports Recall@1/3/5/10 and, if larger, Recall@`k_value` as well (ensure `m_value` ≥ `k_value`). Reranker models (`bge_model`, `electra_model`, `mpnet_model`) are toggled via the `rerankers` section.
  Device placement can be tuned per module via `image_device`, `bge_device`, and `nli_device`, and pairwise NLI scoring uses `nli_batch_size` to control batching.  NLI backends (`deberta_nli_model`, `roberta_nli_model`, `deberta_v3_nli_model`) are toggled through the `nli_models` section.
- **bge_nli_graph_dataset.py** – processes a dataset slice using weighted graph-based NLI clustering. After ranking, the top cluster is passed to LLaVA for answer generation and evaluated with `evaluate_example`. Summary includes Recall@1/3/5/10 and VLM accuracy.
- **image_search_dataset.py** – measures image-only retrieval performance on the dataset and reports Recall@K without running rerankers, NLI, or VLM steps.
- **src/config.py** – dataclasses for configuration and a YAML loader that drops unknown keys.
- **src/pipeline.py** – orchestrates image search, section generation, reranking (via `src/rerankers.py`), and entropy‑based confidence filtering.
- **src/rerankers.py** – `Reranker` base class and implementations: `BGEReranker`, `ElectraReranker`, `JinaReranker`.
- **src/encoders.py** – wrappers around text encoders used in Contriever and Jina embedding modes.
- **src/models.py** – model loading utilities (image/text/VLM/NLI) and device helpers (`resolve_device`).
- **src/dataloader.py** – reads the EVQA CSV and yields question–image samples with metadata.
- **src/embedding.py** – helper functions to encode images with EVA‑CLIP and text with Vicuna.
- **src/segmenter.py** – splits Wikipedia articles into sections, paragraphs, or sentences for ranking.
- **src/nli_cluster.py** – constructs an NLI relation graph and clusters sections; expects a preloaded NLI model/tokenizer.
- **src/utils.py** – FAISS loading, image fetching, and general utility helpers.

## Rerankers
- BGE: set `rerankers.bge: true` and configure `bge_model`, `bge_max_length`, `bge_device`.
- Electra cross-encoder: set `rerankers.electra: true` and configure `electra_model`. The implementation adapts to regression/binary/3‑class heads.
- Jina M0: set `rerankers.jina_m0: true`.

Only enable one reranker at a time; later rerankers override earlier similarity scores.

## NLI Backend Selection
Choose an NLI model in `config.yaml` via `nli_models` toggles. The script calls `models.load_nli_model` and the clustering module clamps tokenization length to the model's supported context automatically.

## Quick Start

Install the requirements and run the evaluation scripts:

```bash
pip install -r requirements.txt
# Evaluate a dataset slice with RAG + LLaVA answer checking
python bge_nli_graph_dataset.py
# Evaluate image only
python image_search_dataset.py
```

`config.yaml` contains `dataset_start` and `dataset_end` to limit the portion of the EVQA dataset evaluated. Leave them blank to process the full test set.
