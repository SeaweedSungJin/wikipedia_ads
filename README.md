# Encyclopedic VQA RAG Pipeline

This project implements a retrieval‑augmented pipeline to answer visual questions using Wikipedia. Given a question and its accompanying image, the system retrieves relevant articles via image similarity, segments them into sections, reranks those sections with text models, and selects high‑confidence sections using entropy of BGE cross‑encoder scores.

## Pipeline Overview
1. Precompute EVA‑CLIP embeddings for Wikipedia images and build a FAISS index.
2. For each query:
   - Embed the question image and search FAISS for top‑k similar images, yielding candidate articles. Non‑article pages (e.g., "list of", "outline of") are filtered and replacements fetched.
   - Split the articles into sections (or paragraphs/sentences depending on the configuration) to form candidate passages.
   - Rerank candidates with any combination of Contriever, Jina M0, and BGE. BGE scores of the top‑M sections are transformed with softmax, entropy is computed, and only high‑confidence sections are returned.
   - Cluster the BGE top‑M sections with an NLI model using a weighted graph approach. Every section pair is scored for entailment and contradiction. Edges are added only when entailment ≥ ``e_min`` and ``entailment - contradiction`` ≥ ``margin`` with a final cutoff ``tau`` on the weight ``max(0, α·P_e - β·P_c)``. The resulting undirected graph's maximal cliques are truncated to the best three‑section subclique and ranked by a weighted blend of normalized BGE similarity and average edge weight.
   - The top cluster's sections (up to three) are fed to the **LLaVA‑1.6‑Mistral‑7B** vision‑language model along with the original question and image. The model is prompted to output only the target entity.
   - Predictions are scored by `evaluate_example`, which first checks for an exact string match and falls back to a BERT‑based answer equivalence model (threshold 0.5).

## Configuration & Files
- **config.yaml** – paths, model options, search parameters, and thresholds such as the BGE confidence, NLI edge gating parameters (`nli_e_min`, `nli_margin`, `nli_tau`), blend weight `nli_lambda`, max length, and cluster size. The `k_value` field sets how many image candidates are retrieved; evaluation reports Recall@1/3/5/10 and, if larger, Recall@`k_value` as well (ensure `m_value` ≥ `k_value`).
  Device placement can be tuned per module via `image_device`, `bge_device`, and `nli_device`, and pairwise NLI scoring uses `nli_batch_size` to control batching.
- **bge_nli_graph_dataset.py** – processes a dataset slice using weighted graph-based NLI clustering. After ranking, the top cluster is passed to LLaVA for answer generation and evaluated with `evaluate_example`. Summary includes Recall@1/3/5/10 and VLM accuracy.
 - **src/config.py** – dataclasses for configuration and a YAML loader that drops unknown keys.
 - **src/pipeline.py** – orchestrates image search, section generation, reranking, and entropy‑based confidence filtering.
- **src/encoders.py** – wrappers around Contriever, Jina M0, and BGE scoring models.
- **src/models.py** – utilities for loading image/text models and computing embeddings.
- **src/dataloader.py** – reads the EVQA CSV and yields question–image samples with metadata.
- **src/embedding.py** – helper functions to encode images with EVA‑CLIP and text with Vicuna.
- **src/segmenter.py** – splits Wikipedia articles into sections, paragraphs, or sentences for ranking.
- **src/nli_cluster.py** – loads an NLI model and groups sections into clusters via pairwise entailment.
- **src/utils.py** – FAISS loading, image fetching, and general utility helpers.

## Quick Start

Install the requirements and run the evaluation scripts:

```bash
pip install -r requirements.txt
# Evaluate a dataset slice with RAG + LLaVA answer checking
python bge_nli_graph_dataset.py
```

`config.yaml` contains `dataset_start` and `dataset_end` to limit the portion of the EVQA dataset evaluated. Leave them blank to process the full test set.