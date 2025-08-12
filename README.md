# Encyclopedic VQA RAG Pipeline

This project implements a retrieval‑augmented pipeline to answer visual questions using Wikipedia. Given a question and its accompanying image, the system retrieves relevant articles via image similarity, segments them into sections, reranks those sections with text models, and selects high‑confidence sections using entropy of BGE cross‑encoder scores.

## Pipeline Overview
1. Precompute EVA‑CLIP embeddings for Wikipedia images and build a FAISS index.
2. For each query:
   - Embed the question image and search FAISS for top‑k similar images, yielding candidate articles. Non‑article pages (e.g., "list of", "outline of") are filtered and replacements fetched.
   - Split the articles into sections (or paragraphs/sentences depending on the configuration) to form candidate passages.
   - Rerank candidates with any combination of Contriever, Jina M0, and BGE. BGE scores of the top‑M sections are transformed with softmax, entropy is computed, and only high‑confidence sections are returned.
   - Optionally, cluster the BGE top‑M sections with an NLI model. Two strategies are available:
     1. **BFS clustering** (`bge_nli_dataset.py`) – sections above an entailment threshold are connected and clusters are grown breadth‑first from the highest BGE scores.
    2. **Graph clique clustering** (`bge_nli_graph_dataset.py`) – every section pair is scored for entailment and contradiction. Edges are added only when entailment ≥ ``e_min`` and ``entailment - contradiction`` ≥ ``margin`` with a final cutoff ``tau`` on the weight ``max(0, α·P_e - β·P_c)``. The resulting undirected graph's maximal cliques are truncated to the best three‑section subclique and ranked by a weighted blend of normalized BGE similarity and average edge weight.
   - Evaluation scripts compare the selected sections or clusters with ground truth to report document and document+section accuracy.

## Configuration & Files
- **config.yaml** – paths, model options, search parameters, and thresholds such as the BGE confidence, NLI edge gating parameters (`nli_e_min`, `nli_margin`, `nli_tau`), blend weight `nli_lambda`, max length, and cluster size. ``nli_threshold`` remains for legacy BFS clustering.
  Device placement can be tuned per module via `image_device`, `bge_device`, and `nli_device`, and pairwise NLI scoring uses `nli_batch_size` to control batching.
- **main.py** – processes a single query end‑to‑end and prints the retrieved sections.
- **image_only.py** – evaluates only the image retrieval stage, listing the top‑k document candidates and reporting how often the ground truth article appears.
- **bge_only.py** – runs image retrieval plus BGE reranking, showing candidate articles, section scores, confidence, and match counts.
 - **bge_nli_dataset.py** – processes a dataset slice with BFS NLI clustering, printing image search results, BGE scores for the top‑M sections, NLI clusters, and a summary of document/section matches including Recall@1/3/5/10 for both BGE and NLI clustering.
 - **bge_nli_graph_dataset.py** – processes a dataset slice using weighted graph-based NLI clustering, printing image search results, BGE scores, relation graph stats, maximal clique indices before truncation, Recall@1/3/5/10 for document and section matches, and separate BGE/NLI timings.
 - **hyde_bge_dataset.py** – performs parallel image and HyDE text retrieval, fuses the candidates, reranks them with BGE, and reports document/section Recall@1/3/5/10 along with image, HyDE, and rerank timings.
 - **score_sections.py** – evaluates a dataset slice and reports Top-k document and section accuracy.
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

Install dependencies and run a sample evaluation:

## Quick Start

Install the requirements and run the evaluation scripts:

```bash
pip install -r requirements.txt
python bge_only.py
python bge_nli_dataset.py
python score_sections.py
```