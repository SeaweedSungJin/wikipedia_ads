# Encyclopedic VQA RAG Pipeline

This project implements a retrieval‑augmented pipeline to answer visual questions using Wikipedia. Given a question and its accompanying image, the system retrieves relevant articles via image similarity, segments them into sections, reranks those sections with text models, and selects high‑confidence sections using entropy of BGE cross‑encoder scores.

## Pipeline Overview
1. Precompute EVA‑CLIP embeddings for Wikipedia images and build a FAISS index.
2. For each query:
   - Embed the question image and search FAISS for top‑k similar images, yielding candidate articles. Non‑article pages (e.g., "list of", "outline of") are filtered and replacements fetched.
   - Split the articles into sections (or paragraphs/sentences depending on the configuration) to form candidate passages.
   - Rerank candidates with any combination of Contriever, Jina M0, ColBERT, and BGE. BGE scores of the top‑M sections are transformed with softmax, entropy is computed, and only high‑confidence sections are returned.
   - Optionally, cluster the BGE top‑M sections with an NLI model. Two strategies are available:
     1. **BFS clustering** (`bge_nli.py`) – sections above an entailment threshold are connected and clusters are grown breadth‑first from the highest BGE scores.
     2. **Graph clique clustering** (`bge_nli_graph.py`) – every section pair is classified as entailment, neutral, or contradiction; an undirected graph is built using non‑contradictory edges and maximal cliques are ranked by their average BGE similarity.
   - Evaluation scripts compare the selected sections or clusters with ground truth to report document and document+section accuracy.

## Configuration & Files
 - **config.yaml** – paths, model options, search parameters, and thresholds such as the BGE confidence, NLI threshold, max length, and cluster size.
- **main.py** – processes a single query end‑to‑end and prints the retrieved sections.
- **bge_only.py** – runs image retrieval plus BGE reranking, showing candidate articles, section scores, confidence, and match counts.
 - **bge_nli.py** – runs the pipeline and clusters the BGE top‑M sections with a simple BFS approach.
 - **bge_nli_graph.py** – alternative execution script that builds an NLI relation graph, finds maximal cliques, and ranks them by average BGE score.
 - **bge_nli_dataset.py** – processes a dataset slice, printing image search results, BGE scores for the top‑M sections, NLI clusters, and a summary of document/section matches including Recall@1/3/5/10 for both BGE and NLI clustering.
 - **bge_nli_graph_dataset.py** – processes a dataset slice using the graph-based NLI clustering, printing image search results, BGE scores, relation graph stats, and Recall@1/3/5/10 for document and section matches.
 - **score_sections.py** – evaluates a dataset slice and reports Top-k document and section accuracy.
 - **src/config.py** – dataclasses for configuration and a YAML loader that drops unknown keys.
 - **src/pipeline.py** – orchestrates image search, section generation, reranking, and entropy‑based confidence filtering.
 - **src/encoders.py** – wrappers around Contriever, Jina M0, ColBERT, and BGE scoring models.
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