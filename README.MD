# Wikipedia RAG Prototype

This repository implements a retrieval‑augmented generation (RAG) pipeline for the **Encyclopedic VQA** dataset. Given a question and its accompanying image, the system searches Wikipedia to identify the relevant article and the most pertinent section within that article.

## How it works

1. **Image retrieval** – Every Wikipedia image is embedded with EVA‑CLIP and stored in a FAISS index. The query image is embedded and the index returns the top‑K most similar images, yielding candidate articles.
2. **Section candidates** – Sections (or paragraphs/sentences) from those articles become candidates for answering the question. Optional TF‑IDF filtering can reduce the pool.
3. **Reranking** – Each candidate section is scored by one or more rerankers. Available modules include:
   - `contriever`: cosine similarity between the question text and the section text
   - `jina_m0`: multimodal cross‑encoder scoring the image and text jointly
   - `qformer`: BLIP‑2 Q‑former with late‑interaction scoring
   - `colbert`: ColBERTv2 token interaction model
   - `bge`: lightweight cross‑encoder whose softmax scores are converted to probabilities; entropy provides a confidence measure that determines how many sections to keep
4. **Evaluation** – Scripts such as `score_sections.py` and `bge_only.py` report how often the correct article and section appear within the top ranks.

See [src/README.md](src/README.md) for detailed configuration options and extended usage notes.

## Quick Start

Install the requirements and run the evaluation scripts:

```bash
pip install -r requirements.txt
python bge_only.py
python score_sections.py
```