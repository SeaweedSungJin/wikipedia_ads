# Wikipedia RAG Prototype

This repository implements a small retrieval augmented generation (RAG)
system for answering questions about Wikipedia images. It encodes a
query image and text question, searches a FAISS index of images, and
ranks text sections from matching pages. Section segmentation can be
configured at the section, paragraph or sentence level. Text sections are
ranked by cosine similarity or optionally with cross-encoder rerankers.
When `jina_m0` is enabled the model creates a multimodal embedding from the
query image and text before comparing it to each candidate section.
A `qformer` option is also provided which computes fine-grained similarity using
Q-former tokens and a late-interaction score.  In this mode both the query and
each candidate section are paired with their corresponding images before
calculating similarity.

## Usage

1. Install the requirements
   ```bash
   pip install -r requirements.txt
   ```
2. Edit `config.yaml` with paths to the EVQA dataset and knowledge base.
   Choose a HuggingFace model via `text_encoder_model` (default
   `facebook/contriever`). Set `segment_level` to `section`, `paragraph` or
   `sentence`. When using paragraph mode you can adjust `chunk_size` to control
   the maximum characters per paragraph.
   The `rerankers` section lets you enable ranking modules such as
  `contriever`, `jina_m0`, or `qformer` independently. When `qformer` is
  enabled the pipeline uses a Q-former model and late interaction similarity
  to rank sections. When loading via LAVIS, set `qformer_model` to
  `blip2_feature_extractor` with `qformer_provider: lavis`.  For a smaller
  setup you can load only the HuggingFace checkpoint
  `Salesforce/blip2-flan-t5-xl` by setting `qformer_provider: hf` and
  `qformer_model: Salesforce/blip2-flan-t5-xl`.  The code maps
  `blip2_feature_extractor` to this checkpoint automatically when falling back
  to the HuggingFace loader. Provide a fine-tuned checkpoint via
  `qformer_weights` if desired.
3. Run the pipeline for a single query
   ```bash
   python main.py
   ```
4. Evaluate over a dataset slice
   ```bash
   python eval_dataset.py
   python score_sections.py
   ```
   The `score_sections.py` script measures retrieval accuracy and expects
   `segment_level` to be `section` or `paragraph`. Sentence-level segments
   are not supported for evaluation.

The heavy models, FAISS index and KB JSON are cached so repeated runs are
fast. Optional TF‑IDF filtering can be enabled in `config.yaml`.