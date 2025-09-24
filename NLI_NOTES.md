NLI Consistency Reranking — Design Notes
=======================================

Overview
- Goal: re-rank the reranker-selected sections into coherent clusters by using pairwise NLI relations. Coherence should help the VLM by presenting mutually consistent facts.
- Inputs per question: top M section candidates (text + metadata) from the reranker stage.
- Outputs: a ranked list of clusters; the top cluster’s texts are fed to the VLM. We also expose cluster scores for analysis.

Pairwise Relations (build_relation_graph)
- For every unique pair (i, j):
  - Run NLI in both directions and average logits → symmetric probabilities Pe (entailment), Pc (contradiction).
  - Compute signed weight: C_ij = max(0, α·Pe − β·Pc) in graph building; when called by consistency mode we disable clamping and use signed weights internally for scoring, but still use a small τ_edge to define connectivity.
- Adjacency: add an undirected edge if C_ij > τ_edge (default from config: `nli_tau`). This prevents giant, weakly-connected components.
- Directionality knobs: `nli_edge_rule` can be "avg" or "both_dir" with a `nli_dir_margin` ≥ 0 enforcing minimal per-direction (ent−contr) before accepting an edge.

Consistency Mode (cluster_sections_consistency)
1) Connected components: split the graph using edges above τ_edge.
2) Greedy pruning inside each component:
   - Global consistency score S(V) = mean of all signed pair weights in V.
   - Iteratively remove the node whose removal increases S the most, until no improvement or the set size reaches `max(2, nli_max_cluster)`.
   - If still oversized, trim by removing nodes with smallest signed contribution until `target_size`.
3) Recycle removals: build subgraphs among removed nodes, run the same pruning; keep clusters that meet `target_size`.
4) Scoring and ranking:
   - Normalise consistency across clusters → `consistency_norm` (min–max within the question).
   - Compute per-cluster section strength `section_norm_avg`: mean of min–max–normalised section scores (prefer `combined_score`, else `rerank_norm`, else `similarity`) across the same question.
   - Blend: `hybrid_score = λ * consistency_norm + (1 − λ) * section_norm_avg`, where λ = `nli_hybrid_lambda` (default 0.5). Final rank uses `hybrid_score`.

Clique Mode (cluster_sections_clique)
- Build graph with gates (`e_min`, `margin`, `tau`) and clamp C_ij to [0,1].
- Enumerate maximal cliques; if oversized, choose the subset of size `max_cluster_size` maximising total edge weight, tie‑broken by section scores.
- Rank with `avg_score = λ * section_norm_avg + (1 − λ) * edge_avg` using `nli_lambda`.

Configuration (config.yaml)
- Model: `nli_models`, `nli_max_length`, `nli_batch_size`, `nli_autocast`, `nli_autocast_dtype`.
- Edge weights: `nli_alpha` (entailment gain), `nli_beta` (contradiction penalty).
- Connectivity: `nli_tau` (τ_edge), `nli_edge_rule` (avg|both_dir), `nli_dir_margin`.
- Selection: `nli_selection` (consistency|clique), `nli_max_cluster` (target cluster size), `nli_hybrid_lambda` (consistency mode), `nli_lambda` (clique mode).
- Reranker feed-in: the pipeline computes section `combined_score` as a calibrated blend of image doc score and reranker/Q‑Former score; NLI consumes the top `m_value` sections.

Practical Tips
- Start with: `nli_selection: consistency`, `nli_hybrid_lambda: 0.3–0.5`, `nli_tau: 0.05–0.12`, `nli_edge_rule: both_dir`, `nli_dir_margin: 0.03`.
- If recall drops sharply, raise M (e.g., `m_value: 15`) while keeping `nli_max_cluster` small (2–3).
- Prefer shorter, atomic section texts for NLI (first 1–2 sentences) to reduce neutral noise; increase `nli_max_length` only if truncation is evident.

Validation Checklist
- The chosen cluster indices match strong pairwise entailments and rank higher when member section scores are strong.
- Disagreements (contradictions) reduce `consistency_norm` and thus the final rank, as intended.

