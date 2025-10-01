NLI Consistency Reranking — Design Notes
=======================================

Overview
- Goal: extract the most self-consistent subset of reranker-selected sections by using pairwise NLI relations so that the VLM reads mutually supportive evidence.
- Inputs per question: top M section candidates (text + metadata) from the reranker stage.
- Output: a single cluster (largest consistent component) whose texts are fed to the VLM, together with diagnostic scores.

Pairwise Relations (build_relation_graph)
- For every unique pair (i, j):
  - Run NLI in both directions and average logits → symmetric probabilities Pe (entailment), Pc (contradiction).
 - Compute non-negative weight: C_ij = max(0, α·Pe − β·Pc). Values below τ_edge are treated as zero and do not form edges.
- Adjacency: add an undirected edge if C_ij ≥ τ_edge (default from config: `nli_tau`). This suppresses extremely weak links.
- Directionality knobs: `nli_edge_rule` can be "avg" or "both_dir" with a `nli_dir_margin` ≥ 0 enforcing minimal per-direction (ent−contr) before accepting an edge.

Consistency Mode (cluster_sections_consistency)
1) Connected components: split the graph using edges where C_ij ≥ τ_edge.
2) Component selection: choose the component with the most nodes (ties broken by the raw consistency score S(V)). Only this component is processed further.
3) Graph reduction: choose between
   - `greedy` (default): reuse precomputed pairwise sums to drop nodes only when the average consistency improves, and optionally trim to `target_size`.
   - `recompute`: follow the Shapley-style pseudo-code, recomputing the global score after removing each candidate node and only applying removals that improve the score by more than `nli_reduction_epsilon`.
4) Scoring: report `consistency_raw`, a unit `consistency_norm` (only one cluster is returned), `section_norm_avg` (mean min–max normalised section strength over the question), and the blended `hybrid_score = λ * consistency_norm + (1 − λ) * section_norm_avg` for diagnostics.

Configuration (config.yaml)
- Model: `nli_models`, `nli_max_length`, `nli_batch_size`, `nli_autocast`, `nli_autocast_dtype`.
- Edge weights: `nli_alpha` (entailment gain), `nli_beta` (contradiction penalty).
- Connectivity: `nli_tau` (τ_edge), `nli_edge_rule` (avg|both_dir), `nli_dir_margin`.
- Selection: `nli_max_cluster` (target cluster size), `nli_hybrid_lambda` (consistency mode diagnostic weight), `nli_reduction_mode` (`greedy` or `recompute`), and `nli_reduction_epsilon` (minimum delta accepted when using `recompute`).
- Reranker feed-in: the pipeline computes section `combined_score` as a calibrated blend of image doc score and reranker/Q‑Former score; NLI consumes the top `m_value` sections.

Practical Tips
- Start with: `nli_hybrid_lambda: 0.3–0.5`, `nli_tau: 0.05–0.12`, `nli_edge_rule: both_dir`, `nli_dir_margin: 0.03`; keep `nli_reduction_mode: greedy` for speed and enable `recompute` when you want exact Shapley-style pruning.
- If recall drops sharply, raise M (e.g., `m_value: 15`) while keeping `nli_max_cluster` small (2–3).
- Prefer shorter, atomic section texts for NLI (first 1–2 sentences) to reduce neutral noise; increase `nli_max_length` only if truncation is evident.

Validation Checklist
- The selected cluster indices should align with strongly entailing pairs, and pruning should favour sections with higher reranker confidence.
- Introducing contradictory evidence should lower `consistency_raw`, making the greedy pass drop that section before the final output.
