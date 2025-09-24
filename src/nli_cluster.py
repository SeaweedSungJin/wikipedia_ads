from __future__ import annotations

"""NLI-based clustering of BGE reranked sections.

The current approach constructs a weighted graph using entailment/contradiction
probabilities and ranks maximal cliques.
"""

from itertools import combinations
from typing import List, Tuple, Set

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

__all__ = [
    "load_nli_model",
    "nli_relation",
    "build_relation_graph",
    "maximal_cliques",
    "cluster_sections_clique",
    "cluster_sections_consistency",
]


"""NLI-based clustering utilities.

This module assumes an already-loaded NLI model and tokenizer are provided
by callers (see models.load_nli_model). It handles tokenization length
clamping and graph construction.
"""


def _effective_max_length(tokenizer, model, requested: int) -> int:
    """Return a safe max_length based on tokenizer/model limits.

    Some models (e.g., RoBERTa) have ``max_position_embeddings`` around 514 and
    tokenizers commonly expose ``model_max_length`` as 512, whereas others (e.g.,
    long-range DeBERTa) support 1024+. This helper caps the requested length to
    what the tokenizer/model can handle to avoid runtime shape errors.
    """
    tok_max = getattr(tokenizer, "model_max_length", None)
    # Guard against tokenizers that set a sentinel very large number
    if tok_max is None or tok_max > 1_000_000:
        tok_max = None
    model_max = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    candidates = [requested]
    if isinstance(tok_max, int) and tok_max > 0:
        candidates.append(tok_max)
    if isinstance(model_max, int) and model_max > 0:
        # ``max_position_embeddings`` counts positions including specials; the
        # tokenizer's ``max_length`` already includes specials, so taking the
        # minimum is safe without manual offsets.
        candidates.append(model_max)
    return int(min(candidates))


# ---------------------------------------------------------------------------
# Graph-based maximal clique clustering
# ---------------------------------------------------------------------------


def nli_relation(
    model,
    tokenizer,
    text1: str,
    text2: str,
    device: torch.device | str = "cpu",
    max_length: int = 512,
) -> Tuple[str, float]:
    """Classify the NLI relationship between two texts.

    Runs the NLI model in both directions (text1→text2 and text2→text1) and
    averages the logits to obtain a symmetric relation label.  Returns the
    label (``"contradiction"``, ``"neutral"``, or ``"entailment"``) and its
    probability.
    """

    def _probs(premise: str, hypothesis: str) -> torch.Tensor:
        eff_max_len = _effective_max_length(tokenizer, model, max_length)
        inputs = tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=eff_max_len,
        ).to(device)
        with torch.no_grad():
            return model(**inputs).logits.softmax(dim=1)[0]

    probs = (_probs(text1, text2) + _probs(text2, text1)) / 2
    label_id = int(torch.argmax(probs).item())
    labels = ["contradiction", "neutral", "entailment"]
    return labels[label_id], float(probs[label_id])


def _nli_probs(
    model,
    tokenizer,
    premise: str,
    hypothesis: str,
    device: torch.device | str,
    max_length: int,
) -> torch.Tensor:
    """Return NLI probabilities (contradiction, neutral, entailment)."""

    eff_max_len = _effective_max_length(tokenizer, model, max_length)
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=eff_max_len,
    ).to(device)
    with torch.no_grad():
        return model(**inputs).logits.softmax(dim=1)[0]


@torch.no_grad()
def build_relation_graph(
    sections: List[dict],
    *,
    model,
    tokenizer,
    max_length: int = 512,
    device: torch.device | str = "cpu",
    alpha: float = 1.0,
    beta: float = 1.0,
    clamp_weights: bool = True,
    e_min: float = 0.5,
    margin: float = 0.15,
    tau: float = 0.25,
    batch_size: int = 32,
    edge_rule: str = "avg",
    dir_margin: float = 0.0,
    autocast: bool = True,
    autocast_dtype: str = "fp16",
) -> Tuple[List[Set[int]], List[List[float]], dict]:
    """Construct a weighted graph of section pairs based on NLI probabilities.

    Edges are added only when entailment exceeds ``e_min`` and the gap between
    entailment and contradiction surpasses ``margin``.  The final weight is
    ``max(0, alpha * entailment - beta * contradiction)``; edges with weight
    below ``tau`` are discarded.  ``stats`` counts how many pairwise relations
    were classified with maximum probability as entailment, neutral, or
    contradiction.
    """

    n = len(sections)
    adj: List[Set[int]] = [set() for _ in range(n)]
    weights: List[List[float]] = [[0.0] * n for _ in range(n)]
    stats = {"entailment": 0, "neutral": 0, "contradiction": 0}

    texts = [s.get("section_text", "") for s in sections]
    # Always score all unique pairs (i < j)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    labels = ["contradiction", "neutral", "entailment"]
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        premises = [texts[i] for i, _ in batch]
        hypotheses = [texts[j] for _, j in batch]
        eff_max_len = _effective_max_length(tokenizer, model, max_length)
        inputs1 = tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            truncation=True,
            max_length=eff_max_len,
            padding=True,
        ).to(device)
        inputs2 = tokenizer(
            hypotheses,
            premises,
            return_tensors="pt",
            truncation=True,
            max_length=eff_max_len,
            padding=True,
        ).to(device)

        # Forward passes (optionally bidirectional, with autocast on CUDA)
        use_cuda = torch.cuda.is_available() and (
            (hasattr(device, "type") and device.type == "cuda") or (isinstance(device, str) and device.startswith("cuda"))
        )
        if autocast and use_cuda:
            amp_dtype = torch.float16 if autocast_dtype == "fp16" else torch.bfloat16
            from contextlib import nullcontext
            ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
        else:
            from contextlib import nullcontext
            ctx = nullcontext()

        with ctx:
            probs1 = model(**inputs1).logits.softmax(dim=1)
            probs2 = model(**inputs2).logits.softmax(dim=1)

        # Average (symmetric) probabilities
        probs_avg = (probs1 + probs2) / 2
        # Directional deltas (entailment - contradiction) per orientation
        ent1 = probs1[:, 2]
        contr1 = probs1[:, 0]
        ent2 = probs2[:, 2]
        contr2 = probs2[:, 0]
        d1 = ent1 - contr1
        d2 = ent2 - contr2
        # Averaged values for legacy thresholds
        ent_avg = probs_avg[:, 2]
        contr_avg = probs_avg[:, 0]
        label_ids = torch.argmax(probs_avg, dim=1).cpu().tolist()
        ents = ent_avg.cpu().tolist()
        contrs = contr_avg.cpu().tolist()

        for idx, ((i, j), ent, contr, lid) in enumerate(zip(batch, ents, contrs, label_ids)):
            stats[labels[lid]] += 1
            # Legacy averaged gating
            if ent < e_min or (ent - contr) < margin:
                continue
            # Optional bidirectional minimum on directional (ent-contr)
            if edge_rule == "both_dir":
                if d1[idx].item() < dir_margin or d2[idx].item() < dir_margin:
                    continue
            weight = alpha * ent - beta * contr
            weight = max(0.0, weight)
            if clamp_weights:
                weight = min(1.0, weight)
            if weight < tau:
                continue
            adj[i].add(j)
            adj[j].add(i)
            weights[i][j] = weights[j][i] = weight

    return adj, weights, stats


def _bron_kerbosch(R: Set[int], P: Set[int], X: Set[int], adj: List[Set[int]], cliques: List[List[int]]):
    """Bron–Kerbosch recursive search for maximal cliques."""
    if not P and not X:
        cliques.append(sorted(R))
        return
    for v in list(P):
        _bron_kerbosch(
            R | {v},
            P & adj[v],
            X & adj[v],
            adj,
            cliques,
        )
        P.remove(v)
        X.add(v)


def maximal_cliques(adj: List[Set[int]]) -> List[List[int]]:
    """Enumerate all maximal cliques in the given graph."""
    P = set(range(len(adj)))
    cliques: List[List[int]] = []
    _bron_kerbosch(set(), P, set(), adj, cliques)
    return cliques


def cluster_sections_clique(
    sections: List[dict],
    *,
    model,
    tokenizer,
    max_length: int = 512,
    device: torch.device | str = "cpu",
    max_cluster_size: int = 3,
    alpha: float = 1.0,
    beta: float = 1.0,
    clamp_weights: bool = True,
    lambda_score: float = 0.7,
    e_min: float = 0.5,
    margin: float = 0.15,
    tau: float = 0.25,
    batch_size: int = 32,
    edge_rule: str = "avg",
    dir_margin: float = 0.0,
    autocast: bool = True,
    autocast_dtype: str = "fp16",
) -> Tuple[List[dict], dict]:
    """Cluster sections by finding maximal cliques with weighted edges.

    Edges are gated by ``e_min``, ``margin`` and ``tau`` before being added to the
    graph.  The final cluster score blends the average normalised BGE similarity
    with the mean edge weight using ``lambda_score``.  Returns ``(clusters, stats)``
    where ``clusters`` is sorted by the final score and ``stats`` contains
    pairwise relation counts from the underlying graph construction.
    """

    if not sections:
        return [], {"entailment": 0, "neutral": 0, "contradiction": 0}

    adj, weights, stats = build_relation_graph(
        sections,
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        device=device,
        alpha=alpha,
        beta=beta,
        clamp_weights=clamp_weights,
        e_min=e_min,
        margin=margin,
        tau=tau,
        batch_size=batch_size,
        edge_rule=edge_rule,
        dir_margin=dir_margin,
        autocast=autocast,
        autocast_dtype=autocast_dtype,
    )

    # normalize BGE similarities to [0,1] so they can be blended with edge weights
    raw_scores = [s.get("similarity", 0.0) for s in sections]
    min_score = min(raw_scores)
    max_score = max(raw_scores)
    if max_score > min_score:
        norm_scores = [(s - min_score) / (max_score - min_score) for s in raw_scores]
    else:
        norm_scores = [0.0 for _ in raw_scores]

    cliques = maximal_cliques(adj)
    raw_cliques = [sorted(list(cl)) for cl in cliques]
    clusters: List[dict] = []
    for cl in cliques:
        best_nodes = list(cl)
        best_edge = 0.0
        best_bge = float("-inf")
        if max_cluster_size and len(cl) > max_cluster_size:
            for combo in combinations(cl, max_cluster_size):
                edge_sum = 0.0
                for a, b in combinations(combo, 2):
                    edge_sum += weights[a][b]
                avg_bge = sum(norm_scores[i] for i in combo) / max_cluster_size
                if (
                    edge_sum > best_edge
                    or (edge_sum == best_edge and avg_bge > best_bge)
                    or (
                        edge_sum == best_edge
                        and avg_bge == best_bge
                        and list(combo) < best_nodes
                    )
                ):
                    best_nodes = list(combo)
                    best_edge = edge_sum
                    best_bge = avg_bge
        else:
            best_nodes = list(cl)
            for a, b in combinations(best_nodes, 2):
                best_edge += weights[a][b]
            best_bge = sum(norm_scores[i] for i in best_nodes) / len(best_nodes)

        members = [sections[i] for i in best_nodes]
        num_edges = len(best_nodes) * (len(best_nodes) - 1) / 2
        edge_avg = best_edge / num_edges if num_edges else 0.0
        final_score = lambda_score * best_bge + (1 - lambda_score) * edge_avg
        members.sort(key=lambda s: s.get("similarity", 0.0), reverse=True)
        clusters.append(
            {"avg_score": final_score, "sections": members, "indices": best_nodes}
        )

    stats["raw_cliques"] = raw_cliques
    clusters.sort(key=lambda c: c["avg_score"], reverse=True)
    return clusters, stats


# ---------------------------------------------------------------------------
# Consistency-based greedy pruning clustering
# ---------------------------------------------------------------------------

def _connected_components(adj: List[Set[int]]) -> List[List[int]]:
    n = len(adj)
    seen = [False] * n
    comps: List[List[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        cur = []
        while stack:
            v = stack.pop()
            cur.append(v)
            for nb in adj[v]:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
        comps.append(sorted(cur))
    return comps


def _global_consistency(indices: List[int], weights: List[List[float]]) -> float:
    """Average signed pairwise weight within a set of nodes.

    Unlike the previous variant that clipped negatives with ``max(0, w)``,
    this version treats contradictions (negative weights) as penalties so that
    incoherent groups receive lower consistency scores.
    """
    m = len(indices)
    if m < 2:
        return 0.0
    total = 0.0
    for a in range(m):
        ia = indices[a]
        for b in range(a + 1, m):
            ib = indices[b]
            total += float(weights[ia][ib])
    return (2.0 / (m * (m - 1))) * total


def _greedy_prune(
    indices: List[int],
    weights: List[List[float]],
    target_size: int,
) -> Tuple[List[int], float, List[int]]:
    # Precompute pairwise sums to enable O(1) removal deltas
    target_size = max(1, target_size)
    cur = list(indices)
    if len(cur) <= max(target_size, 2):
        kept = sorted(cur)
        return kept, _global_consistency(kept, weights), []
    m = len(cur)
    # sumC: total sum over pairs i<j of signed weights
    sumC = 0.0
    # contrib[k]: sum over j!=k of signed w_{k,j} within current set
    contrib = {k: 0.0 for k in cur}
    for x in range(m):
        i = cur[x]
        for y in range(x + 1, m):
            j = cur[y]
            cij = float(weights[i][j])
            sumC += cij
            contrib[i] += cij
            contrib[j] += cij
    def score_of(size: int, sC: float) -> float:
        if size < 2:
            return 0.0
        return (2.0 / (size * (size - 1))) * sC
    min_size = max(target_size, 2)
    improved = True
    removed: List[int] = []
    while improved and len(cur) > min_size:
        improved = False
        base_score = score_of(len(cur), sumC)
        best_gain = 0.0
        best_rm = None
        for k in cur:
            size2 = len(cur) - 1
            sC2 = sumC - contrib[k]
            score2 = score_of(size2, sC2)
            gain = score2 - base_score
            if gain > best_gain + 1e-12:  # small epsilon to avoid flaps
                best_gain = gain
                best_rm = k
        if best_rm is not None and best_gain > 0.0:
            # Remove best_rm and update structures
            cur.remove(best_rm)
            # Update sumC and contrib for remaining nodes
            for j in cur:
                if j == best_rm:
                    continue
                cij = float(weights[best_rm][j])
                contrib[j] -= cij
            sumC -= contrib.pop(best_rm, 0.0)
            removed.append(best_rm)
            improved = True
    while len(cur) > target_size:
        best_rm = min(cur, key=lambda k: contrib.get(k, 0.0))
        cur.remove(best_rm)
        for j in cur:
            if j == best_rm:
                continue
            cij = float(weights[best_rm][j])
            contrib[j] -= cij
        sumC -= contrib.pop(best_rm, 0.0)
        removed.append(best_rm)
    final_set = sorted(cur)
    final_score = score_of(len(final_set), sumC)
    return final_set, final_score, removed


def cluster_sections_consistency(
    sections: List[dict],
    *,
    model,
    tokenizer,
    max_length: int = 512,
    device: torch.device | str = "cpu",
    alpha: float = 1.0,
    beta: float = 1.0,
    batch_size: int = 32,
    # adjacency threshold (C_ij > tau_edge forms an edge for component split)
    tau_edge: float = 1e-8,
    hybrid_lambda: float = 0.5,
    target_size: int = 3,
    autocast: bool = True,
    autocast_dtype: str = "fp16",
    edge_rule: str = "avg",
    dir_margin: float = 0.0,
) -> Tuple[List[dict], dict]:
    """Select section clusters by maximizing global consistency via greedy pruning.

    Steps:
    - Compute symmetric NLI probabilities for all pairs and derive C_ij = max(0, alpha*Pe - beta*Pc)
    - Build an adjacency with edges for C_ij > tau_edge, find connected components
    - For each component, greedily remove sentences that increase global consistency until no gain
    - Return all pruned components as clusters, sorted by global consistency score (descending)
    """
    if not sections:
        return [], {"entailment": 0, "neutral": 0, "contradiction": 0}

    target_size = max(1, target_size)

    # Reuse relation graph builder but disable gates, avoid clamping weights.
    # Set e_min=0, margin=-1 so all pairs pass; tau=tau_edge defines connectivity only.
    adj, weights, stats = build_relation_graph(
        sections,
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        device=device,
        alpha=alpha,
        beta=beta,
        clamp_weights=False,
        e_min=0.0,
        margin=-1.0,
        tau=tau_edge,
        batch_size=batch_size,
        edge_rule=edge_rule,
        dir_margin=dir_margin,
        autocast=autocast,
        autocast_dtype=autocast_dtype,
    )

    comps = _connected_components(adj)
    if not comps:
        comps = [[i] for i in range(len(sections))]
    comps.sort(key=len, reverse=True)

    raw_clusters: List[dict] = []
    for comp in comps:
        # Greedy prune main component
        kept_idx, cons_score, removed_idx = _greedy_prune(comp, weights, target_size)
        if kept_idx:
            members = [sections[i] for i in kept_idx]
            members.sort(key=lambda s: s.get("similarity", 0.0), reverse=True)
            raw_clusters.append({
                "type": "kept",
                "indices": kept_idx,
                "sections": members,
                "consistency_raw": float(cons_score),
            })

        # Build subgraphs among removed nodes (if any)
        if removed_idx:
            removed_set = set(removed_idx)
            # Build adjacency restricted to removed nodes
            sub_adj = [set() for _ in removed_idx]
            idx_map = {node: k for k, node in enumerate(removed_idx)}
            for a_i in removed_idx:
                ai_m = idx_map[a_i]
                for b_j in adj[a_i]:
                    if b_j in removed_set:
                        sub_adj[ai_m].add(idx_map[b_j])
            # Connected components within removed nodes (preserve singletons too)
            comps_rm = _connected_components(sub_adj)
            for comp_local in comps_rm:
                orig_nodes = [removed_idx[k] for k in comp_local]
                if len(orig_nodes) > target_size:
                    trimmed, g, _ = _greedy_prune(orig_nodes, weights, target_size)
                else:
                    trimmed = sorted(orig_nodes)
                    g = _global_consistency(trimmed, weights)
                if len(trimmed) < target_size:
                    continue
                members = [sections[i] for i in trimmed]
                members.sort(key=lambda s: s.get("similarity", 0.0), reverse=True)
                raw_clusters.append({
                    "type": "removed",
                    "indices": trimmed,
                    "sections": members,
                    "consistency_raw": float(g),
                })

    # Normalise consistency over all clusters (min-max)
    cons_vals = [c.get("consistency_raw", 0.0) for c in raw_clusters]
    if cons_vals:
        cmin, cmax = min(cons_vals), max(cons_vals)
    else:
        cmin = cmax = 0.0

    # Prepare per-section strength normalisation (prefer combined_score -> rerank_norm -> similarity)
    sec_raw = [
        float(s.get("combined_score", s.get("rerank_norm", s.get("similarity", 0.0))))
        for s in sections
    ]
    if sec_raw:
        smin, smax = min(sec_raw), max(sec_raw)
    else:
        smin = smax = 0.0

    for c in raw_clusters:
        raw = float(c.get("consistency_raw", 0.0))
        cons_norm = 0.0 if cmax <= cmin else (raw - cmin) / (cmax - cmin)
        c["consistency_norm"] = float(cons_norm)
        # Mean section strength within the cluster (min-max over this sample)
        idxs = c.get("indices", [])
        if idxs:
            vals = [sec_raw[i] for i in idxs]
            sec_norm = 0.0 if smax <= smin else sum((v - smin) / (smax - smin) for v in vals) / len(vals)
        else:
            sec_norm = 0.0
        c["section_norm_avg"] = float(sec_norm)
        # Blend consistency and section strength
        lam = max(0.0, min(1.0, float(hybrid_lambda)))
        hybrid = lam * cons_norm + (1.0 - lam) * sec_norm
        c["hybrid_score"] = float(hybrid)
        c["avg_score"] = float(hybrid)

    raw_clusters.sort(key=lambda c: c.get("avg_score", 0.0), reverse=True)
    return raw_clusters, stats
