from __future__ import annotations

"""NLI-based clustering of reranked sections using consistency pruning."""

from typing import List, Tuple, Set

import torch

__all__ = [
    "nli_relation",
    "build_relation_graph",
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
# Pairwise relation graph helpers
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
    # adjacency threshold (C_ij ≥ tau_edge forms an edge for component split)
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
    - Build an adjacency with edges for C_ij ≥ tau_edge and find connected components
    - Select the connected component with the largest node count (ties broken by raw consistency)
    - Greedily prune that component until removing any node no longer improves global consistency,
      then trim down to ``target_size`` if needed
    - Return a single cluster describing the kept component members
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

    def _component_key(indices: List[int]) -> Tuple[int, float]:
        return (len(indices), _global_consistency(indices, weights))

    selected = max(comps, key=_component_key)
    kept_idx, cons_score, _ = _greedy_prune(selected, weights, target_size)

    if not kept_idx:
        kept_idx = sorted(selected[:target_size])
        cons_score = _global_consistency(kept_idx, weights)

    members = [sections[i] for i in kept_idx]
    members.sort(key=lambda s: s.get("similarity", 0.0), reverse=True)

    # Compute section strength normalisation (prefer combined_score -> rerank_norm -> similarity)
    sec_raw = [
        float(s.get("combined_score", s.get("rerank_norm", s.get("similarity", 0.0))))
        for s in sections
    ]
    if sec_raw:
        smin, smax = min(sec_raw), max(sec_raw)
    else:
        smin = smax = 0.0

    if kept_idx:
        vals = [sec_raw[i] for i in kept_idx]
        sec_norm = 0.0 if smax <= smin else sum((v - smin) / (smax - smin) for v in vals) / len(vals)
    else:
        sec_norm = 0.0

    lam = max(0.0, min(1.0, float(hybrid_lambda)))
    cons_norm = 1.0 if kept_idx else 0.0
    hybrid = lam * cons_norm + (1.0 - lam) * sec_norm

    cluster = {
        "type": "selected",
        "indices": kept_idx,
        "sections": members,
        "consistency_raw": float(cons_score),
        "consistency_norm": float(cons_norm),
        "section_norm_avg": float(sec_norm),
        "hybrid_score": float(hybrid),
        "avg_score": float(hybrid),
    }

    return [cluster], stats
