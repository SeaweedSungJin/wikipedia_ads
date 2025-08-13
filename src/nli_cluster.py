from __future__ import annotations

"""NLI-based clustering of BGE reranked sections.

Two clustering strategies are provided:

* ``cluster_sections`` – LEGACY BFS approach based on an entailment threshold.
* ``cluster_sections_clique`` – current method constructing a weighted graph
  using entailment/contradiction probabilities and ranking maximal cliques.
"""

from collections import deque
from itertools import combinations
from typing import List, Tuple, Set

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_nli_model(
    model_name: str, device: torch.device | str = "cpu"
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Load an NLI model and tokenizer."""
    print(f"Loading NLI model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer


def entailment_score(
    model,
    tokenizer,
    premise: str,
    hypothesis: str,
    device: torch.device | str = "cpu",
    max_length: int = 512,
) -> float:
    """Return entailment probability for a premise/hypothesis pair."""
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits.softmax(dim=1)[0]
    # label order: contradiction, neutral, entailment
    return float(logits[2].item())


def pairwise_entailments(
    sections: List[dict],
    *,
    model,
    tokenizer,
    threshold: float,
    max_length: int,
    device: torch.device | str,
    batch_size: int,
    ) -> List[List[int]]:
    """LEGACY: Build adjacency lists of sections whose entailment exceeds ``threshold``.

    This BFS/threshold-based clustering path is kept for backward compatibility and
    is not used in the main pipeline.  Prefer :func:`cluster_sections_clique`.
    """
    n = len(sections)
    adj = [[] for _ in range(n)]
    texts = [s.get("section_text", "") for s in sections]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        premises = [texts[i] for i, _ in batch]
        hypotheses = [texts[j] for _, j in batch]
        inputs1 = tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)
        inputs2 = tokenizer(
            hypotheses,
            premises,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)
        with torch.no_grad():
            probs1 = model(**inputs1).logits.softmax(dim=1)
            probs2 = model(**inputs2).logits.softmax(dim=1)
        ent = torch.max(probs1[:, 2], probs2[:, 2]).cpu().tolist()
        for (i, j), score in zip(batch, ent):
            if score >= threshold:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def cluster_sections(
    sections: List[dict],
    *,
    model,
    tokenizer,
    threshold: float = 0.6,
    max_length: int = 512,
    device: torch.device | str = "cpu",
    max_cluster_size: int = 3,
    batch_size: int = 32,
) -> List[dict]:
    """LEGACY: Cluster sections using a BFS on a thresholded entailment graph.

    This path is no longer used in the main pipeline; it remains for older
    experiments.  Returns clusters sorted by average BGE similarity, represented as
    ``{"avg_score": float, "sections": List[dict], "indices": List[int]}``.
    """
    if not sections:
        return []

    adj = pairwise_entailments(
        sections,
        model=model,
        tokenizer=tokenizer,
        threshold=threshold,
        max_length=max_length,
        device=device,
        batch_size=batch_size,
    )

    order = sorted(
        range(len(sections)),
        key=lambda i: sections[i].get("similarity", 0.0),
        reverse=True,
    )
    visited = set()
    clusters_idx: List[List[int]] = []

    for idx in order:
        if idx in visited:
            continue
        visited.add(idx)
        cluster = [idx]
        q = deque([idx])
        while q and len(cluster) < max_cluster_size:
            cur = q.popleft()
            for nb in adj[cur]:
                if nb not in visited and len(cluster) < max_cluster_size:
                    visited.add(nb)
                    cluster.append(nb)
                    q.append(nb)
        clusters_idx.append(cluster)

    clusters: List[dict] = []
    for cl in clusters_idx:
        members = [sections[i] for i in cl]
        avg_score = sum(m.get("similarity", 0.0) for m in members) / len(members)
        clusters.append({"avg_score": avg_score, "sections": members, "indices": cl})

    clusters.sort(key=lambda c: c["avg_score"], reverse=True)
    return clusters


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
        inputs = tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
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

    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
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
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    labels = ["contradiction", "neutral", "entailment"]
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        premises = [texts[i] for i, _ in batch]
        hypotheses = [texts[j] for _, j in batch]
        inputs1 = tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)
        inputs2 = tokenizer(
            hypotheses,
            premises,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)
        probs1 = model(**inputs1).logits.softmax(dim=1)
        probs2 = model(**inputs2).logits.softmax(dim=1)
        probs = (probs1 + probs2) / 2
        ents = probs[:, 2].cpu().tolist()
        contrs = probs[:, 0].cpu().tolist()
        label_ids = torch.argmax(probs, dim=1).cpu().tolist()
        for (i, j), ent, contr, lid in zip(batch, ents, contrs, label_ids):
            stats[labels[lid]] += 1
            if ent < e_min or (ent - contr) < margin:
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