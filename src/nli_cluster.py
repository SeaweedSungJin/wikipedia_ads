from __future__ import annotations

"""NLI-based clustering of BGE reranked sections.

The algorithm operates on the top-m sections returned from the BGE reranker.
For every pair of sections we compute the bidirectional entailment score using
an NLI model.  Sections are connected if their entailment probability exceeds
``threshold``.  Starting from the highest BGE-scored section we perform a BFS on
this graph to build clusters, limiting each cluster to at most
``max_cluster_size`` sections.  Clusters are ranked by the average BGE
similarity of their members.
"""

from collections import deque
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
) -> List[List[int]]:
    """Build adjacency lists of sections whose entailment exceeds ``threshold``."""
    n = len(sections)
    adj = [[] for _ in range(n)]
    for i in range(n):
        text_i = sections[i].get("section_text", "")
        for j in range(i + 1, n):
            text_j = sections[j].get("section_text", "")
            s1 = entailment_score(model, tokenizer, text_i, text_j, device, max_length)
            s2 = entailment_score(model, tokenizer, text_j, text_i, device, max_length)
            score = max(s1, s2)
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
) -> List[dict]:
    """Cluster BGE reranked sections using pairwise NLI entailment.

    Returns a list of clusters sorted by average BGE similarity.  Each cluster is
    represented as ``{"avg_score": float, "sections": List[dict]}``.
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
        clusters.append({"avg_score": avg_score, "sections": members})

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


def build_relation_graph(
    sections: List[dict],
    *,
    model,
    tokenizer,
    max_length: int = 512,
    device: torch.device | str = "cpu",
) -> Tuple[List[Set[int]], dict]:
    """Construct an undirected graph of non-contradictory section pairs.

    Returns a tuple ``(adjacency, stats)`` where ``adjacency`` is a list of
    neighbor sets and ``stats`` counts how many pairwise relations were
    classified as entailment, neutral, or contradiction.  Individual pairwise
    classifications are commented out below for brevity; remove the leading ``#``
    to inspect them during debugging.
    """

    n = len(sections)
    adj: List[Set[int]] = [set() for _ in range(n)]
    stats = {"entailment": 0, "neutral": 0, "contradiction": 0}

    for i in range(n):
        text_i = sections[i].get("section_text", "")
        for j in range(i + 1, n):
            text_j = sections[j].get("section_text", "")
            label, _ = nli_relation(
                model, tokenizer, text_i, text_j, device=device, max_length=max_length
            )
            stats[label] += 1
            # print(f"Relation({i},{j}) = {label}")  # Uncomment for pairwise details
            if label != "contradiction":
                adj[i].add(j)
                adj[j].add(i)

    return adj, stats


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
) -> List[dict]:
    """Cluster sections by finding maximal cliques of non‑contradictory pairs."""

    if not sections:
        return []

    adj, _ = build_relation_graph(
        sections,
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        device=device,
    )

    cliques = maximal_cliques(adj)
    clusters: List[dict] = []
    for cl in cliques:
        members = [sections[i] for i in cl]
        avg_score = sum(m.get("similarity", 0.0) for m in members) / len(members)
        members.sort(key=lambda s: s.get("similarity", 0.0), reverse=True)
        if max_cluster_size and len(members) > max_cluster_size:
            members = members[:max_cluster_size]
        clusters.append({"avg_score": avg_score, "sections": members})

    clusters.sort(key=lambda c: c["avg_score"], reverse=True)
    return clusters