"""Reranker interfaces and implementations.

This module standardizes how cross-encoders and bi-encoders score
query/section pairs so the pipeline can swap models with minimal changes.
"""
from __future__ import annotations

from typing import List, Tuple

import torch

from .models import (
    load_bge_reranker,
    load_electra_reranker,
    load_jina_reranker,
)


class Reranker:
    """Base interface for scoring query/section pairs."""

    def score(self, query: str, sections: List[str]) -> List[float]:
        raise NotImplementedError


class BGEReranker(Reranker):
    def __init__(self, model_name: str, device: str | int = "cpu", max_length: int = 512, batch_size: int = 32):
        dev = f"cuda:{device}" if isinstance(device, int) else device
        if not torch.cuda.is_available() and isinstance(dev, str) and "cuda" in dev:
            dev = "cpu"
        self.model, self.tokenizer = load_bge_reranker(model_name, dev)
        self.device = dev
        self.max_length = max_length
        self.batch_size = batch_size

    @torch.no_grad()
    def score(self, query: str, sections: List[str]) -> List[float]:
        scores: List[float] = []
        for i in range(0, len(sections), self.batch_size):
            batch = sections[i : i + self.batch_size]
            pairs = [[query, s] for s in batch]
            if not pairs:
                continue
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            ).to(self.device)
            logits = self.model(**inputs, return_dict=True).logits
            scores.extend(logits.view(-1).cpu().float().tolist())
        return scores


class ElectraReranker(Reranker):
    def __init__(self, model_name: str, device: str | int = "cpu", max_length: int = 512, batch_size: int = 32):
        dev = f"cuda:{device}" if isinstance(device, int) else device
        if not torch.cuda.is_available() and isinstance(dev, str) and "cuda" in dev:
            dev = "cpu"
        self.model, self.tokenizer = load_electra_reranker(model_name, dev)
        self.device = dev
        self.max_length = max_length
        self.batch_size = batch_size

    @torch.no_grad()
    def score(self, query: str, sections: List[str]) -> List[float]:
        scores: List[float] = []
        for i in range(0, len(sections), self.batch_size):
            batch = sections[i : i + self.batch_size]
            pairs = [[query, s] for s in batch]
            if not pairs:
                continue
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            ).to(self.device)
            logits = self.model(**inputs, return_dict=True).logits
            if logits.ndim == 2 and logits.shape[1] == 1:
                batch_scores = logits.view(-1)
            elif logits.ndim == 2 and logits.shape[1] == 2:
                batch_scores = logits[:, 1]
            elif logits.ndim == 2 and logits.shape[1] >= 3:
                batch_scores = logits[:, 2]
            else:
                raise ValueError(f"Unexpected Electra logits shape: {tuple(logits.shape)}")
            scores.extend(batch_scores.cpu().float().tolist())
        return scores


class JinaReranker(Reranker):
    def __init__(self, device: str | int = "cpu"):
        dev = f"cuda:{device}" if isinstance(device, int) else device
        if not torch.cuda.is_available() and isinstance(dev, str) and "cuda" in dev:
            dev = "cpu"
        self.model = load_jina_reranker(dev)

    @torch.no_grad()
    def score(self, query: str, sections: List[str]) -> List[float]:
        pairs = [[query, s] for s in sections]
        if not pairs:
            return []
        scores = self.model.compute_score(pairs, max_length=8192, doc_type="text")
        # compute_score may return a list of floats; normalize type
        return [float(s) for s in scores]

