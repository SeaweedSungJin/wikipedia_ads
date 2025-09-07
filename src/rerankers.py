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


def _device_index(device: str | int | torch.device) -> int | None:
    if isinstance(device, int):
        return device
    if isinstance(device, torch.device):
        device = str(device)
    if isinstance(device, str) and "cuda" in device:
        try:
            return int(device.split(":")[1]) if ":" in device else 0
        except Exception:
            return None
    return None


class BGEReranker(Reranker):
    def __init__(self, model_name: str, device: str | int = "cpu", max_length: int = 512, batch_size: int = 64):
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
        use_cuda = torch.cuda.is_available() and isinstance(self.device, str) and "cuda" in self.device
        prefetch_stream = torch.cuda.Stream(device=_device_index(self.device)) if use_cuda else None

        next_inputs = None
        next_pairs = None
        total = len(sections)
        i = 0
        while i < total:
            if next_inputs is None:
                batch = sections[i : i + self.batch_size]
                pairs = [[query, s] for s in batch]
                tok = self.tokenizer(
                    pairs, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length
                )
                if use_cuda:
                    assert prefetch_stream is not None
                    with torch.cuda.stream(prefetch_stream):
                        moved = {}
                        for k, v in tok.items():
                            moved[k] = v.to(self.device, non_blocking=True)
                    next_inputs = moved
                else:
                    next_inputs = tok.to(self.device)

            # Prefetch next batch while computing current
            j = i + self.batch_size
            if j < total:
                batch2 = sections[j : j + self.batch_size]
                next_pairs = [[query, s] for s in batch2]
                tok2 = self.tokenizer(
                    next_pairs, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length
                )
                if use_cuda:
                    with torch.cuda.stream(prefetch_stream):
                        moved2 = {}
                        for k, v in tok2.items():
                            moved2[k] = v.to(self.device, non_blocking=True)
                    after_inputs = moved2
                else:
                    after_inputs = tok2.to(self.device)
            else:
                after_inputs = None

            if use_cuda:
                torch.cuda.current_stream().wait_stream(prefetch_stream)  # ensure inputs ready

            if use_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(**next_inputs, return_dict=True).logits
            else:
                logits = self.model(**next_inputs, return_dict=True).logits

            if use_cuda:
                idx = _device_index(self.device)
                try:
                    if idx is not None:
                        torch.cuda.synchronize(idx)
                    else:
                        torch.cuda.synchronize()
                except Exception:
                    pass

            scores.extend(logits.view(-1).cpu().float().tolist())

            next_inputs = after_inputs
            i = j

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
        use_cuda = torch.cuda.is_available() and isinstance(self.device, str) and "cuda" in self.device
        prefetch_stream = torch.cuda.Stream(device=_device_index(self.device)) if use_cuda else None

        next_inputs = None
        total = len(sections)
        i = 0
        while i < total:
            if next_inputs is None:
                batch = sections[i : i + self.batch_size]
                pairs = [[query, s] for s in batch]
                tok = self.tokenizer(
                    pairs, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length
                )
                if use_cuda:
                    assert prefetch_stream is not None
                    with torch.cuda.stream(prefetch_stream):
                        moved = {}
                        for k, v in tok.items():
                            moved[k] = v.to(self.device, non_blocking=True)
                    next_inputs = moved
                else:
                    next_inputs = tok.to(self.device)

            j = i + self.batch_size
            if j < total:
                batch2 = sections[j : j + self.batch_size]
                pairs2 = [[query, s] for s in batch2]
                tok2 = self.tokenizer(
                    pairs2, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length
                )
                if use_cuda:
                    with torch.cuda.stream(prefetch_stream):
                        moved2 = {}
                        for k, v in tok2.items():
                            moved2[k] = v.to(self.device, non_blocking=True)
                    after_inputs = moved2
                else:
                    after_inputs = tok2.to(self.device)
            else:
                after_inputs = None

            if use_cuda:
                torch.cuda.current_stream().wait_stream(prefetch_stream)

            if use_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(**next_inputs, return_dict=True).logits
            else:
                logits = self.model(**next_inputs, return_dict=True).logits

            if logits.ndim == 2 and logits.shape[1] == 1:
                batch_scores = logits.view(-1)
            elif logits.ndim == 2 and logits.shape[1] == 2:
                batch_scores = logits[:, 1]
            elif logits.ndim == 2 and logits.shape[1] >= 3:
                batch_scores = logits[:, 2]
            else:
                raise ValueError(f"Unexpected Electra logits shape: {tuple(logits.shape)}")

            if use_cuda:
                idx = _device_index(self.device)
                try:
                    if idx is not None:
                        torch.cuda.synchronize(idx)
                    else:
                        torch.cuda.synchronize()
                except Exception:
                    pass

            scores.extend(batch_scores.cpu().float().tolist())
            next_inputs = after_inputs
            i = j
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
