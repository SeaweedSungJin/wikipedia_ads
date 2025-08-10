"""Modular encoder definitions used by the pipeline."""
from __future__ import annotations

from typing import List

import torch
from transformers import AutoModel, AutoTokenizer

from .embedding import get_text_embedding


class TextEncoder:
    """Interface for text encoding backends."""

    def encode(self, texts: List[str]) -> torch.Tensor:
        raise NotImplementedError


class HFTextEncoder(TextEncoder):
    """HuggingFace model wrapper implementing :class:`TextEncoder`."""

    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def encode(self, texts: List[str]) -> torch.Tensor:
        return get_text_embedding(texts, self.model, self.tokenizer)

class JinaM0Encoder(TextEncoder):
    """Encode queries and texts with the Jina reranker model."""

    def __init__(self, model):
        self.model = model

    def encode_query(self, image_path: str, query: str) -> torch.Tensor:
        from .models import jina_encode

        return jina_encode(self.model, query=query, image=image_path)

    def encode(self, texts: List[str]) -> torch.Tensor:
        from .models import jina_encode

        return torch.stack([jina_encode(self.model, query=t) for t in texts])


class ColBERTEncoder(TextEncoder):
    """Encoder for ColBERT-style token embeddings."""

    def __init__(self, model, tokenizer, *, max_length: int = 180):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def encode_tokens(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
        tokens = out.last_hidden_state.squeeze(0)
        mask = inputs["attention_mask"].squeeze(0).bool()
        return tokens[mask].cpu()

    def encode(self, texts: List[str]) -> torch.Tensor:
        # Mean pool for compatibility with the TextEncoder interface
        embs = [self.encode_tokens(t).mean(dim=0) for t in texts]
        return torch.stack(embs)