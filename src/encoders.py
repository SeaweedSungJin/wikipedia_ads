"""Modular encoder definitions used by the pipeline."""
from __future__ import annotations

from typing import List, Tuple

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
