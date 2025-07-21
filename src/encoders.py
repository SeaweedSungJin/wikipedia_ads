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

class QFormerEncoder(TextEncoder):
    """Encoder wrapper for a Q-former model producing token embeddings."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def encode_pair(self, image_path: str | None, text: str) -> torch.Tensor:
        """Return Q-former token embeddings for an image/text pair."""
        from .utils import load_image

        image = load_image(image_path) if image_path else None
        # LAVIS processors are provided as a dict with separate image/text
        # components.  HuggingFace processors mimic ``__call__``.
        device = next(self.model.parameters()).device

        if isinstance(self.processor, dict):
            image_tensor = self.processor["image"](image) if image is not None else None
            text_tensor = self.processor["text"](text)
            sample = {
                "image": image_tensor.to(device) if hasattr(image_tensor, "to") else image_tensor,
                "text_input": text_tensor.to(device) if hasattr(text_tensor, "to") else text_tensor,
            }
            with torch.no_grad():
                out = self.model.extract_features(sample, mode="qformer")
            return out["qformer_output"]
        else:
            try:
                inputs = self.processor(images=image, text=text, return_tensors="pt")
            except TypeError:
                inputs = self.processor(image=image, text=text, return_tensors="pt")

            for k, v in inputs.items():
                if hasattr(v, "to"):
                    inputs[k] = v.to(device)

            with torch.no_grad():
                out = self.model(**inputs)
            return out.get("qformer_output", out.last_hidden_state)