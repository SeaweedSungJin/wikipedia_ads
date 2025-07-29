"""Modular encoder definitions used by the pipeline."""
from __future__ import annotations

from typing import List, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

from .embedding import get_text_embedding
from PIL import Image


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

    def __init__(self, model, processor, *, max_length: int = 512):
        self.model = model
        self.processor = processor
        self.max_length = max_length

    def encode_pair(self, image_path: str | None, text: str) -> torch.Tensor:
        """Return Q-former token embeddings for an image/text pair."""
        from .utils import load_image

        if image_path:
            image = load_image(image_path)
        else:
            image = Image.new("RGB", (224, 224), color=0)
        # LAVIS processors are provided as a dict with separate image/text
        # components.  HuggingFace processors mimic ``__call__``.
        device = next(self.model.parameters()).device

        if isinstance(self.processor, dict):
            image_tensor = self.processor["image"](image) if image is not None else None
            text_tensor = self.processor["text"](text, max_length=self.max_length)
            sample = {
                "image": image_tensor.to(device) if hasattr(image_tensor, "to") else image_tensor,
                "text_input": text_tensor.to(device) if hasattr(text_tensor, "to") else text_tensor,
            }
            with torch.no_grad():
                out = self.model.extract_features(sample, mode="qformer")
            return out["qformer_output"]
        else:
            try:
                inputs = self.processor(
                    images=image,
                    text=text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                )
            except TypeError:
                inputs = self.processor(
                    image=image,
                    text=text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                )

            # ``Blip2Model`` requires ``decoder_input_ids`` for the language
            # model even when we only need the Q-former outputs. Some processor
            # variants may return ``None`` for ``input_ids`` when no tokenizer is
            # provided, so fall back to calling the tokenizer explicitly.
            input_ids = inputs.get("input_ids")
            if input_ids is None and hasattr(self.processor, "tokenizer"):
                input_ids = self.processor.tokenizer(text, return_tensors="pt").input_ids
                inputs["input_ids"] = input_ids
            if input_ids is not None and "decoder_input_ids" not in inputs:
                inputs["decoder_input_ids"] = input_ids

            for k, v in inputs.items():
                if hasattr(v, "to"):
                    inputs[k] = v.to(device)

            with torch.no_grad():
                out = self.model(**inputs)

            # HuggingFace ``Blip2Model`` returns ``qformer_output`` while other
            # variants might expose ``last_hidden_state``. Support both to avoid
            # attribute errors.
            if hasattr(out, "qformer_output") and out.qformer_output is not None:
                return out.qformer_output
            if hasattr(out, "last_hidden_state"):
                return out.last_hidden_state

            # Fall back to the first element for plain tuple outputs
            return out[0]


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