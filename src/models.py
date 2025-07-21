"""Utilities for loading machine learning models."""
from __future__ import annotations

import os
from typing import Tuple

# Cached models and tokenizers.  These are initialised on first use so
# that subsequent pipeline runs do not reload them from disk.
_IMAGE_MODEL = None
_IMAGE_PROCESSOR = None
_TEXT_MODEL = None
_TOKENIZER = None
_RERANKER_MODEL = None


import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    CLIPImageProcessor,
    BitsAndBytesConfig,
)

def load_jina_reranker(device_map="auto"):
    """Load the Jina cross-modal reranker model.

    This model scores a (query, document) pair directly rather than
    producing embeddings, which can yield more accurate ranking.
    """
    global _RERANKER_MODEL
    if _RERANKER_MODEL is None:
        print("텍스트 리랭커 모델 로딩중...")
        model = AutoModel.from_pretrained(
            "jinaai/jina-reranker-m0",
            torch_dtype="auto",
            trust_remote_code=True,
            device_map=device_map,
        )
        model.eval()
        _RERANKER_MODEL = model
    return _RERANKER_MODEL



def get_device() -> str:
    """Return the name of the available torch device."""

    # Prefer CUDA when available
    return "cuda" if torch.cuda.is_available() else "cpu"


def setup_cuda() -> None:
    """Apply CUDA specific optimisations if available."""

    if get_device() == "cuda":
        # Clear any cached memory
        torch.cuda.empty_cache()
        # Enable segmented memory allocation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def load_image_model(device_map="auto") -> Tuple[AutoModel, CLIPImageProcessor]:
    """Load EVA-CLIP image model, caching the result."""

    global _IMAGE_MODEL, _IMAGE_PROCESSOR
    if _IMAGE_MODEL is None or _IMAGE_PROCESSOR is None:
        print("이미지 모델 로딩중...")
        # Load the model in 8-bit mode to reduce memory usage
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModel.from_pretrained(
            "BAAI/EVA-CLIP-8B",
            trust_remote_code=True,
            quantization_config=quant_cfg,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
        processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        model.eval()
        _IMAGE_MODEL, _IMAGE_PROCESSOR = model, processor
    return _IMAGE_MODEL, _IMAGE_PROCESSOR


def load_text_model(model_name: str, device_map: int | str = "auto") -> Tuple[AutoModel, AutoTokenizer]:
    """Load a HuggingFace text model, caching the result."""

    global _TEXT_MODEL, _TOKENIZER
    if _TEXT_MODEL is None or _TOKENIZER is None:
        print("텍스트 모델 로딩중...")
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map=device_map,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()
        _TEXT_MODEL, _TOKENIZER = model, tokenizer
    return _TEXT_MODEL, _TOKENIZER


def load_text_encoder(model_name: str, device_map: int | str = "auto"):
    """Return a :class:`HFTextEncoder` instance for ``model_name``."""
    model, tokenizer = load_text_model(model_name, device_map)
    from .encoders import HFTextEncoder

    return HFTextEncoder(model, tokenizer)