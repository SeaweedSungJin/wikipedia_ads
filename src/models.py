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
_QFORMER_MODEL = None
_QFORMER_PROCESSOR = None


import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    CLIPImageProcessor,
    BitsAndBytesConfig,
)

def load_jina_reranker(device: str | None = None):
    """Load the Jina cross-modal reranker model."""
    global _RERANKER_MODEL
    if _RERANKER_MODEL is None:
        print("텍스트 리랭커 모델 로딩중...")
        # 1) device_map으로 바로 올리기
        model = AutoModel.from_pretrained(
            "jinaai/jina-reranker-m0",
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        model.to(device)
        model.eval()
        _RERANKER_MODEL = model

    return _RERANKER_MODEL

def load_qformer(
    model_name: str = "Salesforce/blip2-flan-t5-xl",
    device: str | None = None,
    *,
    provider: str = "hf",
    weights_path: str | None = None,
) -> Tuple[object, object]:
    """Load a Q-former model and its processor.

    Parameters
    ----------
    model_name: str
        Name of the pretrained model to load.
    device: str | None
        Optional device for the model.
    provider: str
        ``"hf"`` to load via HuggingFace or ``"lavis"`` to load using the
        LAVIS library.
    weights_path: str | None
        Optional path to fine-tuned weights that will be loaded after the model
        is initialised.
    """

    global _QFORMER_MODEL, _QFORMER_PROCESSOR
    if _QFORMER_MODEL is None or _QFORMER_PROCESSOR is None:
        print("Q-former 모델 로딩중...")

        if provider == "lavis":
            try:
                from lavis.models import load_model_and_preprocess
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "LAVIS library is required for provider='lavis'"
                ) from exc

            try:
                model, vis_proc, txt_proc = load_model_and_preprocess(
                    name=model_name,
                    model_type="pretrain",
                    is_eval=True,
                    device=device or "cpu",
                )
                processor = {
                    "image": vis_proc.get("eval"),
                    "text": txt_proc.get("eval"),
                }
            except Exception as err:
                # Some versions of LAVIS may have incompatible weights.  Rather
                # than failing, we log the error and continue loading via
                # HuggingFace using a compatible checkpoint.
                print(
                    f"LAVIS 모델 로딩 실패: {err}. HuggingFace 로더로 대체합니다."
                )
                provider = "hf"

        if provider == "hf":
            from transformers import AutoProcessor

            # ``blip2_feature_extractor`` is a LAVIS alias and does not exist as
            # a HuggingFace repository.  Map it to a compatible checkpoint when
            # falling back to HF loading.
            hf_name = (
                "Salesforce/blip2-flan-t5-xl" if model_name == "blip2_feature_extractor" else model_name
            )

            model = AutoModel.from_pretrained(
                hf_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            processor = AutoProcessor.from_pretrained(hf_name)

        if weights_path:
            state = torch.load(weights_path, map_location=device or "cpu")
            model.load_state_dict(state, strict=False)

        if device:
            model.to(device)
        model.eval()
        _QFORMER_MODEL, _QFORMER_PROCESSOR = model, processor

    return _QFORMER_MODEL, _QFORMER_PROCESSOR


def compute_late_interaction_similarity(q_tokens: torch.Tensor, c_tokens: torch.Tensor) -> torch.Tensor:
    """Return late interaction similarity score for two token sequences."""

    # Normalize embeddings
    q_tokens = torch.nn.functional.normalize(q_tokens, dim=-1)
    c_tokens = torch.nn.functional.normalize(c_tokens, dim=-1)
    # Compute pairwise dot products
    sim_matrix = torch.bmm(q_tokens, c_tokens.transpose(1, 2))
    # Take best candidate token per query token
    max_sim = sim_matrix.max(dim=2).values
    # Sum over query tokens
    return max_sim.sum(dim=1)

def jina_encode(model, query: str | None = None, image: str | None = None):
    """Return a multimodal embedding using the Jina reranker model."""

    # This helper mirrors the `compute_score` API but yields an embedding
    # vector so we can compare queries and sections with cosine similarity.
    inputs = {"text": query, "image": image}
    with torch.no_grad():
        # The Jina model exposes ``get_multimodal_embeddings`` which
        # accepts a list of {"text": ..., "image": ...} dictionaries.
        return model.get_multimodal_embeddings([inputs])[0]

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
            device_map=3,
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