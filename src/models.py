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
_COLBERT_MODEL = None
_COLBERT_TOKENIZER = None
_BGE_MODEL = None
_BGE_TOKENIZER = None
_VLM_MODEL = None
_VLM_PROCESSOR = None

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    CLIPImageProcessor,
    BitsAndBytesConfig,
)

def load_jina_reranker(device: str | None = None):
    """Load the Jina cross-encoder reranker model."""
    global _RERANKER_MODEL
    if _RERANKER_MODEL is None:
        print("Jina reranker 모델 로딩중...")
        model = AutoModel.from_pretrained(
            "jinaai/jina-reranker-m0-GGUF",
            trust_remote_code=True,
        )
        if device:
            model.to(device)
        model.eval()
        _RERANKER_MODEL = model

    return _RERANKER_MODEL

def load_bge_reranker(model_name: str = "BAAI/bge-reranker-v2-m3", device: str | None = None):
    """Load the BGE cross-encoder reranker."""

    global _BGE_MODEL, _BGE_TOKENIZER
    if _BGE_MODEL is None or _BGE_TOKENIZER is None:
        print("BGE reranker 모델 로딩중...")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if device:
            model.to(device)
        model.eval()
        _BGE_MODEL, _BGE_TOKENIZER = model, tokenizer

    return _BGE_MODEL, _BGE_TOKENIZER

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
        model = AutoModel.from_pretrained(
            "BAAI/EVA-CLIP-8B",
            trust_remote_code=True,
            torch_dtype=torch.float16,
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


def load_vlm_model(
    model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
    device_map: int | str | dict = "auto",
):
    """Load the LLaVA VLM model and processor."""

    global _VLM_MODEL, _VLM_PROCESSOR
    if _VLM_MODEL is None or _VLM_PROCESSOR is None:
        print("LLaVA 모델 로딩중...")
        from transformers import (
            AutoConfig,
            AutoProcessor,
            LlavaForConditionalGeneration,
            LlavaNextForConditionalGeneration,
        )

        cfg = AutoConfig.from_pretrained(model_name)
        if cfg.model_type == "llava_next":
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device_map,
            )
        else:
            model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device_map,
            )
        processor = AutoProcessor.from_pretrained(model_name)
        model.eval()

        # ``llava-next`` expects ``image_sizes`` to compute the number of image
        # patches, but some vision towers (e.g. EVA-CLIP) do not accept this
        # argument.  Strip it before forwarding to the vision tower so the model
        # can still reason about the image dimensions without raising a runtime
        # error.
        # Some checkpoints expose ``get_vision_tower`` while others expose a
        # ``vision_tower`` attribute directly.  Support both so we can patch the
        # forward method regardless of the underlying API surface.
        vision = getattr(model, "get_vision_tower", None)
        if callable(vision):
            vision = vision()
        else:
            vision = getattr(model, "vision_tower", None)

        if vision is not None and hasattr(vision, "forward"):
            orig_forward = vision.forward

            def forward_without_image_sizes(*args, **kwargs):
                kwargs.pop("image_sizes", None)
                return orig_forward(*args, **kwargs)

            vision.forward = forward_without_image_sizes

        _VLM_MODEL, _VLM_PROCESSOR = model, processor

    return _VLM_MODEL, _VLM_PROCESSOR


def generate_vlm_answer(
    model,
    processor,
    question: str,
    image_path: str,
    sections: list[str],
    max_new_tokens: int = 32,
) -> str:
    """Run the VLM on a question, image, and up to three context sections."""

    from PIL import Image

    ctx = "\n".join(sections[:3])
    prompt = (
        f"Question: {question}\nContext: {ctx}\n"
        "Answer with a short entity only."
    )
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
    ]
    # ``apply_chat_template`` returns token ids by default which causes a
    # mismatch between the number of image tokens and visual features.  Force
    # it to output a plain string so we can pass it back into the processor for
    # tokenisation together with the image.
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    img_token_count = prompt.count("<image>")
    if img_token_count != 1:
        raise ValueError(
            f"Prompt has {img_token_count} <image> tokens; expected exactly 1."
        )
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    # ``processor`` may return a five dimensional tensor of pixel values with a
    # leading image batch dimension.  LLaVA's vision tower expects a 4D tensor
    # (batch, channels, height, width) so squeeze out the redundant dimension
    # for a single image, otherwise fail fast for unexpected shapes.
    pixel_values = inputs.get("pixel_values")
    if pixel_values is not None:
        # ``processor`` may emit a 5D tensor when the image has multiple
        # frames (e.g. GIF).  In such cases use only the first frame to keep
        # a single-image shape.  Fail fast on any other unexpected layout.
        if pixel_values.ndim == 5:
            inputs["pixel_values"] = pixel_values[:, 0]
        elif pixel_values.ndim != 4:
            raise ValueError(f"Unexpected pixel_values shape {pixel_values.shape}")
    # Move tensor inputs to the VLM device but keep non-tensors (e.g. image_sizes)
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0
        )
    answer = processor.batch_decode(
        output[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    )[0]
    return answer.strip()