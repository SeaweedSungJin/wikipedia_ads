"""Embedding utilities."""
from __future__ import annotations

from typing import List

import torch
from torch import nn


@torch.no_grad()
def encode_image(image, model, processor) -> torch.Tensor:
    """Encode an image using EVA-CLIP."""

    # Determine the model's device
    device = next(model.parameters()).device

    # Preprocess image and move to device
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device).half()

    # Run the forward pass and normalize
    embedding = model.encode_image(pixel_values)
    embedding = nn.functional.normalize(embedding, dim=-1)

    # Ensure the returned vector matches FAISS expectations
    embedding = embedding.to(dtype=torch.float32, device="cpu")

    # Free GPU memory used for intermediate tensors
    del pixel_values
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    # Return embedding on CPU for convenience
    return embedding

@torch.no_grad()
def get_text_embedding(texts: List[str], model, tokenizer, batch_size: int = 32) -> torch.Tensor:
    """Compute text embeddings using Vicuna."""

    # Early exit for empty inputs
    if not texts:
        return torch.empty(0, model.config.hidden_size, device="cpu")

    device = next(model.parameters()).device
    embeddings = []

    # Tokenize and encode in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)

        try:
            batch_inputs = inputs.to(device)
            outputs = model(**batch_inputs, output_hidden_states=False, output_attentions=False)
        except RuntimeError as e:
            # ``bitsandbytes`` may raise CUDA errors on some hardware.
            # In that case we fall back to CPU inference instead of
            # aborting the entire pipeline.
            if "CUDA" in str(e):
                print("CUDA 오류 발생, CPU에서 임베딩 재시도...")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    # Ignore failures while clearing CUDA cache
                    pass
                model = model.to("cpu")
                batch_inputs = inputs.to("cpu")
                outputs = model(**batch_inputs, output_hidden_states=False, output_attentions=False)
                device = "cpu"
            else:
                raise
        last_hidden = outputs.last_hidden_state

        # Mean pooling with attention mask
        mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        # Normalize each embedding vector
        normed = nn.functional.normalize(mean_pooled, p=2, dim=1)
        embeddings.append(normed.cpu())

        # Clear temporary tensors to reduce GPU memory growth
        del batch_inputs, outputs, last_hidden, mask, sum_embeddings, sum_mask, mean_pooled, normed
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Concatenate all batches and release any cached memory
    all_emb = torch.cat(embeddings, dim=0)
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return all_emb