"""Shared utilities for running the EchoSight Q-Former reranker."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

# These imports require the EchoSight repository to be available.
from omegaconf import OmegaConf  # type: ignore
from lavis.models import load_preprocess  # type: ignore
from lavis.models.blip2_models.blip2_qformer_reranker import (  # type: ignore
    Blip2QformerReranker,
)

ROOT = Path(__file__).resolve().parent
_ECHOSIGHT_CANDIDATES = [ROOT / "EchoSight", ROOT.parent / "EchoSight"]
ECHOSIGHT_ROOT: Path | None = None
for _cand in _ECHOSIGHT_CANDIDATES:
    if _cand.exists():
        ECHOSIGHT_ROOT = _cand
        break
if ECHOSIGHT_ROOT is None:
    raise FileNotFoundError(
        "EchoSight project not found; expected at one of: "
        + ", ".join(str(c) for c in _ECHOSIGHT_CANDIDATES)
    )
if str(ECHOSIGHT_ROOT) not in sys.path:
    sys.path.insert(0, str(ECHOSIGHT_ROOT))

try:
    from data_utils import targetpad_transform  # type: ignore
except Exception:
    targetpad_transform = None


@dataclass(frozen=True)
class QFormerResources:
    model: torch.nn.Module
    vis_processor: Callable | None
    text_processor: Callable | None


@dataclass(frozen=True)
class ScoringResults:
    cls_max: torch.Tensor
    maxsim: torch.Tensor
    logsumexp: torch.Tensor


_RESOURCE_CACHE: Dict[Tuple[str, str, int], QFormerResources] = {}


def _device_to_str(device: str | int | torch.device | None) -> str:
    if isinstance(device, torch.device):
        return str(device)
    if isinstance(device, int):
        return f"cuda:{device}"
    if isinstance(device, str):
        return device
    return "cpu"


def load_qformer_resources(
    ckpt_path: Path,
    device: str | int | torch.device = "cuda:0",
    max_text_length: int | None = 256,
) -> QFormerResources:
    """Load the EchoSight Q-Former reranker and preprocessing utilities."""

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Q-Former checkpoint not found: {ckpt_path}")

    device_str = _device_to_str(device)
    if "cuda" in device_str and not torch.cuda.is_available():
        device_str = "cpu"
    key = (str(ckpt_path.resolve()), device_str, int(max_text_length or 0))
    cached = _RESOURCE_CACHE.get(key)
    if cached is not None:
        return cached

    cfg_path = (
        ECHOSIGHT_ROOT
        / "lavis"
        / "configs"
        / "models"
        / "blip2"
        / "blip2_pretrain.yaml"
    )
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_readonly(cfg, False)
    model_cfg = cfg.model
    OmegaConf.set_readonly(model_cfg, False)
    model_cfg.load_finetuned = False
    model_cfg.load_pretrained = False
    model_cfg.pretrained = ""

    qformer = Blip2QformerReranker.from_config(model_cfg)
    qformer.eval()
    qformer.use_vanilla_qformer = True

    vis_processors, txt_processors = load_preprocess(cfg.preprocess)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    filtered = [k for k in list(state_dict.keys()) if k.startswith("Qformer.cls.predictions")]
    for key_name in filtered:
        state_dict.pop(key_name, None)
    msg = qformer.load_state_dict(state_dict, strict=False)
    meta_params = [
        name
        for name, param in qformer.named_parameters()
        if getattr(param, "is_meta", False) or param.device.type == "meta"
    ]
    if meta_params:
        qformer = qformer.to_empty(device=torch.device("cpu"))
        qformer.load_state_dict(state_dict, strict=False)
        meta_params = [
            name
            for name, param in qformer.named_parameters()
            if getattr(param, "is_meta", False) or param.device.type == "meta"
        ]
        if meta_params:
            raise RuntimeError(
                f"Q-Former still has meta parameters after reload: {meta_params[:5]}"
            )
    if msg.missing_keys:
        missing = [k for k in msg.missing_keys if not k.startswith("Qformer.cls.predictions")]
        if missing:
            print(f"[WARN] Missing keys when loading Q-Former: {missing[:5]}...")
    if msg.unexpected_keys:
        unexpected = [k for k in msg.unexpected_keys if not k.startswith("Qformer.cls.predictions")]
        if unexpected:
            print(f"[WARN] Unexpected keys: {unexpected[:5]}...")

    device_obj = torch.device(device_str)
    qformer = qformer.to(device_obj)

    tokenizer = qformer.tokenizer
    if tokenizer is not None:
        limits: List[int] = []
        if max_text_length is not None and max_text_length > 0:
            limits.append(max_text_length)
        txt_cfg_len = getattr(model_cfg, "max_txt_len", None)
        if txt_cfg_len:
            limits.append(int(txt_cfg_len))
        tok_current = getattr(tokenizer, "model_max_length", None)
        if tok_current and tok_current < 1_000_000:
            limits.append(int(tok_current))
        if limits:
            tokenizer.model_max_length = min(limits)

    vis_proc = None
    image_size = getattr(model_cfg, "image_size", 224)
    if targetpad_transform is not None:
        try:
            vis_proc = targetpad_transform(1.25, image_size)
        except Exception:
            vis_proc = None
    if vis_proc is None and vis_processors:
        vis_proc = vis_processors.get("eval")
    txt_proc = txt_processors.get("eval") if txt_processors else None

    resources = QFormerResources(qformer, vis_proc, txt_proc)
    _RESOURCE_CACHE[key] = resources
    return resources


@torch.no_grad()
def encode_query_multimodal(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    question: str,
) -> torch.Tensor:
    """Return normalized multimodal query embeddings (num_query_token, dim)."""

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    sample = {
        "image": image_tensor.to(device=device, dtype=dtype),
        "text_input": [question],
    }
    multimodal = model.extract_features(sample, mode="multimodal").multimodal_embeds
    return multimodal[0].to(torch.float32).cpu()


@torch.no_grad()
def encode_sections_text(
    model: torch.nn.Module,
    texts: List[str],
    batch_size: int,
    token_keep: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode section texts using the EchoSight Q-Former forward path."""

    device = next(model.parameters()).device
    tokenizer = model.tokenizer

    cls_chunks: List[torch.Tensor] = []
    token_chunks: List[torch.Tensor] = []
    mask_chunks: List[torch.Tensor] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        outputs = model.extract_features({"text_input": batch}, mode="text")
        proj = outputs.text_embeds_proj
        if proj is None:
            raise RuntimeError("Q-Former text features not returned by extract_features")
        proj = proj.to(torch.float32).cpu()

        max_len = getattr(tokenizer, "model_max_length", None)
        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_len if isinstance(max_len, int) and max_len > 0 else None,
        )
        mask = tokens.attention_mask[:, 1:].to(torch.float32)

        cls_chunks.append(proj[:, 0, :])
        token_part = proj[:, 1:, :]

        if token_part.shape[1] < token_keep:
            pad_len = token_keep - token_part.shape[1]
            token_part = F.pad(token_part, (0, 0, 0, pad_len))
            mask = F.pad(mask, (0, pad_len))
        else:
            token_part = token_part[:, :token_keep, :]
            mask = mask[:, :token_keep]

        token_chunks.append(token_part)
        mask_chunks.append(mask)

    cls_embs = torch.cat(cls_chunks, dim=0)
    token_embs = torch.cat(token_chunks, dim=0)
    token_masks = torch.cat(mask_chunks, dim=0).bool()
    return cls_embs, token_embs, token_masks


def compute_cls_max_scores(
    fusion_tokens: torch.Tensor, cls_embs: torch.Tensor
) -> torch.Tensor:
    """Original Q-Former score using CLS embeddings and max similarity."""

    sim = cls_embs @ fusion_tokens.T  # (N, num_tokens)
    scores, _ = torch.max(sim, dim=1)
    return scores


def _pairwise_similarity(
    section_tokens: torch.Tensor,
    fusion_tokens: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("nld,qd->nlq", section_tokens, fusion_tokens)


def compute_maxsim_scores(
    fusion_tokens: torch.Tensor,
    section_tokens: torch.Tensor,
    token_masks: torch.Tensor,
) -> torch.Tensor:
    sim = _pairwise_similarity(section_tokens, fusion_tokens)
    mask = token_masks.unsqueeze(-1)
    sim = sim.masked_fill(~mask, float("-inf"))
    per_query_max = torch.max(sim, dim=1).values
    per_query_max = torch.nan_to_num(per_query_max, nan=-1e4, neginf=-1e4)
    return per_query_max.mean(dim=1)


def compute_logsumexp_scores(
    fusion_tokens: torch.Tensor,
    section_tokens: torch.Tensor,
    token_masks: torch.Tensor,
) -> torch.Tensor:
    sim = _pairwise_similarity(section_tokens, fusion_tokens)
    mask = token_masks.unsqueeze(-1)
    sim = sim.masked_fill(~mask, float("-inf"))
    logsum = torch.logsumexp(sim, dim=1)
    token_counts = mask.sum(dim=1).clamp(min=1).to(logsum.dtype)
    logsum = logsum - torch.log(token_counts)
    logsum = torch.nan_to_num(logsum, nan=-1e4, neginf=-1e4)
    return logsum.mean(dim=1)


def score_sections(
    fusion_tokens: torch.Tensor,
    section_tokens: torch.Tensor,
    cls_embs: torch.Tensor,
    token_masks: torch.Tensor,
) -> ScoringResults:
    scores_cls = compute_cls_max_scores(fusion_tokens, cls_embs)
    scores_maxsim = compute_maxsim_scores(fusion_tokens, section_tokens, token_masks)
    scores_logsumexp = compute_logsumexp_scores(
        fusion_tokens, section_tokens, token_masks
    )
    return ScoringResults(scores_cls, scores_maxsim, scores_logsumexp)


