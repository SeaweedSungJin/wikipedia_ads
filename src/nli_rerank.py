from __future__ import annotations

from typing import List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_nli_model(
    model_name: str, device: torch.device | str = "cpu"
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Load an NLI model and tokenizer."""
    print(f"Loading NLI model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer


def entailment_score(
    model,
    tokenizer,
    premise: str,
    hypothesis: str,
    device: torch.device | str = "cpu",
    max_length: int = 512,
) -> float:
    """Return entailment probability for a premise/hypothesis pair."""
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits.softmax(dim=1)[0]
    # label order: contradiction, neutral, entailment
    return float(logits[2].item())


def score_sections_with_nli(
    question: str,
    sections: List[dict],
    model_name: str | None = None,
    *,
    model: AutoModelForSequenceClassification | None = None,
    tokenizer: AutoTokenizer | None = None,
    max_length: int = 512,
    device: torch.device | str = "cpu",
) -> List[dict]:
    """Score each section against the question using NLI.

    Returns a list of section dicts extended with an ``nli_score`` field.
    Either ``model_name`` or preloaded ``model``/``tokenizer`` must be provided.
    """
    if not sections:
        return []

    if model is None or tokenizer is None:
        if model_name is None:
            raise ValueError("Either model_name or (model, tokenizer) must be supplied")
        model, tokenizer = load_nli_model(model_name, device)

    scored_sections: List[dict] = []
    for sec in sections:
        text = sec.get("section_text", "")
        score1 = entailment_score(
            model, tokenizer, question, text, device, max_length=max_length
        )
        score2 = entailment_score(
            model, tokenizer, text, question, device, max_length=max_length
        )
        score = max(score1, score2)
        sec_copy = dict(sec)
        sec_copy["nli_score"] = score
        scored_sections.append(sec_copy)

    return scored_sections