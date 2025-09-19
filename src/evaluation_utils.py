"""Shared helper routines for evaluation scripts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

from .utils import normalize_title, normalize_url_to_title

_BASE_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)


@dataclass(frozen=True)
class GroundTruth:
    """Container with normalised titles and optional section ids."""

    title_set: set[str]
    section_pairs: set[tuple[str, int]]


def compute_k_values(max_k: Optional[int]) -> list[int]:
    """Return the sorted list of K values that should be reported."""

    values = {k for k in _BASE_K_VALUES if max_k is None or k <= max_k}
    if max_k:
        values.add(max_k)
    return sorted(values)


def init_recall_dict(k_values: Sequence[int]) -> dict[int, int]:
    """Initialise a recall counter dictionary."""

    return {k: 0 for k in k_values}


def build_ground_truth(sample) -> Optional[GroundTruth]:
    """Extract normalised titles and section ids from a dataset sample."""

    raw_titles = str(sample.wikipedia_title or "").split("|")
    raw_urls = str(sample.wikipedia_url or "").split("|")

    titles: list[str] = []
    title_set: set[str] = set()

    for value in raw_titles:
        if not value.strip():
            continue
        normalised = normalize_title(value)
        if not normalised:
            continue
        title_set.add(normalised)
        if normalised not in titles:
            titles.append(normalised)

    for value in raw_urls:
        if not value.strip():
            continue
        normalised = normalize_url_to_title(value)
        if not normalised:
            continue
        title_set.add(normalised)
        if normalised not in titles:
            titles.append(normalised)

    if not title_set:
        return None

    section_raw = sample.metadata.get("evidence_section_id")
    section_ids: list[int] = []
    if isinstance(section_raw, str):
        for value in section_raw.split("|"):
            value = value.strip()
            if not value:
                continue
            try:
                section_ids.append(int(value))
            except ValueError:
                continue
    elif section_raw is not None:
        try:
            section_ids.append(int(section_raw))
        except ValueError:
            pass

    section_pairs: set[tuple[str, int]] = set()
    if section_ids:
        for title, section_id in zip(titles, section_ids):
            section_pairs.add((title, section_id))

    return GroundTruth(title_set=title_set, section_pairs=section_pairs)


def update_recall_from_rank(
    recall: dict[int, int],
    rank: Optional[int],
    k_values: Sequence[int],
) -> None:
    """Increment recall counters when a document is found at ``rank``."""

    if rank is None:
        return
    for k in k_values:
        if rank <= k:
            recall[k] += 1


def update_section_hits(
    indices: Iterable[int],
    sections: Sequence,
    ground_truth: GroundTruth,
    doc_hits: dict[int, int],
    section_hits: dict[int, int],
    k_values: Sequence[int],
    get_title: Callable[[object], str],
    get_section_idx: Callable[[object], Optional[int]],
) -> None:
    """Update recall counters for section-based rankings."""

    idx_list = list(indices)
    for k in k_values:
        top_idx = idx_list[:k]
        if top_idx and _matches_title(top_idx, sections, ground_truth, get_title):
            doc_hits[k] += 1
        if ground_truth.section_pairs and top_idx:
            if _matches_section(top_idx, sections, ground_truth, get_title, get_section_idx):
                section_hits[k] += 1


def _matches_title(
    indices: Sequence[int],
    sections: Sequence,
    ground_truth: GroundTruth,
    get_title: Callable[[object], str],
) -> bool:
    for idx in indices:
        title = normalize_title(get_title(sections[idx]))
        if title in ground_truth.title_set:
            return True
    return False


def _matches_section(
    indices: Sequence[int],
    sections: Sequence,
    ground_truth: GroundTruth,
    get_title: Callable[[object], str],
    get_section_idx: Callable[[object], Optional[int]],
) -> bool:
    for idx in indices:
        section_id = get_section_idx(sections[idx])
        if section_id is None:
            continue
        title = normalize_title(get_title(sections[idx]))
        if (title, section_id) in ground_truth.section_pairs:
            return True
    return False
