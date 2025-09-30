"""Helpers for caching FAISS document retrieval results."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import json

SCHEMA_VERSION = 1


def _safe_token(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def _rounded_mtime(path: Path | None) -> Optional[float]:
    if path is None:
        return None
    try:
        return round(path.stat().st_mtime, 3)
    except FileNotFoundError:
        return None


def _file_size(path: Path | None) -> Optional[int]:
    if path is None:
        return None
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return None


def _build_filename(meta: Dict) -> str:
    encoder = _safe_token(str(meta.get("image_encoder", "encoder")))
    dataset = _safe_token(Path(str(meta.get("dataset_csv", "dataset"))).name)
    kb = _safe_token(Path(str(meta.get("kb_json_path", "kb"))).name)
    start = meta.get("dataset_start", 0)
    end = meta.get("dataset_end", "None")
    k_max = meta.get("clip_k_max", 0)
    return f"{encoder}__{dataset}_{start}-{end}__{kb}__kmax={k_max}.json"


@dataclass
class ClipCacheHandle:
    path: Path
    payload: Dict
    dirty: bool = False

    @property
    def meta(self) -> Dict:
        return self.payload.setdefault("meta", {})

    @property
    def data(self) -> Dict[str, Dict]:
        return self.payload.setdefault("data", {})

    def bump_kmax(self, value: int) -> None:
        cur = int(self.meta.get("clip_k_max", 0))
        if value > cur:
            self.meta["clip_k_max"] = int(value)
            self.dirty = True


def build_meta(
    *,
    dataset_csv: Optional[str],
    dataset_start: int,
    dataset_end: Optional[int],
    kb_json_path: str,
    kb_doc_count: int,
    image_encoder: str,
    base_path: str,
    clip_k_max: int,
) -> Dict:
    dataset_path = Path(dataset_csv) if dataset_csv else None
    kb_path = Path(kb_json_path)
    index_path = Path(base_path) / "kb_index.faiss"
    ids_path = Path(base_path) / "kb_index_ids.pkl"
    return {
        "dataset_csv": str(dataset_path) if dataset_path else None,
        "dataset_start": dataset_start,
        "dataset_end": dataset_end,
        "dataset_mtime": _rounded_mtime(dataset_path),
        "dataset_size": _file_size(dataset_path),
        "kb_json_path": str(kb_path),
        "kb_mtime": _rounded_mtime(kb_path),
        "kb_size_bytes": _file_size(kb_path),
        "kb_doc_count": int(kb_doc_count),
        "faiss_index_path": str(index_path),
        "faiss_index_mtime": _rounded_mtime(index_path),
        "faiss_index_size": _file_size(index_path),
        "faiss_ids_path": str(ids_path),
        "faiss_ids_mtime": _rounded_mtime(ids_path),
        "faiss_ids_size": _file_size(ids_path),
        "image_encoder": image_encoder,
        "clip_k_max": int(clip_k_max),
        "schema_version": SCHEMA_VERSION,
    }


def open_cache(cache_dir: str | Path, meta: Dict) -> ClipCacheHandle:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / _build_filename(meta)

    payload: Optional[Dict] = None
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = None

    if not isinstance(payload, dict) or "meta" not in payload or "data" not in payload:
        payload = {"meta": meta, "data": {}}
        return ClipCacheHandle(path=path, payload=payload, dirty=True)

    handle = ClipCacheHandle(path=path, payload=payload, dirty=False)
    on_disk = handle.meta
    keys = [
        "dataset_csv",
        "dataset_start",
        "dataset_end",
        "dataset_mtime",
        "dataset_size",
        "kb_json_path",
        "kb_mtime",
        "kb_size_bytes",
        "kb_doc_count",
        "faiss_index_path",
        "faiss_index_mtime",
        "faiss_index_size",
        "faiss_ids_path",
        "faiss_ids_mtime",
        "faiss_ids_size",
        "image_encoder",
        "schema_version",
    ]
    if any(on_disk.get(k) != meta.get(k) for k in keys):
        payload = {"meta": meta, "data": {}}
        return ClipCacheHandle(path=path, payload=payload, dirty=True)

    requested = int(meta.get("clip_k_max", 0))
    if requested > int(on_disk.get("clip_k_max", 0)):
        on_disk["clip_k_max"] = requested
        handle.dirty = True
    return handle


def get_row(handle: ClipCacheHandle, row_key: str, min_k: int) -> Optional[Dict[str, List]]:
    entry = handle.data.get(row_key)
    if not entry:
        return None
    doc_indices = entry.get("doc_indices")
    doc_scores = entry.get("doc_scores")
    if not doc_indices or not doc_scores or len(doc_indices) < min_k:
        return None
    length = len(doc_indices)
    image_offsets = entry.get("image_offsets") or [-1] * length
    faiss_positions = entry.get("faiss_positions") or [-1] * length
    if len(image_offsets) != length or len(faiss_positions) != length:
        return None
    return {
        "doc_indices": [int(x) for x in doc_indices],
        "doc_scores": [float(x) for x in doc_scores],
        "image_offsets": [int(x) for x in image_offsets],
        "faiss_positions": [int(x) for x in faiss_positions],
    }


def put_row(
    handle: ClipCacheHandle,
    row_key: str,
    doc_indices: List[int],
    doc_scores: List[float],
    image_offsets: List[int],
    faiss_positions: List[int],
) -> None:
    if not (len(doc_indices) == len(doc_scores) == len(image_offsets) == len(faiss_positions)):
        raise ValueError("cache row length mismatch")
    handle.data[row_key] = {
        "doc_indices": [int(x) for x in doc_indices],
        "doc_scores": [float(x) for x in doc_scores],
        "image_offsets": [int(x) for x in image_offsets],
        "faiss_positions": [int(x) for x in faiss_positions],
    }
    handle.bump_kmax(len(doc_indices))
    handle.dirty = True


def save_cache(handle: ClipCacheHandle) -> None:
    if not handle.dirty:
        return
    tmp = handle.path.with_suffix(handle.path.suffix + ".tmp")
    tmp.write_text(json.dumps(handle.payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(handle.path)
    handle.dirty = False
