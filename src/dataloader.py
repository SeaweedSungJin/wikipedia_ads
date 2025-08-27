"""Dataset loader utilities for Encyclopedic-VQA."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import csv
import json
import os
import pickle

IMAGE_EXTS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".WEBP",
)

INAT_SUBDIRS = [
    "public_test",
    "test",
    "train",
    "val",
    "public_test/images",
    "test/images",
    "train/images",
    "val/images",
    "",
]


@dataclass
class VQASample:
    """Single question-answer-image entry."""

    # Basic Q/A data extracted from the CSV
    question: str  # Textual question
    answer: str  # Corresponding answer string

    # Resolved image paths associated with the question
    image_paths: List[str]

    # Metadata used for bookkeeping and analysis
    row_idx: int
    metadata: Dict[str, str]

    # Convenience accessors for common metadata columns
    wikipedia_title: str
    wikipedia_url: str
    dataset_name: str


class VQADataset:
    """Lightweight iterable dataset for Encyclopedic-VQA."""

    def __init__(
        self,
        csv_path: str,
        id2name_path: Optional[str | List[str]],
        image_root: Optional[str],
        googlelandmark_root: Optional[str] = None,
        start: int = 0,
        end: Optional[int] = None,
    ) -> None:
        """Initialize dataset reader with paths and slicing options."""
        # Paths to the CSV and image directories are kept so that the
        # iterator can lazily resolve them on demand.
        self.csv_path = csv_path  # Path to the EVQA CSV file
        self.image_root = image_root  # Inaturalist image root
        self.googlelandmark_root = googlelandmark_root  # Google Landmark image root

        # Mapping from numeric ID to image filename.  Accept either a single
        # JSON file or a list of files to merge, mirroring the behaviour of
        # ``evaluate_pipeline.py``.
        self.id2name: Dict[str, str] = {}
        if id2name_path:
            paths = (
                id2name_path
                if isinstance(id2name_path, (list, tuple))
                else [id2name_path]
            )
            for path in paths:
                if path and os.path.exists(path):
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        data = {str(k): v for k, v in data.items()}
                        self.id2name.update(data)
                        print(f"[INFO] Loaded id2name: {path} (+{len(data):,})")
                    except Exception as e:
                        print(f"[WARN] id2name 로딩 실패({path}): {e}")

        # Pre-load or build an iNaturalist ID→path cache for robust lookups
        self._inat_cache: Dict[str, str] = {}
        if self.image_root and os.path.exists(self.image_root):
            cache_path = os.path.join(self.image_root, "_id_to_path_cache.pkl")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "rb") as f:
                        self._inat_cache = pickle.load(f)
                    print(
                        f"[INFO] Loaded iNat ID cache: {len(self._inat_cache):,} entries from {cache_path}"
                    )
                except Exception as e:
                    print(f"[WARN] iNat cache 로딩 실패({cache_path}): {e}")
            else:
                try:
                    self._inat_cache = self._build_inat_cache(self.image_root, cache_path)
                except Exception as e:
                    print(f"[WARN] iNat cache 생성 실패: {e}")

        self.start = start  # Starting row index
        self.end = end  # Optional stopping row index

        # Compute dataset length for debugging
        with open(csv_path, newline="", encoding="utf-8") as f:
            self.total_rows = sum(1 for _ in f) - 1
        print(
            f"CSV 파일 '{csv_path}' 로딩 완료. 총 {self.total_rows}개 행 (헤더 제외). 시작={start}, 끝={end}"
        )
    
    def __len__(self):
         return self.end - self.start if self.end is not None else self.total_rows - self.start

    # ------------------------------------------------------------------
    # Image path resolution helpers
    # ------------------------------------------------------------------

    def _build_inat_cache(
        self, base_dir: str, cache_path: str, max_files: int = 10_000_000
    ) -> Dict[str, str]:
        """Scan the iNaturalist directory and build an ID→path cache."""

        print(f"[INFO] Building iNat ID cache from: {base_dir}")
        id2path: Dict[str, str] = {}
        visited = 0
        for root, _, files in os.walk(base_dir):
            for fn in files:
                visited += 1
                if visited % 200000 == 0:
                    print(
                        f"[INFO] scanned {visited:,} files... cache size={len(id2path):,}"
                    )
                if visited > max_files:
                    print("[WARN] Reached max_files limit during scan.")
                    break
                name, ext = os.path.splitext(fn)
                if ext in IMAGE_EXTS and name.isdigit():
                    id2path[name] = os.path.join(root, fn)
            else:
                continue
            break

        with open(cache_path, "wb") as f:
            pickle.dump(id2path, f)
        print(
            f"[INFO] iNat cache built. entries={len(id2path):,} -> {cache_path}"
        )
        return id2path

    def _get_image_path(self, dataset_name: str, image_id: str) -> Optional[str]:
        """Resolve an image path based on dataset name and ID."""

        ds = str(dataset_name).strip().lower()
        sid = str(image_id).strip()

        if ds.startswith("inat"):
            if not self.image_root:
                return None
            name = self.id2name.get(sid)
            if name:
                cand = os.path.join(self.image_root, name)
                if os.path.exists(cand):
                    return cand
                for ext in IMAGE_EXTS:
                    cand2 = os.path.join(self.image_root, name + ext)
                    if os.path.exists(cand2):
                        return cand2

            p = self._inat_cache.get(sid)
            if p and os.path.exists(p):
                return p

            for sd in INAT_SUBDIRS:
                base = os.path.join(self.image_root, sd) if sd else self.image_root
                for ext in IMAGE_EXTS:
                    cand = os.path.join(base, f"{sid}{ext}")
                    if os.path.exists(cand):
                        return cand
            return None

        if ds in {
            "landmarks",
            "googlelandmarks",
            "googlelandmark",
            "google_landmarks",
            "google-landmarks",
            "glr",
            "glr-v2",
            "landmarks-v2",
        }:
            if not self.googlelandmark_root:
                return None
            if len(sid) < 3:
                return None
            c0, c1, c2 = sid[0], sid[1], sid[2]
            for split in ("train", "test", "val", "index"):
                for ext in IMAGE_EXTS:
                    p1 = os.path.join(
                        self.googlelandmark_root, split, c0, c1, c2, f"{sid}{ext}"
                    )
                    if os.path.exists(p1):
                        return p1
                    p2 = os.path.join(
                        self.googlelandmark_root,
                        split,
                        "images",
                        c0,
                        c1,
                        c2,
                        f"{sid}{ext}",
                    )
                    if os.path.exists(p2):
                        return p2
            for ext in IMAGE_EXTS:
                p = os.path.join(self.googlelandmark_root, c0, c1, c2, f"{sid}{ext}")
                if os.path.exists(p):
                    return p
            return None

        return None


    def _resolve_paths(self, dataset_name: str, ids: List[str]) -> List[str]:
        """Convert image IDs to full file paths."""

        paths: List[str] = []
        ds = str(dataset_name).strip().lower()
        for id_ in ids:
            p = self._get_image_path(ds, id_)
            if not p:
                # Try both datasets if name was ambiguous or lookup failed
                if not ds.startswith("inat"):
                    p = self._get_image_path("inaturalist", id_)
                if not p:
                    p = self._get_image_path("landmarks", id_)
            if p and os.path.exists(p):
                paths.append(p)
        return paths

    def _parse_ids(self, field: str) -> List[str]:
        """Extract image IDs from the CSV value of ``dataset_image_ids``."""

        if field is None or field == "":
            return []

        try:
            ids = json.loads(field)
        except json.JSONDecodeError:
            # Field may be a simple pipe separated string like "1|2|3"
            return [s for s in field.split("|") if s]

        # ``ids`` can be an int, a string, or a list/tuple. Normalise to list.
        if isinstance(ids, (list, tuple)):
            values = ids
        else:
            values = [ids]

        # Convert each element to string for downstream lookup
        return [str(i) for i in values]

    def __iter__(self) -> Iterator[VQASample]:
        """Iterate over VQA samples as :class:`VQASample` objects."""

        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                # Skip rows before the configured start index
                if idx < self.start:
                    continue
                # Stop iterating at the end index if provided
                if self.end is not None and idx >= self.end:
                    break

                dataset_name = row.get("dataset_name", "inaturalist")
                ids = self._parse_ids(row.get("dataset_image_ids", ""))
                image_paths = self._resolve_paths(dataset_name, ids)

                if not image_paths:
                    print(f"[Row {idx}] 이미지 경로를 찾을 수 없습니다. 건너뜁니다.")
                    continue

                # Yield structured sample for downstream processing
                yield VQASample(
                    question=row.get("question", ""),
                    answer=row.get("answer", ""),
                    image_paths=image_paths,
                    row_idx=idx,
                    metadata=row,
                    wikipedia_title=row.get("wikipedia_title", ""),
                    wikipedia_url=row.get("wikipedia_url", ""),
                    dataset_name=dataset_name,
                )