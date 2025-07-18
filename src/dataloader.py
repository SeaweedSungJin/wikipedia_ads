"""Dataset loader utilities for Encyclopedic-VQA."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import csv
import json
import os


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
        id2name_path: Optional[str],
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

        # Mapping from numeric ID to image filename
        self.id2name = {}
        if id2name_path:
            print("id2name JSON 로딩중...")
            with open(id2name_path, "r", encoding="utf-8") as f:
                self.id2name = json.load(f)

        self.start = start  # Starting row index
        self.end = end  # Optional stopping row index

        # Compute dataset length for debugging
        with open(csv_path, newline="", encoding="utf-8") as f:
            self.total_rows = sum(1 for _ in f) - 1
        print(
            f"CSV 파일 '{csv_path}' 로딩 완료. 총 {self.total_rows}개 행 (헤더 제외). 시작={start}, 끝={end}"
        )

    def _get_image_path(self, dataset_name: str, image_id: str) -> Optional[str]:
        """Resolve an image path based on dataset name and ID."""

        if dataset_name == "inaturalist":
            if not self.image_root:
                return None
            name = self.id2name.get(str(image_id))
            if name:
                path = os.path.join(self.image_root, name)
                if os.path.exists(path):
                    return path

            # In test splits the mapping may be unavailable. Try common
            # fallback patterns such as ``<root>/<id>.jpg`` or searching
            # under train/val/test subfolders.
            candidate = os.path.join(self.image_root, f"{image_id}.jpg")
            if os.path.exists(candidate):
                return candidate
            for split in ("train", "val", "test"):
                candidate = os.path.join(self.image_root, split, f"{image_id}.jpg")
                if os.path.exists(candidate):
                    return candidate
            # If the ID itself looks like a relative path, use it directly
            candidate = os.path.join(self.image_root, str(image_id))
            if os.path.exists(candidate):
                return candidate
        elif dataset_name.lower() in ("googlelandmarks", "googlelandmark", "landmarks"):
            # Accept several naming conventions for Google Landmark
            # to match variations in the CSV.
            if not self.googlelandmark_root:
                return None
            image_id_str = str(image_id)
            if len(image_id_str) < 3:
                return None

            # Google Landmark images are stored under split folders (train/index/test)
            # followed by three nested directories derived from the image id digits.
            for split in ("train", "index", "test"):
                for ext in (".jpg", ".jpeg", ".JPG", ".JPEG"):
                    candidate = os.path.join(
                        self.googlelandmark_root,
                        split,
                        image_id_str[0],
                        image_id_str[1],
                        image_id_str[2],
                        f"{image_id_str}{ext}",
                    )
                    if os.path.exists(candidate):
                        return candidate

            # Fallback: some datasets omit the split folder entirely
            for ext in (".jpg", ".jpeg", ".JPG", ".JPEG"):
                candidate = os.path.join(
                    self.googlelandmark_root,
                    image_id_str[0],
                    image_id_str[1],
                    image_id_str[2],
                    f"{image_id_str}{ext}",
                )
                if os.path.exists(candidate):
                    return candidate
        return None


    def _resolve_paths(self, dataset_name: str, ids: List[str]) -> List[str]:
        """Convert image IDs to full file paths."""

        paths: List[str] = []
        for id_ in ids:
            p = self._get_image_path(dataset_name, id_)
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