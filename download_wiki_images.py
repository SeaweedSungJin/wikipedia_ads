#!/usr/bin/env python3
import argparse
import os
import json
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image, UnidentifiedImageError
from PIL.Image import DecompressionBombError
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
USER_AGENT = "wikipedia_ads/1.0 (https://example.com/contact)"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# PIL DecompressionBomb 보호 해제
Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sanitize_title(title: str) -> str:
    """Return a filesystem‐friendly version of a Wikipedia title."""
    return title.replace("/", "_").replace(" ", "_").replace(":", "_")


def download_image(url: str, path: str) -> None:
    """
    Download an image from `url` to local `path`, then verify with PIL.
    Raises on HTTP errors, timeouts, or corrupted images.
    Even very large images (decompression bombs) will be retried once.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, stream=True, timeout=10, headers=headers)
    resp.raise_for_status()
    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    # Validate image integrity
    try:
        with Image.open(path) as img:
            img.verify()
    except DecompressionBombError:
        # 재시도: 제한 해제된 상태로 다시 검증
        with Image.open(path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError) as e:
        os.remove(path)
        raise RuntimeError("Corrupted or unsupported image") from e


def process_entry(entry: dict,
                  output_dir: str,
                  failed_urls: set,
                  failure_log_path: str,
                  stats: dict,
                  max_workers: int) -> None:
    """
    Schedule downloads for all new, non‐failed images in one KB entry.
    """
    title = entry.get("title", "unknown")
    urls = entry.get("image_urls", [])
    sec_indices = entry.get("image_section_indices", [])
    folder = os.path.join(output_dir, sanitize_title(title))

    tasks = []
    for i, url in enumerate(urls):
        if url in failed_urls:
            continue
        ext = os.path.splitext(urlparse(url).path)[1].lower()
        if ext not in VALID_EXTENSIONS:
            ext = ".jpg"
        sec_idx = sec_indices[i] if i < len(sec_indices) else 0
        fname = f"img_{i:03d}_sec{sec_idx}{ext}"
        out_path = os.path.join(folder, fname)
        if not os.path.exists(out_path):
            tasks.append((url, out_path, sec_idx))

    stats["total"] += len(tasks)

    if not tasks:
        return

    # 병렬 다운로드 with progress bar per entry
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(download_image, url, path): (title, url, path, sec_idx)
            for url, path, sec_idx in tasks
        }
        for future in tqdm(
            as_completed(future_to_task),
            total=len(future_to_task),
            desc=f"Downloading [{sanitize_title(title)}]",
            leave=False,
            unit="img"
        ):
            title, url, path, sec_idx = future_to_task[future]
            try:
                future.result()
            except Exception as e:
                stats["failed"] += 1
                failed_urls.add(url)
                log_entry = {
                    "title": title,
                    "url": url,
                    "section_index": sec_idx,
                    "error_type": type(e).__name__,
                    "error_msg": str(e)
                }
                with open(failure_log_path, "a", encoding="utf-8") as flog:
                    flog.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                print(f"[ERROR] Failed to download {url}: {e}")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------
def main():
    KB_PATH = "/dataset/evqa/encyclopedic_kb_wiki_aug.jsonl"  # 고정된 경로

    parser = argparse.ArgumentParser(description="Parallel image downloader for Wikipedia KB")
    parser.add_argument("--output_dir", default="datasets/wiki_images", help="Image output directory")
    parser.add_argument("--img_workers", type=int, default=8, help="Threads per entry for image download")
    parser.add_argument("--entry_workers", type=int, default=16, help="Document-level parallelism")
    parser.add_argument("--start", type=int, default=0, help="Start line (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End line (exclusive)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    failure_log_path = os.path.join(args.output_dir, "failed_urls.jsonl")
    downloaded_titles_path = os.path.join(args.output_dir, "downloaded_titles.txt")

    failed_urls_set = load_failed_urls(failure_log_path)
    downloaded_titles = load_downloaded_titles(downloaded_titles_path)

    stats = {"total": 0, "failed": 0} 
    
    # 전체 KB 로딩 (혹은 슬라이스)
    with open(KB_PATH, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    sliced_lines = all_lines[args.start:args.end] if args.end else all_lines[args.start:]
    parsed_entries = []
    for line in sliced_lines:
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        title = entry.get("title", "")
        if title in downloaded_titles:
            continue
        parsed_entries.append(entry)

    print(f"Processing {len(parsed_entries)} entries (start={args.start}, end={args.end})...")

    with ThreadPoolExecutor(max_workers=args.entry_workers) as executor:
        futures = [
            executor.submit(
                process_entry,
                entry,
                args.output_dir,
                failed_urls_set,
                failure_log_path,
                stats,               # ← 이 줄이 빠졌던 것이 문제
                args.img_workers,
            )
            for entry in parsed_entries
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing entries", unit="entry"):
            result = future.result()
            if result:
                with open(downloaded_titles_path, "a", encoding="utf-8") as fout:
                    fout.write(result + "\n")
def load_failed_urls(path: str) -> set:
    """Load failed URLs from a log file (jsonl)."""
    failed = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    failed.add(record.get("url"))
                except json.JSONDecodeError:
                    continue
    return failed


def load_downloaded_titles(path: str) -> set:
    """Load already processed titles from file."""
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())



if __name__ == "__main__":
    main()
