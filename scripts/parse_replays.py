#!/usr/bin/env python3
"""Parse raw replay logs into BC PyTorch dataset (doubles or singles)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm

from config.settings import (
    BC_DATASET_PATH,
    PROCESSED_DATA_DIR,
    RAW_LOGS_DIR,
    SINGLES_BC_DATASET_PATH,
    SINGLES_RAW_LOGS_DIR,
)
from src.doubles.data.replay_parser import build_dataset, parse_log_file
from src.core.training.mmap_dataset import mmap_dataset_dir, save_mmap_dataset
from src.singles.meta_database import load_meta_database
from src.singles.replay_parser import build_singles_dataset, parse_singles_log_file


def _discover_log_paths(input_dir: Path, *, include_html: bool) -> list[Path]:
    """All replay files under input_dir (recursive), de-duplicated by resolved path."""
    seen: set[Path] = set()
    paths: list[Path] = []
    for pattern in ("*.log",):
        for path in sorted(input_dir.rglob(pattern)):
            key = path.resolve()
            if key not in seen:
                seen.add(key)
                paths.append(path)
    if include_html:
        for path in sorted(input_dir.rglob("*.html")):
            key = path.resolve()
            if key not in seen:
                seen.add(key)
                paths.append(path)
    return paths


def _manifest_saved_count(input_dir: Path) -> int | None:
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return int(payload.get("stats", {}).get("saved", 0)) or None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse replay logs to bc_dataset.pt")
    parser.add_argument(
        "--format",
        choices=("doubles", "singles"),
        default="doubles",
        help="Dataset format (doubles=VGC, singles=BSS)",
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--skip-rating", action="store_true", help="For local HTML/log smoke tests")
    parser.add_argument(
        "--include-html",
        action="store_true",
        help="Also parse .html replays anywhere under input dir",
    )
    args = parser.parse_args()

    if args.format == "singles":
        input_dir = args.input or SINGLES_RAW_LOGS_DIR
        out_path = args.out or SINGLES_BC_DATASET_PATH
        parse_fn = parse_singles_log_file
        build_fn = build_singles_dataset
        meta_db = load_meta_database(format="singles", live_fetch=False)
        summary_name = "singles_parse_summary.json"
    else:
        input_dir = args.input or RAW_LOGS_DIR
        out_path = args.out or BC_DATASET_PATH
        parse_fn = parse_log_file
        build_fn = build_dataset
        meta_db = load_meta_database(format="doubles", live_fetch=False)
        summary_name = "parse_summary.json"

    paths = _discover_log_paths(input_dir, include_html=args.include_html)
    if not paths:
        raise SystemExit(f"No replay files found under {input_dir}")

    manifest_saved = _manifest_saved_count(input_dir)
    log_only = sum(1 for p in paths if p.suffix.lower() == ".log")
    print(f"Discovered {len(paths)} replay file(s) under {input_dir} ({log_only} .log)")
    if manifest_saved is not None:
        print(f"Manifest stats.saved={manifest_saved}")
        if log_only < manifest_saved:
            raise SystemExit(
                f"Missing logs: found {log_only} .log files but manifest expects {manifest_saved}. "
                "Run scripts/scrape_replays.py or check raw logs before parsing."
            )

    all_samples = []
    for path in tqdm(paths, desc=f"Parsing {args.format} replays", unit="log"):
        all_samples.extend(
            parse_fn(
                path,
                skip_rating=args.skip_rating,
                meta_db=meta_db,
            )
        )

    dataset = build_fn(all_samples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, out_path, pickle_protocol=4)
    if args.format == "doubles":
        mmap_path = save_mmap_dataset(dataset, mmap_dataset_dir(out_path))
        print(f"Mmap dataset -> {mmap_path}", flush=True)

    sample_count = int(dataset["token_ids"].shape[0])
    summary = {
        "format": args.format,
        "files": len(paths),
        "log_files": log_only,
        "manifest_saved": manifest_saved,
        "samples": sample_count,
        "out": str(out_path),
        "hash_fn": "stable_hash_md5",
    }
    summary_path = PROCESSED_DATA_DIR / summary_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
