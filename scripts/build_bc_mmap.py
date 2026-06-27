#!/usr/bin/env python3
"""Stream-parse replay logs into chunked mmap .npy files (low peak RAM)."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from config.settings import BC_DATASET_PATH, RAW_LOGS_DIR
from src.core.training.mmap_dataset import (
    mmap_dataset_dir,
    save_mmap_chunk,
    write_mmap_manifest,
)
from src.doubles.data.replay_parser import build_dataset, parse_log_file
from src.singles.meta_database import load_meta_database


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_LOGS_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--chunk-logs",
        type=int,
        default=250,
        help="Flush a mmap chunk every N log files",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue after existing chunks in --out (skip already-parsed logs)",
    )
    parser.add_argument("--skip-rating", action="store_true")
    args = parser.parse_args()

    out_root = args.out or mmap_dataset_dir(BC_DATASET_PATH)
    chunks_dir = out_root / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.input.glob("*.log"))
    if not paths:
        raise SystemExit(f"No .log files in {args.input}")

    meta_db = load_meta_database(format="doubles", live_fetch=False)
    chunk_dirs: list[Path] = []
    batch_samples: list = []
    chunk_idx = 0
    total = 0
    skip_logs = 0
    flush_logs = args.chunk_logs

    if args.resume:
        existing = sorted(chunks_dir.glob("chunk_*"))
        if existing:
            summary_path = out_root / "build_summary.json"
            if summary_path.is_file():
                prev = json.loads(summary_path.read_text(encoding="utf-8"))
                skip_logs = int(prev.get("logs_parsed", len(existing) * flush_logs))
            else:
                skip_logs = len(existing) * flush_logs
            chunk_dirs = existing
            chunk_idx = len(existing)
            for chunk_dir in existing:
                meta = json.loads((chunk_dir / "manifest.json").read_text(encoding="utf-8"))
                total += int(meta["samples"])
            paths = paths[skip_logs:]
            print(
                f"Resume: {chunk_idx} chunks, {total:,} samples, skipping {skip_logs:,} logs "
                f"(flush every {flush_logs} logs)",
                flush=True,
            )

    def _flush() -> None:
        nonlocal chunk_idx, total, batch_samples
        if not batch_samples:
            return
        dataset = build_dataset(batch_samples)
        n = int(dataset["token_ids"].shape[0])
        if n == 0:
            batch_samples = []
            return
        name = f"chunk_{chunk_idx:04d}"
        chunk_dir = chunks_dir / name
        save_mmap_chunk(dataset, chunk_dir, chunk_name=name)
        chunk_dirs.append(chunk_dir)
        total += n
        chunk_idx += 1
        batch_samples = []
        del dataset
        gc.collect()
        write_mmap_manifest(out_root, chunk_dirs)
        print(f"  wrote {name}: {n:,} samples (total {total:,})", flush=True)

    for i, path in enumerate(tqdm(paths, desc="Parsing to mmap chunks", unit="log")):
        batch_samples.extend(
            parse_log_file(path, skip_rating=args.skip_rating, meta_db=meta_db)
        )
        if (i + 1) % flush_logs == 0:
            _flush()
    _flush()

    summary = {
        "files": len(sorted(args.input.glob("*.log"))),
        "chunks": len(chunk_dirs),
        "samples": total,
        "flush_logs": flush_logs,
        "logs_parsed": skip_logs + len(paths),
        "out": str(out_root),
    }
    (out_root / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_mmap_manifest(out_root, chunk_dirs)
    print(f"Done: {total:,} samples in {len(chunk_dirs)} chunks -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
