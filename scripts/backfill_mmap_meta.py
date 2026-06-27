#!/usr/bin/env python3
"""Write per-chunk meta.json by re-parsing logs (no bc_dataset.pt load)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from config.settings import BC_DATASET_PATH, RAW_LOGS_DIR
from src.core.training.mmap_dataset import _chunk_path, has_mmap_dataset, mmap_dataset_dir
from src.doubles.data.replay_parser import parse_log_file
from src.singles.meta_database import load_meta_database

INITIAL_FLUSH = 250
RESUME_FLUSH = 100
RESUME_CHUNK_IDX = 97


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_LOGS_DIR)
    parser.add_argument("--mmap", type=Path, default=None)
    parser.add_argument("--skip-rating", action="store_true")
    args = parser.parse_args()

    mmap_dir = args.mmap or mmap_dataset_dir(BC_DATASET_PATH)
    if not has_mmap_dataset(mmap_dir):
        raise SystemExit(f"No mmap dataset at {mmap_dir}")

    root = json.loads((mmap_dir / "manifest.json").read_text(encoding="utf-8"))
    chunk_entries = root["chunks"]

    paths = sorted(args.input.glob("*.log"))
    if not paths:
        raise SystemExit(f"No logs in {args.input}")

    meta_db = load_meta_database(format="doubles", live_fetch=False)
    batch_meta: list[dict] = []
    chunk_idx = 0
    logs_in_chunk = 0

    def _flush_size(idx: int) -> int:
        return INITIAL_FLUSH if idx < RESUME_CHUNK_IDX else RESUME_FLUSH

    def _write_chunk() -> None:
        nonlocal chunk_idx, batch_meta, logs_in_chunk
        if chunk_idx >= len(chunk_entries):
            return
        entry = chunk_entries[chunk_idx]
        want = int(entry["samples"])
        if len(batch_meta) != want:
            raise SystemExit(
                f"{entry['dir']}: parsed {len(batch_meta)} meta rows, expected {want}"
            )
        chunk_dir = _chunk_path(mmap_dir, entry["dir"])
        (chunk_dir / "meta.json").write_text(
            json.dumps(batch_meta, ensure_ascii=False), encoding="utf-8"
        )
        manifest_path = chunk_dir / "manifest.json"
        chunk_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunk_manifest["has_meta"] = True
        manifest_path.write_text(json.dumps(chunk_manifest, indent=2), encoding="utf-8")
        chunk_idx += 1
        batch_meta = []
        logs_in_chunk = 0

    for path in tqdm(paths, desc="Backfill meta", unit="log"):
        for s in parse_log_file(path, skip_rating=args.skip_rating, meta_db=meta_db):
            batch_meta.append(
                {
                    "replay_id": s.replay_id,
                    "turn": s.turn,
                    "side": s.side,
                    "sample_kind": s.sample_kind,
                }
            )
        logs_in_chunk += 1
        if logs_in_chunk >= _flush_size(chunk_idx):
            _write_chunk()

    if batch_meta:
        _write_chunk()

    if chunk_idx != len(chunk_entries):
        raise SystemExit(f"Wrote {chunk_idx} chunks, expected {len(chunk_entries)}")

    print(f"Done: meta.json on {chunk_idx} chunks", flush=True)


if __name__ == "__main__":
    main()
