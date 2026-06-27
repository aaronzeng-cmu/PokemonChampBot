#!/usr/bin/env python3
"""Extract one team-preview sample per replay log (6v6 species -> leads + brought)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tqdm import tqdm

from config.settings import (
    PREVIEW_DATASET_PATH,
    PROCESSED_DATA_DIR,
    RAW_LOGS_DIR,
    SINGLES_PREVIEW_DATASET_PATH,
    SINGLES_RAW_LOGS_DIR,
)
from src.core.data.perspective import hash_token
from src.doubles.data.replay_parser import parse_log_lines
from src.core.data.roster_profile import BROUGHT_BY_FORMAT, build_match_rosters, roster_species_key


def _lead_species(lines: list[str], side: str, *, format: str) -> set[str]:
    leads: set[str] = set()
    for line in lines:
        if line.startswith("|turn|"):
            break
        if not (line.startswith("|switch|") or line.startswith("|drag|")):
            continue
        parts = line.split("|")
        if len(parts) < 4 or not parts[2].startswith(side):
            continue
        label = parts[3].split(":")[-1].strip().split(",")[0]
        from poke_env.data import to_id_str

        leads.add(roster_species_key(to_id_str(label)))
    if format == "singles":
        return set(list(leads)[:1])
    return leads


def _team_order(lines: list[str], side: str) -> list[str]:
    order: list[str] = []
    for line in lines:
        if not line.startswith(f"|poke|{side}|"):
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        from src.core.data.log_tracker import _species_name

        order.append(roster_species_key(_species_name(parts[3])))
    return order


def parse_preview_sample(path: Path, *, format: str = "doubles") -> dict | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = parse_log_lines(text)
    rosters = build_match_rosters(lines)

    our_side = "p1"
    opp_side = "p2"
    our_order = _team_order(lines, our_side)
    opp_order = _team_order(lines, opp_side)
    if len(our_order) != 6 or len(opp_order) != 6:
        return None

    expected_brought = BROUGHT_BY_FORMAT.get(format, 4)
    our_brought = {
        roster_species_key(e.species)
        for e in rosters.for_side(our_side).entries.values()
        if e.brought
    }
    if len(our_brought) != expected_brought:
        return None

    our_leads = _lead_species(lines, our_side, format=format)
    expected_leads = 1 if format == "singles" else 2
    if len(our_leads) != expected_leads:
        return None
    if not our_leads.issubset(our_brought):
        return None

    species_ids = np.array(
        [hash_token(s) for s in our_order + opp_order],
        dtype=np.int64,
    )
    leads = np.zeros(6, dtype=np.float32)
    brought = np.zeros(6, dtype=np.float32)
    for i, species in enumerate(our_order):
        if species in our_leads:
            leads[i] = 1.0
        if species in our_brought:
            brought[i] = 1.0

    return {
        "species_ids": species_ids,
        "leads": leads,
        "brought": brought,
        "replay_id": path.stem,
        "format": format,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build preview_dataset.pt")
    parser.add_argument(
        "--format",
        choices=("doubles", "singles"),
        default="doubles",
        help="Preview format (doubles=bring-4, singles=bring-3)",
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.format == "singles":
        input_dir = args.input or SINGLES_RAW_LOGS_DIR
        out_path = args.out or SINGLES_PREVIEW_DATASET_PATH
        summary_name = "singles_preview_parse_summary.json"
    else:
        input_dir = args.input or RAW_LOGS_DIR
        out_path = args.out or PREVIEW_DATASET_PATH
        summary_name = "preview_parse_summary.json"

    paths = sorted(input_dir.rglob("*.log"))
    if not paths:
        raise SystemExit(f"No .log files under {input_dir}")

    rows: list[dict] = []
    skipped = 0
    for path in tqdm(paths, desc=f"Parsing {args.format} preview", unit="log"):
        row = parse_preview_sample(path, format=args.format)
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    if not rows:
        raise SystemExit("No valid preview samples")

    dataset = {
        "species_ids": torch.tensor(np.stack([r["species_ids"] for r in rows]), dtype=torch.long),
        "leads": torch.tensor(np.stack([r["leads"] for r in rows]), dtype=torch.float32),
        "brought": torch.tensor(np.stack([r["brought"] for r in rows]), dtype=torch.float32),
        "meta": [{"replay_id": r["replay_id"], "format": r["format"]} for r in rows],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, out_path, pickle_protocol=4)

    summary = {
        "format": args.format,
        "files": len(paths),
        "samples": len(rows),
        "skipped": skipped,
        "out": str(out_path),
    }
    summary_path = PROCESSED_DATA_DIR / summary_name
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
