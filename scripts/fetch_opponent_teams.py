#!/usr/bin/env python3
"""Download Reg M-A opponent teams into teams/opponents/ (for training diversity)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import OPPONENT_POOL_DIR, ROOT_DIR
from src.doubles.teams.fetch_sources import fetch_opponent_teams

DEFAULT_URLS_FILE = ROOT_DIR / "teams" / "sources" / "extra_paste_urls.txt"
DEFAULT_CSV_CACHE = ROOT_DIR / "teams" / "sources" / "vgcpastes_ma.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Showdown teams from public Reg M-A paste sources.",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=50,
        help="Number of unique valid teams to save (default: 50)",
    )
    parser.add_argument(
        "--source",
        choices=["vgcpastes", "smogon"],
        default="vgcpastes",
        help=(
            "vgcpastes: 50 newest from VGCPastes sheet (EVs=Yes, Replica=✔); "
            "smogon: forum scrape"
        ),
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=150,
        help="Max Smogon metagame thread pages (smogon source only)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OPPONENT_POOL_DIR,
        help=f"Output directory (default: {OPPONENT_POOL_DIR})",
    )
    parser.add_argument(
        "--urls-file",
        type=Path,
        default=DEFAULT_URLS_FILE,
        help="Extra pokepast URLs (smogon source only)",
    )
    parser.add_argument(
        "--csv-cache",
        type=Path,
        default=DEFAULT_CSV_CACHE,
        help=f"Cache VGCPastes CSV export (default: {DEFAULT_CSV_CACHE})",
    )
    args = parser.parse_args()

    urls_file = args.urls_file if args.urls_file.is_file() else None
    csv_cache = args.csv_cache if args.source == "vgcpastes" else None

    manifest = fetch_opponent_teams(
        out_dir=args.out_dir,
        target=args.target,
        max_pages=args.max_pages,
        extra_urls_file=urls_file,
        source=args.source,
        csv_cache=csv_cache,
    )

    print(f"Saved {manifest['saved']} teams to {args.out_dir.resolve()}")
    if "filter" in manifest:
        print("Filter:", manifest["filter"])
        print("Rows matching filter:", manifest.get("discovered_rows_matching_filter"))
    if "discovered_ids" in manifest:
        print("Discovered paste IDs:", manifest["discovered_ids"])
    if manifest["saved"] < args.target:
        print(f"Warning: only {manifest['saved']}/{args.target} teams saved.")
    if manifest["errors"]:
        print(f"Skipped {len(manifest['errors'])} pastes (see manifest.json)")
    print(f"Manifest: {(args.out_dir / 'manifest.json').resolve()}")


if __name__ == "__main__":
    main()
