#!/usr/bin/env python3
"""Download Reg M-A Pikalytics meta stats into teams/meta/."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import (
    PIKALYTICS_META_PATH,
    ROOT_DIR,
    SINGLES_META_DATABASE_PATH,
    SINGLES_PIKALYTICS_FORMAT,
)
from src.doubles.teams.pikalytics_meta import (
    DEFAULT_FORMAT,
    discover_species_from_battle_usage,
    discover_species_targets,
    fetch_all_meta,
    fetch_format_meta,
    fetch_missing_species,
    fetch_species_meta,
    save_meta_json,
)

FORMAT_PRESETS = {
    "doubles": {
        "pikalytics_format": DEFAULT_FORMAT,
        "out": PIKALYTICS_META_PATH,
    },
    "singles": {
        "pikalytics_format": SINGLES_PIKALYTICS_FORMAT,
        "out": SINGLES_META_DATABASE_PATH,
    },
}

DEFAULT_OUT = ROOT_DIR / "teams" / "meta" / "pikalytics_reg_ma.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Pikalytics AI markdown meta stats for Reg M-A.",
    )
    parser.add_argument(
        "--format",
        choices=sorted(FORMAT_PRESETS),
        default="doubles",
        help="Battle format preset (doubles=VGC, singles=BSS Champions)",
    )
    parser.add_argument(
        "--pikalytics-format",
        default=None,
        help="Override Pikalytics format code (default: preset for --format)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: preset for --format)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Only fetch detailed stats for top N species by usage (default: all indexed species)",
    )
    parser.add_argument(
        "--all-species",
        action="store_true",
        help="Fetch every species on the format index (default when no --top/--species)",
    )
    parser.add_argument(
        "--battle-usage",
        action="store_true",
        help="Discover species from Pikalytics Battle Usage API (~270, default for bulk fetch)",
    )
    parser.add_argument(
        "--discover-all",
        action="store_true",
        help="Battle usage list + Showdown learnsets + opponent pool + teammate crawl",
    )
    parser.add_argument(
        "--legal-only",
        action="store_true",
        help="Fetch only Showdown Champions learnsets legal list (~215)",
    )
    parser.add_argument(
        "--species",
        nargs="*",
        default=None,
        help="Explicit species names (Showdown-style, e.g. Kingambit Charizard-Mega-Y)",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Only fetch format overview (cores, top 20), no per-species pages",
    )
    parser.add_argument(
        "--fill-missing",
        action="store_true",
        help="Fetch only species missing from --out cache (battle usage list)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds between HTTP requests (default: 0.5)",
    )
    parser.add_argument(
        "--print-sample",
        action="store_true",
        help="Print a short summary to stdout after fetch",
    )
    args = parser.parse_args()

    preset = FORMAT_PRESETS[args.format]
    pikalytics_format = args.pikalytics_format or preset["pikalytics_format"]
    out_path = args.out or preset["out"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.fill_missing:
        payload = fetch_missing_species(out_path, format_code=pikalytics_format, delay_s=args.delay)
        missing = payload.pop("missing_fetched", [])
        skipped = payload.pop("skipped", None)
        save_meta_json(payload, out_path, merge=True)
        n_ok = len(payload.get("pokemon", {}))
        n_err = len(payload.get("errors", {}))
        if skipped is not None:
            print(f"Cache already complete ({skipped} species); nothing to fetch.")
        else:
            print(f"Filled {n_ok} missing species (requested {len(missing)}) into {out_path}")
        if n_err:
            print(f"  {n_err} species failed: {', '.join(sorted(payload['errors']))}")
        return

    if args.format_only:
        meta = fetch_format_meta(pikalytics_format, delay_s=0.0)
        payload = {
            "format": asdict(meta),
            "pokemon": {},
            "errors": {},
        }
        save_meta_json(payload, out_path)
        print(f"Wrote format overview to {out_path}")
        print(f"  species indexed: {len(meta.species)}")
        print(f"  top usage rows: {len(meta.top_usage)}")
        return

    if args.species and len(args.species) == 1 and not args.top:
        one = fetch_species_meta(args.species[0], pikalytics_format, delay_s=0.0)
        payload = {"pokemon": {args.species[0]: asdict(one)}}
        save_meta_json(payload, out_path)
        print(f"Wrote {args.species[0]} meta to {out_path}")
        if args.print_sample:
            print(json.dumps(asdict(one), indent=2)[:2000])
        return

    species_list = args.species
    top_n = args.top
    bulk_fetch = (
        args.battle_usage
        or args.discover_all
        or args.legal_only
        or (species_list is None and top_n is None)
    )
    if bulk_fetch and not species_list:
        if args.legal_only:
            from src.doubles.planning.champions_legal import load_legal_species_names

            species_list = load_legal_species_names()
            label = "Showdown learnsets"
        elif args.discover_all:
            species_list = discover_species_targets(
                format_code=pikalytics_format,
                use_battle_usage=True,
                use_legal_list=True,
                crawl_teammates=True,
            )
            label = "battle usage + learnsets + pool + teammates"
        else:
            species_list = discover_species_from_battle_usage(pikalytics_format)
            label = "Pikalytics Battle Usage"
        print(f"Discovered {len(species_list)} species targets ({label})")

    payload = fetch_all_meta(
        format_code=pikalytics_format,
        species=species_list,
        top_n=top_n,
        delay_s=args.delay,
    )
    save_meta_json(payload, out_path)

    n_ok = len(payload["pokemon"])
    n_err = len(payload["errors"])
    print(f"Wrote meta for {n_ok} species to {out_path}")
    if n_err:
        print(f"  {n_err} species failed: {', '.join(sorted(payload['errors']))}")
    if args.print_sample and payload["pokemon"]:
        first = next(iter(payload["pokemon"]))
        print(f"\nSample ({first}):")
        print(json.dumps(payload["pokemon"][first], indent=2)[:2500])


if __name__ == "__main__":
    main()
