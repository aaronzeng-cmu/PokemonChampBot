#!/usr/bin/env python3
"""Audit live vs training move-slot ordering (meta-imputation desync)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poke_env.data import to_id_str
from poke_env.teambuilder import Teambuilder

from config.settings import TEAM_PATH
from src.core.data.move_utils import canonical_move_list
from src.doubles.data.meta_move_imputation import impute_moves_to_four
from src.doubles.planning.meta_database import MetaDatabase
from src.core.planning.species_normalize import clean_species_name


def _species_moves_from_team(team_text: str) -> dict[str, list[str]]:
    mons = Teambuilder.parse_showdown_team(team_text)
    out: dict[str, list[str]] = {}
    for mon in mons:
        raw_name = mon.nickname or mon.species or ""
        species = to_id_str(clean_species_name(raw_name))
        if not species:
            continue
        moves = [to_id_str(m) for m in (mon.moves or [])]
        out[species] = moves
    return out


def _top_n_meta_moves(meta_db: MetaDatabase, species: str, n: int = 4) -> list[str]:
    prior = meta_db.get_species_prior(species)
    ranked = sorted(prior.moves.items(), key=lambda x: -x[1])[:n]
    return [to_id_str(name) for name, _ in ranked]


def _slot_table(live: list[str], training: list[str]) -> list[str]:
    width = max(len(live), len(training), 4)
    live = (live + [""] * width)[:width]
    training = (training + [""] * width)[:width]
    lines = ["  Slot | Live (inference)     | Training (imputed)   | Match"]
    lines.append("  -----+----------------------+----------------------+------")
    mismatches = 0
    for i in range(4):
        a = live[i] or "(zero-pad)"
        b = training[i] or "(zero-pad)"
        ok = a == b
        if not ok:
            mismatches += 1
        lines.append(f"  [{i}]  | {a:20} | {b:20} | {'OK' if ok else 'DESYNC'}")
    lines.append(f"  => {mismatches}/4 slots mismatched")
    return lines


def audit_species(
    *,
    species: str,
    live_moves: list[str],
    meta_db: MetaDatabase,
    rng: random.Random,
) -> str:
    live_canonical = canonical_move_list(live_moves)

    top4 = _top_n_meta_moves(meta_db, species, n=4)
    meta_top4_alpha = canonical_move_list(top4)

    # Training path: P1 often has <4 revealed moves early; meta fills the rest.
    scenarios: list[tuple[str, list[str]]] = [
        ("0 known moves (cold impute)", []),
        ("1 known (heatwave used turn 1)", [live_moves[0]] if live_moves else []),
        ("3 known (missing last paste move)", live_moves[:-1] if len(live_moves) >= 2 else live_moves),
        ("4 known (full roster, no impute)", live_moves),
    ]
    # Species-specific realistic partial sets
    if species == "charizard":
        scenarios.extend(
            [
                ("3 known: heatwave/protect/weatherball (solarbeam unused in log)", ["heatwave", "protect", "weatherball"]),
            ]
        )
    if species == "garchomp":
        scenarios.extend(
            [
                ("3 known: eq/claw/slide (poisonjab unused in log)", ["earthquake", "dragonclaw", "rockslide"]),
            ]
        )

    chunks = [
        f"{'=' * 72}",
        f"SPECIES: {species}",
        f"{'=' * 72}",
        "",
        "LIVE (TransformerPlayer / team paste, alphabetized):",
        f"  {live_canonical}",
        "",
        "META top-4 by Pikalytics usage (alphabetized only -- diagnostic):",
        f"  raw top-4: {top4}",
        f"  alphabetized: {meta_top4_alpha}",
        "",
        *_slot_table(live_canonical, meta_top4_alpha),
        "",
    ]

    for label, known in scenarios:
        imputed = impute_moves_to_four(species, known, meta_db, rng)
        chunks.extend(
            [
                f"TRAINING impute_moves_to_four -- {label}:",
                f"  known:   {canonical_move_list(known) if known else []}",
                f"  imputed: {imputed}",
                "",
                *_slot_table(live_canonical, imputed),
                "",
            ]
        )

    return "\n".join(chunks)


def main() -> None:
    team_text = TEAM_PATH.read_text(encoding="utf-8")
    team_moves = _species_moves_from_team(team_text)
    meta_db = MetaDatabase(live_fetch=False)
    rng = random.Random(0)

    targets = ["charizard", "garchomp"]
    report_parts = [
        "Move Slot Alignment Audit",
        f"Team file: {TEAM_PATH}",
        "",
        "Compares live canonical move slots (inference) vs training tensors",
        "built via meta-imputation + alphabetical canonical_move_list.",
        "",
    ]

    for species in targets:
        live = team_moves.get(species)
        if not live:
            report_parts.append(f"WARNING: {species} not found in team paste")
            continue
        report_parts.append(audit_species(species=species, live_moves=live, meta_db=meta_db, rng=rng))

    report = "\n".join(report_parts)
    out_dir = Path("logs/eval/move_slot_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "move_slot_alignment_latest.txt"
    out_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nSaved -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
