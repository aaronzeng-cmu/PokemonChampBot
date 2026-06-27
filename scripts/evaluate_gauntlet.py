#!/usr/bin/env python3
"""Meta Gauntlet: evaluate Hybrid bot vs opponent pool with weighted win rate."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config.settings as settings
from config.settings import BATTLE_FORMAT, LOGS_DIR, OPPONENT_POOL_DIR
from src.doubles.evaluation.gauntlet_runner import run_gauntlet


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Hybrid bot against the meta gauntlet opponent pool",
    )
    parser.add_argument("--games-per-team", type=int, default=2)
    parser.add_argument("--max-teams", type=int, default=None, help="Limit pool size")
    parser.add_argument("--pool-dir", type=Path, default=OPPONENT_POOL_DIR)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON report path (default: logs/gauntlet/<timestamp>.json)",
    )
    parser.add_argument("--equal-weights", action="store_true", help="Use 1/N weights instead of meta")
    parser.add_argument("--ismcts-dets", type=int, default=None)
    parser.add_argument("--ismcts-ms", type=int, default=None)
    parser.add_argument(
        "--ismcts-value-weight",
        type=float,
        default=None,
        help="Override ISMCTS_RL_VALUE_WEIGHT",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    ismcts_cfg: dict = {
        "determinizations": settings.ISMCTS_DETERMINIZATIONS,
        "time_budget_ms": settings.ISMCTS_TIME_BUDGET_MS,
        "use_fast_damage": settings.ISMCTS_USE_FAST_DAMAGE,
        "rl_value_weight": settings.ISMCTS_RL_VALUE_WEIGHT,
    }
    if args.ismcts_dets is not None:
        settings.ISMCTS_DETERMINIZATIONS = args.ismcts_dets
        ismcts_cfg["determinizations"] = args.ismcts_dets
    if args.ismcts_ms is not None:
        settings.ISMCTS_TIME_BUDGET_MS = args.ismcts_ms
        ismcts_cfg["time_budget_ms"] = args.ismcts_ms
    if args.ismcts_value_weight is not None:
        settings.ISMCTS_RL_VALUE_WEIGHT = args.ismcts_value_weight
        ismcts_cfg["rl_value_weight"] = args.ismcts_value_weight

    print(f"Format: {BATTLE_FORMAT}")
    print(f"Gauntlet pool: {args.pool_dir.resolve()}")
    print(f"ISMCTS config: {ismcts_cfg}")
    print(f"Weighting: {'equal' if args.equal_weights else 'meta (Pikalytics species usage)'}")
    print(f"Games per team: {args.games_per_team}\n")

    report = run_gauntlet(
        games_per_team=args.games_per_team,
        pool_dir=args.pool_dir,
        max_teams=args.max_teams,
        equal_weights=True if args.equal_weights else None,
        ismcts_config=ismcts_cfg,
        max_steps=args.max_steps,
        verbose=not args.quiet,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = LOGS_DIR.parent / "gauntlet"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (out_dir / f"gauntlet_{stamp}.json")
    payload = report.to_dict()
    payload["timestamp_utc"] = stamp
    payload["pool_dir"] = str(args.pool_dir.resolve())
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nWeighted win rate: {report.weighted_win_rate:.1%}")
    print(f"Raw win rate: {report.raw_win_rate:.1%}")
    print("\nBy archetype:")
    for arch, stats in sorted(report.by_archetype().items()):
        print(
            f"  {arch}: {stats['wins']}/{stats['games']} "
            f"({stats['win_rate']:.1%}) weighted={stats['weighted_win_rate']:.1%} "
            f"[{stats['team_count']} teams]"
        )
    print(f"\nReport: {out_path.resolve()}")


if __name__ == "__main__":
    main()
