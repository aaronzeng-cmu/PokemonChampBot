#!/usr/bin/env python3
"""Collect asymmetrical value-network training data (Hybrid vs gauntlet pool)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from config.settings import LOGS_DIR, OPPONENT_POOL_DIR
from archive.ismcts.evaluation.value_collector import ValueDataCollector, save_value_dataset
from src.planning.macro_strategist import HeuristicMacroStrategist
from archive.ismcts.players.hybrid_player import HybridPlayer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect belief-augmented state/outcome data for value network training",
    )
    parser.add_argument("--games", type=int, default=100, help="Total games to play")
    parser.add_argument("--max-teams", type=int, default=None)
    parser.add_argument("--pool-dir", type=Path, default=OPPONENT_POOL_DIR)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: logs/value_data/)",
    )
    parser.add_argument("--ismcts-ms", type=int, default=1500)
    parser.add_argument("--ismcts-dets", type=int, default=None)
    parser.add_argument(
        "--llm-macro",
        action="store_true",
        help="Use DeepSeek macro strategist (slower; default is heuristic macro)",
    )
    args = parser.parse_args()

    import config.settings as settings

    settings.ISMCTS_TIME_BUDGET_MS = args.ismcts_ms
    if args.ismcts_dets is not None:
        settings.ISMCTS_DETERMINIZATIONS = args.ismcts_dets

    out_dir = args.out_dir or (LOGS_DIR.parent / "value_data")
    hybrid = (
        HybridPlayer(start_listening=False, macro=HeuristicMacroStrategist())
        if not args.llm_macro
        else None
    )
    collector = ValueDataCollector(pool_dir=args.pool_dir, hybrid=hybrid)

    print(f"Collecting {args.games} asymmetrical games (Hybrid vs gauntlet pool)")
    print(f"Macro: {'DeepSeek' if args.llm_macro else 'heuristic (fast)'}")
    print(f"Pool: {args.pool_dir.resolve()}")
    print(f"ISMCTS budget: {settings.ISMCTS_TIME_BUDGET_MS}ms\n")

    session = collector.run(
        num_games=args.games,
        max_teams=args.max_teams,
        max_steps=args.max_steps,
        verbose=True,
    )

    meta_path = save_value_dataset(session, out_dir)
    print(f"\nGames: {session.games_played} | Wins: {session.wins}")
    print(f"Win rate: {session.wins / session.games_played:.1%}")
    print(f"Turn records: {len(session.records)}")
    print(f"Metadata: {meta_path.resolve()}")


if __name__ == "__main__":
    main()
