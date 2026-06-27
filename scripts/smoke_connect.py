#!/usr/bin/env python3
"""Smoke test: connect to local Showdown and play one battle."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import BATTLE_FORMAT, TEAM_PATH
from src.core.battle.battle_runner import run_battles
from src.doubles.players.max_damage_player import MaxDamagePlayer
from src.doubles.players.vgc_random_player import VGCRandomPlayer


def main():
    team = TEAM_PATH.read_text(encoding="utf-8")
    print(f"Format: {BATTLE_FORMAT}")
    agent = VGCRandomPlayer(
        battle_format=BATTLE_FORMAT,
        team=team,
        max_concurrent_battles=1,
    )
    opponent = MaxDamagePlayer(
        battle_format=BATTLE_FORMAT,
        team=team,
        max_concurrent_battles=1,
    )
    result = run_battles(agent, opponent, n_battles=1)
    print(f"Battles: {result.total}, wins: {result.wins}, losses: {result.losses}")
    print(f"Win rate: {result.win_rate:.1%}")


if __name__ == "__main__":
    main()
