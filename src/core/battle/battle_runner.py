"""Thin async battle runner for local Showdown evaluation (no Gymnasium)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from poke_env.player.player import Player
from poke_env.ps_client import LocalhostServerConfiguration


@dataclass
class BattleResult:
    wins: int
    losses: int
    draws: int
    total: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total else 0.0


async def run_battles_async(
    agent: Player,
    opponent: Player,
    *,
    n_battles: int = 1,
) -> BattleResult:
    await agent.battle_against(opponent, n_battles=n_battles)
    wins = sum(1 for b in agent.battles.values() if b.won)
    losses = sum(1 for b in agent.battles.values() if b.lost)
    draws = len(agent.battles) - wins - losses
    return BattleResult(wins=wins, losses=losses, draws=draws, total=len(agent.battles))


def run_battles(
    agent: Player,
    opponent: Player,
    *,
    n_battles: int = 1,
    server_configuration=LocalhostServerConfiguration,
) -> BattleResult:
    """Run n_battles between two Players on the local Showdown server."""
    agent.reset_battles()
    opponent.reset_battles()
    return asyncio.run(run_battles_async(agent, opponent, n_battles=n_battles))


def run_battle_series(
    agent: Player,
    opponent: Player,
    *,
    n_battles: int,
    on_progress: Optional[callable] = None,
) -> BattleResult:
    """Run battles in batches so progress callbacks can fire."""
    wins = losses = draws = 0
    batch = min(10, n_battles)
    done = 0
    while done < n_battles:
        k = min(batch, n_battles - done)
        agent.reset_battles()
        opponent.reset_battles()
        result = asyncio.run(run_battles_async(agent, opponent, n_battles=k))
        wins += result.wins
        losses += result.losses
        draws += result.draws
        done += k
        if on_progress:
            on_progress(done, n_battles, wins)
    return BattleResult(wins=wins, losses=losses, draws=draws, total=n_battles)
