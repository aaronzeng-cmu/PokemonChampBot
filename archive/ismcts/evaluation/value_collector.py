"""Asymmetrical value-network training data collection."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import OPPONENT_POOL_DIR
from archive.rl.env.champions_vgc_env import ChampionsVGCRLEnv, load_team
from archive.rl.env.hybrid_agent_wrapper import HybridAgentWrapper
from src.evaluation.gauntlet_runner import GauntletTeam, load_gauntlet_pool, run_gauntlet_battle
from archive.ismcts.planning.value_state import VALUE_STATE_DIM, embed_value_state
from src.players.gauntlet_opponent import GauntletOpponentPlayer
from archive.ismcts.players.hybrid_player import HybridPlayer


@dataclass
class ValueTurnRecord:
    game_id: str
    turn: int
    team_file: str
    archetype: str
    state: list[float]
    outcome: float | None = None  # filled post-game: 1.0 win, 0.0 loss

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "turn": self.turn,
            "team_file": self.team_file,
            "archetype": self.archetype,
            "state": self.state,
            "outcome": self.outcome,
        }


@dataclass
class ValueCollectionSession:
    records: list[ValueTurnRecord] = field(default_factory=list)
    games_played: int = 0
    wins: int = 0

    def add_game_records(self, rows: list[ValueTurnRecord], *, won: bool) -> None:
        outcome = 1.0 if won else 0.0
        for row in rows:
            row.outcome = outcome
            self.records.append(row)
        self.games_played += 1
        self.wins += int(won)


class ValueDataCollector:
    """Hybrid (fixed team) vs gauntlet pool — record belief-augmented states each turn."""

    def __init__(
        self,
        *,
        hybrid: HybridPlayer | None = None,
        opponent: GauntletOpponentPlayer | None = None,
        pool_dir: Path | None = None,
    ):
        self.hybrid = hybrid or HybridPlayer(start_listening=False)
        self.opponent = opponent or GauntletOpponentPlayer(start_listening=False)
        self.pool_dir = pool_dir or OPPONENT_POOL_DIR
        self.session = ValueCollectionSession()

    def play_game(self, team: GauntletTeam, *, max_steps: int = 500) -> bool:
        game_id = uuid.uuid4().hex[:12]
        env = ChampionsVGCRLEnv(
            team=load_team(),
            opponent_team=team.packed,
            log_level=40,
            open_timeout=None,
        )
        gym_env = HybridAgentWrapper(env, self.hybrid, self.opponent)
        turn_records: list[ValueTurnRecord] = []

        obs, _ = gym_env.reset()
        steps = 0
        terminated = truncated = False

        while not (terminated or truncated):
            obs, reward, terminated, truncated, _ = gym_env.step(0)
            steps += 1

            battle = env.battle1
            if battle is not None and not battle.teampreview and not battle.finished:
                belief = self.hybrid._ctx(battle).get("belief")
                vec = embed_value_state(battle, belief)
                turn_records.append(
                    ValueTurnRecord(
                        game_id=game_id,
                        turn=int(battle.turn),
                        team_file=team.file,
                        archetype=team.archetype.primary,
                        state=vec.tolist(),
                    )
                )

            if steps >= max_steps:
                break

        won = bool(env.battle1.won) if env.battle1 else False
        gym_env.close()
        self.session.add_game_records(turn_records, won=won)
        return won

    def run(
        self,
        *,
        num_games: int,
        max_teams: int | None = None,
        max_steps: int = 500,
        verbose: bool = True,
    ) -> ValueCollectionSession:
        teams = load_gauntlet_pool(self.pool_dir, max_teams=max_teams)
        if not teams:
            raise FileNotFoundError(f"No teams in {self.pool_dir}")

        for i in range(num_games):
            team = teams[i % len(teams)]
            won = self.play_game(team, max_steps=max_steps)
            if verbose and (i + 1) % 10 == 0:
                s = self.session
                print(
                    f"  [{i + 1}/{num_games}] last={team.file} "
                    f"session_wr={s.wins}/{s.games_played} "
                    f"records={len(s.records)}"
                )
        return self.session


def save_value_dataset(
    session: ValueCollectionSession,
    out_dir: Path,
    *,
    prefix: str = "value_data",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    jsonl_path = out_dir / f"{prefix}_{stamp}.jsonl"
    meta_path = out_dir / f"{prefix}_{stamp}_meta.json"

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for rec in session.records:
            fh.write(json.dumps(rec.to_dict()) + "\n")

    states = np.array([rec.state for rec in session.records], dtype=np.float32)
    outcomes = np.array([rec.outcome for rec in session.records], dtype=np.float32)
    npz_path = out_dir / f"{prefix}_{stamp}.npz"
    np.savez_compressed(
        npz_path,
        states=states,
        outcomes=outcomes,
        turns=np.array([rec.turn for rec in session.records], dtype=np.int32),
    )

    meta = {
        "timestamp_utc": stamp,
        "games_played": session.games_played,
        "wins": session.wins,
        "win_rate": session.wins / session.games_played if session.games_played else 0.0,
        "num_records": len(session.records),
        "state_dim": VALUE_STATE_DIM,
        "jsonl": str(jsonl_path),
        "npz": str(npz_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta_path
