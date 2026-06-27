"""Meta Gauntlet: hybrid bot vs weighted opponent pool."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from poke_env.teambuilder import Teambuilder

from config.settings import (
    GAUNTLET_EQUAL_WEIGHTS,
    GAUNTLET_OPPONENT_STATUS_CHANCE,
    OPPONENT_POOL_DIR,
    SHUFFLE_TEAM_ORDER,
)
from src.doubles.planning.meta_database import MetaDatabase
from src.doubles.teams.gauntlet_weights import normalize_team_weights, team_meta_score
from archive.rl.env.champions_vgc_env import ChampionsVGCRLEnv, load_team
from archive.rl.env.hybrid_agent_wrapper import HybridAgentWrapper
from archive.ismcts.players.hybrid_player import HybridPlayer
from src.doubles.players.gauntlet_opponent import GauntletOpponentPlayer
from src.doubles.teams.archetype import TeamArchetype, classify_team_export
from src.core.teams.roster import shuffle_showdown_team
from src.doubles.teams.team_pool import load_team_text


@dataclass
class GauntletTeam:
    file: str
    path: Path
    export: str
    packed: str
    weight: float
    archetype: TeamArchetype
    team_id: str = ""
    description: str = ""


@dataclass
class GauntletMatchResult:
    team_file: str
    archetype: str
    archetype_tags: list[str]
    wins: int
    games: int
    weight: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0


@dataclass
class GauntletReport:
    games_per_team: int
    teams: list[GauntletMatchResult] = field(default_factory=list)
    ismcts_config: dict[str, Any] = field(default_factory=dict)

    @property
    def weighted_win_rate(self) -> float:
        total_w = sum(r.weight for r in self.teams)
        if total_w <= 0:
            return 0.0
        return sum(r.weight * r.win_rate for r in self.teams) / total_w

    @property
    def raw_win_rate(self) -> float:
        wins = sum(r.wins for r in self.teams)
        games = sum(r.games for r in self.teams)
        return wins / games if games else 0.0

    def by_archetype(self) -> dict[str, dict[str, float]]:
        buckets: dict[str, list[GauntletMatchResult]] = {}
        for row in self.teams:
            buckets.setdefault(row.archetype, []).append(row)
        out: dict[str, dict[str, float]] = {}
        for arch, rows in buckets.items():
            wins = sum(r.wins for r in rows)
            games = sum(r.games for r in rows)
            w_total = sum(r.weight for r in rows)
            w_wr = (
                sum(r.weight * r.win_rate for r in rows) / w_total if w_total else 0.0
            )
            out[arch] = {
                "wins": wins,
                "games": games,
                "win_rate": wins / games if games else 0.0,
                "weighted_win_rate": w_wr,
                "team_count": len(rows),
            }
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "games_per_team": self.games_per_team,
            "weighted_win_rate": self.weighted_win_rate,
            "raw_win_rate": self.raw_win_rate,
            "ismcts_config": self.ismcts_config,
            "by_archetype": self.by_archetype(),
            "teams": [
                {
                    "team_file": r.team_file,
                    "archetype": r.archetype,
                    "archetype_tags": r.archetype_tags,
                    "wins": r.wins,
                    "games": r.games,
                    "win_rate": r.win_rate,
                    "weight": r.weight,
                }
                for r in self.teams
            ],
        }


def _packed_team(showdown_export: str) -> str:
    mons = Teambuilder.parse_showdown_team(showdown_export)
    if len(mons) != 6:
        raise ValueError(f"Expected 6 Pokémon, got {len(mons)}")
    return Teambuilder.join_team(mons)


def load_gauntlet_pool(
    pool_dir: Path | None = None,
    *,
    max_teams: int | None = None,
    equal_weights: bool | None = None,
    meta_db: MetaDatabase | None = None,
) -> list[GauntletTeam]:
    """Load gauntlet teams with equal or Pikalytics-based meta weights."""
    pool_dir = pool_dir or OPPONENT_POOL_DIR
    use_equal = GAUNTLET_EQUAL_WEIGHTS if equal_weights is None else equal_weights
    db = meta_db or MetaDatabase(live_fetch=False)

    manifest_path = pool_dir / "manifest.json"
    manifest: dict = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    meta_by_file = {t["file"]: t for t in manifest.get("teams", [])}
    paths = sorted(pool_dir.glob("*.txt"))
    if max_teams is not None:
        paths = paths[:max_teams]
    if not paths:
        raise FileNotFoundError(f"No opponent teams in {pool_dir}")

    exports: list[str] = []
    for path in paths:
        exports.append(load_team_text(path))

    if use_equal:
        weights = normalize_team_weights([1.0] * len(paths))
    else:
        scores = [team_meta_score(exp, db) for exp in exports]
        weights = normalize_team_weights(scores)

    teams: list[GauntletTeam] = []
    for path, export, weight in zip(paths, exports, weights):
        arch = classify_team_export(export)
        meta = meta_by_file.get(path.name, {})
        teams.append(
            GauntletTeam(
                file=path.name,
                path=path,
                export=export,
                packed=_packed_team(
                    shuffle_showdown_team(export) if SHUFFLE_TEAM_ORDER else export
                ),
                weight=weight,
                archetype=arch,
                team_id=str(meta.get("team_id", "")),
                description=str(meta.get("description", "")),
            )
        )
    return teams


def run_gauntlet_battle(
    *,
    opponent_packed_team: str,
    hybrid: HybridPlayer | None = None,
    opponent: GauntletOpponentPlayer | None = None,
    max_steps: int = 500,
    log_level: int = 40,
) -> tuple[bool, float, int]:
    """Single battle: hybrid (fixed team) vs specific opponent packed team."""
    env = ChampionsVGCRLEnv(
        team=load_team(),
        opponent_team=opponent_packed_team,
        log_level=log_level,
        open_timeout=None,
    )
    agent = hybrid or HybridPlayer(start_listening=False)
    opp = opponent or GauntletOpponentPlayer(
        start_listening=False,
        status_chance=GAUNTLET_OPPONENT_STATUS_CHANCE,
    )
    gym_env = HybridAgentWrapper(env, agent, opp)

    obs, _ = gym_env.reset()
    total_reward = 0.0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, _ = gym_env.step(0)
        total_reward += float(reward)
        steps += 1
        if steps >= max_steps:
            break

    won = bool(env.battle1.won) if env.battle1 else False
    gym_env.close()
    return won, total_reward, steps


def run_gauntlet(
    *,
    games_per_team: int = 2,
    pool_dir: Path | None = None,
    max_teams: int | None = None,
    equal_weights: bool | None = None,
    hybrid: HybridPlayer | None = None,
    opponent: GauntletOpponentPlayer | None = None,
    ismcts_config: dict[str, Any] | None = None,
    max_steps: int = 500,
    verbose: bool = True,
) -> GauntletReport:
    teams = load_gauntlet_pool(pool_dir, max_teams=max_teams, equal_weights=equal_weights)
    agent = hybrid or HybridPlayer(start_listening=False)
    opp = opponent or GauntletOpponentPlayer(
        start_listening=False,
        status_chance=GAUNTLET_OPPONENT_STATUS_CHANCE,
    )

    report = GauntletReport(
        games_per_team=games_per_team,
        ismcts_config=ismcts_config or {},
    )

    for idx, team in enumerate(teams, start=1):
        wins = 0
        for g in range(games_per_team):
            won, reward, steps = run_gauntlet_battle(
                opponent_packed_team=team.packed,
                hybrid=agent,
                opponent=opp,
                max_steps=max_steps,
            )
            wins += int(won)
            if verbose:
                print(
                    f"  [{idx}/{len(teams)}] {team.file} game {g + 1}/{games_per_team}: "
                    f"{'WIN' if won else 'LOSS'} reward={reward:+.2f} steps={steps}"
                )

        report.teams.append(
            GauntletMatchResult(
                team_file=team.file,
                archetype=team.archetype.primary,
                archetype_tags=list(team.archetype.tags),
                wins=wins,
                games=games_per_team,
                weight=team.weight,
            )
        )
        if verbose:
            print(
                f"  -> {team.file} ({team.archetype.primary}): "
                f"{wins}/{games_per_team} | running weighted WR {report.weighted_win_rate:.1%}"
            )

    return report
