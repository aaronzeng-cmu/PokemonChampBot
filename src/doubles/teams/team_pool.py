"""Random opponent team selection from a downloaded meta pool."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Union

from poke_env.teambuilder import Teambuilder

from config.settings import (
    LOGS_DIR,
    OPPONENT_POOL_CURRICULUM,
    OPPONENT_POOL_DIR,
    OPPONENT_POOL_START_TEAMS,
    SHUFFLE_TEAM_ORDER,
    TEAM_PATH,
)
from src.core.teams.roster import shuffle_showdown_team

POOL_SIZE_PATH = LOGS_DIR / "opponent_pool_size.json"


def read_pool_team_limit(default: int = OPPONENT_POOL_START_TEAMS) -> int:
    if not POOL_SIZE_PATH.is_file():
        return default
    try:
        data = json.loads(POOL_SIZE_PATH.read_text(encoding="utf-8"))
        return int(data.get("team_limit", default))
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


def load_team_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_pool_manifest(pool_dir: Path) -> dict:
    manifest_path = pool_dir / "manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    files = sorted(pool_dir.glob("*.txt"))
    return {
        "count": len(files),
        "teams": [f.name for f in files],
    }


def _packed_team(showdown_export: str) -> str:
    mons = Teambuilder.parse_showdown_team(showdown_export)
    if len(mons) != 6:
        raise ValueError(f"Expected 6 Pokémon, got {len(mons)}")
    return Teambuilder.join_team(mons)


class PoolTeambuilder(Teambuilder):
    """Yield a random packed team from a pool of Showdown exports."""

    def __init__(
        self,
        team_exports: list[str],
        *,
        use_curriculum: bool | None = None,
    ):
        self._exports = team_exports
        if use_curriculum is None:
            self._use_curriculum = OPPONENT_POOL_CURRICULUM
        else:
            self._use_curriculum = use_curriculum

    @property
    def team_count(self) -> int:
        return len(self._active_exports())

    @property
    def pool_size(self) -> int:
        return len(self._exports)

    def _active_exports(self) -> list[str]:
        if not self._use_curriculum:
            return self._exports
        limit = read_pool_team_limit(len(self._exports))
        return self._exports[:limit]

    def yield_team(self) -> str:
        pool = self._active_exports()
        export = random.choice(pool)
        if SHUFFLE_TEAM_ORDER:
            export = shuffle_showdown_team(export)
        return _packed_team(export)

    @classmethod
    def from_directory(
        cls,
        pool_dir: Path,
        *,
        use_curriculum: bool | None = None,
    ) -> PoolTeambuilder:
        exports = []
        for path in sorted(pool_dir.glob("*.txt")):
            text = load_team_text(path)
            if text:
                exports.append(text)
        if not exports:
            raise FileNotFoundError(f"No .txt teams in {pool_dir}")
        return cls(exports, use_curriculum=use_curriculum)


def load_opponent_team_builder(
    *,
    use_pool: bool = True,
    pool_dir: Path | None = None,
    use_curriculum: bool | None = None,
) -> Union[str, PoolTeambuilder]:
    """Agent uses fixed team; opponents sample from pool when available."""
    pool_dir = pool_dir or OPPONENT_POOL_DIR
    if use_pool and pool_dir.is_dir():
        try:
            return PoolTeambuilder.from_directory(
                pool_dir,
                use_curriculum=use_curriculum,
            )
        except FileNotFoundError:
            pass
    return load_team_text(TEAM_PATH)


def load_agent_team() -> str:
    return load_team_text(TEAM_PATH)


def opponent_pool_description(team: Union[str, PoolTeambuilder]) -> dict:
    if isinstance(team, PoolTeambuilder):
        return {
            "mode": "pool",
            "pool_size": team.pool_size,
            "active_teams": team.team_count,
            "use_curriculum": team._use_curriculum,
        }
    return {"mode": "mirror", "pool_size": 0, "active_teams": 1, "use_curriculum": False}
