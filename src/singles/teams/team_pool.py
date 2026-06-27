"""Random opponent team selection for BSS / Champions Singles."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from config.settings import (
    SINGLES_OPPONENT_POOL_DIR,
    SINGLES_TEAM_PATH,
    USE_SINGLES_OPPONENT_TEAM_POOL,
)
from src.doubles.teams.team_pool import (
    PoolTeambuilder,
    load_pool_manifest,
    load_team_text,
    opponent_pool_description,
)

__all__ = [
    "PoolTeambuilder",
    "load_agent_team",
    "load_meta_team_pool",
    "load_opponent_team_builder",
    "load_pool_manifest",
    "opponent_pool_description",
]


def load_agent_team() -> str:
    return load_team_text(SINGLES_TEAM_PATH)


def load_meta_team_pool(
    *,
    pool_dir: Path | None = None,
    use_curriculum: bool = False,
) -> PoolTeambuilder:
    """Random BSS meta team each battle (full pool, no curriculum cap)."""
    pool_dir = pool_dir or SINGLES_OPPONENT_POOL_DIR
    return PoolTeambuilder.from_directory(pool_dir, use_curriculum=use_curriculum)


def load_opponent_team_builder(
    *,
    use_pool: bool = True,
    pool_dir: Path | None = None,
    use_curriculum: bool = False,
) -> Union[str, PoolTeambuilder]:
    """Agent uses fixed singles team; opponents sample from pool when available."""
    pool_dir = pool_dir or SINGLES_OPPONENT_POOL_DIR
    if use_pool and USE_SINGLES_OPPONENT_TEAM_POOL and pool_dir.is_dir():
        try:
            return PoolTeambuilder.from_directory(pool_dir, use_curriculum=use_curriculum)
        except FileNotFoundError:
            pass
    return load_agent_team()


def load_opponent_team_fallback() -> str:
    """Mirror fallback when pool is empty."""
    return load_agent_team()
