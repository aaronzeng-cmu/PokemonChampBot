"""File-backed opponent pool expansion (read by workers without IPC)."""

from __future__ import annotations

import json
from pathlib import Path

from config.settings import (
    LOGS_DIR,
    OPPONENT_POOL_GROW_EVERY_STEPS,
    OPPONENT_POOL_GROW_SIZE,
    OPPONENT_POOL_MAX_TEAMS,
    OPPONENT_POOL_START_TEAMS,
)

POOL_SIZE_PATH = LOGS_DIR / "opponent_pool_size.json"


def read_pool_team_limit(default: int = OPPONENT_POOL_START_TEAMS) -> int:
    if not POOL_SIZE_PATH.is_file():
        return default
    try:
        data = json.loads(POOL_SIZE_PATH.read_text(encoding="utf-8"))
        return int(data.get("team_limit", default))
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


def write_pool_team_limit(team_limit: int, *, total_available: int) -> None:
    POOL_SIZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "team_limit": int(team_limit),
        "total_available": int(total_available),
    }
    POOL_SIZE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def init_pool_curriculum(total_available: int) -> None:
    write_pool_team_limit(
        min(OPPONENT_POOL_START_TEAMS, total_available),
        total_available=total_available,
    )


def pool_limit_for_timesteps(timesteps: int, total_available: int) -> int:
    expansions = timesteps // max(OPPONENT_POOL_GROW_EVERY_STEPS, 1)
    limit = OPPONENT_POOL_START_TEAMS + expansions * OPPONENT_POOL_GROW_SIZE
    return min(limit, OPPONENT_POOL_MAX_TEAMS, total_available)
