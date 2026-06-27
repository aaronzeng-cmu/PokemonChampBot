"""Helpers for HTML replay output from poke-env battles."""

from __future__ import annotations

from pathlib import Path


def keep_agent_replays(run_dir: Path, agent_username: str) -> list[Path]:
    """
    poke-env writes one HTML replay per env player (agent1 + agent2) for the same battle.
    Keep only the agent perspective and remove duplicates.
    """
    kept: list[Path] = []
    for html in sorted(run_dir.glob("*.html")):
        if html.name.startswith(f"{agent_username} - "):
            kept.append(html)
        else:
            html.unlink(missing_ok=True)
    return kept
