"""Macro strategist game plan schema."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GamePlan:
    primary_threats: list[str] = field(default_factory=list)
    optimal_lead: list[str] = field(default_factory=list)
    opponent_likely_lead: list[str] = field(default_factory=list)
    win_condition: str = ""
    priority_kos: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> GamePlan:
        return cls(
            primary_threats=list(data.get("primary_threats") or []),
            optimal_lead=list(data.get("optimal_lead") or []),
            opponent_likely_lead=list(data.get("opponent_likely_lead") or []),
            win_condition=str(data.get("win_condition") or ""),
            priority_kos=list(data.get("priority_kos") or []),
        )
