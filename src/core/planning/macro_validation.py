"""Validate LLM macro strategist output against preview rosters."""

from __future__ import annotations

import logging

from poke_env.data import to_id_str

from src.core.planning.game_plan import GamePlan
from src.core.planning.species_normalize import clean_species_name

logger = logging.getLogger(__name__)


def _species_id(name: str) -> str:
    return to_id_str(clean_species_name(name))


def _build_allowed_index(names: list[str]) -> dict[str, str]:
    index: dict[str, str] = {}
    for name in names:
        if not name:
            continue
        display = clean_species_name(name)
        index[_species_id(display)] = display
    return index


def resolve_species(name: str, allowed_index: dict[str, str]) -> str | None:
    """Return canonical allowed display name, or None if hallucinated."""
    if not name or not str(name).strip():
        return None
    key = _species_id(str(name))
    return allowed_index.get(key)


def validate_and_normalize_game_plan(
    plan: GamePlan,
    our_team: list[str],
    opp_team: list[str],
) -> GamePlan | None:
    """
    Ensure every species reference is in the preview rosters.
    Returns normalized plan, or None if any list field contains a hallucination.
    """
    our_index = _build_allowed_index(our_team)
    opp_index = _build_allowed_index(opp_team)

    def check_list(field: str, names: list[str], allowed: dict[str, str]) -> list[str] | None:
        out: list[str] = []
        for raw in names:
            resolved = resolve_species(raw, allowed)
            if resolved is None:
                logger.warning(
                    "Macro plan rejected: %s contains hallucinated species %r (allowed: %s)",
                    field,
                    raw,
                    sorted(allowed.values()),
                )
                return None
            out.append(resolved)
        return out

    primary = check_list("primary_threats", plan.primary_threats, opp_index)
    if primary is None:
        return None
    lead = check_list("optimal_lead", plan.optimal_lead, our_index)
    if lead is None:
        return None
    opp_lead = check_list("opponent_likely_lead", plan.opponent_likely_lead, opp_index)
    if opp_lead is None:
        return None
    kos = check_list("priority_kos", plan.priority_kos, opp_index)
    if kos is None:
        return None

    return GamePlan(
        primary_threats=primary,
        optimal_lead=lead,
        opponent_likely_lead=opp_lead,
        win_condition=plan.win_condition,
        priority_kos=kos,
    )
