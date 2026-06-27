"""Team preview helpers for Champions VGC bring-4 / lead selection."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poke_env.battle.double_battle import DoubleBattle

from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player.battle_order import DoubleBattleOrder

from src.doubles.battle.action_space import combo_to_ndarray, enumerate_legal_combos


def random_teampreview_command(
    battle: DoubleBattle,
    *,
    rng: random.Random | None = None,
) -> str:
    """Return a random legal /team command for this battle's roster (VGC: 4 mons)."""
    r = rng or random
    members = list(range(1, len(battle.team) + 1))
    r.shuffle(members)
    if battle.format is not None and "vgc" in battle.format:
        members = members[:4]
    # Selection flags are set by SingleAgentWrapper across two preview steps.
    return "/team " + "".join(str(c) for c in members)


def parse_team_command(command: str) -> list[int]:
    """Parse Showdown /team digits into 1-based roster slot indices."""
    body = command.strip()
    if body.startswith("/team"):
        body = body[5:].strip()
    return [int(ch) for ch in body if ch.isdigit()]


def slots_to_species(battle: DoubleBattle, slots: list[int]) -> list[str]:
    team_list = list(battle.team.values())
    return [team_list[i - 1].species for i in slots]


def decode_combo_index(
    battle: DoubleBattle,
    combo_idx: int,
    *,
    combos: list[tuple[int, int]] | None = None,
) -> tuple[int, int]:
    """Decode a preview-step combo index to two 1-based roster slots."""
    if combos is None:
        combos = enumerate_legal_combos(battle)
    pair = combo_to_ndarray(combo_idx, combos)
    order = DoublesEnv.action_to_order(pair, battle, fake=True, strict=False)
    if not isinstance(order, DoubleBattleOrder):
        raise TypeError(f"Expected DoubleBattleOrder during preview, got {type(order)}")
    species = [p.base_species for p in battle.team.values()]
    a = order.first_order.order
    b = order.second_order.order
    return species.index(a.base_species) + 1, species.index(b.base_species) + 1


def describe_preview_combos(battle: DoubleBattle) -> list[dict]:
    """Legal preview actions with decoded slot pairs (for smoke / debugging)."""
    try:
        combos = enumerate_legal_combos(battle)
    except ValueError:
        return []
    rows: list[dict] = []
    for idx in range(len(combos)):
        s1, s2 = decode_combo_index(battle, idx, combos=combos)
        rows.append(
            {
                "combo_idx": idx,
                "slots": [s1, s2],
                "species": slots_to_species(battle, [s1, s2]),
            }
        )
    return rows


def _side_team_summary(team_values) -> dict:
    team_list = list(team_values)
    brought = [
        p.species
        for p in team_list
        if getattr(p, "selected_in_teampreview", False)
    ]
    if not brought:
        brought = [p.species for p in team_list]
    return {"brought": brought}


def battle_team_summary(battle: DoubleBattle) -> dict:
    """Species brought and turn-1 leads (after preview completes)."""
    our = _side_team_summary(battle.team.values())
    leads = [p.species for p in battle.active_pokemon if p is not None]
    return {"leads": leads, "brought": our["brought"]}


def opponent_team_summary(battle: DoubleBattle) -> dict:
    """Opponent roster species brought to the match."""
    return _side_team_summary(battle.opponent_team.values())


def full_team_species(team_values) -> list[str]:
    """All six species on a side (full pasted team, regardless of preview)."""
    return [p.species for p in team_values]


def opponent_full_team_summary(battle: DoubleBattle) -> dict:
    """Full 6-mon opponent paste species (pool team identity)."""
    return {"full_team": full_team_species(battle.opponent_team.values())}


def policy_teampreview_command(
    battle: DoubleBattle,
    *,
    predict_combo_index,
    fallback_command: str | None = None,
) -> str:
    """Two-step masked preview, same semantics as poke-env env._teampreview."""
    if fallback_command is not None:
        return fallback_command

    species = [p.base_species for p in battle.team.values()]
    team_list = list(battle.team.values())
    slots: list[int] = []

    for _ in range(2):
        combo_idx = predict_combo_index(battle)
        pair = combo_to_ndarray(combo_idx, enumerate_legal_combos(battle))
        order = DoublesEnv.action_to_order(pair, battle, strict=False)
        if not isinstance(order, DoubleBattleOrder):
            return random_teampreview_command(battle)
        a = order.first_order.order
        b = order.second_order.order
        i1 = species.index(a.base_species) + 1
        i2 = species.index(b.base_species) + 1
        slots.extend([i1, i2])

    for s in slots:
        team_list[s - 1]._selected_in_teampreview = True
    return "/team " + "".join(str(s) for s in slots)
