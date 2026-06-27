#!/usr/bin/env python3
"""
Compare BC-example decoding vs live inference decoding on a synthetic
turn-1 state built from reg_ma_team.txt (Charizard + Whimsicott leads).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from poke_env.data import to_id_str
from poke_env.teambuilder import Teambuilder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import BC_MODEL_PATH, TEAM_PATH
from src.doubles.battle.move_order import (
    canonical_move_list,
    decode_move_action_index,
    format_live_canonical_action,
)
from src.doubles.data.action_codec import decode_log_slot_action, format_log_action_pair
from src.doubles.data.action_space_spec import target_offset_label
from src.doubles.data.log_action_mask import (
    log_turn_slot_mask,
    pick_masked_log_actions,
    slot_mask_for_eval,
)
from src.core.data.log_tracker import BattleLogState
from src.core.data.perspective import MonPerspective
from src.core.data.state_tokenizer import (
    TRAJECTORY_DEPTH,
    encode_log_state,
    human_readable_state,
    stack_trajectory,
)
from src.doubles.evaluation.bc_examples import _topk_choices
from src.core.model.transformer_bot import load_model
from src.core.planning.species_normalize import clean_species_name


def _team_from_paste() -> tuple[list[str], dict[str, list[str]]]:
    mons = Teambuilder.parse_showdown_team(TEAM_PATH.read_text(encoding="utf-8"))
    roster: list[str] = []
    moves_by_species: dict[str, list[str]] = {}
    for mon in mons:
        species = to_id_str(clean_species_name(mon.nickname or mon.species or ""))
        roster.append(species)
        moves_by_species[species] = canonical_move_list(
            [to_id_str(m) for m in (mon.moves or [])]
        )
    return roster, moves_by_species


def _mon(
    slot: str,
    species: str,
    moves: list[str],
    *,
    hp: int = 100,
    max_hp: int = 100,
) -> MonPerspective:
    return MonPerspective(
        slot=slot,
        species=species,
        hp=hp,
        max_hp=max_hp,
        active=True,
        seen=True,
        moves=list(moves),
        item_revealed=True,
        ability_revealed=True,
    )


def build_reg_ma_turn1_log_state(
    *,
    leads: tuple[str, str] = ("charizard", "whimsicott"),
    opp: tuple[str, str] = ("kangaskhan", "farigiraf"),
) -> BattleLogState:
    """Synthetic p1 turn-1 view matching live trace 20260613_055827 decision 1."""
    roster, moves_by = _team_from_paste()
    state = BattleLogState()
    state.turn = 1
    state.field.weather = "sandstorm"
    state.team_roster["p1"] = roster

    lead_a, lead_b = leads
    state.mons["p1a"] = _mon(
        "p1a",
        lead_a,
        moves_by[lead_a],
        hp=167 if lead_a == "charizard" else 149,
        max_hp=167 if lead_a == "charizard" else 149,
    )
    state.mons["p1b"] = _mon("p1b", lead_b, moves_by[lead_b], hp=149, max_hp=149)

    for suffix, species in zip(("a", "b"), opp):
        state.mons[f"p2{suffix}"] = MonPerspective(
            slot=f"p2{suffix}",
            species=species,
            hp=100,
            max_hp=100,
            active=True,
            seen=False,
            moves=[],
        )
    return state


class _FakeMon:
    """Minimal poke-env mon stub for format_live_canonical_action."""

    def __init__(self, species: str, moves: list[str]):
        self.species = species
        self.moves = {str(i): _FakeMove(m) for i, m in enumerate(moves)}


class _FakeMove:
    def __init__(self, move_id: str):
        self.id = move_id


class _FakeTeam:
    def __init__(self, roster: list[str], moves_by: dict[str, list[str]]):
        self._mons = [_FakeMon(s, moves_by.get(s, [])) for s in roster]

    def values(self):
        return self._mons


class _FakeBattle:
    def __init__(self, state: BattleLogState, side: str = "p1"):
        _, moves_by = _team_from_paste()
        roster = state.team_roster[side]
        self.team = _FakeTeam(roster, moves_by)
        self.active_pokemon = [
            self._active_mon(state, side, "a"),
            self._active_mon(state, side, "b"),
        ]

    @staticmethod
    def _active_mon(state: BattleLogState, side: str, suffix: str):
        mon = state.mons[f"{side}{suffix}"]
        return _FakeMon(mon.species, mon.moves)


def decode_live_style(
    state: BattleLogState,
    side: str,
    slot_suffix: str,
    action_idx: int,
) -> str:
    battle = _FakeBattle(state, side)
    pos = 0 if slot_suffix == "a" else 1
    return format_live_canonical_action(battle, pos, action_idx)


def _decode_breakdown(
    state: BattleLogState,
    side: str,
    slot_suffix: str,
    action_idx: int,
) -> str:
    mon = state.mons[f"{side}{slot_suffix}"]
    moves = canonical_move_list(mon.moves)
    if action_idx <= 6:
        return f"switch idx={action_idx}"
    move_slot, target_offset, mega, tera = decode_move_action_index(action_idx)
    move_name = moves[move_slot - 1] if 0 < move_slot <= len(moves) else "?"
    return (
        f"index={action_idx} slot={move_slot} target={target_offset}"
        f" ({target_offset_label(target_offset)}) move={move_name}"
        f" canonical={moves}"
    )


def _compare_decoders(state: BattleLogState, indices: list[int]) -> list[str]:
    lines = ["DECODER PARITY (BC decode_log_slot_action vs live format_live_canonical_action)"]
    for suffix in ("a", "b"):
        lines.append(f"  Slot {suffix.upper()} ({state.mons[f'p1{suffix}'].species}):")
        for idx in indices:
            bc = decode_log_slot_action(state, "p1", suffix, idx)
            live = decode_live_style(state, "p1", suffix, idx)
            match = "OK" if bc == live else "MISMATCH"
            lines.append(f"    [{idx:3d}] BC:   {bc}")
            lines.append(f"         Live: {live}  [{match}]")
    return lines


def main() -> None:
    state = build_reg_ma_turn1_log_state()
    snapshot = encode_log_state(state, "p1")
    stacked = stack_trajectory([], snapshot, depth=TRAJECTORY_DEPTH)
    x = torch.as_tensor(stacked, dtype=torch.long).unsqueeze(0)

    device = "cpu"
    model = load_model(BC_MODEL_PATH, device=device)
    with torch.no_grad():
        logits0, logits1 = model(x.to(device))

    row0, row1 = logits0[0], logits1[0]
    raw0, raw1 = int(row0.argmax().item()), int(row1.argmax().item())
    pred0, pred1 = pick_masked_log_actions(row0, row1, view=state, side="p1", sample_kind="turn")

    mask0 = slot_mask_for_eval(state, side="p1", sample_kind="turn", slot_suffix="a")
    mask1 = slot_mask_for_eval(
        state,
        side="p1",
        sample_kind="turn",
        slot_suffix="b",
        slot0_pred=pred0,
    )

    lines: list[str] = [
        "BC vs Live Decode Simulation — reg_ma_team turn 1",
        f"Team file: {TEAM_PATH}",
        f"Model: {BC_MODEL_PATH}",
        "",
        "Synthetic state (matches live trace leads: Charizard + Whimsicott):",
        human_readable_state(state, "p1").get("our_actives") and "",
    ]
    brief = []
    for key in ("our_actives", "opp_actives"):
        for mon in human_readable_state(state, "p1")[key]:
            if mon.get("present"):
                brief.append(
                    f"  {mon['slot']}: {mon['species']} [{', '.join(mon.get('moves') or [])}]"
                )
    lines.extend(brief)
    lines.append("")

    probe_indices = sorted(
        {raw0, raw1, pred0, pred1, 11, 14, 19, 9, 24}
        | set(int(i) for i in mask0.nonzero()[0].tolist()[:12])
    )
    lines.extend(_compare_decoders(state, probe_indices))
    lines.append("")

    for label, suffix, raw, picked, row in [
        ("A (Charizard)", "a", raw0, pred0, row0),
        ("B (Whimsicott)", "b", raw1, pred1, row1),
    ]:
        lines.append(f"--- Slot {label} ---")
        lines.append(_decode_breakdown(state, "p1", suffix, raw))
        lines.append(f"Raw argmax [{raw}]:")
        lines.append(f"  BC:   {decode_log_slot_action(state, 'p1', suffix, raw)}")
        lines.append(f"  Live: {decode_live_style(state, 'p1', suffix, raw)}")
        lines.append(f"  legal (log mask): {bool(mask0[raw] if suffix == 'a' else mask1[raw])}")
        lines.append(f"Masked pick [{picked}]:")
        lines.append(f"  BC:   {decode_log_slot_action(state, 'p1', suffix, picked)}")
        lines.append(f"  Live: {decode_live_style(state, 'p1', suffix, picked)}")
        lines.append("")

    topk0 = _topk_choices(
        row0, view=state, side="p1", slot_suffix="a", k=5, legal_mask=mask0
    )
    topk1 = _topk_choices(
        row1,
        view=state,
        side="p1",
        slot_suffix="b",
        k=5,
        legal_mask=mask1,
    )
    lines.append("BC-style top-5 (masked, decode_log_slot_action):")
    for rank, c in enumerate(topk0, 1):
        lines.append(f"  Slot A {rank}. {100*c.probability:5.1f}% | [{c.index}] {c.label}")
    for rank, c in enumerate(topk1, 1):
        lines.append(f"  Slot B {rank}. {100*c.probability:5.1f}% | [{c.index}] {c.label}")
    lines.append("")

    lines.append("Live-style top-5 labels (same indices, format_live_canonical_action):")
    for rank, c in enumerate(topk0, 1):
        live_lbl = decode_live_style(state, "p1", "a", c.index)
        lines.append(f"  Slot A {rank}. [{c.index}] {live_lbl}")
    for rank, c in enumerate(topk1, 1):
        live_lbl = decode_live_style(state, "p1", "b", c.index)
        lines.append(f"  Slot B {rank}. [{c.index}] {live_lbl}")
    lines.append("")

    lines.append(f"Joint BC decode:  {format_log_action_pair(state, 'p1', pred0, pred1)}")
    lines.append(
        "Joint live decode: "
        f"[{decode_live_style(state, 'p1', 'a', pred0)}] | "
        f"[{decode_live_style(state, 'p1', 'b', pred1)}]"
    )
    lines.append("")
    lines.append("Live trace reference (20260613_055827 decision 1):")
    lines.append("  Raw:  A[11] heatwave->opp B (illegal) | B[11] encore->opp B (legal)")
    lines.append("  Pick: A[14] protect->default | B[11] encore->opp B")

    report = "\n".join(lines)
    out_dir = Path("logs/eval/decode_compare")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bc_live_decode_compare_latest.txt"
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
