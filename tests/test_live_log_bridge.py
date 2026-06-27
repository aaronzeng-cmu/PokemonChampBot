"""Live inference must match BC log encoding on the same protocol snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from config.settings import BC_MODEL_PATH
from src.doubles.data.live_log_bridge import (
    pick_masked_dual_force_actions,
    pick_masked_live_log_actions,
    replay_view_at_force_switch,
    replay_view_at_turn_start,
    sync_our_mons_from_battle,
)
from src.doubles.data.log_action_mask import _forced_switch_suffix, pick_masked_log_actions
from src.core.data.log_tracker import BattleLogState
from src.core.data.perspective import MonPerspective
from src.doubles.data.replay_parser import parse_replay_log
from src.core.data.state_tokenizer import encode_log_state, trajectory_frame_fingerprints
from src.core.model.transformer_bot import load_model
from src.doubles.planning.meta_database import MetaDatabase

TRACE_JSON = Path(
    "logs/eval/inference_trace/20260613_145621/inference_trace_latest.json"
)


def _load_protocol() -> tuple[str, list[str], list[dict]]:
    if not TRACE_JSON.is_file():
        return "", [], []
    battle = json.loads(TRACE_JSON.read_text(encoding="utf-8"))["battles"][0]
    protocol = battle["protocol_log"]
    return battle["battle_tag"], protocol, battle.get("decisions") or []


def test_turn1_view_matches_parser_tensor():
    replay_id, protocol, _ = _load_protocol()
    if not protocol:
        return

    meta_db = MetaDatabase(live_fetch=False)
    samples = parse_replay_log(
        "\n".join(protocol),
        replay_id=replay_id,
        skip_rating=True,
        keep_view_state=True,
    )
    s1 = next(
        s for s in samples if s.side == "p1" and s.turn == 1 and s.sample_kind == "turn"
    )

    view = replay_view_at_turn_start(
        protocol,
        side="p1",
        turn=1,
        replay_id=replay_id,
        meta_db=meta_db,
    )
    assert view is not None
    snap = encode_log_state(view, "p1")
    assert np.array_equal(snap, s1.tokens[-13:])
    assert trajectory_frame_fingerprints(s1.tokens)[-1] == trajectory_frame_fingerprints(
        np.concatenate([np.zeros((26, 16), dtype=np.int64), snap])
    )[-1]


def test_turn2_voluntary_view_matches_parser():
    replay_id, protocol, _ = _load_protocol()
    if not protocol:
        return

    meta_db = MetaDatabase(live_fetch=False)
    samples = parse_replay_log(
        "\n".join(protocol),
        replay_id=replay_id,
        skip_rating=True,
        keep_view_state=True,
    )
    s = next(
        s
        for s in samples
        if s.side == "p1" and s.turn == 2 and s.sample_kind == "turn"
    )

    view = replay_view_at_turn_start(
        protocol,
        side="p1",
        turn=2,
        replay_id=replay_id,
        meta_db=meta_db,
    )
    assert view is not None
    snap = encode_log_state(view, "p1")
    assert np.array_equal(snap, s.tokens[-13:])


def test_live_and_bc_masked_preds_agree_on_parser_samples():
    replay_id, protocol, _ = _load_protocol()
    if not protocol:
        return

    meta_db = MetaDatabase(live_fetch=False)
    samples = parse_replay_log(
        "\n".join(protocol),
        replay_id=replay_id,
        skip_rating=True,
        keep_view_state=True,
    )
    p1 = [s for s in samples if s.side == "p1"]
    model = load_model(BC_MODEL_PATH, device="cpu")

    class _FakeBattle:
        def __init__(self, turn: int, force_switch: list[bool]):
            self.turn = turn
            self.force_switch = force_switch
            self.battle_tag = replay_id

    for sample in p1:
        view = sample.view_state
        assert view is not None
        x = torch.as_tensor(sample.tokens, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            l0, l1 = model(x)
        bc0, bc1 = pick_masked_log_actions(
            l0[0], l1[0], view=view, side="p1", sample_kind=sample.sample_kind
        )

        if sample.sample_kind == "force_switch":
            suf = _forced_switch_suffix(view, "p1")
            fs = [suf == "a", suf == "b"]
        else:
            fs = [False, False]
        fake = _FakeBattle(sample.turn, fs)
        live0, live1 = pick_masked_live_log_actions(
            l0[0],
            l1[0],
            battle=fake,
            view=view,
            side="p1",
            sample_kind=sample.sample_kind,
        )
        assert (bc0, bc1) == (live0, live1), (
            f"turn {sample.turn} {sample.sample_kind}: bc={(bc0, bc1)} live={(live0, live1)}"
        )


def test_dual_force_switch_matches_two_parser_samples():
    replay_id, protocol, _ = _load_protocol()
    if not protocol:
        return

    meta_db = MetaDatabase(live_fetch=False)
    samples = parse_replay_log(
        "\n".join(protocol),
        replay_id=replay_id,
        skip_rating=True,
        keep_view_state=True,
    )
    fs_samples = [
        s
        for s in samples
        if s.side == "p1" and s.turn == 2 and s.sample_kind == "force_switch"
    ]
    if len(fs_samples) < 2:
        return

    class _FakeBattle:
        def __init__(self):
            self.battle_tag = replay_id
            self.turn = 2
            self.force_switch = [True, True]

    model = load_model(BC_MODEL_PATH, device="cpu")
    fake = _FakeBattle()

    s1 = next(s for s in samples if s.side == "p1" and s.turn == 1 and s.sample_kind == "turn")
    s2turn = next(
        s for s in samples if s.side == "p1" and s.turn == 2 and s.sample_kind == "turn"
    )
    hist = [s1.tokens[-13:], s2turn.tokens[-13:]]

    live_pred = pick_masked_dual_force_actions(
        model,
        battle=fake,
        protocol_lines=protocol,
        side="p1",
        meta_db=meta_db,
        history=hist,
        last_push_turn=2,
        device="cpu",
    )

    with torch.no_grad():
        l0, l1 = model(torch.as_tensor(fs_samples[0].tokens).unsqueeze(0))
    bc0 = pick_masked_log_actions(
        l0[0], l1[0], view=fs_samples[0].view_state, side="p1", sample_kind="force_switch"
    )
    with torch.no_grad():
        l0, l1 = model(torch.as_tensor(fs_samples[1].tokens).unsqueeze(0))
    bc1 = pick_masked_log_actions(
        l0[0], l1[0], view=fs_samples[1].view_state, side="p1", sample_kind="force_switch"
    )
    assert live_pred == (bc0[0], bc1[1])


def test_force_switch_view_matches_first_parser_sample():
    replay_id, protocol, _ = _load_protocol()
    if not protocol:
        return

    meta_db = MetaDatabase(live_fetch=False)
    samples = parse_replay_log(
        "\n".join(protocol),
        replay_id=replay_id,
        skip_rating=True,
        keep_view_state=True,
    )
    fs = [
        s
        for s in samples
        if s.side == "p1" and s.turn == 2 and s.sample_kind == "force_switch"
    ]
    if not fs:
        return

    view = replay_view_at_force_switch(
        protocol,
        side="p1",
        turn=2,
        replay_id=replay_id,
        meta_db=meta_db,
    )
    assert view is not None
    snap = encode_log_state(view, "p1")
    assert np.array_equal(snap, fs[0].tokens[-13:])


def test_sync_our_mons_overrides_meta_imputed_moves():
    class _Move:
        def __init__(self, mid: str):
            self.id = mid

    class _Mon:
        def __init__(self, species: str, moves: list[str]):
            self.species = species
            self.moves = {i: _Move(m) for i, m in enumerate(moves)}

    class _Battle:
        team = {
            "0": _Mon(
                "Garchomp",
                ["dragonclaw", "earthquake", "poisonjab", "rockslide"],
            )
        }

    view = BattleLogState(
        mons={
            "p1a": MonPerspective(
                slot="p1a",
                species="garchomp",
                moves=["dragonclaw", "earthquake", "rockslide", "stompingtantrum"],
            )
        }
    )
    sync_our_mons_from_battle(_Battle(), view, side="p1")
    assert view.mons["p1a"].moves == [
        "dragonclaw",
        "earthquake",
        "poisonjab",
        "rockslide",
    ]
