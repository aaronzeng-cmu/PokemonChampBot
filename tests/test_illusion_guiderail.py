"""Tests for Zoroark Illusion guiderails."""

from __future__ import annotations

from pathlib import Path

from src.doubles.data.action_codec import encode_log_move, format_log_action_pair
from src.doubles.data.replay_parser import parse_log_file
from src.core.data.state_tokenizer import human_readable_state


def test_illusion_replace_reveals_zoroark_turn3():
    path = Path("data/raw_logs/gen9championsvgc2026regma-2629508270.log")
    samples = parse_log_file(path, keep_view_state=True)
    sample = next(
        s
        for s in samples
        if s.turn == 3 and s.side == "p1" and s.sample_kind == "turn"
    )
    hr = human_readable_state(sample.view_state, "p1")
    slot_a = hr["our_actives"][0]
    assert slot_a["species"] == "zoroarkhisui", slot_a
    assert "bittermalice" in (slot_a["moves"] or [])
    assert "closecombat" not in (slot_a["moves"] or [])
    assert slot_a["illusion_broken"] is True
    assert slot_a["illusion_disguise"] == "hawlucha"
    gt = format_log_action_pair(sample.view_state, "p1", sample.action_slot0, sample.action_slot1)
    assert "zoroarkhisui" in gt or "zoroark" in gt
    assert "hawlucha" not in gt.split("|")[0].lower()


def test_encode_log_move_guiderail_zoroark_move():
    from src.core.data.log_tracker import BattleLogState
    from src.core.data.perspective import MonPerspective

    state = BattleLogState(
        team_roster={"p1": ["hawlucha", "zoroarkhisui", "froslass", "dedenne"]},
        mons={
            "p1a": MonPerspective(
                slot="p1a",
                species="hawlucha",
                active=True,
                moves=[],
            )
        },
    )
    idx = encode_log_move(state, "p1a: Hawlucha", "Bitter Malice", "p2b: Sneasler")
    assert idx != -100
