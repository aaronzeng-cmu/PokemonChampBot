"""Singles BC eval vs live inference trace alignment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.singles.evaluation.live_bc_alignment import (
    build_bc_audit_report,
    pick_masked_bc_action,
)
from src.singles.log_action_mask import singles_mask_for_eval
from src.singles.replay_parser import parse_singles_replay_log


def _minimal_trace_payload() -> dict:
    protocol = [
        "|init|battle",
        "|title|test",
        "|j|☆Agent",
        "|j|☆Opp",
        "|gametype|singles",
        "|player|p1|Agent|100",
        "|player|p2|Opp|100",
        "|gen|9",
        "|tier|[Gen 9 Champions] BSS Reg M-A",
        "|clearpoke",
        "|poke|p1|Garchomp, L50, M|",
        "|poke|p1|Rotom-Wash, L50|",
        "|poke|p1|Floette-Eternal, L50, F|",
        "|poke|p2|Floette-Eternal, L50, F|",
        "|teampreview|3",
        "|teamsize|p1|3",
        "|teamsize|p2|3",
        "|start",
        "|switch|p1a: Garchomp|Garchomp, L50, M|215/215",
        "|switch|p2a: Floette|Floette-Eternal, L50, F|100/100",
        "|turn|1",
        "|move|p2a: Floette|Moonblast|p1a: Garchomp",
        "|faint|p1a: Garchomp",
        (
            '|request|{"forceSwitch":[true],"side":{"name":"Agent","id":"p1",'
            '"pokemon":[{"ident":"p1: Garchomp","details":"Garchomp, L50, M",'
            '"condition":"0 fnt","active":true},{"ident":"p1: Rotom",'
            '"details":"Rotom-Wash, L50","condition":"157/157","active":false},'
            '{"ident":"p1: Floette","details":"Floette-Eternal, L50, F",'
            '"condition":"149/149","active":false}]}}'
        ),
        "|switch|p1a: Rotom|Rotom-Wash, L50|157/157",
        "|turn|2",
    ]
    return {
        "battle_tag": "battle-alignment-test",
        "won": False,
        "turn": 2,
        "brought": ["rotomwash", "floetteeternal", "garchomp"],
        "protocol_log": protocol,
        "decisions": [
            {
                "kind": "inference",
                "decision_index": 1,
                "turn": 1,
                "force_switch": False,
                "trajectory_frames": ["empty", "empty", "our=(683,0)"],
                "raw_top1": {"index": 5, "label": "move"},
                "picked": {"index": 5, "label": "move"},
            },
            {
                "kind": "inference",
                "decision_index": 2,
                "turn": 1,
                "force_switch": True,
                "trajectory_frames": ["empty", "empty", "our=(683,0)"],
                "raw_top1": {"index": 0, "label": "switch"},
                "picked": {"index": 0, "label": "switch"},
            },
        ],
    }


def test_parser_samples_have_eval_masks():
    battle = _minimal_trace_payload()
    samples = parse_singles_replay_log(
        "\n".join(battle["protocol_log"]),
        replay_id=battle["battle_tag"],
        skip_rating=True,
        keep_view_state=True,
    )
    p1 = [s for s in samples if s.side == "p1"]
    if not p1:
        pytest.skip("minimal fixture did not pass validate_log")
    for sample in p1:
        view = sample.view_state
        assert view is not None
        mask = singles_mask_for_eval(view, side="p1", sample_kind=sample.sample_kind)
        assert mask is not None and mask.any()
        assert mask[sample.action] or sample.action < 0


def test_audit_report_structure(tmp_path: Path):
    from src.core.model.transformer_bot import (
        SINGLES_ACTION_SIZE,
        VGCBehaviorCloner,
        VGCBehaviorClonerConfig,
        save_model,
    )

    model_path = tmp_path / "model.pt"
    model = VGCBehaviorCloner(
        VGCBehaviorClonerConfig(action_space="singles", action_size=SINGLES_ACTION_SIZE)
    )
    save_model(model, model_path)

    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps({"battles": [_minimal_trace_payload()]}), encoding="utf-8")
    report = build_bc_audit_report(trace_path, model_path=model_path, device="cpu")
    assert "AUDIT SUMMARY" in report
    assert "Parser samples:" in report
