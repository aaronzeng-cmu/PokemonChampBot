"""Tensor parity: encode_live_as_log vs replay parser on shared protocol."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.core.data.state_tokenizer import stack_trajectory, TRAJECTORY_DEPTH
from src.singles.battle.live_log_bridge import encode_live_as_log, teampreview_protocol_line
from src.singles.evaluation.live_bc_alignment import (
    _simulate_live_trajectory,
    tensor_digest,
)
from src.singles.evaluation.tensor_diff import format_tensor_diff
from src.singles.replay_parser import parse_singles_replay_log


class _FakeBattle:
    def __init__(self, *, tag: str, turn: int, force_switch: bool = False):
        self.battle_tag = tag
        self.turn = turn
        self.force_switch = force_switch
        self.team = {}
        self.active_pokemon = None


def test_live_log_bridge_snapshot_matches_parser_turn():
    protocol = [
        "|player|p1|A|1000",
        "|player|p2|B|1000",
        "|poke|p1|Floette-Eternal, L50|",
        "|poke|p1|Lucario, L50|",
        "|poke|p1|Meowscarada, L50|",
        "|poke|p2|Gengar, L50|",
        "|poke|p2|Dragonite, L50|",
        "|poke|p2|Tyranitar, L50|",
        "|start",
        "|switch|p1a: Floette|Floette-Eternal, L50|149/149",
        "|switch|p2a: Gengar|Gengar, L50|100/100",
        "|turn|1",
        "|move|p1a: Floette|Moonblast|p2a: Gengar",
        "|move|p2a: Gengar|Shadow Ball|p1a: Floette",
        "|turn|2",
        "|turn|3",
        "|turn|4",
        "|turn|5",
    ]
    tag = "parity-protocol-test"
    samples = parse_singles_replay_log(
        "\n".join(protocol),
        replay_id=tag,
        skip_rating=True,
        keep_view_state=True,
    )
    p1_turn1 = next(s for s in samples if s.side == "p1" and s.turn == 1 and s.sample_kind == "turn")

    battle = _FakeBattle(tag=tag, turn=1, force_switch=False)
    encoded = encode_live_as_log(battle, protocol_lines=protocol, side="p1")
    assert encoded is not None
    live_snap, _, kind = encoded
    assert kind == "turn"

    parser_stacked = p1_turn1.tokens
    live_stacked = stack_trajectory([], live_snap, depth=TRAJECTORY_DEPTH)
    assert np.array_equal(parser_stacked, live_stacked), format_tensor_diff(
        live_stacked, parser_stacked
    )


_SMOKE_TRACE = (
    Path(__file__).resolve().parents[1]
    / "logs/eval/singles/alignment_smoke/inference_trace_20260617_012847.json"
)


@pytest.mark.skipif(not _SMOKE_TRACE.is_file(), reason="smoke trace fixture missing")
def test_simulated_live_trajectory_matches_parser_on_smoke_trace():
    """Full stacked tensor parity: replay decisions through live bridge + trajectory sim."""
    data = json.loads(_SMOKE_TRACE.read_text(encoding="utf-8"))
    battle = (data.get("battles") or [data])[0]
    protocol = list(battle.get("protocol_log") or [])
    tp = battle.get("teampreview")
    if tp:
        line = teampreview_protocol_line("p1", tp)
        if line not in protocol:
            protocol.append(line)
    tag = battle["battle_tag"]
    decisions = [d for d in battle.get("decisions") or [] if d.get("kind") == "inference"]
    if not decisions or any(d.get("protocol_len") is None for d in decisions):
        pytest.skip("trace missing protocol_len (re-run inference trace)")

    samples = parse_singles_replay_log(
        "\n".join(protocol),
        replay_id=tag,
        skip_rating=True,
        keep_view_state=True,
    )
    p1_samples = [s for s in samples if s.side == "p1"]
    simulated = _simulate_live_trajectory(
        protocol, decisions, tag=tag, side="p1", teampreview_cmd=tp
    )

    used_live: set[int] = set()
    mismatches: list[str] = []
    for sample in p1_samples:
        for di, dec in enumerate(decisions):
            if di in used_live:
                continue
            if dec.get("kind") != "inference":
                continue
            if int(dec.get("turn", -1)) != sample.turn:
                continue
            fs = bool(dec.get("force_switch"))
            if sample.sample_kind == "force_switch" and not fs:
                continue
            if sample.sample_kind == "turn" and fs:
                continue
            used_live.add(di)
            di_key = int(dec.get("decision_index", -1))
            stacked = simulated.get(di_key)
            assert stacked is not None, f"missing sim stack turn={sample.turn} kind={sample.sample_kind}"
            if not np.array_equal(stacked, sample.tokens):
                mismatches.append(
                    f"turn={sample.turn} {sample.sample_kind} "
                    f"digest sim={tensor_digest(stacked)} parser={tensor_digest(sample.tokens)}\n"
                    + format_tensor_diff(stacked, sample.tokens)
                )
            break

    assert not mismatches, "\n\n".join(mismatches)
