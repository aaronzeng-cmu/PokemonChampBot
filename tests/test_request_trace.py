"""Server |request| capture helpers for inference traces."""

from __future__ import annotations

from src.doubles.evaluation.battle_inference_trace import (
    encode_request_protocol_line,
    format_server_request_block,
    parse_request_protocol_line,
    summarize_server_request,
)


class _FakeMove:
    def __init__(self, move_id: str) -> None:
        self.id = move_id


class _FakeBattle:
    def __init__(self) -> None:
        self.turn = 13
        self._last_request = {
            "wait": False,
            "forceSwitch": [False, False],
            "side": {
                "pokemon": [
                    {"ident": "p1a: Whimsicott"},
                    {"ident": "p1b: Garchomp"},
                ]
            },
            "active": [
                {
                    "moves": [
                        {
                            "id": "moonblast",
                            "move": "Moonblast",
                            "pp": 10,
                            "maxpp": 16,
                            "disabled": False,
                            "target": "adjacentFoe",
                        }
                    ]
                },
                {
                    "moves": [
                        {
                            "id": "rockslide",
                            "move": "Rock Slide",
                            "pp": 0,
                            "maxpp": 16,
                            "disabled": True,
                            "target": "allAdjacentFoes",
                        },
                        {
                            "id": "dragonclaw",
                            "move": "Dragon Claw",
                            "pp": 16,
                            "maxpp": 16,
                            "disabled": False,
                            "target": "normal",
                        },
                    ]
                },
            ],
        }
        self.available_moves = [
            [_FakeMove("moonblast")],
            [_FakeMove("dragonclaw"), _FakeMove("earthquake")],
        ]


def test_encode_and_parse_request_protocol_line():
    req = {"turn": 13, "active": [{"moves": [{"id": "rockslide", "pp": 4}]}]}
    line = encode_request_protocol_line(req)
    assert line.startswith("|request|{")
    assert parse_request_protocol_line(line) == req
    assert parse_request_protocol_line("|request|...") is None


def test_summarize_server_request():
    summary = summarize_server_request(_FakeBattle())
    assert summary["turn"] == 13
    slot_b = summary["active"][1]
    assert slot_b["ident"] == "p1b: Garchomp"
    assert slot_b["available_move_ids"] == ["dragonclaw", "earthquake"]
    rock = slot_b["moves"][0]
    assert rock["id"] == "rockslide"
    assert rock["pp"] == 0
    assert rock["disabled"] is True


def test_format_server_request_block():
    summary = summarize_server_request(_FakeBattle())
    text = format_server_request_block(summary)
    assert "Server request:" in text
    assert "rockslide: pp=0/16" in text
    assert "DISABLED" in text
    assert "poke-env available: ['dragonclaw', 'earthquake']" in text
