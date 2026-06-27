"""TransformerPlayer must capture protocol for BC live-log bridge."""

from __future__ import annotations

from pathlib import Path

from config.settings import BC_MODEL_PATH
from src.doubles.players.transformer_player import TransformerPlayer


def test_capture_battle_log_enabled_by_default():
    if not Path(BC_MODEL_PATH).is_file():
        return
    player = TransformerPlayer(model_path=BC_MODEL_PATH, device="cpu")
    assert player.capture_battle_log is True


def test_record_replays_agent_enables_protocol_capture():
    if not Path(BC_MODEL_PATH).is_file():
        return
    from scripts.record_replays import _make_agent

    out = Path("logs/replays/_test_stub")
    agent = _make_agent(
        "transformer",
        model_path=Path(BC_MODEL_PATH),
        team="",
        device="cpu",
        out_dir=out,
    )
    assert agent.capture_battle_log is True
