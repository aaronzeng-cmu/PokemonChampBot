"""Shared Showdown protocol capture + BC-aligned trajectory for live singles players."""

from __future__ import annotations

import json
from collections import deque

import numpy as np
from poke_env.battle.battle import Battle

from src.core.data.state_tokenizer import (
    TRAJECTORY_DEPTH,
    encode_singles_battle,
    push_trajectory,
    stack_trajectory,
)
from src.singles.battle.live_log_bridge import (
    encode_champchoice_protocol_line,
    encode_live_as_log,
    protocol_through_flush_turn,
    singles_mask_for_live_decision,
    snapshot_for_turn_flush,
    teampreview_protocol_line,
    turn_trajectory_should_push,
)


class SinglesLiveProtocolSupport:
    """Mixin: protocol logging, deferred trajectory flush, BC-aligned obs/mask."""

    def _init_live_protocol_state(self) -> None:
        self._trajectory_history: dict[str, deque[np.ndarray]] = {}
        self._last_trajectory_push_turn: dict[str, int] = {}
        self._last_flushed_turn: dict[str, int] = {}
        self._protocol_logs: dict[str, list[str]] = {}
        self._teampreview_cmds: dict[str, str] = {}

    def _clear_live_protocol_state(self) -> None:
        self._trajectory_history.clear()
        self._last_trajectory_push_turn.clear()
        self._last_flushed_turn.clear()
        self._protocol_logs.clear()
        self._teampreview_cmds.clear()

    def _record_teampreview(self, battle: Battle, cmd: str, *, capture_protocol: bool = True) -> None:
        tag = battle.battle_tag
        self._teampreview_cmds[tag] = cmd
        if not capture_protocol:
            return
        line = teampreview_protocol_line("p1", cmd)
        log = self._protocol_logs.setdefault(tag, [])
        if line not in log:
            log.append(line)

    def _record_teampreview_protocol(self, battle: Battle, cmd: str) -> None:
        """Always capture teampreview cmd and protocol line (RL path)."""
        self._record_teampreview(battle, cmd, capture_protocol=True)

    def _current_snapshot(self, battle: Battle) -> np.ndarray:
        protocol = self._protocol_for_encoding(battle)
        encoded = (
            encode_live_as_log(battle, protocol_lines=protocol, side="p1")
            if protocol
            else None
        )
        if encoded is None:
            return encode_singles_battle(battle)
        return encoded[0]

    def _stacked_obs(self, battle: Battle) -> np.ndarray:
        snapshot = self._current_snapshot(battle)
        history_list = list(self._history_for_tag(battle.battle_tag))
        return stack_trajectory(history_list, snapshot, depth=TRAJECTORY_DEPTH)

    @staticmethod
    def _battle_tag_from_split(split_messages) -> str | None:
        if not split_messages or not split_messages[0]:
            return None
        room = split_messages[0][0]
        if not room:
            return None
        return room[1:] if room.startswith(">") else room

    async def _capture_protocol_message(self, split_messages) -> None:
        from src.singles.evaluation.inference_trace import encode_request_protocol_line

        tag = self._battle_tag_from_split(split_messages)
        if tag is None:
            return

        log = self._protocol_logs.setdefault(tag, [])
        try:
            turns_to_flush: list[int] = []
            for split_message in split_messages[1:]:
                if not split_message or len(split_message) < 2:
                    continue
                if split_message[1] == "request":
                    continue
                if split_message[1] == "turn" and len(split_message) >= 3:
                    try:
                        new_turn = int(split_message[2])
                        if new_turn > 1:
                            turns_to_flush.append(new_turn - 1)
                    except ValueError:
                        pass
                if split_message[1] == "":
                    log.append("|" + "|".join(split_message[1:]))
                else:
                    log.append("|" + "|".join(split_message[1:]))

            for flushed_turn in turns_to_flush:
                self._try_flush_turn_trajectory(tag, flushed_turn)

            for split_message in split_messages[1:]:
                if not split_message or len(split_message) < 2:
                    continue
                if split_message[1] != "request":
                    continue
                if len(split_message) >= 3 and split_message[2]:
                    try:
                        request = json.loads(split_message[2])
                        log.append(encode_request_protocol_line(request))
                    except json.JSONDecodeError:
                        log.append("|request|...")
                else:
                    log.append("|request|...")
        except Exception:
            pass

    def _history_for_tag(self, tag: str) -> deque[np.ndarray]:
        if tag not in self._trajectory_history:
            self._trajectory_history[tag] = deque(maxlen=TRAJECTORY_DEPTH)
        return self._trajectory_history[tag]

    def _protocol_snapshot_lines(
        self,
        lines: list[str],
        tag: str,
        *,
        ensure_turn: int | None = None,
    ) -> list[str]:
        protocol = list(lines)
        cmd = self._teampreview_cmds.get(tag)
        if cmd:
            line = teampreview_protocol_line("p1", cmd)
            if line not in protocol:
                protocol.append(line)
        if ensure_turn is not None and ensure_turn >= 1:
            turn_line = f"|turn|{ensure_turn}"
            if not any(line.startswith(turn_line) for line in protocol):
                protocol.append(turn_line)
        return protocol

    def _protocol_snapshot(self, tag: str, *, ensure_turn: int | None = None) -> list[str]:
        return self._protocol_snapshot_lines(
            list(self._protocol_logs.get(tag, [])),
            tag,
            ensure_turn=ensure_turn,
        )

    def _protocol_for_flush(self, tag: str, flushed_turn: int) -> list[str]:
        raw = list(self._protocol_logs.get(tag, []))
        capped = protocol_through_flush_turn(raw, flushed_turn, len(raw))
        return self._protocol_snapshot_lines(capped, tag, ensure_turn=flushed_turn)

    def _protocol_for_encoding(self, battle: Battle) -> list[str]:
        return self._protocol_snapshot(battle.battle_tag, ensure_turn=int(battle.turn))

    def _try_flush_turn_trajectory(self, tag: str, flushed_turn: int) -> None:
        if flushed_turn < 1:
            return
        if self._last_flushed_turn.get(tag, 0) >= flushed_turn:
            return

        protocol = self._protocol_for_flush(tag, flushed_turn)
        if not turn_trajectory_should_push(
            protocol,
            turn=flushed_turn,
            side="p1",
            replay_id=tag,
        ):
            self._last_flushed_turn[tag] = flushed_turn
            return

        snapshot = snapshot_for_turn_flush(
            protocol,
            turn=flushed_turn,
            side="p1",
            replay_id=tag,
            meta_db=None,
        )
        if snapshot is None:
            self._last_flushed_turn[tag] = flushed_turn
            return

        dq = self._history_for_tag(tag)
        history_list = list(dq)
        push_trajectory(
            history_list,
            snapshot,
            depth=TRAJECTORY_DEPTH,
            maxlen=TRAJECTORY_DEPTH,
        )
        dq.clear()
        dq.extend(history_list)
        self._last_trajectory_push_turn[tag] = flushed_turn
        self._last_flushed_turn[tag] = flushed_turn

    def _flush_pending_turn_trajectories(self, tag: str) -> None:
        log = self._protocol_logs.get(tag, [])
        max_turn = 0
        for line in log:
            if line.startswith("|turn|"):
                try:
                    max_turn = max(max_turn, int(line.split("|")[2]))
                except (IndexError, ValueError):
                    pass
        for turn in range(self._last_flushed_turn.get(tag, 0) + 1, max_turn + 1):
            self._try_flush_turn_trajectory(tag, turn)

    def _stacked_obs(self, battle: Battle) -> np.ndarray:
        snapshot = self._current_snapshot(battle)
        history_list = list(self._history_for_tag(battle.battle_tag))
        return stack_trajectory(history_list, snapshot, depth=TRAJECTORY_DEPTH)

    def _live_action_mask(self, battle: Battle) -> np.ndarray:
        protocol = self._protocol_for_encoding(battle)
        return singles_mask_for_live_decision(battle, protocol, side="p1")

    def _record_champchoice(self, battle: Battle, action: int) -> None:
        tag = battle.battle_tag
        choice_line = encode_champchoice_protocol_line(battle, action, side="p1")
        if choice_line is not None:
            log = self._protocol_logs.setdefault(tag, [])
            log.append(choice_line)
