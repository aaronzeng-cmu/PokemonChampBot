"""Parse BSS / Champions Singles replay logs into BC training samples."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.data.log_tracker import BattleLogState, LogStateTracker, project_first_person
from src.core.data.perspective import stable_seed_int
from src.core.data.roster_profile import build_match_rosters
from src.core.data.state_tokenizer import (
    N_FIELDS,
    TRAJECTORY_DEPTH,
    encode_log_state,
    push_trajectory,
    stack_trajectory,
)
from src.core.model.transformer_bot import SINGLES_ACTION_SIZE
from src.doubles.data.replay_parser import (
    MIN_RATING,
    MIN_TURN,
    extract_log_text,
    parse_log_lines,
    validate_log,
)
from src.singles.log_action_codec import (
    ACTION_UNKNOWN,
    is_force_switch_decision,
    parse_singles_force_switch,
    parse_singles_turn_action,
    training_action_mask,
)
from src.singles.meta_database import load_meta_database


@dataclass
class SinglesParsedSample:
    tokens: np.ndarray
    action: int
    replay_id: str
    turn: int
    side: str
    sample_kind: str = "turn"
    view_state: object | None = None
    action_mask: np.ndarray | None = None


def _rng_for_sample(replay_id: str, turn: int, side: str, sample_kind: str) -> random.Random:
    seed = stable_seed_int(replay_id, turn, side, sample_kind, "singles")
    return random.Random(seed)


def _emit_sample(
    samples: list[SinglesParsedSample],
    *,
    trajectory: list[np.ndarray],
    view: BattleLogState,
    side: str,
    action: int,
    replay_id: str,
    turn: int,
    sample_kind: str,
    keep_view_state: bool,
    update_trajectory: bool,
) -> None:
    if action == ACTION_UNKNOWN:
        return
    mask = np.array(
        training_action_mask(
            view,
            side,
            ground_truth=action,
            sample_kind=sample_kind,
        ),
        dtype=np.bool_,
    )
    snapshot = encode_log_state(
        view,
        side,
        format="singles",
        force_switch=(sample_kind == "force_switch"),
    )
    if update_trajectory:
        tokens = push_trajectory(trajectory, snapshot, depth=TRAJECTORY_DEPTH, maxlen=TRAJECTORY_DEPTH)
    else:
        tokens = stack_trajectory(trajectory, snapshot, depth=TRAJECTORY_DEPTH)

    samples.append(
        SinglesParsedSample(
            tokens=tokens,
            action=action,
            replay_id=replay_id,
            turn=turn,
            side=side,
            sample_kind=sample_kind,
            view_state=view if keep_view_state else None,
            action_mask=mask,
        )
    )


def _new_trajectories() -> dict[str, list[np.ndarray]]:
    return {"p1": [], "p2": []}


def _try_force_switch_samples(
    samples: list[SinglesParsedSample],
    *,
    tracker: LogStateTracker,
    turn_lines: list[str],
    line_idx: int,
    parts: list[str],
    lines_so_far: list[str],
    line: str,
    replay_id: str,
    current_turn: int,
    trajectories: dict[str, list[np.ndarray]],
    keep_view_state: bool,
    meta_db,
) -> bool:
    """Emit pivot/faint force_switch samples on the replacement ``|switch|`` line."""
    from src.singles.battle.live_log_bridge import rosters_through_turn

    if not is_force_switch_decision(parts, turn_lines, line_idx):
        return False
    if len(parts) < 4 or parts[1] != "switch":
        return False

    pre_switch = tracker.state.clone()
    rosters = rosters_through_turn(lines_so_far, current_turn)
    for side in ("p1", "p2"):
        rng = _rng_for_sample(replay_id, current_turn, side, "force_switch")
        view = project_first_person(
            pre_switch,
            side,
            rosters=rosters,
            meta_db=meta_db,
            rng=rng,
            format="singles",
        )
        action = parse_singles_force_switch(parts, view, side)
        if action is None:
            continue
        _emit_sample(
            samples,
            trajectory=trajectories[side],
            view=view,
            side=side,
            action=action,
            replay_id=replay_id,
            turn=current_turn,
            sample_kind="force_switch",
            keep_view_state=keep_view_state,
            update_trajectory=False,
        )
    return False


def parse_singles_replay_log(
    log_text: str,
    replay_id: str = "",
    *,
    skip_rating: bool = False,
    keep_view_state: bool = False,
    meta_db=None,
) -> list[SinglesParsedSample]:
    lines = parse_log_lines(log_text)
    tracker = LogStateTracker()
    if meta_db is None:
        from src.singles.battle.live_log_bridge import _meta_db

        meta_db = _meta_db()

    for line in lines:
        tracker.process_line(line)
    ok, _reason = validate_log(tracker, skip_rating=skip_rating)
    if not ok:
        return []

    tracker = LogStateTracker()
    current_turn = 0
    turn_lines: list[str] = []
    pre_turn_state = None
    trajectories = _new_trajectories()
    samples: list[SinglesParsedSample] = []
    lines_so_far: list[str] = []

    def _flush_turn() -> None:
        nonlocal turn_lines, pre_turn_state
        if current_turn < 1 or pre_turn_state is None or not turn_lines:
            turn_lines = []
            return
        from src.singles.battle.live_log_bridge import (
            replay_view_at_turn_start,
            snapshot_for_turn_flush,
        )

        for side in ("p1", "p2"):
            view = replay_view_at_turn_start(
                lines_so_far,
                turn=current_turn,
                side=side,
                replay_id=replay_id,
                meta_db=meta_db,
                deterministic_moves=False,
            )
            if view is None:
                continue
            action = parse_singles_turn_action(turn_lines, view, side)
            if action == ACTION_UNKNOWN:
                continue
            current_snap = encode_log_state(
                view,
                side,
                format="singles",
                force_switch=False,
            )
            hist_snap = snapshot_for_turn_flush(
                lines_so_far,
                turn=current_turn,
                side=side,
                replay_id=replay_id,
                meta_db=meta_db,
            )
            if hist_snap is None:
                continue
            mask = np.array(
                training_action_mask(
                    view,
                    side,
                    ground_truth=action,
                    sample_kind="turn",
                ),
                dtype=np.bool_,
            )
            tokens = push_trajectory(
                trajectories[side],
                current_snap,
                depth=TRAJECTORY_DEPTH,
                maxlen=TRAJECTORY_DEPTH,
                history_snapshot=hist_snap,
            )
            samples.append(
                SinglesParsedSample(
                    tokens=tokens,
                    action=action,
                    replay_id=replay_id,
                    turn=current_turn,
                    side=side,
                    sample_kind="turn",
                    view_state=view if keep_view_state else None,
                    action_mask=mask,
                )
            )
        turn_lines = []

    for line in lines:
        if line.startswith("|turn|"):
            _flush_turn()
            parts = line.split("|")
            if len(parts) >= 3:
                current_turn = int(parts[2])
                tracker.process_line(line)
                pre_turn_state = tracker.state.clone()
            lines_so_far.append(line)
            continue

        if current_turn >= 1:
            turn_lines.append(line)
            parts = line.split("|")
            if len(parts) >= 2:
                _try_force_switch_samples(
                    samples,
                    tracker=tracker,
                    turn_lines=turn_lines,
                    line_idx=len(turn_lines) - 1,
                    parts=parts,
                    lines_so_far=lines_so_far,
                    line=line,
                    replay_id=replay_id,
                    current_turn=current_turn,
                    trajectories=trajectories,
                    keep_view_state=keep_view_state,
                    meta_db=meta_db,
                )

        lines_so_far.append(line)
        tracker.process_line(line)

    _flush_turn()
    return samples


def parse_singles_log_file(
    path: Path,
    *,
    skip_rating: bool = False,
    keep_view_state: bool = False,
    meta_db=None,
) -> list[SinglesParsedSample]:
    replay_id = path.stem
    log_text = extract_log_text(path) if path.suffix == ".html" else path.read_text(
        encoding="utf-8", errors="replace"
    )
    return parse_singles_replay_log(
        log_text,
        replay_id=replay_id,
        skip_rating=skip_rating,
        keep_view_state=keep_view_state,
        meta_db=meta_db,
    )


def find_sample_view_state(
    log_dir: Path,
    *,
    replay_id: str,
    turn: int,
    side: str,
    sample_kind: str = "turn",
) -> BattleLogState | None:
    """Re-parse a singles replay log and return the view state for one sample."""
    path = log_dir / f"{replay_id}.log"
    if not path.is_file():
        matches = list(log_dir.glob(f"*{replay_id}*.log"))
        path = matches[0] if matches else path
    if not path.is_file():
        return None
    for sample in parse_singles_log_file(path, skip_rating=True, keep_view_state=True):
        if (
            sample.turn == turn
            and sample.side == side
            and sample.sample_kind == sample_kind
            and sample.view_state is not None
        ):
            return sample.view_state
    return None


def build_singles_dataset(samples: list[SinglesParsedSample]) -> dict:
    if not samples:
        empty = np.zeros(0, dtype=np.int64)
        return {
            "token_ids": np.zeros((0, TRAJECTORY_DEPTH * 13, N_FIELDS), dtype=np.int64),
            "action": empty,
            "action_mask": np.zeros((0, SINGLES_ACTION_SIZE), dtype=np.bool_),
            "meta": [],
        }
    tokens = np.stack([s.tokens for s in samples])
    actions = np.array([s.action for s in samples], dtype=np.int64)
    masks = np.stack([s.action_mask for s in samples])
    meta = [
        {
            "replay_id": s.replay_id,
            "turn": s.turn,
            "side": s.side,
            "sample_kind": s.sample_kind,
        }
        for s in samples
    ]
    return {
        "token_ids": tokens,
        "action": actions,
        "action_mask": masks,
        "meta": meta,
    }
