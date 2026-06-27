"""Parse Showdown replay logs into BC training samples (Turn 1+ only)."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.doubles.data.action_codec import (
    ACTION_UNKNOWN,
    describe_action,
    is_force_switch_decision,
    parse_force_switch_actions,
    parse_side_turn_actions,
)
from src.doubles.data.action_space_spec import ACTION_PASS, ACTION_SIZE, encode_combo
from src.core.data.log_tracker import BattleLogState, LogStateTracker, project_first_person
from src.core.data.perspective import stable_seed_int
from src.core.data.roster_profile import build_match_rosters
from src.core.data.state_tokenizer import (
    N_FIELDS,
    TRAJECTORY_DEPTH,
    encode_log_state,
    human_readable_state,
    push_trajectory,
    stack_trajectory,
)
from src.doubles.data.log_action_mask import training_slot_masks
from src.doubles.planning.meta_database import MetaDatabase

MIN_RATING = 1350
MIN_TURN = 3


@dataclass
class ParsedSample:
    tokens: np.ndarray
    action_combo: int
    action_slot0: int
    action_slot1: int
    replay_id: str
    turn: int
    side: str
    sample_kind: str = "turn"
    view_state: object | None = None  # BattleLogState after projection (debug)
    mask_slot0: np.ndarray | None = None
    mask_slot1: np.ndarray | None = None


def _rng_for_sample(replay_id: str, turn: int, side: str, sample_kind: str) -> random.Random:
    seed = stable_seed_int(replay_id, turn, side, sample_kind)
    return random.Random(seed)


def extract_log_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(
        r'<script[^>]*class="battle-log-data"[^>]*>(.*?)</script>',
        text,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    return text


def parse_log_lines(log_text: str) -> list[str]:
    return [ln.strip() for ln in log_text.splitlines() if ln.strip()]


def validate_log(tracker: LogStateTracker, *, skip_rating: bool = False) -> tuple[bool, str]:
    s = tracker.state
    if not skip_rating:
        r1 = s.ratings.get("p1", 0)
        r2 = s.ratings.get("p2", 0)
        if r1 < MIN_RATING or r2 < MIN_RATING:
            return False, f"rating_below_{MIN_RATING}"
    if s.max_turn < MIN_TURN:
        return False, f"ended_before_turn_{MIN_TURN}"
    return True, "ok"


def _emit_sample(
    samples: list[ParsedSample],
    *,
    trajectory: list[np.ndarray],
    view: BattleLogState,
    side: str,
    a0: int,
    a1: int,
    replay_id: str,
    turn: int,
    sample_kind: str,
    keep_view_state: bool,
    update_trajectory: bool,
) -> None:
    if a0 == ACTION_PASS and a1 == ACTION_PASS:
        return
    mask0, mask1 = training_slot_masks(
        view, side, sample_kind, ground_truth_a0=a0, ground_truth_a1=a1
    )
    snapshot = encode_log_state(view, side)
    if update_trajectory:
        tokens = push_trajectory(trajectory, snapshot, depth=TRAJECTORY_DEPTH, maxlen=TRAJECTORY_DEPTH)
    else:
        tokens = stack_trajectory(trajectory, snapshot, depth=TRAJECTORY_DEPTH)

    samples.append(
        ParsedSample(
            tokens=tokens,
            action_combo=encode_combo(a0, a1),
            action_slot0=a0,
            action_slot1=a1,
            replay_id=replay_id,
            turn=turn,
            side=side,
            sample_kind=sample_kind,
            view_state=view if keep_view_state else None,
            mask_slot0=mask0,
            mask_slot1=mask1,
        )
    )


def _new_trajectories() -> dict[str, list[np.ndarray]]:
    return {"p1": [], "p2": []}


def _try_force_switch_samples(
    samples: list[ParsedSample],
    *,
    tracker: LogStateTracker,
    turn_lines: list[str],
    line_idx: int,
    parts: list[str],
    rosters,
    replay_id: str,
    current_turn: int,
    trajectories: dict[str, list[np.ndarray]],
    keep_view_state: bool,
    meta_db: MetaDatabase,
) -> None:
    if not is_force_switch_decision(parts, turn_lines, line_idx):
        return
    pre_switch = tracker.state.clone()
    for side in ("p1", "p2"):
        rng = _rng_for_sample(replay_id, current_turn, side, "force_switch")
        view = project_first_person(
            pre_switch, side, rosters=rosters, meta_db=meta_db, rng=rng
        )
        pair = parse_force_switch_actions(parts, view, side)
        if pair is None:
            continue
        a0, a1 = pair
        _emit_sample(
            samples,
            trajectory=trajectories[side],
            view=view,
            side=side,
            a0=a0,
            a1=a1,
            replay_id=replay_id,
            turn=current_turn,
            sample_kind="force_switch",
            keep_view_state=keep_view_state,
            update_trajectory=False,
        )


def parse_replay_log(
    log_text: str,
    replay_id: str = "",
    *,
    skip_rating: bool = False,
    keep_view_state: bool = False,
    meta_db: MetaDatabase | None = None,
) -> list[ParsedSample]:
    lines = parse_log_lines(log_text)
    rosters = build_match_rosters(lines)
    tracker = LogStateTracker()
    if meta_db is None:
        meta_db = MetaDatabase(live_fetch=False)

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
    samples: list[ParsedSample] = []

    def _flush_turn() -> None:
        nonlocal turn_lines, pre_turn_state
        if current_turn < 1 or pre_turn_state is None or not turn_lines:
            turn_lines = []
            return
        for side in ("p1", "p2"):
            rng = _rng_for_sample(replay_id, current_turn, side, "turn")
            view = project_first_person(
                pre_turn_state, side, rosters=rosters, meta_db=meta_db, rng=rng
            )
            a0, a1 = parse_side_turn_actions(turn_lines, view, side)
            _emit_sample(
                samples,
                trajectory=trajectories[side],
                view=view,
                side=side,
                a0=a0,
                a1=a1,
                replay_id=replay_id,
                turn=current_turn,
                sample_kind="turn",
                keep_view_state=keep_view_state,
                update_trajectory=True,
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
                    rosters=rosters,
                    replay_id=replay_id,
                    current_turn=current_turn,
                    trajectories=trajectories,
                    keep_view_state=keep_view_state,
                    meta_db=meta_db,
                )

        tracker.process_line(line)

    _flush_turn()
    return samples


def parse_log_file(
    path: Path,
    *,
    skip_rating: bool = False,
    keep_view_state: bool = False,
    meta_db: MetaDatabase | None = None,
) -> list[ParsedSample]:
    replay_id = path.stem
    log_text = extract_log_text(path) if path.suffix == ".html" else path.read_text(
        encoding="utf-8", errors="replace"
    )
    return parse_replay_log(
        log_text,
        replay_id=replay_id,
        skip_rating=skip_rating,
        keep_view_state=keep_view_state,
        meta_db=meta_db,
    )


def build_dataset(samples: list[ParsedSample]) -> dict:
    if not samples:
        empty = np.zeros(0, dtype=np.int64)
        return {
            "token_ids": np.zeros((0, TRAJECTORY_DEPTH * 13, N_FIELDS), dtype=np.int64),
            "action_combo": empty,
            "action_slot0": empty,
            "action_slot1": empty,
            "mask_slot0": np.zeros((0, ACTION_SIZE), dtype=np.bool_),
            "mask_slot1": np.zeros((0, ACTION_SIZE), dtype=np.bool_),
            "meta": [],
        }
    tokens = np.stack([s.tokens for s in samples])
    combo = np.array([s.action_combo for s in samples], dtype=np.int64)
    a0 = np.array([s.action_slot0 for s in samples], dtype=np.int64)
    a1 = np.array([s.action_slot1 for s in samples], dtype=np.int64)
    m0 = np.stack([s.mask_slot0 for s in samples])
    m1 = np.stack([s.mask_slot1 for s in samples])
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
        "action_combo": combo,
        "action_slot0": a0,
        "action_slot1": a1,
        "mask_slot0": m0,
        "mask_slot1": m1,
        "meta": meta,
    }


def find_sample_view_state(
    log_dir: Path,
    *,
    replay_id: str,
    turn: int,
    side: str,
    sample_kind: str = "turn",
) -> BattleLogState | None:
    """Re-parse a replay log and return the view state for one sample."""
    path = log_dir / f"{replay_id}.log"
    if not path.is_file():
        matches = list(log_dir.glob(f"*{replay_id}*.log"))
        path = matches[0] if matches else path
    if not path.is_file():
        return None
    for sample in parse_log_file(path, skip_rating=True, keep_view_state=True):
        if (
            sample.turn == turn
            and sample.side == side
            and sample.sample_kind == sample_kind
            and sample.view_state is not None
        ):
            return sample.view_state
    return None


def sample_audit_dict(sample: ParsedSample) -> dict:
    view = sample.view_state
    if view is None:
        return {}
    return {
        "replay_id": sample.replay_id,
        "turn": sample.turn,
        "side": sample.side,
        "sample_kind": sample.sample_kind,
        "state": human_readable_state(view, sample.side),
        "action": describe_action(sample.action_slot0, sample.action_slot1),
        "tensor_shape": list(sample.tokens.shape),
        "tensor": sample.tokens.tolist(),
    }
