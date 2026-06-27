"""Singles action index layout (poke-env 0–21, BSS Reg M-A)."""

from __future__ import annotations

from dataclasses import dataclass

from src.core.model.transformer_bot import SINGLES_ACTION_SIZE
from src.singles.log_action_codec import (
    DMAX_TERA_BASE,
    MEGA_BASE,
    MOVE_BASE,
)

# Canonical singles layout (Bring-3 bench):
# 0-1 switch | 2-5 move | 6-9 mega | 10-13 z-move | 14-17 dynamax
Z_MOVE_BASE = 10
DYNAMAX_BASE = DMAX_TERA_BASE


@dataclass(frozen=True)
class SinglesActionDecode:
    """Decoded semantics for a flat singles action index."""

    index: int
    is_switch: bool
    move_slot: int | None
    mega: bool
    zmove: bool
    dynamax: bool
    tera: bool

    @property
    def is_move(self) -> bool:
        return not self.is_switch

    @property
    def illegal_gimmick(self) -> bool:
        """Z-Move, Dynamax, or Terastallize — banned in Champions M-A singles."""
        return self.zmove or self.dynamax or self.tera


def decode_singles_action_index(action_idx: int) -> SinglesActionDecode:
    """Decode flat index 0..SINGLES_ACTION_SIZE-1 into move/gimmick flags."""
    idx = int(action_idx)
    if idx < 0 or idx >= SINGLES_ACTION_SIZE:
        return SinglesActionDecode(
            index=idx,
            is_switch=False,
            move_slot=None,
            mega=False,
            zmove=False,
            dynamax=False,
            tera=False,
        )

    if idx < MOVE_BASE:
        return SinglesActionDecode(
            index=idx,
            is_switch=True,
            move_slot=None,
            mega=False,
            zmove=False,
            dynamax=False,
            tera=False,
        )

    move_slot: int
    mega = zmove = dynamax = tera = False

    if MOVE_BASE <= idx < MEGA_BASE:
        move_slot = idx - MOVE_BASE
    elif MEGA_BASE <= idx < Z_MOVE_BASE:
        move_slot = idx - MEGA_BASE
        mega = True
    elif Z_MOVE_BASE <= idx < DYNAMAX_BASE:
        move_slot = idx - Z_MOVE_BASE
        zmove = True
    else:
        move_slot = idx - DYNAMAX_BASE
        dynamax = True
        tera = True

    return SinglesActionDecode(
        index=idx,
        is_switch=False,
        move_slot=move_slot,
        mega=mega,
        zmove=zmove,
        dynamax=dynamax,
        tera=tera,
    )


def is_mega_action(action_idx: int) -> bool:
    return decode_singles_action_index(action_idx).mega


def is_zmove_action(action_idx: int) -> bool:
    return decode_singles_action_index(action_idx).zmove
