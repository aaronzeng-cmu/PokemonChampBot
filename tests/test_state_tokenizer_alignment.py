"""Training-inference alignment for encode_battle vs encode_log_state."""

from __future__ import annotations

import numpy as np

from src.core.data.log_tracker import BattleLogState
from src.core.data.perspective import MonPerspective, boost_id
from src.core.data.state_tokenizer import (
    FIELD_BOOST_START,
    FIELD_HP,
    FIELD_ROLE,
    FIELD_SPECIES,
    FIELD_STATUS,
    N_FIELDS,
    N_TOKENS,
    TOKEN_OPP_ACTIVE,
    TOKEN_OPP_BENCH,
    TOKEN_OUR_ACTIVE,
    TOKEN_OUR_BENCH,
    TRAJECTORY_DEPTH,
    empty_slot_token,
    _live_bench_members,
    empty_snapshot,
    encode_log_state,
    push_trajectory,
    stack_trajectory,
)


class _FakeMon:
    def __init__(
        self,
        *,
        species: str,
        active: bool = False,
        revealed: bool = False,
        selected: bool = False,
        team_index: int = 0,
    ):
        self.species = species
        self.active = active
        self._revealed = revealed
        self._selected_in_teampreview = selected
        self._team_index = team_index

    @property
    def revealed(self) -> bool:
        return self._revealed

    @property
    def selected_in_teampreview(self) -> bool:
        return self._selected_in_teampreview


def test_push_trajectory_copies_snapshot():
    snap = np.arange(N_TOKENS * N_FIELDS, dtype=np.int64).reshape(N_TOKENS, N_FIELDS)
    hist: list[np.ndarray] = []
    push_trajectory(hist, snap, depth=TRAJECTORY_DEPTH, maxlen=TRAJECTORY_DEPTH)
    snap[:] = 0
    assert hist[0].sum() > 0
    assert np.all(hist[0] == np.arange(N_TOKENS * N_FIELDS, dtype=np.int64).reshape(N_TOKENS, N_FIELDS))


def test_trajectory_chronological_order_oldest_first():
    """Order must be [t-2, t-1, t0] with oldest frame in the lowest token rows."""
    f0 = np.full((N_TOKENS, N_FIELDS), 1, dtype=np.int64)
    f1 = np.full((N_TOKENS, N_FIELDS), 2, dtype=np.int64)
    f2 = np.full((N_TOKENS, N_FIELDS), 3, dtype=np.int64)
    hist: list[np.ndarray] = []
    push_trajectory(hist, f0, depth=TRAJECTORY_DEPTH, maxlen=TRAJECTORY_DEPTH)
    push_trajectory(hist, f1, depth=TRAJECTORY_DEPTH, maxlen=TRAJECTORY_DEPTH)
    stacked = push_trajectory(hist, f2, depth=TRAJECTORY_DEPTH, maxlen=TRAJECTORY_DEPTH)
    assert stacked[0, 0] == 1
    assert stacked[N_TOKENS, 0] == 2
    assert stacked[2 * N_TOKENS, 0] == 3


def test_force_switch_stack_does_not_append_history():
    f0 = np.full((N_TOKENS, N_FIELDS), 5, dtype=np.int64)
    f1 = np.full((N_TOKENS, N_FIELDS), 6, dtype=np.int64)
    hist: list[np.ndarray] = []
    push_trajectory(hist, f0, depth=TRAJECTORY_DEPTH, maxlen=TRAJECTORY_DEPTH)
    assert len(hist) == 1
    stack_trajectory(hist, f1, depth=TRAJECTORY_DEPTH)
    assert len(hist) == 1
    assert hist[0][0, 0] == 5


def test_turn1_trajectory_zero_padding_matches_parser():
    snapshot = np.arange(N_TOKENS * N_FIELDS, dtype=np.int64).reshape(N_TOKENS, N_FIELDS)
    stacked = stack_trajectory([], snapshot, depth=TRAJECTORY_DEPTH)
    assert stacked.shape == (TRAJECTORY_DEPTH * N_TOKENS, N_FIELDS)
    assert np.all(stacked[: 2 * N_TOKENS] == 0)

    history: list[np.ndarray] = []
    via_push = push_trajectory(history, snapshot, depth=TRAJECTORY_DEPTH, maxlen=TRAJECTORY_DEPTH)
    assert np.array_equal(stacked, via_push)
    assert len(history) == 1


def test_brought_team_never_falls_back_to_full_six():
    from src.core.data.state_tokenizer import _brought_team_members

    team = [
        _FakeMon(species="lead-a", active=True, selected=True),
        _FakeMon(species="lead-b", active=True, selected=True),
        _FakeMon(species="bench-a", selected=True),
        _FakeMon(species="bench-b", selected=True),
        _FakeMon(species="ghost-a", selected=False),
        _FakeMon(species="ghost-b", selected=False),
    ]
    brought = _brought_team_members(team)
    assert len(brought) == 4
    assert "ghost-a" not in [p.species for p in brought]


def test_brought_team_uses_active_and_switches_without_preview_flags():
    from src.core.data.state_tokenizer import _brought_team_members

    class _FakeBattle:
        def __init__(self, switches):
            self.available_switches = switches

    team = [
        _FakeMon(species="lead-a", active=True),
        _FakeMon(species="lead-b", active=True),
        _FakeMon(species="bench-a"),
        _FakeMon(species="ghost-a"),
    ]
    battle = _FakeBattle(switches=[team[2]])
    brought = _brought_team_members(team, battle=battle)
    assert [p.species for p in brought] == ["lead-a", "lead-b", "bench-a"]


def test_live_bench_filters_unbrought_and_unrevealed():
    team = [
        _FakeMon(species="lead-a", active=True, selected=True),
        _FakeMon(species="lead-b", active=True, selected=True),
        _FakeMon(species="bench-a", selected=True),
        _FakeMon(species="bench-b", selected=True),
        _FakeMon(species="ghost-a", selected=False),
        _FakeMon(species="ghost-b", selected=False),
    ]
    ours = _live_bench_members(team, is_ours=True)
    assert [p.species for p in ours] == ["bench-a", "bench-b"]

    opp_team = [
        _FakeMon(species="opp-lead", active=True, revealed=True),
        _FakeMon(species="opp-bench", revealed=True),
        _FakeMon(species="opp-unseen", revealed=False),
    ]
    opp = _live_bench_members(opp_team, is_ours=False, battle=None)
    assert [p.species for p in opp] == ["opp-bench"]


def test_log_bench_max_two_with_zero_pad_slots():
    state = BattleLogState()
    for slot, species, active in [
        ("p1a", "a", True),
        ("p1b", "b", True),
        ("p1c", "c", False),
        ("p1d", "d", False),
    ]:
        state.mons[slot] = MonPerspective(
            slot=slot,
            species=species,
            hp=100,
            max_hp=100,
            active=active,
            seen=True,
            moves=["protect"],
        )
    tokens = encode_log_state(state, "p1")
    assert tokens[5, FIELD_SPECIES] != 0
    assert tokens[6, FIELD_SPECIES] != 0
    assert tokens[7, FIELD_SPECIES] == 0
    assert tokens[8, FIELD_SPECIES] == 0


def test_hp_and_boost_encoding_uses_fraction_and_stage_map():
    from src.core.data.state_tokenizer import _encode_mon_token
    from src.core.data.perspective import status_id

    mon = MonPerspective(
        species="pikachu",
        hp=37,
        max_hp=149,
        status="par",
        boosts={"atk": 2, "spe": -1},
        moves=["thunderbolt", "protect", "fakeout", "voltswitch"],
    )
    tok = _encode_mon_token(mon, role=1, is_ours=True)
    assert tok[FIELD_HP] == int(mon.hp_fraction * 20)
    assert tok[FIELD_STATUS] == status_id("par")
    assert tok[FIELD_BOOST_START] == boost_id(2)
    assert tok[FIELD_BOOST_START + 4] == boost_id(-1)


def test_singles_format_populates_active_and_two_bench_slots():
    """Singles uses tokens 1, 3, 5-6, 9-10; pads 2, 4, 7-8, 11-12 with EMPTY_SLOT."""
    state = BattleLogState()
    for slot, species, active in [
        ("p1a", "active-us", True),
        ("p1b", "bench-us-1", False),
        ("p1c", "bench-us-2", False),
        ("p2a", "active-them", True),
        ("p2b", "bench-them-1", False),
        ("p2c", "bench-them-2", False),
    ]:
        state.mons[slot] = MonPerspective(
            slot=slot,
            species=species,
            hp=100,
            max_hp=100,
            active=active,
            seen=True,
            moves=["protect"],
        )

    tokens = encode_log_state(state, "p1", format="singles")
    assert tokens.shape == (N_TOKENS, N_FIELDS)

    assert tokens[TOKEN_OUR_ACTIVE, FIELD_SPECIES] != 0
    assert tokens[TOKEN_OPP_ACTIVE, FIELD_SPECIES] != 0
    assert tokens[TOKEN_OUR_BENCH, FIELD_SPECIES] != 0
    assert tokens[TOKEN_OUR_BENCH + 1, FIELD_SPECIES] != 0
    assert tokens[TOKEN_OPP_BENCH, FIELD_SPECIES] != 0
    assert tokens[TOKEN_OPP_BENCH + 1, FIELD_SPECIES] != 0

    empty_our_active = empty_slot_token(TOKEN_OUR_ACTIVE)
    empty_opp_active = empty_slot_token(TOKEN_OPP_ACTIVE)
    empty_our_bench = empty_slot_token(TOKEN_OUR_BENCH)
    empty_opp_bench = empty_slot_token(TOKEN_OPP_BENCH)

    assert np.array_equal(tokens[TOKEN_OUR_ACTIVE + 1], empty_our_active)
    assert np.array_equal(tokens[TOKEN_OPP_ACTIVE + 1], empty_opp_active)
    assert np.array_equal(tokens[TOKEN_OUR_BENCH + 2], empty_our_bench)
    assert np.array_equal(tokens[TOKEN_OUR_BENCH + 3], empty_our_bench)
    assert np.array_equal(tokens[TOKEN_OPP_BENCH + 2], empty_opp_bench)
    assert np.array_equal(tokens[TOKEN_OPP_BENCH + 3], empty_opp_bench)

    stacked = stack_trajectory([], tokens, depth=TRAJECTORY_DEPTH)
    assert stacked.shape == (TRAJECTORY_DEPTH * N_TOKENS, N_FIELDS)
