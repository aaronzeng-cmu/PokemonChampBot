"""Tests for singles semantic canonical inference bridge."""

from __future__ import annotations

from poke_env.battle.battle import Battle
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.player.player import Player

from src.core.data.log_tracker import BattleLogState
from src.core.data.perspective import MonPerspective
from src.core.data.roster_profile import roster_species_key
from src.core.data.state_tokenizer import (
    SINGLES_BENCH_SLOTS,
    TOKEN_OUR_BENCH,
    encode_singles_battle,
)
from src.singles.action_space_spec import MEGA_BASE, Z_MOVE_BASE
from src.singles.battle.canonical_inference import (
    canonical_index_to_battle_order,
    pick_masked_canonical_index,
)
from src.singles.bench_slots import live_our_bench_species, log_our_bench_species, species_to_bench_switch_index
from src.singles.battle.live_legality import legal_switch_indices, live_brought_species
from src.singles.action_mask import singles_action_mask
from src.singles.log_action_codec import MOVE_BASE, SWITCH_BASE, encode_singles_log_switch


def _team_mon(species: str, *, active: bool = False, selected: bool = False) -> Pokemon:
    mon = Pokemon(gen=9, species=species)
    mon._active = active
    mon._selected_in_teampreview = selected
    mon._revealed = True
    mon._current_hp = 100
    mon._max_hp = 100
    return mon


def _mock_battle_bring3() -> Battle:
    battle = Battle("tag", "gen9championsbssregma", None, None)
    lucario = _team_mon("lucario", selected=False)
    rotom = _team_mon("rotomwash", selected=False)
    floette = _team_mon("floetteeternal", active=True, selected=True)
    volcarona = _team_mon("volcarona", selected=True)
    meow = _team_mon("meowscarada", selected=True)
    garchomp = _team_mon("garchomp", selected=False)

    moves = [
        Move("calmmind", gen=9),
        Move("moonblast", gen=9),
        Move("drainingkiss", gen=9),
        Move("lightofruin", gen=9),
    ]
    battle._team = {
        "p1a": lucario,
        "p1b": rotom,
        "p1c": floette,
        "p1d": volcarona,
        "p1e": meow,
        "p1f": garchomp,
    }
    battle._active_pokemon = floette
    battle._opponent_active_pokemon = Pokemon(gen=9, species="hippowdon")
    battle._opponent_team = {"p2a": battle._opponent_active_pokemon}
    battle._available_moves = moves
    battle._available_switches = [volcarona, meow]
    battle._can_mega_evolve = True
    battle._force_switch = False
    battle._wait = False

    move_order = Player.create_order(moves[0])
    mega_order = Player.create_order(moves[0], mega=True)
    switch_meow = Player.create_order(meow)
    switch_volc = Player.create_order(volcarona)
    battle._valid_orders = [move_order, mega_order, switch_meow, switch_volc]
    return battle


def _mock_log_view_bring3() -> BattleLogState:
    view = BattleLogState(turn=2)
    view.team_roster = {
        "p1": ["lucario", "rotomwash", "floetteeternal", "volcarona", "meowscarada", "garchomp"],
    }
    view.brought_species = {
        "p1": {"floetteeternal", "volcarona", "meowscarada"},
    }
    view.mons["p1a"] = MonPerspective(
        slot="p1a",
        species="floetteeternal",
        hp=53,
        max_hp=149,
        active=True,
        moves=["calmmind", "moonblast", "drainingkiss", "lightofruin"],
    )
    view.mons["p1b"] = MonPerspective(
        slot="p1b",
        species="volcarona",
        hp=100,
        max_hp=100,
        active=False,
        moves=["fierydance"],
    )
    view.mons["p1c"] = MonPerspective(
        slot="p1c",
        species="meowscarada",
        hp=155,
        max_hp=155,
        active=False,
        moves=["uturn"],
    )
    return view


def test_log_switch_encodes_bench_slot_not_paste_index():
    view = _mock_log_view_bring3()
    # volcarona is paste slot 3 but bench token 1
    assert species_to_bench_switch_index(view, "p1", "Volcarona, L50") == 0
    assert encode_singles_log_switch(view, "p1a: Floette", "Volcarona, L50") == SWITCH_BASE + 0
    assert species_to_bench_switch_index(view, "p1", "Meowscarada, L50") == 1
    assert encode_singles_log_switch(view, "p1a: Floette", "Meowscarada, L50") == SWITCH_BASE + 1
    assert encode_singles_log_switch(view, "p1a: Floette", "Lucario, L50") == -100


def test_live_brought_excludes_unselected_paste_mons():
    battle = _mock_battle_bring3()
    brought = live_brought_species(battle)
    assert brought == {
        roster_species_key("floetteeternal"),
        roster_species_key("volcarona"),
        roster_species_key("meowscarada"),
    }
    assert roster_species_key("lucario") not in brought


def test_legal_switch_indices_are_bench_slots():
    battle = _mock_battle_bring3()
    legal = legal_switch_indices(battle)
    assert legal == {SWITCH_BASE, SWITCH_BASE + 1}
    bench_species = live_our_bench_species(battle)
    assert bench_species == [
        roster_species_key("volcarona"),
        roster_species_key("meowscarada"),
    ]


def test_live_mask_blocks_z_moves():
    battle = _mock_battle_bring3()
    mask = singles_action_mask(battle)
    assert not any(mask[Z_MOVE_BASE:])
    assert mask[MEGA_BASE]


def test_encode_singles_bench_only_brought_off_field():
    battle = _mock_battle_bring3()
    tokens = encode_singles_battle(battle)
    bench_species_hashes = [int(tokens[TOKEN_OUR_BENCH + i, 1]) for i in range(SINGLES_BENCH_SLOTS)]
    assert all(h != 0 for h in bench_species_hashes)


def test_canonical_move_submission_by_move_id():
    battle = _mock_battle_bring3()
    order = canonical_index_to_battle_order(battle, MOVE_BASE)
    assert "calmmind" in str(order).lower()


def test_canonical_switch_submission_uses_bench_token():
    battle = _mock_battle_bring3()
    battle._force_switch = True
    order = canonical_index_to_battle_order(battle, SWITCH_BASE + 1)
    assert "meowscarada" in str(order).lower()


def test_pick_masked_skips_illegal_paste_switch():
    battle = _mock_battle_bring3()
    battle._force_switch = True
    mask = singles_action_mask(battle)
    legal_sw = legal_switch_indices(battle)
    assert legal_sw.issubset({SWITCH_BASE, SWITCH_BASE + 1})
    assert any(mask[i] for i in legal_sw)
    logits = [10.0] * len(mask)
    picked = pick_masked_canonical_index(logits, mask)
    assert picked in legal_sw
