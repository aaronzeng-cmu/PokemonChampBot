"""Singles live vs log token parity (mega, force switch, items, bench preview)."""

from __future__ import annotations

from poke_env.battle.battle import Battle
from poke_env.battle.pokemon import Pokemon

from src.core.data.log_tracker import BattleLogState, project_first_person
from src.core.data.perspective import MonPerspective, hash_token
from src.core.data.roster_profile import build_match_rosters
from src.core.data.state_tokenizer import (
    DECISION_FORCE_SWITCH,
    FIELD_ABILITY,
    FIELD_DECISION_FLAGS,
    FIELD_FLAGS,
    FIELD_ITEM,
    FIELD_LAST_MOVE_ID,
    FLAG_CAN_MEGA,
    TOKEN_OPP_BENCH,
    encode_log_state,
    encode_singles_battle,
)


def _mock_singles_battle(
    *,
    can_mega: bool = False,
    force_switch: bool = False,
    with_opp_preview: bool = False,
) -> Battle:
    battle = Battle("test", "gen9", None, None)
    mon = Pokemon(gen=9, species="floetteeternal")
    mon._active = True
    mon.item = "floettite"
    mon.ability = "flowerveil"
    opp = Pokemon(gen=9, species="gengar")
    opp._active = True
    battle._team = {"p1a": mon}
    battle._opponent_team = {"p2a": opp}
    if with_opp_preview:
        for i, sp in enumerate(["dragonite", "tyranitar"], start=1):
            p = Pokemon(gen=9, species=sp)
            p._active = False
            p._revealed = False
            battle._opponent_team[f"p2b{i}"] = p
    battle._active_pokemon = mon
    battle._opponent_active_pokemon = opp
    battle._can_mega_evolve = can_mega
    battle._force_switch = force_switch
    return battle


def test_encode_singles_battle_sets_can_mega_flag():
    off = encode_singles_battle(_mock_singles_battle(can_mega=False))
    on = encode_singles_battle(_mock_singles_battle(can_mega=True))
    assert int(off[1, FIELD_FLAGS]) & FLAG_CAN_MEGA == 0
    assert int(on[1, FIELD_FLAGS]) & FLAG_CAN_MEGA == FLAG_CAN_MEGA


def test_encode_singles_battle_sets_force_switch_field_bit():
    normal = encode_singles_battle(_mock_singles_battle(force_switch=False))
    forced = encode_singles_battle(_mock_singles_battle(force_switch=True))
    assert int(normal[0, FIELD_DECISION_FLAGS]) == 0
    assert int(forced[0, FIELD_DECISION_FLAGS]) & DECISION_FORCE_SWITCH


def test_encode_log_state_singles_force_switch_field_bit():
    view = BattleLogState(turn=2)
    view.mons["p1a"] = MonPerspective(
        slot="p1a",
        species="volcarona",
        hp=0,
        max_hp=191,
        fainted=True,
        active=True,
        moves=["fierydance"],
        can_mega=False,
    )
    normal = encode_log_state(view, "p1", format="singles", force_switch=False)
    forced = encode_log_state(view, "p1", format="singles", force_switch=True)
    assert int(normal[0, FIELD_DECISION_FLAGS]) == 0
    assert int(forced[0, FIELD_DECISION_FLAGS]) & DECISION_FORCE_SWITCH


def test_live_encodes_our_item_and_ability():
    view = BattleLogState(turn=2)
    view.mons["p1a"] = MonPerspective(
        slot="p1a",
        species="floetteeternal",
        hp=100,
        max_hp=149,
        active=True,
        moves=["moonblast"],
        item="floettite",
        item_revealed=True,
        ability="flowerveil",
        ability_revealed=True,
    )
    view.mons["p2a"] = MonPerspective(
        slot="p2a", species="gengar", hp=100, max_hp=100, active=True, seen=True
    )
    log_tok = encode_log_state(view, "p1", format="singles")
    live_tok = encode_singles_battle(_mock_singles_battle())

    assert int(live_tok[1, FIELD_ITEM]) == int(log_tok[1, FIELD_ITEM])
    assert int(live_tok[1, FIELD_ABILITY]) == int(log_tok[1, FIELD_ABILITY])
    assert int(live_tok[1, FIELD_ITEM]) == hash_token("floettite")


def test_live_opp_bench_preview_matches_log():
    lines = [
        "|player|p1|Bot|1000",
        "|player|p2|Opp|1000",
        "|poke|p1|Floette, L50|",
        "|poke|p1|Volcarona, L50|",
        "|poke|p1|Rotom-Wash, L50|",
        "|poke|p2|Gengar, L50|",
        "|poke|p2|Dragonite, L50|",
        "|poke|p2|Tyranitar, L50|",
        "|poke|p2|Excadrill, L50|",
        "|poke|p2|Clefable, L50|",
        "|poke|p2|Corviknight, L50|",
        "|start",
        "|switch|p1a|Floette|149/149",
        "|switch|p2a|Gengar|100/100",
    ]
    rosters = build_match_rosters(lines)
    state = BattleLogState(turn=1)
    state.mons["p1a"] = MonPerspective(
        slot="p1a", species="floette", hp=149, max_hp=149, active=True, moves=["moonblast"]
    )
    state.mons["p2a"] = MonPerspective(
        slot="p2a", species="gengar", hp=100, max_hp=100, active=True, seen=True, moves=["shadowball"]
    )
    state.team_roster = {
        "p1": ["floette", "volcarona", "rotomwash"],
        "p2": ["gengar", "dragonite", "tyranitar", "excadrill", "clefable", "corviknight"],
    }
    state.brought_species = {
        "p1": set(state.team_roster["p1"]),
        "p2": set(state.team_roster["p2"]),
    }
    view = project_first_person(state, "p1", rosters=rosters, format="singles")
    log_tok = encode_log_state(view, "p1", format="singles")

    battle = _mock_singles_battle(with_opp_preview=True)
    live_tok = encode_singles_battle(battle)

    assert int(live_tok[TOKEN_OPP_BENCH, 1]) == int(log_tok[TOKEN_OPP_BENCH, 1])
    assert int(live_tok[TOKEN_OPP_BENCH + 1, 1]) == int(log_tok[TOKEN_OPP_BENCH + 1, 1])
    assert int(live_tok[TOKEN_OPP_BENCH, 1]) != 0


def test_last_move_id_on_active_log_slot():
    from src.core.data.perspective import move_vocab_id

    view = BattleLogState(turn=3)
    view.mons["p1a"] = MonPerspective(
        slot="p1a",
        species="floetteeternal",
        hp=100,
        max_hp=149,
        active=True,
        moves=["moonblast", "calmmind", "drainingkiss", "lightofruin"],
        last_move_id=move_vocab_id("moonblast"),
    )
    tok = encode_log_state(view, "p1", format="singles")
    assert int(tok[1, FIELD_LAST_MOVE_ID]) == move_vocab_id("moonblast")
