"""Tests for singles damage evaluation and training masks."""

from __future__ import annotations

from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon

from src.core.model.transformer_bot import SINGLES_ACTION_SIZE
from src.singles.log_action_mask import training_singles_mask
from src.singles.planning.damage_eval import estimated_damage_to_defender


def test_training_singles_mask_allows_multiple_actions():
    from src.core.data.log_tracker import BattleLogState
    from src.core.data.perspective import MonPerspective

    view = BattleLogState(turn=3)
    view.team_roster = {
        "p1": ["garchomp", "rotomwash", "floette", "volcarona", "meowscarada", "lucario"],
    }
    view.brought_species = {"p1": {"garchomp", "rotomwash", "floette"}}
    view.mons["p1a"] = MonPerspective(
        slot="p1a",
        species="garchomp",
        hp=100,
        max_hp=100,
        moves=["earthquake", "outrage", "stealthrock", "swordsdance"],
        active=True,
        can_mega=False,
    )
    mask = training_singles_mask(view, "p1", "turn", ground_truth=6)
    assert mask.shape == (SINGLES_ACTION_SIZE,)
    assert int(mask.sum()) >= 3
    assert mask[6]


def test_estimated_damage_uses_identifier_api(monkeypatch):
    from poke_env.battle.battle import Battle

    battle = Battle("tag", "gen9", None, None)
    attacker = Pokemon(gen=9, species="lucario")
    attacker._stats = {"hp": 150, "atk": 150, "def": 100, "spa": 100, "spd": 100, "spe": 120}
    defender = Pokemon(gen=9, species="garchomp")
    defender._stats = {"hp": 150, "atk": 120, "def": 100, "spa": 80, "spd": 80, "spe": 100}
    move = Move("closecombat", gen=9)
    battle._team = {"p1a": attacker}
    battle._opponent_team = {"p2a": defender}

    calls: list[tuple] = []

    def fake_calc(attacker_id, defender_id, mv, btl):
        calls.append((attacker_id, defender_id, mv.id))
        return 80, 95

    monkeypatch.setattr("src.singles.planning.damage_eval.calculate_damage", fake_calc)
    monkeypatch.setattr(
        "src.singles.planning.damage_eval.team_identifier",
        lambda _battle, mon: "p1a" if mon is attacker else "p2a",
    )

    score = estimated_damage_to_defender(battle, move, attacker, defender)
    assert score == 87.5
    assert calls == [("p1a", "p2a", "closecombat")]
