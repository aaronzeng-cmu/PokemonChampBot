"""Offline tests for planning / belief / macro strategist modules."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.doubles.planning.belief_state import BeliefState, Distribution
from src.core.planning.game_plan import GamePlan
from src.doubles.planning.macro_strategist import HeuristicMacroStrategist
from src.doubles.planning.meta_database import MetaDatabase
from src.doubles.planning.spread_priors import spread_key


class TestDistribution(unittest.TestCase):
    def test_normalize_and_sample(self):
        dist = Distribution({"A": 1.0, "B": 3.0}).normalized()
        self.assertAlmostEqual(sum(dist.options.values()), 1.0)
        rng = random.Random(42)
        samples = [dist.sample(rng) for _ in range(100)]
        self.assertIn("A", samples)
        self.assertIn("B", samples)

    def test_collapse(self):
        dist = Distribution({"A": 0.5, "B": 0.5})
        dist.collapse("A")
        self.assertEqual(dist.options, {"A": 1.0})


class TestMetaDatabase(unittest.TestCase):
    def test_load_and_species_prior(self):
        db = MetaDatabase(live_fetch=False)
        prior = db.get_species_prior("Incineroar")
        self.assertTrue(prior.moves)
        top = max(prior.moves.values())
        self.assertGreater(top, 50.0)  # raw Pikalytics % (e.g. Fake Out ~99%)
        self.assertTrue(prior.items)
        self.assertTrue(prior.abilities)

    def test_species_name_resolution(self):
        db = MetaDatabase(live_fetch=True)
        prior = db.get_species_prior("Charizard", item="Charizardite Y")
        self.assertEqual(prior.pikalytics_key, "Charizard-Mega-Y")
        self.assertGreater(prior.moves.get("Heat Wave", 0), 10.0)

    def test_mega_family_blended_with_usage(self):
        db = MetaDatabase(live_fetch=True)
        prior = db.get_species_prior("Charizard")
        self.assertIn("Charizard-Mega-Y", prior.form_variants)
        self.assertGreater(
            prior.form_variants.get("Charizard-Mega-Y", 0),
            prior.form_variants.get("Charizard-Mega-X", 0),
        )
        self.assertEqual(prior.pikalytics_key, "Charizard-Mega-Y")
        self.assertGreater(prior.moves.get("Heat Wave", 0), 50.0)
        self.assertGreater(prior.items.get("Charizardite Y", 0), 50.0)

    def test_matchup_context_nonempty_for_known_species(self):
        db = MetaDatabase()
        ctx = db.get_matchup_context(
            ["Incineroar", "Rillaboom"],
            ["Kingambit", "Incineroar"],
        )
        self.assertIsInstance(ctx, str)


class TestBeliefState(unittest.TestCase):
    def test_collapse_move_and_speed_floor(self):
        belief = BeliefState()
        from src.doubles.planning.belief_state import BeliefPokemon

        mon = BeliefPokemon(
            species="Ogerpon",
            moves=[Distribution({"Ivy Cudgel": 0.6, "Spiky Shield": 0.4}) for _ in range(4)],
            item=Distribution({"Choice Scarf": 0.3, "Sitrus Berry": 0.7}),
        )
        belief._mons["Ogerpon"] = mon
        belief.collapse_move("Ogerpon", "spikyshield", slot_idx=0)
        self.assertIn("spikyshield", {m.lower() for m in mon.revealed_moves} | {"Spiky Shield"})

        belief.update_speed_floor("Ogerpon", 151)
        self.assertEqual(mon.speed_floor, 151)

    def test_collapse_mega_form_updates_priors(self):
        db = MetaDatabase(live_fetch=True)
        belief = BeliefState()
        from src.doubles.planning.belief_state import BeliefPokemon

        blended = db.get_species_prior("Charizard")
        belief._mons["charizard"] = BeliefPokemon(
            species="Charizard",
            moves=[Distribution(dict(blended.moves)) for _ in range(4)],
            item=Distribution(dict(blended.items)),
            ability=Distribution(dict(blended.abilities)),
        )
        belief.collapse_mega_form("charizard", db, item="Charizardite Y")
        mon = belief.get("charizard")
        self.assertTrue(mon.mega_confirmed)
        self.assertEqual(mon.mega_form, "Charizard-Mega-Y")
        self.assertGreater(mon.moves[0].options.get("Heat Wave", 0), 50.0)
        self.assertGreater(mon.item.options.get("Charizardite Y", 0), 0.0)

        concrete = belief.sample_determinization(random.Random(0))
        self.assertTrue(concrete["charizard"].mega)

    def test_mega_form_for_ability(self):
        from src.core.planning.species_normalize import infer_mega_stone, mega_form_for_ability

        self.assertEqual(
            mega_form_for_ability("Charizard", "drought"),
            "Charizard-Mega-Y",
        )
        self.assertEqual(
            mega_form_for_ability("Charizard", "toughclaws"),
            "Charizard-Mega-X",
        )

        class _FakeMon:
            base_species = "charizard"
            species = "charizard"
            item = "charizarditey"
            ability = "drought"
            _last_details = "Charizard, L50, F"
            forme_change_ability = "drought"

        self.assertEqual(infer_mega_stone(_FakeMon()), "Charizardite Y")

    def test_brought_roster_belief(self):
        db = MetaDatabase(live_fetch=False)

        class _Mon:
            def __init__(self, species: str, *, revealed: bool = False):
                self.species = species
                self.base_species = species
                self.item = ""
                self.ability = ""
                self.moves = {}
                self.stats = {}
                self.current_hp = 100
                self.max_hp = 100
                self.fainted = False
                self.revealed = revealed
                self._last_details = ""

        class _Battle:
            teampreview_opponent_team = [
                _Mon("Incineroar"),
                _Mon("Rillaboom"),
                _Mon("Kingambit"),
                _Mon("Flutter Mane"),
                _Mon("Urshifu"),
                _Mon("Amoonguss"),
            ]
            _opponent_team = {}
            opponent_active_pokemon = []
            opponent_team = {}
            turn = 0
            team = {}

        belief = BeliefState()
        belief.initialize_from_preview(_Battle(), db)
        self.assertEqual(len(belief.pokemon), 6)
        uncertain = [m for m in belief.pokemon if not m.confirmed_brought]
        self.assertEqual(len(uncertain), 6)
        self.assertAlmostEqual(sum(m.brought_prob for m in uncertain), 4.0, places=5)

        belief.confirm_brought("Incineroar", db, battle_mon=_Mon("Incineroar", revealed=True))
        incin = belief.get("Incineroar")
        self.assertTrue(incin.confirmed_brought)
        self.assertFalse(incin.preview_only)
        self.assertEqual(incin.brought_prob, 1.0)
        remaining = [m for m in belief.pokemon if not m.confirmed_brought]
        self.assertAlmostEqual(sum(m.brought_prob for m in remaining), 3.0, places=5)

        for sp in ("Rillaboom", "Kingambit", "Flutter Mane", "Urshifu"):
            belief.confirm_brought(sp, db)
        absent = [m for m in belief.pokemon if m.confirmed_absent]
        self.assertEqual(len(absent), 2)
        self.assertTrue(all(m.brought_prob == 0.0 for m in absent))
        absent_names = {m.species for m in absent}
        self.assertEqual(len(absent_names), 2)

        rng = random.Random(0)
        samples = [belief.sample_determinization(rng) for _ in range(20)]
        for sample in samples:
            for name in absent_names:
                self.assertNotIn(name, sample)

    def test_sample_determinization(self):
        belief = BeliefState()
        from src.doubles.planning.belief_state import BeliefPokemon

        belief._mons["Incineroar"] = BeliefPokemon(
            species="Incineroar",
            moves=[Distribution({"Fake Out": 1.0}) for _ in range(4)],
            item=Distribution({"Sitrus Berry": 1.0}),
            ability=Distribution({"Intimidate": 1.0}),
            ev_spread=Distribution({spread_key("Careful", [32, 0, 14, 0, 20, 0]): 1.0}),
            tera_type=Distribution({"Fire": 1.0}),
        )
        concrete = belief.sample_determinization(random.Random(0))
        self.assertIn("Incineroar", concrete)
        self.assertEqual(len(concrete["Incineroar"].moves), 4)


class TestMacroStrategist(unittest.TestCase):
    def test_game_plan_from_dict(self):
        plan = GamePlan.from_dict(
            {
                "primary_threats": ["Kingambit"],
                "optimal_lead": ["Incineroar", "Rillaboom"],
                "win_condition": "KO Kingambit early",
                "priority_kos": ["Kingambit"],
            }
        )
        self.assertEqual(plan.primary_threats, ["Kingambit"])
        self.assertEqual(len(plan.optimal_lead), 2)


class TestPikalyticsBattleUsage(unittest.TestCase):
    def test_battle_usage_list_url(self):
        from src.doubles.teams.pikalytics_meta import (
            battle_format_key,
            battle_usage_list_url,
            resolve_data_date,
        )

        self.assertEqual(battle_format_key(), "gen9championsvgc2026regma-1760")
        self.assertEqual(
            battle_usage_list_url(data_date="2026-05"),
            "https://www.pikalytics.com/api/l/2026-05/gen9championsvgc2026regma-1760",
        )
        md = "**Data Date**: 2026-05\n"
        self.assertEqual(resolve_data_date(markdown=md), "2026-05")

    def test_discover_species_from_battle_usage_live(self):
        from src.doubles.teams.pikalytics_meta import discover_species_from_battle_usage

        names = discover_species_from_battle_usage()
        self.assertGreater(len(names), 200)
        self.assertIn("Mimikyu", names)
        self.assertIn("Kingambit", names)


class TestMacroValidation(unittest.TestCase):
    def test_rejects_hallucinated_lead(self):
        from src.core.planning.game_plan import GamePlan
        from src.core.planning.macro_validation import validate_and_normalize_game_plan

        our = ["Incineroar", "Rillaboom", "Whimsicott"]
        opp = ["Kingambit", "Flutter Mane"]
        plan = GamePlan(
            primary_threats=["Kingambit"],
            optimal_lead=["Arcanine", "Incineroar"],
            opponent_likely_lead=["Kingambit", "Flutter Mane"],
            priority_kos=["Kingambit"],
        )
        self.assertIsNone(validate_and_normalize_game_plan(plan, our, opp))

    def test_accepts_valid_plan(self):
        from src.core.planning.game_plan import GamePlan
        from src.core.planning.macro_validation import validate_and_normalize_game_plan

        our = ["Incineroar", "Rillaboom"]
        opp = ["Kingambit", "Flutter Mane"]
        plan = GamePlan(
            primary_threats=["Kingambit"],
            optimal_lead=["Incineroar", "Rillaboom"],
            opponent_likely_lead=["Kingambit", "Flutter Mane"],
            priority_kos=["Flutter Mane"],
        )
        out = validate_and_normalize_game_plan(plan, our, opp)
        self.assertIsNotNone(out)
        self.assertEqual(out.optimal_lead, ["Incineroar", "Rillaboom"])


class TestGauntlet(unittest.TestCase):
    def test_equal_team_weights(self):
        from src.doubles.evaluation.gauntlet_runner import load_gauntlet_pool

        teams = load_gauntlet_pool(max_teams=5, equal_weights=True)
        self.assertEqual(len(teams), 5)
        self.assertAlmostEqual(sum(t.weight for t in teams), 1.0, places=5)
        for t in teams:
            self.assertAlmostEqual(t.weight, 0.2, places=5)

    def test_meta_team_weights(self):
        from src.doubles.evaluation.gauntlet_runner import load_gauntlet_pool

        teams = load_gauntlet_pool(max_teams=10, equal_weights=False)
        self.assertEqual(len(teams), 10)
        self.assertAlmostEqual(sum(t.weight for t in teams), 1.0, places=5)
        weights = [t.weight for t in teams]
        self.assertGreater(max(weights) - min(weights), 1e-6)

    def test_team_meta_score(self):
        from src.doubles.teams.gauntlet_weights import parse_team_species_names, team_meta_score

        export = (Path(__file__).resolve().parents[1] / "teams" / "opponents").glob("*.txt")
        path = next(export, None)
        self.assertIsNotNone(path)
        text = path.read_text(encoding="utf-8")
        species = parse_team_species_names(text)
        self.assertEqual(len(species), 6)
        score = team_meta_score(text)
        self.assertGreater(score, 0.0)

    def test_value_mlp_roundtrip(self):
        import tempfile

        import torch

        from archive.ismcts.planning.value_mlp import (
            ValueMLP,
            ValueMLPConfig,
            load_value_mlp,
            save_value_mlp,
        )

        config = ValueMLPConfig(input_dim=32, hidden_dims=(16, 8), dropout=0.0)
        model = ValueMLP(config)
        model.eval()
        x = torch.randn(4, 32)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (4,))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_mlp.pt"
            save_value_mlp(model, path, extra={"test": True})
            loaded = load_value_mlp(path)
            with torch.no_grad():
                out2 = loaded(x)
            self.assertTrue(torch.allclose(out, out2, atol=1e-5))

    def test_archetype_trick_room(self):
        from src.doubles.teams.archetype import classify_team_export

        text = "Amoonguss @ Sitrus Berry\n- Trick Room\n- Spore\n"
        arch = classify_team_export(text)
        self.assertIn("trick_room", arch.tags)

    def test_archetype_tailwind(self):
        from src.doubles.teams.archetype import classify_team_export

        text = "Whimsicott @ Focus Sash\n- Tailwind\n- Encore\n"
        arch = classify_team_export(text)
        self.assertIn("tailwind", arch.tags)

    def test_belief_roster_embedding(self):
        from src.doubles.planning.belief_state import BeliefPokemon, BeliefState
        from archive.ismcts.planning.value_state import embed_belief_roster

        belief = BeliefState()
        belief._mons["a"] = BeliefPokemon(
            species="Incineroar", slot=1, brought_prob=0.8, confirmed_brought=True
        )
        vec = embed_belief_roster(belief)
        self.assertEqual(vec.shape[0], 24)
        self.assertAlmostEqual(vec[0], 0.8)
        self.assertAlmostEqual(vec[1], 1.0)


class TestSpreadPriors(unittest.TestCase):
    def test_aggregate_from_pool(self):
        from src.doubles.planning.spread_priors import aggregate_spread_priors

        priors = aggregate_spread_priors()
        self.assertIsInstance(priors, dict)
        if priors:
            species, spreads = next(iter(priors.items()))
            self.assertAlmostEqual(sum(spreads.values()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
