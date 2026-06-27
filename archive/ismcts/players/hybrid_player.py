"""Knowledge-augmented hybrid player: BeliefState + DeepSeek + ISMCTS."""

from __future__ import annotations

from typing import Any, Optional

from poke_env.battle.double_battle import DoubleBattle
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder
from poke_env.player.player import Player

from config.settings import (
    BATTLE_FORMAT,
    ISMCTS_RL_VALUE_MODEL,
    ISMCTS_VALUE_MLP_PATH,
    TEAM_PATH,
)
from archive.rl.env.champions_vgc_env import ChampionsVGCRLEnv
from archive.rl.env.combo_action_space import enumerate_legal_combos
from src.doubles.planning.belief_state import BeliefState
from archive.ismcts.planning.ismcts import search as ismcts_search
from src.doubles.planning.macro_strategist import GamePlan, MacroStrategist
from src.doubles.planning.meta_database import MetaDatabase
from src.doubles.planning.observation_tracker import BattleSnapshot, ObservationTracker
from archive.ismcts.planning.rl_value import build_value_evaluator, build_value_fn_for_hybrid
from src.doubles.teams.teampreview import decode_combo_index, random_teampreview_command


class HybridPlayer(Player):
    def __init__(
        self,
        *,
        battle_format: str = BATTLE_FORMAT,
        team: Optional[str] = None,
        meta_db: MetaDatabase | None = None,
        macro: MacroStrategist | None = None,
        rl_value_model: str | None = None,
        value_mlp_path: str | None = None,
        **kwargs: Any,
    ):
        if team is None:
            team = TEAM_PATH.read_text(encoding="utf-8")
        super().__init__(battle_format=battle_format, team=team, **kwargs)
        self.meta_db = meta_db or MetaDatabase()
        self.macro = macro or MacroStrategist()
        self._tracker = ObservationTracker()
        self._battle_ctx: dict[str, dict] = {}
        self._value_evaluator = build_value_evaluator(
            mlp_path=value_mlp_path or ISMCTS_VALUE_MLP_PATH,
            ppo_path=rl_value_model or ISMCTS_RL_VALUE_MODEL or None,
        )
        self._value_fn = build_value_fn_for_hybrid(
            self._value_evaluator,
            lambda b: self._ctx(b).get("belief"),
        )

    def _ctx(self, battle: DoubleBattle) -> dict:
        tag = battle.battle_tag
        if tag not in self._battle_ctx:
            self._battle_ctx[tag] = {
                "belief": None,
                "game_plan": None,
                "prev_snapshot": None,
                "macro_done": False,
                "preview_step": 0,
            }
        return self._battle_ctx[tag]

    def _sync_belief(self, battle: DoubleBattle) -> None:
        """Apply observation tracker updates (gym path does not use _handle_battle_message)."""
        ctx = self._ctx(battle)
        belief = ctx.get("belief")
        if belief is None:
            return
        prev = ctx.get("prev_snapshot")
        self._tracker.process(battle, prev, belief, self.meta_db)
        ctx["prev_snapshot"] = BattleSnapshot.from_battle(battle)

    async def _handle_battle_message(self, split_messages):
        await super()._handle_battle_message(split_messages)
        for battle in self.battles.values():
            if battle.finished:
                continue
            ctx = self._ctx(battle)
            belief = ctx.get("belief")
            if belief is None and not battle.teampreview:
                belief = BeliefState()
                belief.initialize_from_preview(battle, self.meta_db)
                ctx["belief"] = belief
            if belief is not None:
                self._tracker.process(
                    battle,
                    ctx.get("prev_snapshot"),
                    belief,
                    self.meta_db,
                )
                ctx["prev_snapshot"] = BattleSnapshot.from_battle(battle)

    def _ensure_belief(self, battle: DoubleBattle) -> BeliefState:
        ctx = self._ctx(battle)
        belief = ctx.get("belief")
        if belief is None:
            belief = BeliefState()
            belief.initialize_from_preview(battle, self.meta_db)
            ctx["belief"] = belief
        return belief

    def _ensure_game_plan(self, battle: DoubleBattle) -> GamePlan:
        ctx = self._ctx(battle)
        if ctx.get("game_plan") is not None:
            return ctx["game_plan"]
        belief = self._ensure_belief(battle)
        plan = self.macro.analyze(battle, belief, self.meta_db)
        ctx["game_plan"] = plan
        ctx["macro_done"] = True
        return plan

    def _lead_combo_from_plan(self, battle: DoubleBattle, plan: GamePlan) -> int | None:
        if len(plan.optimal_lead) < 2:
            return None
        lead_a, lead_b = plan.optimal_lead[0], plan.optimal_lead[1]
        species = [p.base_species for p in battle.team.values()]
        try:
            combos = enumerate_legal_combos(battle)
        except ValueError:
            return None
        best_idx = None
        best_score = -1.0
        for idx in range(len(combos)):
            try:
                s1, s2 = decode_combo_index(battle, idx, combos=combos)
            except (ValueError, TypeError):
                continue
            sp1 = species[s1 - 1]
            sp2 = species[s2 - 1]
            score = 0.0
            if sp1 in (lead_a, lead_b):
                score += 1.0
            if sp2 in (lead_a, lead_b):
                score += 1.0
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx if best_score > 0 else None

    def choose_preview_combo(self, battle: DoubleBattle) -> int:
        """Return combo index for gym-style team preview (masked combos)."""
        self._ensure_belief(battle)
        self._sync_belief(battle)
        plan = self._ensure_game_plan(battle)
        idx = self._lead_combo_from_plan(battle, plan)
        if idx is not None:
            return idx
        belief = self._ensure_belief(battle)
        try:
            combos = enumerate_legal_combos(battle)
        except ValueError:
            return 0
        if not combos:
            return 0
        return ismcts_search(battle, belief, plan, value_fn=self._value_fn)

    def teampreview(self, battle: DoubleBattle) -> str:
        ctx = self._ctx(battle)
        belief = self._ensure_belief(battle)
        plan = self._ensure_game_plan(battle)

        if ctx["preview_step"] == 0:
            ctx["preview_step"] = 1
            combo_idx = self._lead_combo_from_plan(battle, plan)
            if combo_idx is not None:
                order = ChampionsVGCRLEnv.action_to_order(combo_idx, battle, strict=False)
                species = [p.base_species for p in battle.team.values()]
                team_list = list(battle.team.values())
                a = order.first_order.order
                b = order.second_order.order
                i1 = species.index(a.base_species) + 1
                i2 = species.index(b.base_species) + 1
                team_list[i1 - 1]._selected_in_teampreview = True
                team_list[i2 - 1]._selected_in_teampreview = True
                remaining = [
                    i
                    for i in range(1, len(team_list) + 1)
                    if not team_list[i - 1].selected_in_teampreview
                ]
                if battle.format and "vgc" in battle.format:
                    bring = remaining[:2]
                else:
                    bring = remaining
                slots = [i1, i2] + bring
                return "/team " + "".join(str(s) for s in slots)

        return random_teampreview_command(battle)

    def choose_move(self, battle: DoubleBattle) -> BattleOrder:
        if battle.wait:
            return DefaultBattleOrder()

        ctx = self._ctx(battle)
        belief = self._ensure_belief(battle)
        self._sync_belief(battle)
        plan = ctx.get("game_plan") or self._ensure_game_plan(battle)

        combo_idx = ismcts_search(battle, belief, plan, value_fn=self._value_fn)
        return ChampionsVGCRLEnv.action_to_order(combo_idx, battle, strict=False)
