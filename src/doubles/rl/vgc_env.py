"""Gymnasium doubles env with BC (39x24) observations and canonical action masks."""

from __future__ import annotations

import json
import random
from collections import deque
from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box, MultiDiscrete
from poke_env.battle.double_battle import DoubleBattle
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)
from poke_env.ps_client import LocalhostServerConfiguration, ServerConfiguration
from poke_env.teambuilder import Teambuilder

from config.settings import (
    BATTLE_FORMAT,
    SHUFFLE_TEAM_ORDER,
    TEAM_PATH,
    USE_OPPONENT_TEAM_POOL,
)
from src.doubles.battle.canonical_inference import (
    canonical_index_to_single_order,
    is_joint_order_valid,
)
from src.doubles.data.action_space_spec import ACTION_PASS, ACTION_SIZE
from src.doubles.data.live_log_bridge import encode_live_as_log, slot_mask_for_live
from src.core.data.log_tracker import BattleLogState
from src.core.data.state_tokenizer import (
    N_FIELDS,
    N_TOKENS,
    STACKED_N_TOKENS,
    TRAJECTORY_DEPTH,
    encode_battle,
    push_trajectory,
    stack_trajectory,
)
from src.doubles.evaluation.battle_inference_trace import encode_request_protocol_line
from src.doubles.planning.meta_database import MetaDatabase
from src.doubles.data.log_action_mask import pick_masked_argmax, pick_masked_log_actions
from src.doubles.players.preview_orchestrator import PreviewOrchestrator
from src.doubles.rl.rewards import BattleSnapshot, calc_step_reward
from src.core.teams.roster import shuffle_showdown_team
from src.doubles.teams.team_pool import load_opponent_team_builder


def load_team() -> str:
    return TEAM_PATH.read_text(encoding="utf-8")


def resolve_opponent_team(
    opponent_team: str | Teambuilder | None = None,
) -> str | Teambuilder:
    if opponent_team is not None:
        return opponent_team
    if USE_OPPONENT_TEAM_POOL:
        return load_opponent_team_builder(use_pool=True)
    return load_team()


class VGCRLEnv(DoublesEnv):
    """
  Doubles RL env: stacked BC token observations, canonical MultiDiscrete actions,
  live-log-bridge action masks, dense battle rewards.
    """

    def __init__(
        self,
        *,
        team: str | Teambuilder | None = None,
        opponent_team: str | Teambuilder | None = None,
        battle_format: str = BATTLE_FORMAT,
        server_configuration: ServerConfiguration | None = None,
        preview_device: str = "cpu",
        **kwargs: Any,
    ):
        if team is None:
            team = load_team()
        self._base_agent_team = team if isinstance(team, str) else None
        if server_configuration is None:
            server_configuration = LocalhostServerConfiguration
        super().__init__(
            team=team,
            battle_format=battle_format,
            server_configuration=server_configuration,
            accept_open_team_sheet=True,
            choose_on_teampreview=False,
            **kwargs,
        )
        self.agent2.update_team(resolve_opponent_team(opponent_team))

        obs_box = Box(
            low=-np.inf,
            high=np.inf,
            shape=(STACKED_N_TOKENS, N_FIELDS),
            dtype=np.float32,
        )
        self.action_spaces = {
            agent: MultiDiscrete([ACTION_SIZE, ACTION_SIZE])
            for agent in self.possible_agents
        }
        self.observation_spaces = {agent: obs_box for agent in self.possible_agents}

        self._protocol_logs: dict[str, list[str]] = {}
        self._trajectory_history: dict[str, deque[np.ndarray]] = {}
        self._last_trajectory_request_key: dict[str, tuple] = {}
        self._pending_snapshot: dict[str, np.ndarray] = {}
        self._log_views: dict[str, BattleLogState | None] = {}
        self._sample_kinds: dict[str, str] = {}
        self._snapshots: dict[str, BattleSnapshot | None] = {}
        self._meta_db = MetaDatabase(live_fetch=False)
        self._preview = PreviewOrchestrator(device=preview_device)
        self.step_count = 0
        self.episode_count = 0
        self._stall_turn = -1
        self._stall_steps = 0
        self.consecutive_rejections = 0
        self._last_step_turn = -1
        self._last_order_str = ""
        self._use_fallback_order: BattleOrder | None = None
        self._install_protocol_hook()
        self._install_preview_hook()

    def _install_protocol_hook(self) -> None:
        original = self.agent1._handle_battle_message

        async def _hooked(split_messages: list[list[str]]) -> None:
            await original(split_messages)
            try:
                battle = await self.agent1._get_battle(split_messages[0][0])
                tag = battle.battle_tag
                log = self._protocol_logs.setdefault(tag, [])
                for split_message in split_messages[1:]:
                    if not split_message:
                        continue
                    if len(split_message) >= 2 and split_message[1] in ("", "request"):
                        if split_message[1] == "":
                            log.append("|" + "|".join(split_message[1:]))
                        elif split_message[1] == "request":
                            if len(split_message) >= 3 and split_message[2]:
                                try:
                                    request = json.loads(split_message[2])
                                    log.append(encode_request_protocol_line(request))
                                except json.JSONDecodeError:
                                    log.append("|request|...")
                            else:
                                log.append("|request|...")
                    elif len(split_message) >= 2:
                        log.append("|" + "|".join(split_message[1:]))
            except Exception:
                pass

        self.agent1._handle_battle_message = _hooked  # type: ignore[method-assign]
        # PSClient captured the original handler at construction; re-bind after hook.
        self.agent1.ps_client._on_battle_message = self.agent1._handle_battle_message

    def _install_preview_hook(self) -> None:
        """Use PreviewOrchestrator for agent1 team preview (like TransformerPlayer)."""

        async def _agent1_teampreview(battle: DoubleBattle) -> str:
            return self._preview.teampreview(battle)

        self.agent1._teampreview = _agent1_teampreview  # type: ignore[method-assign]

    def teampreview(self, battle: DoubleBattle) -> str:
        """Public hook mirroring TransformerPlayer; used by PreviewOrchestrator wiring."""
        return self._preview.teampreview(battle)

    def reset_battles(self) -> None:
        super().reset_battles()
        self.step_count = 0
        self._stall_turn = -1
        self._stall_steps = 0
        self.consecutive_rejections = 0
        self._last_step_turn = -1
        self._last_order_str = ""
        self._use_fallback_order = None
        self._protocol_logs.clear()
        self._trajectory_history.clear()
        self._last_trajectory_request_key.clear()
        self._pending_snapshot.clear()
        self._log_views.clear()
        self._sample_kinds.clear()
        self._snapshots.clear()

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ):
        if SHUFFLE_TEAM_ORDER and self._base_agent_team is not None:
            rng = None
            if seed is not None:
                rng = random.Random(seed)
            elif getattr(self, "_np_random", None) is not None:
                rng = random.Random(int(self._np_random.integers(0, 2**31 - 1)))
            self.agent1.update_team(
                shuffle_showdown_team(self._base_agent_team, rng=rng)
            )
        return super().reset(seed=seed, options=options)

    def _history_for(self, battle: DoubleBattle) -> deque[np.ndarray]:
        tag = battle.battle_tag
        if tag not in self._trajectory_history:
            self._trajectory_history[tag] = deque(maxlen=TRAJECTORY_DEPTH)
        return self._trajectory_history[tag]

    def _current_snapshot(
        self, battle: DoubleBattle
    ) -> tuple[np.ndarray, BattleLogState | None, str]:
        tag = battle.battle_tag
        protocol = self._protocol_logs.get(tag, [])
        encoded = encode_live_as_log(
            battle,
            protocol_lines=protocol,
            side="p1",
            meta_db=self._meta_db,
        )
        if encoded is not None:
            snapshot, view, sample_kind = encoded
            self._log_views[tag] = view
            from src.doubles.battle.move_order import effective_force_switch_flags

            if any(effective_force_switch_flags(battle)):
                sample_kind = "force_switch"
            self._sample_kinds[tag] = sample_kind
            return snapshot, view, sample_kind

        try:
            snapshot = encode_battle(battle)
        except Exception:
            snapshot = np.zeros((N_TOKENS, N_FIELDS), dtype=np.int64)
        self._log_views[tag] = None
        self._sample_kinds[tag] = "turn"
        return snapshot, None, "turn"

    def _request_fingerprint(self, battle: DoubleBattle) -> tuple:
        """Uniquely identify a Showdown decision request (turn + phase + legal set)."""
        from src.doubles.battle.move_order import effective_force_switch_flags

        return (
            int(battle.turn),
            effective_force_switch_flags(battle),
            bool(battle.teampreview),
            tuple(str(o) for o in battle.valid_orders[0]),
            tuple(str(o) for o in battle.valid_orders[1]),
        )

    def _is_new_request(self, battle: DoubleBattle) -> bool:
        tag = battle.battle_tag
        fp = self._request_fingerprint(battle)
        return self._last_trajectory_request_key.get(tag) != fp

    def commit_trajectory(self, battle: DoubleBattle) -> None:
        tag = battle.battle_tag
        if not self._is_new_request(battle):
            return
        snapshot = self._pending_snapshot.get(tag)
        if snapshot is None:
            return
        dq = self._history_for(battle)
        history_list = list(dq)
        push_trajectory(
            history_list,
            snapshot,
            depth=TRAJECTORY_DEPTH,
            maxlen=TRAJECTORY_DEPTH,
        )
        dq.clear()
        dq.extend(history_list)
        self._last_trajectory_request_key[tag] = self._request_fingerprint(battle)

    def embed_battle(self, battle: DoubleBattle) -> npt.NDArray[np.float32]:
        try:
            snapshot, _, _ = self._current_snapshot(battle)
            tag = battle.battle_tag
            self._pending_snapshot[tag] = snapshot

            history = self._history_for(battle)
            history_list = list(history)
            stacked = stack_trajectory(history_list, snapshot, depth=TRAJECTORY_DEPTH)
            return stacked.astype(np.float32)
        except Exception:
            return np.zeros((STACKED_N_TOKENS, N_FIELDS), dtype=np.float32)

    def calc_reward(self, battle: DoubleBattle) -> float:
        tag = battle.battle_tag
        last = self._snapshots.get(tag)
        reward, snap = calc_step_reward(last, battle)
        self._snapshots[tag] = snap
        return reward

    def _sample_kind(self, battle: DoubleBattle) -> str:
        from src.doubles.battle.move_order import effective_force_switch_flags

        if any(effective_force_switch_flags(battle)):
            return "force_switch"
        return self._sample_kinds.get(battle.battle_tag, "turn")

    def _force_switch_slot_masks(
        self,
        battle: DoubleBattle,
        *,
        slot0_pred: int | None = None,
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        """Strict masks during forced-switch: switches only / pass only per slot."""
        from src.doubles.battle.move_order import (
            apply_joint_slot1_mask_numpy,
            canonical_force_switch_mask,
            constrain_mask_to_valid_orders,
            effective_force_switch_flags,
        )
        from src.doubles.data.live_log_bridge import live_force_switch_slot_masks

        flags = effective_force_switch_flags(battle)
        mask0 = canonical_force_switch_mask(battle, 0)
        mask1 = canonical_force_switch_mask(battle, 1)

        view = self._log_views.get(battle.battle_tag)
        if view is not None:
            log0, log1 = live_force_switch_slot_masks(battle, view, "p1")
            if log0.any():
                mask0 = mask0 & log0.astype(bool)
            if log1.any():
                mask1 = mask1 & log1.astype(bool)

        if not flags[0]:
            mask0 = np.zeros(ACTION_SIZE, dtype=bool)
            mask0[ACTION_PASS] = True
        if not flags[1]:
            mask1 = np.zeros(ACTION_SIZE, dtype=bool)
            mask1[ACTION_PASS] = True

        if slot0_pred is not None:
            mask1 = apply_joint_slot1_mask_numpy(
                mask1,
                a0_canonical=slot0_pred,
                force_switch=True,
            )

        if not mask0.any():
            mask0[ACTION_PASS] = True
        if not mask1.any():
            mask1[ACTION_PASS] = True
        mask0 = constrain_mask_to_valid_orders(battle, 0, mask0)
        mask1 = constrain_mask_to_valid_orders(battle, 1, mask1)
        return mask0, mask1

    def _pokeenv_canonical_masks(
        self,
        battle: DoubleBattle,
        *,
        slot0_pred: int | None = None,
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        from src.doubles.battle.move_order import (
            apply_joint_slot1_mask_numpy,
            canonical_force_switch_mask,
            effective_force_switch_flags,
            pokeenv_action_mask_to_canonical,
        )

        flags = effective_force_switch_flags(battle)
        if any(flags):
            mask0 = canonical_force_switch_mask(battle, 0)
            mask1 = canonical_force_switch_mask(battle, 1)
        else:
            pe0 = DoublesEnv.get_action_mask_individual(battle, 0)
            pe1 = DoublesEnv.get_action_mask_individual(battle, 1)
            mask0 = np.array(
                pokeenv_action_mask_to_canonical(battle, 0, pe0), dtype=bool
            )
            mask1 = np.array(
                pokeenv_action_mask_to_canonical(battle, 1, pe1), dtype=bool
            )
        if slot0_pred is not None:
            mask1 = apply_joint_slot1_mask_numpy(
                mask1,
                a0_canonical=slot0_pred,
                force_switch=any(flags),
            )
        return mask0, mask1

    def slot_action_masks(
        self,
        battle: DoubleBattle,
        *,
        slot0_pred: int | None = None,
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        """Per-slot canonical masks via live-log bridge (shape 2 x 107)."""
        from src.doubles.battle.move_order import effective_force_switch_flags

        if any(effective_force_switch_flags(battle)):
            return self._force_switch_slot_masks(battle, slot0_pred=slot0_pred)

        tag = battle.battle_tag
        view = self._log_views.get(tag)
        sample_kind = self._sample_kind(battle)
        mask0: npt.NDArray[np.bool_] | None = None
        mask1: npt.NDArray[np.bool_] | None = None

        if view is not None:
            mask0 = slot_mask_for_live(
                battle,
                view,
                side="p1",
                sample_kind=sample_kind,
                slot_suffix="a",
            )
            mask1 = slot_mask_for_live(
                battle,
                view,
                side="p1",
                sample_kind=sample_kind,
                slot_suffix="b",
                slot0_pred=slot0_pred,
            )
            if mask0 is not None and mask1 is not None:
                mask0 = mask0.astype(bool)
                mask1 = mask1.astype(bool)
            else:
                mask0 = mask1 = None

        if mask0 is None or mask1 is None:
            from src.doubles.battle.move_order import (
                apply_joint_slot1_mask_numpy,
                pokeenv_action_mask_to_canonical,
            )

            pe0 = DoublesEnv.get_action_mask_individual(battle, 0)
            pe1 = DoublesEnv.get_action_mask_individual(battle, 1)
            mask0 = np.array(
                pokeenv_action_mask_to_canonical(battle, 0, pe0), dtype=bool
            )
            mask1 = np.array(
                pokeenv_action_mask_to_canonical(battle, 1, pe1), dtype=bool
            )
            if slot0_pred is not None:
                mask1 = apply_joint_slot1_mask_numpy(
                    mask1,
                    a0_canonical=slot0_pred,
                    force_switch=False,
                )

        pe0, pe1 = self._pokeenv_canonical_masks(battle, slot0_pred=slot0_pred)
        mask0 = mask0 & pe0
        mask1 = mask1 & pe1
        if not mask0.any():
            mask0 = pe0.copy()
        if not mask1.any():
            mask1 = pe1.copy()
        from src.doubles.battle.move_order import constrain_mask_to_valid_orders

        mask0 = constrain_mask_to_valid_orders(battle, 0, mask0)
        mask1 = constrain_mask_to_valid_orders(battle, 1, mask1)
        return mask0, mask1

    def action_masks(
        self, battle: DoubleBattle | None = None
    ) -> npt.NDArray[np.bool_]:
        battle = battle or self.battle1
        assert battle is not None
        mask0, mask1 = self.slot_action_masks(battle)
        flat = np.concatenate([mask0, mask1])
        if not flat.any():
            flat[0] = True
        return flat

    def get_action_mask(self, battle: DoubleBattle) -> list[int]:
        """Flattened canonical mask for poke-env observation dict + SB3."""
        return [int(x) for x in self.action_masks(battle)]

    def _autocorrect_action(self, battle: DoubleBattle) -> tuple[int, int]:
        """Pick guaranteed-legal actions via live-log bridge masks (zero logits)."""
        import torch

        tag = battle.battle_tag
        view = self._log_views.get(tag)
        sample_kind = self._sample_kind(battle)
        logits0 = torch.zeros(ACTION_SIZE)
        logits1 = torch.zeros(ACTION_SIZE)

        if view is not None:
            return pick_masked_log_actions(
                logits0,
                logits1,
                view=view,
                side="p1",
                sample_kind=sample_kind,
            )

        from src.doubles.battle.move_order import (
            apply_joint_slot1_mask_numpy,
            effective_force_switch_flags,
        )

        mask0, mask1 = self.slot_action_masks(battle)
        a0 = pick_masked_argmax(logits0, mask0)
        mask1 = apply_joint_slot1_mask_numpy(
            mask1,
            a0_canonical=a0,
            force_switch=any(effective_force_switch_flags(battle)),
        )
        a1 = pick_masked_argmax(logits1, mask1)
        return a0, a1

    def _random_valid_order(self, battle: DoubleBattle) -> BattleOrder:
        """Sample a joint order guaranteed legal per battle.valid_orders."""
        import random

        legal0 = list(battle.valid_orders[0]) or [PassBattleOrder()]
        legal1 = list(battle.valid_orders[1]) or [PassBattleOrder()]
        for _ in range(32):
            o0 = random.choice(legal0)
            o1 = random.choice(legal1)
            joined = DoubleBattleOrder.join_orders([o0], [o1])
            if joined:
                return joined[0]
        return DefaultBattleOrder()

    def _sanitize_action(
        self, battle: DoubleBattle, action: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.int64]:
        """Ensure RL action is legal under live-log masks before submitting."""
        from src.doubles.battle.move_order import (
            apply_joint_slot1_mask_numpy,
            effective_force_switch_flags,
        )

        pair = np.asarray(action, dtype=np.int64).reshape(2)
        a0, a1 = int(pair[0]), int(pair[1])
        mask0, mask1 = self.slot_action_masks(battle)
        if 0 <= a0 < ACTION_SIZE and mask0[a0]:
            mask1_joint = apply_joint_slot1_mask_numpy(
                mask1,
                a0_canonical=a0,
                force_switch=any(effective_force_switch_flags(battle)),
            )
            if 0 <= a1 < ACTION_SIZE and mask1_joint[a1]:
                return np.array([a0, a1], dtype=np.int64)
        return np.array(self._autocorrect_action(battle), dtype=np.int64)

    def _recover_rejections(self) -> float:
        """Break [Unavailable choice] deadlocks with poke-env-legal orders."""
        penalty = 0.0
        timeout = min(5.0, float(self._challenge_timeout or 10.0))
        max_fix = 5
        for _ in range(max_fix):
            if not self.battle1 or self.battle1.finished:
                break
            if not self.agent1._trying_again.is_set():
                break
            self.agent1._trying_again.clear()
            penalty -= 1.0
            pair = self._autocorrect_action(self.battle1)
            order = self.action_to_order(
                np.array(pair, dtype=np.int64),
                self.battle1,
                fake=self._fake,
                strict=False,
            )
            self.agent1.order_queue.put(order)
            try:
                battle1 = self.agent1.battle_queue.get(timeout=timeout)
            except Exception:
                break
            self.battle1 = battle1
            self.agent1_to_move = True
        return penalty

    def _forfeit_if_stalled(self) -> bool:
        """Forfeit when the battle turn stops advancing to avoid infinite loops."""
        if not self.battle1 or self.battle1.finished:
            self._stall_steps = 0
            return False
        turn = int(self.battle1.turn)
        if turn == self._stall_turn:
            self._stall_steps += 1
        else:
            self._stall_turn = turn
            self._stall_steps = 0
        if self._stall_steps < 8:
            return False
        from poke_env.player.battle_order import ForfeitBattleOrder, _EmptyBattleOrder

        if self.agent1_to_move:
            self.agent1_to_move = False
            self.agent1.order_queue.put(ForfeitBattleOrder())
            if self.agent2_to_move:
                self.agent2_to_move = False
                self.agent2.order_queue.put(_EmptyBattleOrder())
        else:
            if self.agent2_to_move:
                self.agent2_to_move = False
                self.agent2.order_queue.put(ForfeitBattleOrder())
            else:
                self.agent1.order_queue.put(ForfeitBattleOrder())
        try:
            self.battle1 = self.agent1.battle_queue.get(
                timeout=min(5.0, float(self._challenge_timeout or 10.0))
            )
            self.battle2 = self.agent2.battle_queue.get(
                timeout=min(5.0, float(self._challenge_timeout or 10.0))
            )
        except Exception:
            self._stall_steps = 0
            return False
        self._stall_steps = 0
        return bool(self.battle1.finished)

    def _order_from_canonical_action(
        self,
        battle: DoubleBattle,
        pair: npt.NDArray[np.int64],
    ) -> BattleOrder:
        """Translate canonical slot indices to a joint order (no fallback)."""
        from src.doubles.battle.move_order import effective_force_switch_flags

        ca0, ca1 = int(pair[0]), int(pair[1])
        flags = effective_force_switch_flags(battle)
        if any(flags):
            if not flags[0]:
                ca0 = ACTION_PASS
            if not flags[1]:
                ca1 = ACTION_PASS

        order0 = self._canonical_slot_order(battle, 0, ca0, flags)
        order1 = self._canonical_slot_order(battle, 1, ca1, flags)
        joined = DoubleBattleOrder.join_orders([order0], [order1])
        if joined:
            return joined[0]
        return DefaultBattleOrder()

    def step(self, actions):
        self.step_count += 1
        agent0 = self.possible_agents[0]
        turn_str = int(self.battle1.turn) if self.battle1 else -1
        print(f"Turn: {turn_str} | Step: {self.step_count}")

        actions = dict(actions)
        turn_before = int(self.battle1.turn) if self.battle1 else -1
        fallback_penalty = 0.0
        self._use_fallback_order = None

        if (
            self.battle1 is not None
            and not self.battle1.teampreview
            and self.agent1_to_move
        ):
            raw = np.asarray(actions.get(agent0, [0, 0]), dtype=np.int64).reshape(2)
            actions[agent0] = self._sanitize_action(self.battle1, raw)
            pair = np.asarray(actions[agent0], dtype=np.int64).reshape(2)

            if self.consecutive_rejections >= 1:
                self._use_fallback_order = self._random_valid_order(self.battle1)
                fallback_penalty = -1.0
                print(
                    f"[FALLBACK] Stalled turn — substituting "
                    f"{self._use_fallback_order}"
                )
            else:
                order = self._order_from_canonical_action(self.battle1, pair)
                if not is_joint_order_valid(self.battle1, order):
                    self._use_fallback_order = self._random_valid_order(self.battle1)
                    fallback_penalty = -1.0
                    print(
                        f"[FALLBACK] Invalid RL order: {order} "
                        f"-> {self._use_fallback_order}"
                    )
                else:
                    self._last_order_str = str(order)

        obs, rewards, terms, truncs, infos = super().step(actions)

        if fallback_penalty != 0.0:
            rewards = dict(rewards)
            rewards[agent0] = float(rewards[agent0]) + fallback_penalty

        if self.battle1 is not None and not self.battle1.finished:
            turn_after = int(self.battle1.turn)
            stalled = (
                turn_after == turn_before
                and not self.battle1.teampreview
                and self._use_fallback_order is None
            )
            if stalled:
                self.consecutive_rejections += 1
            else:
                self.consecutive_rejections = 0
            self._last_step_turn = turn_after

        extra_penalty = self._recover_rejections()
        if extra_penalty != 0.0:
            rewards = dict(rewards)
            rewards[agent0] = float(rewards.get(agent0, 0.0)) + extra_penalty
            if self.battle1 is not None:
                obs = dict(obs)
                obs[agent0] = {
                    "observation": self.embed_battle(self.battle1),
                    "action_mask": np.array(self.get_action_mask(self.battle1)),
                }

        if terms.get(agent0) or truncs.get(agent0):
            self.episode_count += 1
            self.consecutive_rejections = 0

        if agent0 in infos and isinstance(infos[agent0], dict):
            infos[agent0]["episode_count"] = self.episode_count

        return obs, rewards, terms, truncs, infos

    def action_to_order(
        self,
        action: npt.NDArray[np.int64] | list[int],
        battle: DoubleBattle,
        fake: bool = False,
        strict: bool = True,
    ) -> BattleOrder:
        if self._use_fallback_order is not None:
            order = self._use_fallback_order
            self._use_fallback_order = None
            return order

        pair = np.asarray(action, dtype=np.int64).reshape(-1)
        if pair.shape[0] != 2:
            return DefaultBattleOrder()
        order = self._order_from_canonical_action(battle, pair)
        if isinstance(order, DefaultBattleOrder) and not fake:
            return self._random_valid_order(battle)
        return order

    @staticmethod
    def _canonical_slot_order(
        battle: DoubleBattle,
        pos: int,
        canonical_idx: int,
        force_flags: tuple[bool, bool],
    ) -> SingleBattleOrder:
        """Translate canonical index to a single-slot order; pass is never a move."""
        if any(force_flags) and not force_flags[pos]:
            return PassBattleOrder()
        if canonical_idx == ACTION_PASS:
            return PassBattleOrder()
        return canonical_index_to_single_order(battle, pos, canonical_idx)

    def get_additional_info(self) -> dict[str, dict[str, Any]]:
        info = super().get_additional_info()
        if self.battle1 is not None:
            tag = self.battle1.battle_tag
            agent = self.possible_agents[0]
            if agent in info:
                info[agent]["battle_won"] = bool(self.battle1.won)
                info[agent]["battle_lost"] = bool(self.battle1.lost)
                info[agent]["battle_turn"] = int(self.battle1.turn)
        return info
