"""RL env encode/decode alignment vs BC live bridge and TransformerPlayer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from config.settings import BC_MODEL_PATH
from poke_env.environment.doubles_env import DoublesEnv

from src.doubles.battle.canonical_inference import (
    canonical_indices_to_battle_order,
    decode_canonical_tuple,
    submission_debug,
)
from src.doubles.battle.move_order import (
    apply_joint_slot1_mask_numpy,
    pokeenv_action_mask_to_canonical,
    remap_canonical_action_to_pokeenv,
)
from src.doubles.data.action_codec import format_log_action_pair
from src.doubles.data.action_space_spec import ACTION_SIZE
from src.doubles.battle.move_order import apply_joint_slot1_mask_numpy
from src.doubles.data.live_log_bridge import encode_live_as_log, pick_masked_live_log_actions
from src.doubles.data.log_action_mask import pick_masked_log_actions
from src.core.data.state_tokenizer import (
    TRAJECTORY_DEPTH,
    push_trajectory,
    stack_trajectory,
    trajectory_frame_fingerprints,
)
from src.doubles.evaluation.live_bc_alignment import (
    _FakeBattle,
    _find_parser_sample,
    _simulate_live_trajectory,
    load_trace_battle,
)
from src.core.model.transformer_bot import load_model
from src.doubles.planning.meta_database import MetaDatabase
from src.doubles.data.replay_parser import parse_replay_log


@dataclass
class RLEnvAlignmentReport:
    """Summary of RL observation/action codec alignment checks."""

    n_decisions: int = 0
    tensor_matches: int = 0
    mask_matches_player: int = 0
    mask_subset_pokeenv: int = 0
    decode_mask_ok: int = 0
    decode_pe_ok: int = 0
    pred_matches_bc: int = 0
    flat_mask_ok: int = 0
    sanitize_unchanged: int = 0
    joint_legal_policy: int = 0
    lines: list[str] = field(default_factory=list)
    json_path: Path | None = None
    txt_path: Path | None = None

    @property
    def tensor_match_rate(self) -> float:
        return self.tensor_matches / max(1, self.n_decisions)

    @property
    def pred_match_rate(self) -> float:
        return self.pred_matches_bc / max(1, self.n_decisions)


def _reference_embed_from_vgc_state(
    vgc_env,
    battle,
) -> tuple[np.ndarray, object | None, str]:
    """Mirror TransformerPlayer._stacked_input using VGCRLEnv internal state."""
    tag = battle.battle_tag
    protocol = vgc_env._protocol_logs.get(tag, [])
    encoded = encode_live_as_log(
        battle,
        protocol_lines=protocol,
        side="p1",
        meta_db=vgc_env._meta_db,
    )
    if encoded is not None:
        snapshot, view, sample_kind = encoded
        vgc_env._log_views[tag] = view
        vgc_env._sample_kinds[tag] = sample_kind
    else:
        from src.core.data.state_tokenizer import encode_battle

        snapshot = encode_battle(battle)
        view = None
        sample_kind = "turn"
        vgc_env._log_views[tag] = None
        vgc_env._sample_kinds[tag] = sample_kind

    vgc_env._pending_snapshot[tag] = snapshot
    history = list(vgc_env._history_for(battle))
    stacked = stack_trajectory(history, snapshot, depth=TRAJECTORY_DEPTH)
    return stacked.astype(np.float32), view, sample_kind


def _player_style_masks(
    vgc_env,
    battle,
    *,
    slot0_pred: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """TransformerPlayer._canonical_mask without poke-env intersection."""
    from src.doubles.data.live_log_bridge import slot_mask_for_live

    tag = battle.battle_tag
    view = vgc_env._log_views.get(tag)
    sample_kind = vgc_env._sample_kind(battle)

    if view is not None:
        mask0 = slot_mask_for_live(
            battle, view, side="p1", sample_kind=sample_kind, slot_suffix="a"
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
            return mask0.astype(bool), mask1.astype(bool)

    from src.doubles.battle.move_order import canonical_force_switch_mask

    if any(battle.force_switch):
        return (
            canonical_force_switch_mask(battle, 0),
            canonical_force_switch_mask(battle, 1),
        )
    pe0 = DoublesEnv.get_action_mask_individual(battle, 0)
    pe1 = DoublesEnv.get_action_mask_individual(battle, 1)
    return (
        np.array(pokeenv_action_mask_to_canonical(battle, 0, pe0), dtype=bool),
        np.array(pokeenv_action_mask_to_canonical(battle, 1, pe1), dtype=bool),
    )


def _pokeenv_legal_canonical(battle, pos: int, canonical_idx: int) -> bool:
    pe = remap_canonical_action_to_pokeenv(canonical_idx, battle, pos)
    pe_mask = DoublesEnv.get_action_mask_individual(battle, pos)
    if 0 <= pe < len(pe_mask):
        return bool(pe_mask[pe])
    return False


def _decode_audit(vgc_env, battle, a0: int, a1: int) -> dict:
    mask0, mask1 = vgc_env.slot_action_masks(battle)
    mask1_joint = apply_joint_slot1_mask_numpy(
        mask1,
        a0_canonical=a0,
        force_switch=any(battle.force_switch),
    )
    in_env_mask = bool(mask0[a0] and mask1_joint[a1])
    pe_ok = _pokeenv_legal_canonical(battle, 0, a0) and _pokeenv_legal_canonical(
        battle, 1, a1
    )
    order = canonical_indices_to_battle_order(battle, a0, a1)
    return {
        "a0": a0,
        "a1": a1,
        "in_env_mask": in_env_mask,
        "pokeenv_legal": pe_ok,
        "order": str(order),
        "slot0_debug": submission_debug(battle, 0, a0),
        "slot1_debug": submission_debug(battle, 1, a1),
    }


def audit_rl_env_on_battle(vgc_env, battle) -> dict:
    """Run alignment checks at the current battle state (pre-step)."""
    env_obs = vgc_env.embed_battle(battle)
    ref_obs, view, sample_kind = _reference_embed_from_vgc_state(vgc_env, battle)
    tensor_match = bool(np.allclose(env_obs, ref_obs, atol=0, rtol=0))

    env_mask0, env_mask1 = vgc_env.slot_action_masks(battle)
    player_mask0, player_mask1 = _player_style_masks(vgc_env, battle)
    mask_match_player = bool(
        np.array_equal(env_mask0, player_mask0 & vgc_env._pokeenv_canonical_masks(battle)[0])
        and np.array_equal(
            env_mask1,
            player_mask1 & vgc_env._pokeenv_canonical_masks(battle)[1],
        )
    )
    pe0, pe1 = vgc_env._pokeenv_canonical_masks(battle)
    mask_subset_pe = bool(
        np.all(~env_mask0 | pe0) and np.all(~env_mask1 | pe1)
    )

    flat = vgc_env.action_masks(battle)
    flat_ok = bool(
        flat.shape[0] == ACTION_SIZE * 2
        and np.array_equal(flat[:ACTION_SIZE], env_mask0)
        and np.array_equal(flat[ACTION_SIZE:], env_mask1)
    )

    pred = {}
    if view is not None:
        import torch as th

        model = getattr(audit_rl_env_on_battle, "_model", None)
        device = getattr(audit_rl_env_on_battle, "_device", "cpu")
        if model is None:
            model = load_model(BC_MODEL_PATH, device=device)
            audit_rl_env_on_battle._model = model
            audit_rl_env_on_battle._device = device
        x = th.as_tensor(ref_obs, dtype=th.long).unsqueeze(0).to(device)
        with th.no_grad():
            l0, l1 = model(x)
        bc_pred = pick_masked_log_actions(
            l0[0], l1[0], view=view, side="p1", sample_kind=sample_kind
        )
        live_pred = pick_masked_live_log_actions(
            l0[0], l1[0], battle=battle, view=view, side="p1", sample_kind=sample_kind
        )
        pred = {
            "bc_pred": bc_pred,
            "live_pred": live_pred,
            "pred_match": bc_pred == live_pred,
            "bc_label": format_log_action_pair(view, "p1", *bc_pred),
            "live_label": format_log_action_pair(view, "p1", *live_pred),
        }
        a0, a1 = live_pred
    else:
        a0 = int(np.where(env_mask0)[0][0]) if env_mask0.any() else 0
        a1 = int(np.where(env_mask1)[0][0]) if env_mask1.any() else 0

    raw = np.array([a0, a1], dtype=np.int64)
    sanitized = vgc_env._sanitize_action(battle, raw)
    decode_raw = _decode_audit(vgc_env, battle, int(raw[0]), int(raw[1]))
    decode_san = _decode_audit(
        vgc_env, battle, int(sanitized[0]), int(sanitized[1])
    )
    autocorrect = vgc_env._autocorrect_action(battle)
    decode_auto = _decode_audit(vgc_env, battle, autocorrect[0], autocorrect[1])

    return {
        "turn": int(battle.turn),
        "sample_kind": sample_kind,
        "tensor_match": tensor_match,
        "env_fps": trajectory_frame_fingerprints(env_obs.astype(np.int64)),
        "ref_fps": trajectory_frame_fingerprints(ref_obs.astype(np.int64)),
        "mask_match_player_pe": mask_match_player,
        "mask_subset_pokeenv": mask_subset_pe,
        "flat_mask_ok": flat_ok,
        "pred": pred,
        "sanitize_unchanged": bool(np.array_equal(raw, sanitized)),
        "decode_raw": decode_raw,
        "decode_sanitized": decode_san,
        "decode_autocorrect": decode_auto,
    }


def run_live_rl_alignment(
    *,
    n_steps: int = 40,
    model_path: Path = BC_MODEL_PATH,
    device: str = "cpu",
    use_policy: bool = True,
) -> RLEnvAlignmentReport:
    """Play steps through VGCRLEnv and audit encode/decode + policy sampling."""
    from scripts.train_rl import make_env
    from src.doubles.rl.gym_wrappers import find_vgc_env
    from src.doubles.rl.custom_policy import VGCBehaviorMaskablePolicy, is_joint_legal

    def mask_env(e):
        cur = e
        while cur is not None:
            if hasattr(cur, "action_masks"):
                return cur
            cur = getattr(cur, "env", None)

    audit_rl_env_on_battle._model = load_model(model_path, device=device)
    audit_rl_env_on_battle._device = device

    policy = None
    if use_policy:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        vec = DummyVecEnv([lambda: make_env(log_level=50)])
        probe = MaskablePPO(
            VGCBehaviorMaskablePolicy,
            vec,
            n_steps=64,
            device=device,
            policy_kwargs=dict(
                bc_model_path=str(model_path),
                net_arch=dict(pi=[], vf=[64, 64]),
            ),
        )
        from src.doubles.rl.custom_policy import init_bc_actor_weights

        init_bc_actor_weights(probe.policy, model_path)
        policy = probe.policy
        vec.close()

    report = RLEnvAlignmentReport()
    env = make_env(log_level=50)
    me = mask_env(env)
    vgc = find_vgc_env(env)
    assert vgc is not None

    obs, _ = env.reset()
    lines = ["RL env live alignment audit", f"steps={n_steps}", ""]

    for step_i in range(n_steps):
        battle = vgc.battle1
        if battle is None or battle.finished or battle.teampreview:
            break
        audit = audit_rl_env_on_battle(vgc, battle)
        report.n_decisions += 1
        masks = me.action_masks()
        mask0 = masks[:ACTION_SIZE]
        mask1_base = masks[ACTION_SIZE:]
        force_sw = not mask0[7:].any()

        policy_action = None
        joint_ok = False
        if policy is not None:
            policy_action, _ = policy.predict(
                obs, deterministic=True, action_masks=masks
            )
            policy_action = np.asarray(policy_action, dtype=np.int64).reshape(2)
            joint_ok = is_joint_legal(
                mask0,
                mask1_base,
                int(policy_action[0]),
                int(policy_action[1]),
                force_switch=force_sw,
            )
            if joint_ok:
                report.joint_legal_policy += 1

        if audit["tensor_match"]:
            report.tensor_matches += 1
        if audit["mask_match_player_pe"]:
            report.mask_matches_player += 1
        if audit["mask_subset_pokeenv"]:
            report.mask_subset_pokeenv += 1
        if audit["flat_mask_ok"]:
            report.flat_mask_ok += 1
        if policy_action is not None:
            san = vgc._sanitize_action(battle, policy_action)
            if np.array_equal(policy_action, san):
                report.sanitize_unchanged += 1
            decode = _decode_audit(
                vgc, battle, int(policy_action[0]), int(policy_action[1])
            )
        else:
            decode = audit["decode_sanitized"]
        if decode["in_env_mask"]:
            report.decode_mask_ok += 1
        if decode["pokeenv_legal"]:
            report.decode_pe_ok += 1
        pred = audit.get("pred") or {}
        if pred.get("pred_match"):
            report.pred_matches_bc += 1

        lines.append(
            f"--- step {step_i + 1} turn {audit['turn']} kind={audit['sample_kind']} ---"
        )
        lines.append(f"tensor match: {audit['tensor_match']}")
        lines.append(f"mask == player AND pe: {audit['mask_match_player_pe']}")
        lines.append(f"flat mask 214: {audit['flat_mask_ok']}")
        if policy_action is not None:
            lines.append(
                f"policy joint-legal: {joint_ok} action={policy_action.tolist()}"
            )
            lines.append(
                f"sanitize unchanged: {np.array_equal(policy_action, vgc._sanitize_action(battle, policy_action))}"
            )
            lines.append(
                f"decode policy: mask={decode['in_env_mask']} pe={decode['pokeenv_legal']} "
                f"order={decode['order']}"
            )
        lines.append("")

        if policy is not None:
            action = policy_action
        else:
            legal0 = np.where(mask0)[0]
            legal1 = np.where(mask1_base)[0]
            action = np.array(
                [int(np.random.choice(legal0)), int(np.random.choice(legal1))],
                dtype=np.int64,
            )
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()

    lines.extend(
        [
            "=== SUMMARY ===",
            f"Decisions audited: {report.n_decisions}",
            f"Tensor matches: {report.tensor_matches}/{report.n_decisions}",
            f"Mask matches (player AND pe): {report.mask_matches_player}/{report.n_decisions}",
            f"Mask subset poke-env: {report.mask_subset_pokeenv}/{report.n_decisions}",
            f"Flat 214-d mask ok: {report.flat_mask_ok}/{report.n_decisions}",
            f"BC/live pred match: {report.pred_matches_bc}/{report.n_decisions}",
            f"Policy joint-legal: {report.joint_legal_policy}/{report.n_decisions}",
            f"Sanitize unchanged (policy): {report.sanitize_unchanged}/{report.n_decisions}",
            f"Sanitized decode in-mask: {report.decode_mask_ok}/{report.n_decisions}",
            f"Sanitized decode poke-env legal: {report.decode_pe_ok}/{report.n_decisions}",
        ]
    )
    report.lines = lines
    env.close()
    return report


def run_trace_rl_alignment(
    trace_json: Path,
    *,
    model_path: Path = BC_MODEL_PATH,
    device: str = "cpu",
) -> RLEnvAlignmentReport:
    """Replay inference trace protocol; audit encode path vs parser BC samples."""
    battle = load_trace_battle(trace_json)
    protocol = battle.get("protocol_log") or []
    tag = battle["battle_tag"]
    decisions = battle.get("decisions") or []
    meta_db = MetaDatabase(live_fetch=False)

    samples = parse_replay_log(
        "\n".join(protocol),
        replay_id=tag,
        skip_rating=True,
        keep_view_state=True,
    )
    p1_samples = [s for s in samples if s.side == "p1"]
    sim = _simulate_live_trajectory(protocol, decisions, tag=tag, meta_db=meta_db)
    used_parser: set[int] = set()
    model = load_model(model_path, device=device)

    report = RLEnvAlignmentReport()
    lines = [
        f"RL env trace alignment — {tag}",
        f"Trace: {trace_json}",
        "",
    ]

    for di, dec in enumerate(decisions):
        if dec.get("kind") != "inference":
            continue
        report.n_decisions += 1
        turn = int(dec["turn"])
        fs = list(dec.get("force_switch") or [False, False])
        if di not in sim:
            lines.append(f"decision {di + 1} turn {turn}: encode_live_as_log FAILED")
            continue

        stacked, view, sample_kind = sim[di]
        fake = _FakeBattle(tag=tag, turn=turn, force_switch=fs)

        parser_sample = _find_parser_sample(
            p1_samples,
            turn=turn,
            sample_kind=sample_kind,
            side="p1",
            view=view,
            live_fs=fs,
            used=used_parser,
        )
        tensor_match = bool(
            parser_sample is not None
            and np.array_equal(stacked, parser_sample.tokens)
        )
        if tensor_match:
            report.tensor_matches += 1

        x = torch.as_tensor(stacked, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            l0, l1 = model(x)
        live_pred = pick_masked_live_log_actions(
            l0[0], l1[0], battle=fake, view=view, side="p1", sample_kind=sample_kind
        )
        if parser_sample is not None and parser_sample.view_state is not None:
            bc_pred = pick_masked_log_actions(
                l0[0],
                l1[0],
                view=parser_sample.view_state,
                side="p1",
                sample_kind=parser_sample.sample_kind,
            )
            pred_match = live_pred == bc_pred
        else:
            bc_pred = (-1, -1)
            pred_match = False
        if pred_match:
            report.pred_matches_bc += 1

        decode_ok = False
        pe_ok = False
        order_str = "n/a (fake battle)"
        if isinstance(fake, object) and hasattr(fake, "team"):
            pe0 = DoublesEnv.get_action_mask_individual(fake, 0)
            pe1 = DoublesEnv.get_action_mask_individual(fake, 1)
            can0 = np.array(pokeenv_action_mask_to_canonical(fake, 0, pe0), dtype=bool)
            can1 = np.array(pokeenv_action_mask_to_canonical(fake, 1, pe1), dtype=bool)
            mask1j = apply_joint_slot1_mask_numpy(
                can1, a0_canonical=live_pred[0], force_switch=any(fake.force_switch)
            )
            decode_ok = bool(can0[live_pred[0]] and mask1j[live_pred[1]])
            pe_ok = _pokeenv_legal_canonical(fake, 0, live_pred[0]) and _pokeenv_legal_canonical(
                fake, 1, live_pred[1]
            )
            order_str = str(canonical_indices_to_battle_order(fake, *live_pred))
        if decode_ok:
            report.decode_mask_ok += 1
        if pe_ok:
            report.decode_pe_ok += 1

        lines.append(
            f"--- decision {di + 1} turn {turn} fs={fs} kind={sample_kind} ---"
        )
        lines.append(f"tensor match parser: {tensor_match}")
        lines.append(
            f"pred bc={format_log_action_pair(parser_sample.view_state, 'p1', *bc_pred) if parser_sample and parser_sample.view_state else 'n/a'}"
        )
        lines.append(f"pred live={format_log_action_pair(view, 'p1', *live_pred)}")
        lines.append(f"pred match: {pred_match}")
        lines.append(
            f"decode live pred: canonical_mask={decode_ok} pokeenv={pe_ok} order={order_str}"
        )
        lines.append("")

    lines.extend(
        [
            "=== SUMMARY ===",
            f"Inference decisions: {report.n_decisions}",
            f"Tensor matches: {report.tensor_matches}/{report.n_decisions}",
            f"Pred matches BC: {report.pred_matches_bc}/{report.n_decisions}",
            f"Decode in-mask: {report.decode_mask_ok}/{report.n_decisions}",
            f"Decode poke-env legal: {report.decode_pe_ok}/{report.n_decisions}",
        ]
    )
    report.lines = lines
    return report


def save_report(report: RLEnvAlignmentReport, out_dir: Path, stem: str) -> RLEnvAlignmentReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"{stem}.txt"
    json_path = out_dir / f"{stem}.json"
    txt_path.write_text("\n".join(report.lines), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "n_decisions": report.n_decisions,
                "tensor_match_rate": report.tensor_match_rate,
                "pred_match_rate": report.pred_match_rate,
                "tensor_matches": report.tensor_matches,
                "pred_matches_bc": report.pred_matches_bc,
                "mask_matches_player": report.mask_matches_player,
                "mask_subset_pokeenv": report.mask_subset_pokeenv,
                "flat_mask_ok": report.flat_mask_ok,
                "decode_mask_ok": report.decode_mask_ok,
                "decode_pe_ok": report.decode_pe_ok,
                "sanitize_unchanged": report.sanitize_unchanged,
                "joint_legal_policy": report.joint_legal_policy,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report.txt_path = txt_path
    report.json_path = json_path
    return report
