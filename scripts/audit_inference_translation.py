#!/usr/bin/env python3
"""1-battle inference translation audit: masks, decodes, and desync report."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from poke_env.ps_client.account_configuration import AccountConfiguration

from config.settings import BATTLE_FORMAT, BC_EVAL_LOG_DIR, BC_MODEL_PATH, TEAM_PATH
from src.doubles.battle.inference_audit import (
    format_legal_mask_lines,
    legal_mask_debug_report,
    translation_audit_for_decision,
)
from src.doubles.battle.canonical_inference import (
    pick_masked_canonical_indices,
    submission_debug,
)
from poke_env.environment.doubles_env import DoublesEnv
from src.doubles.players.max_damage_player import MaxDamagePlayer
from src.doubles.players.transformer_player import TransformerPlayer


class AuditTransformerPlayer(TransformerPlayer):
    """TransformerPlayer that captures turn-1 mask audit on first real decision."""

    def __init__(self, *args, audit_out: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit_out = audit_out
        self._audit_captured = False
        self.audit_payload: dict = {}

    def choose_move(self, battle):
        if (
            not self._audit_captured
            and int(battle.turn) >= 1
            and not (battle.wait and not any(battle.force_switch))
        ):
            x = self._stacked_input(battle)
            with torch.no_grad():
                logits0, logits1 = self.model(x)
            row0 = logits0[0]
            row1 = logits1[0]
            raw0 = int(row0.argmax().item())
            raw1 = int(row1.argmax().item())
            ca0, ca1 = pick_masked_canonical_indices(battle, row0, row1)
            sub0 = submission_debug(battle, 0, ca0)
            sub1 = submission_debug(battle, 1, ca1)

            slot_reports = [
                legal_mask_debug_report(battle, 0),
                legal_mask_debug_report(battle, 1),
            ]
            decision_audit = {
                "turn": int(battle.turn),
                "slots": [
                    {
                        "pos": 0,
                        "raw_argmax": raw0,
                        "canonical_picked": ca0,
                        "submission": sub0,
                        "fallback": raw0 != ca0,
                    },
                    {
                        "pos": 1,
                        "raw_argmax": raw1,
                        "canonical_picked": ca1,
                        "submission": sub1,
                        "fallback": raw1 != ca1,
                    },
                ],
            }

            text_lines = [
                f"=== Inference Translation Audit | turn {battle.turn} ===",
                "",
                "--- Translation trace ---",
                json.dumps(decision_audit, indent=2),
                "",
            ]
            for rep in slot_reports:
                text_lines.extend(format_legal_mask_lines(rep))
                text_lines.append("")

            # Check specific indices from prior trace failure
            for rep in slot_reports:
                pe_mask = list(DoublesEnv.get_action_mask_individual(battle, rep["pos"]))
                ca_mask = pokeenv_action_mask_to_canonical(battle, rep["pos"], pe_mask)
                for idx in (19, 21, 24):
                    text_lines.append(
                        f"  CHECK idx {idx} slot {rep['pos']}: "
                        f"legal={ca_mask[idx] if idx < len(ca_mask) else False} "
                        f"(canonical moves={rep['canonical_moves']})"
                    )

            text = "\n".join(text_lines)
            self.audit_payload = {
                "turn": int(battle.turn),
                "battle_tag": battle.battle_tag,
                "decision": decision_audit,
                "slot_masks": slot_reports,
                "text": text,
            }
            self.audit_out.parent.mkdir(parents=True, exist_ok=True)
            self.audit_out.write_text(text, encoding="utf-8")
            (self.audit_out.with_suffix(".json")).write_text(
                json.dumps(self.audit_payload, indent=2), encoding="utf-8"
            )
            print(text, flush=True)
            self._audit_captured = True

        return super().choose_move(battle)


async def _run_audit(*, model: Path, device: str, out_dir: Path) -> dict:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    audit_txt = out_dir / f"inference_translation_audit_{stamp}.txt"

    agent = AuditTransformerPlayer(
        model_path=model,
        battle_format=BATTLE_FORMAT,
        team=TEAM_PATH.read_text(encoding="utf-8"),
        device=device,
        trace_inference=True,
        trace_top_k=5,
        capture_battle_log=True,
        max_concurrent_battles=1,
        audit_out=audit_txt,
        account_configuration=AccountConfiguration.generate("AuditXform", rand=True),
    )
    opponent = MaxDamagePlayer(
        battle_format=BATTLE_FORMAT,
        team=TEAM_PATH.read_text(encoding="utf-8"),
        max_concurrent_battles=1,
        account_configuration=AccountConfiguration.generate("AuditMaxD", rand=True),
    )
    await agent.battle_against(opponent, n_battles=1)
    battle = next(iter(agent.battles.values()))
    trace = agent.drain_inference_trace(battle.battle_tag)
    return {
        "won": bool(battle.won),
        "turn": battle.turn,
        "audit": agent.audit_payload,
        "audit_path": str(audit_txt),
        "trace_decisions": len(trace.get("decisions", [])),
        "fallbacks": sum(1 for d in trace.get("decisions", []) if d.get("any_fallback")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="1-battle inference translation audit")
    parser.add_argument("--model", type=Path, default=BC_MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=BC_EVAL_LOG_DIR / "inference_audit",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = asyncio.run(_run_audit(model=args.model, device=args.device, out_dir=args.out_dir))
    print("\n=== Summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
