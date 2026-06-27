"""Build SinglesTransformer or SinglesRLReplay eval agents with shared team setup."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.teambuilder import Teambuilder
from sb3_contrib import MaskablePPO

from config.settings import SINGLES_BATTLE_FORMAT, SINGLES_BC_MODEL_PATH
from src.doubles.teams.team_pool import opponent_pool_description
from src.singles.preview_orchestrator import SinglesPreviewOrchestrator
from src.singles.rl.rl_replay_player import SinglesRLReplayPlayer
from src.singles.teams.team_pool import (
    load_agent_team,
    load_meta_team_pool,
    load_opponent_team_builder,
)
from src.singles.transformer_player import SinglesTransformerPlayer

AgentTeam = Union[str, Teambuilder]


def load_eval_agent_team(*, use_meta_pool: bool = False) -> AgentTeam:
    if use_meta_pool:
        return load_meta_team_pool(use_curriculum=False)
    return load_agent_team()


def describe_agent_team(team: AgentTeam) -> dict:
    return opponent_pool_description(team)


def build_singles_eval_agent(
    *,
    rl_checkpoint: Path | None = None,
    bc_model_path: Path = SINGLES_BC_MODEL_PATH,
    preview_model_path: Path,
    device: str,
    team: AgentTeam,
    trace_inference: bool = False,
    trace_top_k: int = 5,
    capture_battle_log: bool = False,
    log_illegal_top1: bool = False,
    save_replays: str | bool = False,
    account_name: str = "SinglesEval",
    deterministic: bool = True,
):
    preview = SinglesPreviewOrchestrator(model_path=preview_model_path, device=device)
    account = AccountConfiguration.generate(account_name, rand=True)
    common = dict(
        battle_format=SINGLES_BATTLE_FORMAT,
        team=team,
        device=device,
        preview=preview,
        trace_inference=trace_inference,
        trace_top_k=trace_top_k,
        capture_battle_log=capture_battle_log,
        log_illegal_top1=log_illegal_top1,
        max_concurrent_battles=1,
        save_replays=save_replays,
        account_configuration=account,
    )
    if rl_checkpoint is not None:
        model = MaskablePPO.load(str(rl_checkpoint), device=device)
        return SinglesRLReplayPlayer(
            model,
            deterministic=deterministic,
            **common,
        )
    return SinglesTransformerPlayer(
        model_path=bc_model_path,
        **common,
    )


def build_opponent_team(*, mirror: bool = False):
    return load_opponent_team_builder(use_pool=not mirror)
