"""Project-wide configuration for Champions VGC RL training."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    """Load secrets from repo-root .env (gitignored — see .env.example)."""
    env_file = ROOT_DIR / ".env"
    if not env_file.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file)
    except ImportError:
        pass


_load_dotenv()

TEAM_PATH = ROOT_DIR / "teams" / "reg_ma_team.txt"
SINGLES_TEAM_PATH = ROOT_DIR / "teams" / "singles_bss_team.txt"
OPPONENT_POOL_DIR = ROOT_DIR / "teams" / "opponents"
USE_OPPONENT_TEAM_POOL = True
# Gradually expand opponent pool during training (file-backed; safe for SubprocVecEnv).
OPPONENT_POOL_CURRICULUM = True
OPPONENT_POOL_START_TEAMS = 10
OPPONENT_POOL_GROW_SIZE = 10
OPPONENT_POOL_GROW_EVERY_STEPS = 100_000
OPPONENT_POOL_MAX_TEAMS = 50
MODELS_DIR = ROOT_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
LOGS_DIR = ROOT_DIR / "logs" / "tensorboard"
REPLAYS_DIR = ROOT_DIR / "logs" / "replays"

# BC Transformer pipeline (active)
DATA_DIR = ROOT_DIR / "data"
RAW_LOGS_DIR = DATA_DIR / "raw_logs"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
BC_MODEL_PATH = MODELS_DIR / "bc_transformer_latest.pt"
BC_TRAINING_LOG_DIR = ROOT_DIR / "logs" / "bc_training"
BC_EVAL_LOG_DIR = ROOT_DIR / "logs" / "eval"
BC_DATASET_PATH = PROCESSED_DATA_DIR / "bc_dataset.pt"
PREVIEW_DATASET_PATH = PROCESSED_DATA_DIR / "preview_dataset.pt"
PREVIEW_MODEL_PATH = MODELS_DIR / "preview_model.pt"

# Singles BC pipeline
META_DATA_DIR = DATA_DIR / "meta"
SINGLES_META_DATABASE_PATH = META_DATA_DIR / "singles_meta_database.json"
SINGLES_RAW_LOGS_DIR = DATA_DIR / "raw_logs_singles"
SINGLES_BC_DATASET_PATH = PROCESSED_DATA_DIR / "singles_bc_dataset.pt"
SINGLES_PREVIEW_DATASET_PATH = PROCESSED_DATA_DIR / "singles_preview_dataset.pt"
SINGLES_BC_MODEL_PATH = MODELS_DIR / "singles_bc_transformer_latest.pt"
SINGLES_PREVIEW_MODEL_PATH = MODELS_DIR / "singles_preview_model.pt"
SINGLES_OPPONENT_POOL_DIR = ROOT_DIR / "teams" / "singles_opponents"
USE_SINGLES_OPPONENT_TEAM_POOL = True

BATTLE_FORMAT = "gen9championsvgc2026regma"
SINGLES_BATTLE_FORMAT = "gen9championsbssregmb"
SINGLES_PIKALYTICS_FORMAT = "gen9championsbssregmb"
MAX_COMBOS = 2048

# Shuffle roster block order each battle so all 6 mons can be brought (Showdown
# uses paste slots 1-2 as temporary preview actives only — not fixed OTS leads).
SHUFFLE_TEAM_ORDER = True

# Team preview: policy chooses bring-4 + leads when True (requires observation v2+).
LEARN_TEAM_PREVIEW = True
# Used only when LEARN_TEAM_PREVIEW is False (scripted fallback).
DEFAULT_TEAM_PREVIEW = "/team 3456"

# Reward when our side completes a mega evolution (active mega form count increases).
MEGA_EVOLUTION_REWARD = 0.35

# --- Legacy RL / ISMCTS (archived — see archive/README.md) ---
# Training defaults
N_ENV = 16
PPO_DEVICE = "auto"  # auto | cuda | cpu
PPO_LEARNING_RATE = 3e-4
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 256
PPO_GAMMA = 0.99
PPO_ENT_COEF = 0.01

STAGE1_TIMESTEPS = 500_000
STAGE2_TIMESTEPS = 1_000_000
STAGE3_TIMESTEPS = 2_000_000

# Periodic saves during model.learn (timesteps; 0 disables)
CHECKPOINT_SAVE_FREQ = 50_000

EVAL_N_BATTLES = 50

# Meta / planning (generative hybrid bot)
META_DIR = ROOT_DIR / "teams" / "meta"
PIKALYTICS_META_PATH = META_DIR / "pikalytics_reg_mb.json"
DEX_CACHE_PATH = META_DIR / "dex_reg_mb.json"
SHOWDOWN_DATA_DIR = Path(
    os.environ.get(
        "SHOWDOWN_DATA_DIR",
        str(ROOT_DIR.parent / "pokemon-showdown" / "data"),
    )
)

# DeepSeek macro strategist (API key via .env; empty = heuristic fallback)
# NEVER commit .env — it is gitignored. Copy .env.example to .env locally.
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
DEEPSEEK_REASONING_EFFORT = os.environ.get("DEEPSEEK_REASONING_EFFORT", "max")
DEEPSEEK_THINKING_ENABLED = os.environ.get("DEEPSEEK_THINKING_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
MACRO_STRATEGIST_ENABLED = True
MACRO_STRATEGIST_FALLBACK = "heuristic"

# ISMCTS tactician
ISMCTS_DETERMINIZATIONS = 8
ISMCTS_TIME_BUDGET_MS = 8000
ISMCTS_USE_FAST_DAMAGE = True
ISMCTS_RL_VALUE_MODEL = os.environ.get("ISMCTS_RL_VALUE_MODEL", "")
ISMCTS_RL_VALUE_WEIGHT = float(os.environ.get("ISMCTS_RL_VALUE_WEIGHT", "0.2"))
ISMCTS_VALUE_MLP_PATH = os.environ.get(
    "ISMCTS_VALUE_MLP_PATH",
    str(MODELS_DIR / "value_mlp.pt"),
)

# Meta Gauntlet evaluation (Pillar 2)
GAUNTLET_GAMES_PER_TEAM = 2
GAUNTLET_EQUAL_WEIGHTS = os.environ.get("GAUNTLET_EQUAL_WEIGHTS", "false").lower() in (
    "1",
    "true",
    "yes",
)
GAUNTLET_OPPONENT_STATUS_CHANCE = 0.10
