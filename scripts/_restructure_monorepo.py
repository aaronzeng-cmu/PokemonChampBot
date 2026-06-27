"""One-shot monorepo restructure: src/ -> src/core, src/doubles, src/singles."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

CORE_REL = {
    "data/perspective.py",
    "data/log_tracker.py",
    "data/mega_items.py",
    "data/roster_profile.py",
    "data/state_tokenizer.py",
    "model/transformer_bot.py",
    "model/preview_model.py",
    "model/__init__.py",
    "planning/game_plan.py",
    "planning/species_normalize.py",
    "planning/dex_cache.py",
    "planning/macro_validation.py",
    "teams/roster.py",
    "battle/battle_runner.py",
    "battle/replay_utils.py",
    "training/__init__.py",
}

CORE_IMPORTS: list[tuple[str, str]] = [
    ("src.data.state_tokenizer", "src.core.data.state_tokenizer"),
    ("src.data.roster_profile", "src.core.data.roster_profile"),
    ("src.data.log_tracker", "src.core.data.log_tracker"),
    ("src.data.perspective", "src.core.data.perspective"),
    ("src.data.mega_items", "src.core.data.mega_items"),
    ("src.model.transformer_bot", "src.core.model.transformer_bot"),
    ("src.model.preview_model", "src.core.model.preview_model"),
    ("src.planning.macro_validation", "src.core.planning.macro_validation"),
    ("src.planning.species_normalize", "src.core.planning.species_normalize"),
    ("src.planning.dex_cache", "src.core.planning.dex_cache"),
    ("src.planning.game_plan", "src.core.planning.game_plan"),
    ("src.teams.roster", "src.core.teams.roster"),
    ("src.battle.battle_runner", "src.core.battle.battle_runner"),
    ("src.battle.replay_utils", "src.core.battle.replay_utils"),
    ("from src.model import", "from src.core.model import"),
]

DOUBLES_IMPORTS: list[tuple[str, str]] = [
    ("src.battle.", "src.doubles.battle."),
    ("src.data.", "src.doubles.data."),
    ("src.rl.", "src.doubles.rl."),
    ("src.players.", "src.doubles.players."),
    ("src.evaluation.", "src.doubles.evaluation."),
    ("src.planning.", "src.doubles.planning."),
    ("src.teams.", "src.doubles.teams."),
    ("src.env.", "src.doubles.env."),
    ("src.model.", "src.doubles.model."),
]

MOVE_UTILS = '''"""Format-agnostic move list helpers shared by core and doubles."""

from __future__ import annotations

from poke_env.data import to_id_str


def canonical_move_list(moves: list[str]) -> list[str]:
    """Stable alphabetical move slots 1-4 (Showdown ids)."""
    seen: list[str] = []
    for move in moves:
        mid = to_id_str(move)
        if mid and mid not in seen:
            seen.append(mid)
    return sorted(seen)[:4]
'''

MERGE_NOTE = '''# PokemonChampBot_Singles — Monorepo workspace

This directory is a **duplicate** of `PokemonChampBot` created so Singles
monorepo work can proceed while the original 500k-step VGC RL training run
continues on the GPU without import-path breakage.

## Merge back when training finishes

1. Stop or complete the RL run in the original `PokemonChampBot` folder.
2. Cherry-pick or merge the monorepo changes from this tree into the main repo.
3. Re-run the full test suite and a short RL eval smoke test.
4. Delete this duplicate workspace once the main tree is updated.

Do **not** run training scripts from both directories against the same
Showdown account simultaneously.
'''


def _ensure_init(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    init = path / "__init__.py"
    if not init.exists():
        init.write_text("", encoding="utf-8")


def move_files() -> None:
    all_files = [p for p in SRC.rglob("*") if p.is_file()]
    for path in sorted(all_files):
        rel = path.relative_to(SRC).as_posix()
        if rel == "__init__.py":
            continue
        if rel in CORE_REL:
            dest = SRC / "core" / rel
        else:
            dest = SRC / "doubles" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(dest))

    for pkg in [
        "battle",
        "data",
        "model",
        "rl",
        "players",
        "teams",
        "evaluation",
        "planning",
        "env",
        "training",
    ]:
        d = SRC / pkg
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    _ensure_init(SRC / "core")
    _ensure_init(SRC / "core" / "data")
    _ensure_init(SRC / "core" / "model")
    _ensure_init(SRC / "core" / "planning")
    _ensure_init(SRC / "core" / "teams")
    _ensure_init(SRC / "core" / "battle")
    _ensure_init(SRC / "core" / "training")
    _ensure_init(SRC / "doubles")
    _ensure_init(SRC / "singles")

    (SRC / "core" / "data" / "move_utils.py").write_text(MOVE_UTILS, encoding="utf-8")
    (SRC / "singles" / "__init__.py").write_text(
        '"""Singles (BSS Bring-3) modules — populated in later phases."""\n',
        encoding="utf-8",
    )
    (ROOT / "MERGE_NOTE.md").write_text(MERGE_NOTE, encoding="utf-8")


def rewrite_imports_in_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    for old, new in CORE_IMPORTS:
        text = text.replace(old, new)
    for old, new in DOUBLES_IMPORTS:
        text = text.replace(old, new)
    text = text.replace(
        "from src.doubles.battle.move_order import canonical_move_list",
        "from src.core.data.move_utils import canonical_move_list",
    )
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_move_order() -> None:
    path = SRC / "doubles" / "battle" / "move_order.py"
    text = path.read_text(encoding="utf-8")
    if "from src.core.data.move_utils import canonical_move_list" not in text:
        text = text.replace(
            "from src.doubles.data.action_space_spec import ACTION_SIZE, target_offset_label",
            "from src.core.data.move_utils import canonical_move_list\n"
            "from src.doubles.data.action_space_spec import ACTION_SIZE, target_offset_label",
        )
        text = re.sub(
            r"\ndef canonical_move_list\(moves: list\[str\]\) -> list\[str\]:.*?\n    return sorted\(seen\)\[:4\]\n\n",
            "\n",
            text,
            count=1,
            flags=re.DOTALL,
        )
        path.write_text(text, encoding="utf-8")


def patch_perspective() -> None:
    path = SRC / "core" / "data" / "perspective.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "from src.doubles.battle.move_order import canonical_move_list",
        "from src.core.data.move_utils import canonical_move_list",
    )
    path.write_text(text, encoding="utf-8")


def patch_state_tokenizer_imports() -> None:
    path = SRC / "core" / "data" / "state_tokenizer.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "from src.doubles.battle.move_order import canonical_move_list",
        "from src.core.data.move_utils import canonical_move_list",
    )
    text = text.replace(
        "from src.core.data.mega_state import live_can_mega_for_pos",
        "from src.doubles.data.mega_state import live_can_mega_for_pos",
    )
    path.write_text(text, encoding="utf-8")


def patch_roster_profile() -> None:
    path = SRC / "core" / "data" / "roster_profile.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "from src.doubles.battle.move_order import canonical_move_list",
        "from src.core.data.move_utils import canonical_move_list",
    )
    path.write_text(text, encoding="utf-8")


def rewrite_all_py_files() -> int:
    count = 0
    for path in ROOT.rglob("*.py"):
        if "_restructure_monorepo.py" in path.name:
            continue
        if "archive" in path.parts:
            continue
        if rewrite_imports_in_file(path):
            count += 1
    return count


def main() -> None:
    move_files()
    n = rewrite_all_py_files()
    patch_move_order()
    patch_perspective()
    patch_state_tokenizer_imports()
    patch_roster_profile()
    print(f"Restructure complete. Updated imports in {n} files.")


if __name__ == "__main__":
    main()
