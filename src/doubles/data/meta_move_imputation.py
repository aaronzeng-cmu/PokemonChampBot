"""Meta-database imputation for unrevealed move slots (P1 training tensors)."""

from __future__ import annotations

import logging
import random

from poke_env.data import to_id_str

from src.core.data.move_utils import canonical_move_list
from src.core.data.perspective import MonPerspective
from src.doubles.planning.meta_database import MetaDatabase

logger = logging.getLogger(__name__)

# Generic, always-legal-ish moves used to pad a move list when the meta has no
# (or too few) candidates for an off-meta species. Prevents the encoder from
# crashing on 0%-usage / meme Pokémon humans may bring in real battles.
_SAFE_FALLBACK_MOVES = ("protect", "tackle", "facade", "substitute")


def _safe_weight(value, default: float = 1.0) -> float:
    """Coerce a usage weight to float without raising on garbage values."""
    try:
        w = float(value)
    except (TypeError, ValueError):
        return default
    return w if w > 0 else 0.0


def _pad_with_safe_moves(known: list[str]) -> list[str]:
    """Top a move list up to four ids using generic safe fallbacks."""
    out = canonical_move_list(list(known))
    for mid in _SAFE_FALLBACK_MOVES:
        if len(out) >= 4:
            break
        if mid not in out:
            out.append(mid)
    return canonical_move_list(out)[:4]


def _weighted_sample_without_replacement(
    candidates: list[tuple[str, float]],
    k: int,
    rng: random.Random,
) -> list[str]:
    if k <= 0 or not candidates:
        return []
    pool = list(candidates)
    picked: list[str] = []
    for _ in range(min(k, len(pool))):
        total = sum(w for _, w in pool)
        if total <= 0:
            break
        roll = rng.random() * total
        acc = 0.0
        chosen_idx = 0
        for i, (_, w) in enumerate(pool):
            acc += w
            if roll <= acc:
                chosen_idx = i
                break
        move_id, _ = pool.pop(chosen_idx)
        picked.append(move_id)
    return picked


def impute_moves_to_four(
    species: str,
    known_moves: list[str],
    meta_db: MetaDatabase,
    rng: random.Random,
) -> list[str]:
    """
    Fill a Pokémon's move list to exactly four Showdown ids.
    Known moves are kept; remaining slots sampled from meta usage weights.
    """
    known = canonical_move_list(list(known_moves))
    if len(known) >= 4:
        return known[:4]

    try:
        prior = meta_db.get_species_prior(species)
        known_set = set(known)
        candidates: list[tuple[str, float]] = []
        for name, weight in prior.moves.items():
            mid = to_id_str(name)
            if not mid or mid in known_set:
                continue
            candidates.append((mid, _safe_weight(weight)))

        need = 4 - len(known)
        sampled = _weighted_sample_without_replacement(candidates, need, rng)
        filled = canonical_move_list(known + sampled)[:4]
    except Exception as exc:
        logger.warning("Move imputation failed for %r: %s; using safe moves", species, exc)
        filled = canonical_move_list(known)[:4]

    if len(filled) < 4:
        filled = _pad_with_safe_moves(filled)
    return filled


def impute_p1_mon_moves(
    mon: MonPerspective,
    meta_db: MetaDatabase,
    rng: random.Random,
) -> None:
    if not mon.species:
        return
    mon.moves = impute_moves_to_four(mon.species, mon.moves, meta_db, rng)


def impute_moves_to_four_deterministic(
    species: str,
    known_moves: list[str],
    meta_db: MetaDatabase,
) -> list[str]:
    """
    Fill to four moves deterministically: known moves plus highest-weight meta
    candidates, tie-broken alphabetically by move id.
    """
    known = canonical_move_list(list(known_moves))
    if len(known) >= 4:
        return known[:4]

    try:
        prior = meta_db.get_species_prior(species)
        known_set = set(known)
        candidates: list[tuple[str, float]] = []
        for name, weight in prior.moves.items():
            mid = to_id_str(name)
            if not mid or mid in known_set:
                continue
            candidates.append((mid, _safe_weight(weight)))

        need = 4 - len(known)
        candidates.sort(key=lambda item: (-item[1], item[0]))
        sampled = [move_id for move_id, _ in candidates[:need]]
        filled = canonical_move_list(known + sampled)[:4]
    except Exception as exc:
        logger.warning("Move imputation failed for %r: %s; using safe moves", species, exc)
        filled = canonical_move_list(known)[:4]

    if len(filled) < 4:
        filled = _pad_with_safe_moves(filled)
    return filled


def impute_p1_mon_moves_deterministic(
    mon: MonPerspective,
    meta_db: MetaDatabase,
) -> None:
    if not mon.species:
        return
    mon.moves = impute_moves_to_four_deterministic(mon.species, mon.moves, meta_db)
