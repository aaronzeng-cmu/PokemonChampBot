"""Translate English battle-log text (from OCR) into structured events.

The Champions HUD prints one human-readable line per protocol event during
animations, e.g. ``"Gyarados's Attack and Speed rose!"``. OCR of that line is
rarely perfect, so the rules here are deliberately lenient: case-insensitive,
punctuation-agnostic, tolerant of a dropped possessive apostrophe, and adverb
position is not assumed.

Events are emitted as plain dicts consumed by ``LiveBattleTracker.apply_log_event``.
Entity resolution (mapping a name -> active slot) happens in the tracker, which is
the only component that knows the current actives; the parser only flags whether
the subject is an opponent ("the opposing X" / "the foe's X").
"""

from __future__ import annotations

import re
from typing import Any

from poke_env.data import to_id_str

# English stat name -> Showdown short key (matches log_tracker boosts dict).
_STAT_KEYS: dict[str, str] = {
    "attack": "atk",
    "defense": "def",
    "defence": "def",
    "special attack": "spa",
    "sp atk": "spa",
    "sp. atk": "spa",
    "sp attack": "spa",
    "sp. attack": "spa",
    "spatk": "spa",
    "special defense": "spd",
    "special defence": "spd",
    "sp def": "spd",
    "sp. def": "spd",
    "sp defense": "spd",
    "sp. defense": "spd",
    "spdef": "spd",
    "speed": "spe",
    "accuracy": "accuracy",
    "evasion": "evasion",
    "evasiveness": "evasion",
    "critical hit ratio": "crit",
    "critical-hit ratio": "crit",
}

# Adverb -> magnitude. Real Pokémon semantics: sharply=2, drastically=3
# (likewise harshly=2, severely=3 for drops). Adjust here if you prefer the
# simplified +2 for "drastically".
_MAGNITUDE_ADVERBS: dict[str, int] = {
    "drastically": 3,
    "severely": 3,
    "sharply": 2,
    "harshly": 2,
}

_OPPONENT_PREFIXES = (
    "the opposing ",
    "the foe's ",
    "the foe ",
    "opposing ",
    "foe's ",
)

# Weather phrase fragments -> Showdown weather id (matches state_tokenizer).
_WEATHER_RULES: list[tuple[str, str]] = [
    ("sunlight turned harsh", "sunnyday"),
    ("sunlight is strong", "sunnyday"),
    ("started to rain", "raindance"),
    ("began to rain", "raindance"),
    ("is raining", "raindance"),
    ("sandstorm kicked up", "sandstorm"),
    ("sandstorm is raging", "sandstorm"),
    ("started to snow", "snowscape"),
    ("began to snow", "snowscape"),
    ("snow started", "snowscape"),
    ("it is snowing", "snowscape"),
    ("started to hail", "hail"),
    ("is hailing", "hail"),
]

# Terrain phrase fragments -> terrain id fragment.
_TERRAIN_RULES: list[tuple[str, str]] = [
    ("electric terrain", "electricterrain"),
    ("grassy terrain", "grassyterrain"),
    ("misty terrain", "mistyterrain"),
    ("psychic terrain", "psychicterrain"),
]


# Stat words used to anchor the "Sp: Atk" -> "Sp. Atk" OCR fix.
_SP_STAT_WORD = r"(atk|atck|attack|def|defense|defence)"


def normalize_text(text: str) -> str:
    """Clean OCR artifacts before regex parsing (case preserved for names).

    - Normalize unicode apostrophes / backticks and collapse whitespace.
    - Fix the colon-for-period stat typo: ``"Sp: Atk" -> "Sp. Atk"``.
    - Force trailing OCR misreads of ``!`` (``1`` / ``I`` / a single stray ``l``)
      back into a real ``!`` so faint / stat / move rules fire. A *single* trailing
      ``l`` is stripped so double-``l`` verbs survive (``"felll" -> "fell!"``,
      not ``"fel!"``).
    """
    text = text.replace("\u2019", "'").replace("`", "'")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return text

    text = re.sub(
        rf"(?i)\bsp\s*[:.]\s*{_SP_STAT_WORD}\b",
        lambda m: "Sp. " + m.group(1),
        text,
    )

    text = re.sub(r"[1I!]+$", "!", text)
    if text.endswith("l"):
        text = text[:-1] + "!"
    return text


# Backwards-compatible alias (older callers / tests).
def _normalize(text: str) -> str:
    return normalize_text(text)


def _strip_trailing_punct(text: str) -> str:
    return text.strip().rstrip("!.").strip()


def _split_subject(raw: str) -> tuple[str, bool]:
    """Return (clean_name, is_opponent) after stripping opponent prefixes."""
    name = raw.strip()
    lowered = name.lower()
    is_opponent = False
    for prefix in _OPPONENT_PREFIXES:
        if lowered.startswith(prefix):
            name = name[len(prefix) :].strip()
            is_opponent = True
            break
    return name, is_opponent


def _stat_key(token: str) -> str | None:
    token = _strip_trailing_punct(token).strip().replace("'", "")
    if not token:
        return None
    key = _STAT_KEYS.get(token)
    if key is not None:
        return key
    # Leniency: collapse spaces / dots / colons (e.g. "sp.atk", "sp:atk", "spatk").
    squashed = token.replace(".", "").replace(":", "").replace(" ", "")
    for name, mapped in _STAT_KEYS.items():
        if name.replace(".", "").replace(" ", "") == squashed:
            return mapped
    return None


def _resolve_stats(stats_blob: str, *, strict: bool = False) -> list[str]:
    """Split a stats phrase ('Attack and Speed') into Showdown stat keys.

    With ``strict=True`` every chunk must map to a stat or the result is empty;
    used when guessing where a dropped-apostrophe subject ends.
    """
    blob = stats_blob.lower().replace(" and ", ",").replace("&", ",")
    keys: list[str] = []
    for chunk in blob.split(","):
        if not chunk.strip():
            continue
        key = _stat_key(chunk)
        if key is None:
            if strict:
                return []
            continue
        if key not in keys:
            keys.append(key)
    return keys


def _parse_stat_change(norm: str) -> dict[str, Any] | None:
    if "rose" not in norm and "fell" not in norm:
        return None
    # Ignore "won't go any higher/lower" no-op messages.
    if "go any higher" in norm or "go any lower" in norm or "can't go" in norm:
        return None

    sign = 1 if "rose" in norm else -1
    magnitude = 1
    for adverb, value in _MAGNITUDE_ADVERBS.items():
        if adverb in norm:
            magnitude = value
            break

    # Remove verbs/adverbs to isolate "<subject>'s <stats>".
    head = re.sub(r"\b(rose|fell|sharply|drastically|harshly|severely)\b", " ", norm)
    head = _strip_trailing_punct(re.sub(r"\s+", " ", head).strip())

    subject_blob = head
    stats_blob = ""
    poss = re.match(r"(?P<subject>.+?)'s\s+(?P<stats>.+)$", head)
    if poss:
        subject_blob = poss.group("subject")
        stats_blob = poss.group("stats")
    else:
        # Possessive apostrophe likely dropped by OCR: the stats are the longest
        # trailing run of tokens that cleanly resolve to stat names.
        tokens = head.split()
        for start in range(1, len(tokens)):
            candidate = " ".join(tokens[start:])
            if _resolve_stats(candidate, strict=True):
                subject_blob = " ".join(tokens[:start]).strip()
                stats_blob = candidate
                break

    stats = _resolve_stats(stats_blob)
    if not stats:
        return None

    target, is_opponent = _split_subject(subject_blob)
    if not target:
        return None
    return {
        "type": "stat_boost",
        "target": to_id_str(target),
        "target_name": target,
        "is_opponent": is_opponent,
        "stats": stats,
        "amount": sign * magnitude,
    }


def _parse_faint(norm: str) -> dict[str, Any] | None:
    # Tolerate OCR garbage glued onto "fainted" (e.g. "faintedl") by not requiring
    # a trailing word boundary after the verb.
    m = re.match(r"(?P<subject>.+?)\s+fainted", norm, re.IGNORECASE)
    if not m:
        return None
    target, is_opponent = _split_subject(m.group("subject"))
    if not target:
        return None
    return {
        "type": "faint",
        "target": to_id_str(target),
        "target_name": target,
        "is_opponent": is_opponent,
    }


def _parse_mega(norm: str) -> dict[str, Any] | None:
    # "Staraptor has Mega Evolved into Mega Staraptor!" -> subject = Staraptor.
    m = re.match(r"(?P<subject>.+?)\s+(?:has\s+)?mega[\s\-]?evolved", norm, re.IGNORECASE)
    if not m:
        return None
    target, is_opponent = _split_subject(m.group("subject"))
    if not target:
        return None
    return {
        "type": "mega_evolve",
        "target": to_id_str(target),
        "target_name": target,
        "is_opponent": is_opponent,
    }


def _parse_move(norm: str) -> dict[str, Any] | None:
    m = re.match(r"(?P<subject>.+?)\s+used\s+(?P<move>.+?)$", norm)
    if not m:
        return None
    user, is_opponent = _split_subject(m.group("subject"))
    move = _strip_trailing_punct(m.group("move"))
    if not user or not move:
        return None
    return {
        "type": "move",
        "user": to_id_str(user),
        "user_name": user,
        "is_opponent": is_opponent,
        "move": to_id_str(move),
        "move_name": move,
    }


def _parse_protect(norm: str) -> dict[str, Any] | None:
    # Protect-family success line, e.g. "Garchomp protected itself!" (also Detect,
    # King's Shield, Spiky Shield). This is the only signal that a Protect actually
    # *activated*; "X used Protect" fires even when it fails, so we key off this.
    m = re.match(r"(?P<subject>.+?)\s+protected\s+itself", norm, re.IGNORECASE)
    if not m:
        return None
    subject, is_opponent = _split_subject(m.group("subject"))
    if not subject:
        return None
    return {
        "type": "protect",
        "target": to_id_str(subject),
        "target_name": subject,
        "is_opponent": is_opponent,
    }


def _parse_weather(norm: str) -> dict[str, Any] | None:
    for fragment, weather_id in _WEATHER_RULES:
        if fragment in norm:
            return {"type": "weather", "weather": weather_id}
    return None


def _parse_terrain(norm: str) -> dict[str, Any] | None:
    for fragment, terrain_id in _TERRAIN_RULES:
        if fragment in norm and ("set" in norm or "covered" in norm or "got weird" in norm or "terrain" in norm):
            return {"type": "terrain", "terrain": terrain_id}
    return None


def _parse_status(norm: str) -> dict[str, Any] | None:
    rules = [
        (r"(?P<subject>.+?)\s+was paralyzed", "par"),
        (r"(?P<subject>.+?)\s+was burned", "brn"),
        (r"(?P<subject>.+?)\s+was poisoned", "psn"),
        (r"(?P<subject>.+?)\s+was badly poisoned", "tox"),
        (r"(?P<subject>.+?)\s+fell asleep", "slp"),
        (r"(?P<subject>.+?)\s+was frozen", "frz"),
    ]
    for pattern, status in rules:
        m = re.match(pattern, norm)
        if m:
            target, is_opponent = _split_subject(m.group("subject"))
            if target:
                return {
                    "type": "status",
                    "target": to_id_str(target),
                    "target_name": target,
                    "is_opponent": is_opponent,
                    "status": status,
                }
    return None


# Order matters: weather/terrain are field-wide; stat/faint/status/move are subject-based.
_PARSERS = (
    _parse_weather,
    _parse_terrain,
    _parse_stat_change,
    _parse_faint,
    _parse_mega,
    _parse_protect,
    _parse_status,
    _parse_move,
)


_ABILITY_IDS: set[str] | None = None


def _ability_ids() -> set[str]:
    """All gen-9 ability ids (from the pokedex), used to tell abilities from items.

    The mid-screen banner ("X's Y") doesn't say whether Y is an ability or an
    item, so we classify by membership: abilities form a closed set derivable
    from the pokedex; anything else is treated as a held item.
    """
    global _ABILITY_IDS
    if _ABILITY_IDS is None:
        ids: set[str] = set()
        try:
            from poke_env.data import GenData

            for entry in GenData.from_gen(9).pokedex.values():
                for ability in (entry.get("abilities") or {}).values():
                    if ability:
                        ids.add(to_id_str(ability))
        except Exception:
            pass
        _ABILITY_IDS = ids
    return _ABILITY_IDS


def parse_ability_item_popup(text: str | None) -> dict[str, Any] | None:
    """Parse a mid-screen ability/item banner ("<Holder>'s <Name>").

    Returns an ``ability_item`` event with ``subtype`` ("ability" or "item").
    The apostrophe is required as an anchor so move lists / stray HUD text don't
    produce false reveals.
    """
    if not text:
        return None
    cleaned = re.sub(r"\s+", " ", text.replace("\u2019", "'").replace("`", "'")).strip()
    m = re.match(r"(?P<holder>.+?)['’]s?\s+(?P<name>.+)$", cleaned)
    if not m:
        return None
    holder, is_opponent = _split_subject(m.group("holder"))
    name = _strip_trailing_punct(m.group("name"))
    if not holder or len(name) < 3:
        return None
    name_id = to_id_str(name)
    if not name_id:
        return None
    subtype = "ability" if name_id in _ability_ids() else "item"
    return {
        "type": "ability_item",
        "subtype": subtype,
        "holder": to_id_str(holder),
        "holder_name": holder,
        "is_opponent": is_opponent,
        "name": name,
        "name_id": name_id,
    }


def parse_string(text: str | None) -> dict[str, Any] | None:
    """Parse a single battle-log line into an event dict, or ``None``.

    Lenient to OCR noise: case-insensitive, tolerates missing punctuation and a
    dropped possessive apostrophe.
    """
    if not text:
        return None
    norm = normalize_text(text)
    if not norm:
        return None
    for parser in _PARSERS:
        event = parser(norm)
        if event is not None:
            return event
    return None
