"""LLM macro strategist for team preview (DeepSeek v4 Pro)."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from config.settings import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_REASONING_EFFORT,
    DEEPSEEK_THINKING_ENABLED,
    MACRO_STRATEGIST_ENABLED,
    MACRO_STRATEGIST_FALLBACK,
)

from src.core.planning.game_plan import GamePlan
from src.core.planning.macro_validation import validate_and_normalize_game_plan
from src.core.planning.species_normalize import clean_species_name, opponent_belief_key

if TYPE_CHECKING:
    from poke_env.battle.double_battle import DoubleBattle

    from src.doubles.planning.belief_state import BeliefState
    from src.doubles.planning.meta_database import MetaDatabase

logger = logging.getLogger(__name__)
_warned_no_key = False


def preview_roster_names(battle: DoubleBattle) -> tuple[list[str], list[str]]:
    """Display names for our 6 and opponent preview 6."""
    our = [
        clean_species_name(str(p.base_species or p.species or ""))
        for p in battle.team.values()
    ]
    preview = list(battle.teampreview_opponent_team) or list(battle.opponent_team.values())
    opp = [clean_species_name(opponent_belief_key(m)) for m in preview if m]
    return our, opp

GAME_PLAN_SCHEMA = {
    "primary_threats": ["species names"],
    "optimal_lead": ["two species from our team"],
    "opponent_likely_lead": ["two species"],
    "win_condition": "short strategy text",
    "priority_kos": ["species to KO first"],
}


def build_preview_prompt(
    battle: DoubleBattle,
    belief: BeliefState,
    meta_db: MetaDatabase,
) -> str:
    our_species, opp_species = preview_roster_names(battle)

    lines = [
        "You are a Gen 9 Pokémon Champions VGC Reg M-A coach.",
        "Analyze team preview and return strict JSON matching this schema:",
        json.dumps(GAME_PLAN_SCHEMA),
        "",
        "CRITICAL CONSTRAINTS:",
        "1. You may ONLY reference Pokémon from the EXACT following lists.",
        f"MY_TEAM: {json.dumps(our_species)}",
        f"OPPONENT_TEAM: {json.dumps(opp_species)}",
        "2. Any deviation from these exact names will cause a fatal system error.",
        "3. optimal_lead must contain exactly two species from MY_TEAM.",
        "4. primary_threats, opponent_likely_lead, and priority_kos must use OPPONENT_TEAM names only.",
        "",
        f"Our team: {', '.join(our_species)}",
        f"Opponent preview (6 shown, 4 brought): {', '.join(opp_species)}",
        "",
        "Opponent roster belief (P in bring-4 until confirmed on field):",
    ]

    for mon in sorted(belief.pokemon, key=lambda m: -m.brought_prob):
        if mon.confirmed_absent:
            lines.append(f"- {mon.species}: ruled out (not brought)")
            continue
        if mon.confirmed_brought:
            lines.append(f"- {mon.species}: confirmed in game")
        else:
            lines.append(f"- {mon.species}: P(brought) {mon.brought_prob * 100:.0f}%")
        if mon.preview_only and not mon.confirmed_brought:
            continue

    lines.append("")
    lines.append("Set priors (confirmed / active candidates):")

    for mon in belief.pokemon:
        if mon.preview_only and not mon.confirmed_brought:
            continue
        top_moves = sorted(
            mon.moves[0].options.items() if mon.moves else [],
            key=lambda x: -x[1],
        )[:8]
        top_items = sorted(mon.item.options.items(), key=lambda x: -x[1])[:5]
        top_abilities = sorted(mon.ability.options.items(), key=lambda x: -x[1])[:3]
        prior = meta_db.get_species_prior(mon.species)
        lines.append(f"- {mon.species}:")
        if mon.mega_confirmed and mon.mega_form:
            lines.append(f"  confirmed mega: {mon.mega_form}")
        elif prior.form_variants and len(prior.form_variants) > 1:
            forms = sorted(prior.form_variants.items(), key=lambda x: -x[1])
            lines.append(
                "  forms (% ladder usage): "
                + ", ".join(f"{f} {p:.2f}%" for f, p in forms[:4])
            )
        if top_moves:
            lines.append(
                "  moves (% of sets): "
                + ", ".join(f"{m} {p:.1f}%" for m, p in top_moves)
            )
        if top_items:
            lines.append(
                "  items (% of sets): "
                + ", ".join(f"{i} {p:.1f}%" for i, p in top_items)
            )
        if top_abilities:
            lines.append(
                "  abilities (% of sets): "
                + ", ".join(f"{a} {p:.1f}%" for a, p in top_abilities)
            )

    matchup = meta_db.get_matchup_context(our_species, opp_species)
    if matchup:
        lines.extend(["", "Meta matchup context:", matchup])

    lines.extend(
        [
            "",
            "Format: bring 4, doubles, Tera and Mega legal, Champions stat points (max 32/stat, 66 total).",
            "Return only valid JSON, no markdown.",
        ]
    )
    return "\n".join(lines)


class HeuristicMacroStrategist:
    def analyze(
        self,
        battle: DoubleBattle,
        belief: BeliefState,
        meta_db: MetaDatabase,
    ) -> GamePlan:
        our_species, opp_species = preview_roster_names(battle)

        threat_scores: list[tuple[str, float]] = []
        for mon in belief.pokemon:
            if mon.confirmed_absent:
                continue
            if mon.confirmed_brought:
                weight = 1.0
            else:
                weight = mon.brought_prob
            prior = meta_db.get_species_prior(mon.species)
            score = (prior.usage_pct or 0.0) * weight
            threat_scores.append((mon.species, score))
        if not threat_scores:
            for species in opp_species:
                prior = meta_db.get_species_prior(species)
                threat_scores.append((species, prior.usage_pct or 0.0))
        threat_scores.sort(key=lambda x: -x[1])
        primary_threats = [s for s, _ in threat_scores[:3]]

        optimal_lead = our_species[:2] if len(our_species) >= 2 else our_species
        opp_lead = opp_species[:2] if len(opp_species) >= 2 else opp_species

        return GamePlan(
            primary_threats=primary_threats,
            optimal_lead=optimal_lead,
            opponent_likely_lead=opp_lead,
            win_condition=f"Neutralize {primary_threats[0]} and maintain board control.",
            priority_kos=primary_threats[:2],
        )


class MacroStrategist:
    def __init__(self) -> None:
        self._heuristic = HeuristicMacroStrategist()
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not DEEPSEEK_API_KEY:
            return None
        try:
            from openai import OpenAI
            import os

            # httpx (used by the OpenAI SDK) relies on SSL_CERT_FILE when verifying TLS.
            # This environment sometimes points SSL_CERT_FILE to a non-existent file.
            ssl_cert_file = os.environ.get("SSL_CERT_FILE")
            if ssl_cert_file and not os.path.exists(ssl_cert_file):
                try:
                    import certifi

                    os.environ["SSL_CERT_FILE"] = certifi.where()
                except Exception:
                    # If certifi isn't available, we'll fall back to the default behavior.
                    pass
            base_url = DEEPSEEK_BASE_URL.rstrip("/")
            # DeepSeek's OpenAI-compatible endpoint is usually under /v1.
            # If config only has the root domain, append /v1 for compatibility.
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"

            self._client = OpenAI(
                api_key=DEEPSEEK_API_KEY,
                base_url=base_url,
            )
        except (ImportError, FileNotFoundError, OSError) as exc:
            if isinstance(exc, ImportError):
                logger.warning("openai package not installed; using heuristic macro strategist")
            else:
                logger.warning(
                    "OpenAI client init failed (%s); using heuristic macro strategist",
                    exc,
                )
            return None
        return self._client

    def analyze(
        self,
        battle: DoubleBattle,
        belief: BeliefState,
        meta_db: MetaDatabase,
    ) -> GamePlan:
        global _warned_no_key
        if not MACRO_STRATEGIST_ENABLED:
            return self._heuristic.analyze(battle, belief, meta_db)

        client = self._get_client()
        if client is None:
            if not _warned_no_key and MACRO_STRATEGIST_FALLBACK == "heuristic":
                logger.warning(
                    "DEEPSEEK_API_KEY not set; using heuristic macro strategist"
                )
                _warned_no_key = True
            return self._heuristic.analyze(battle, belief, meta_db)

        prompt = build_preview_prompt(battle, belief, meta_db)
        try:
            kwargs: dict = {
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a VGC strategist. Respond with JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.3,
            }
            if DEEPSEEK_THINKING_ENABLED:
                kwargs["reasoning_effort"] = DEEPSEEK_REASONING_EFFORT or "high"
                kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            else:
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or "{}"
            data = json.loads(content)
            plan = GamePlan.from_dict(data)
            our_team, opp_team = preview_roster_names(battle)
            validated = validate_and_normalize_game_plan(plan, our_team, opp_team)
            if validated is None:
                logger.warning(
                    "Macro strategist LLM hallucinated species; using heuristic fallback"
                )
                return self._heuristic.analyze(battle, belief, meta_db)
            return validated
        except Exception as exc:
            logger.warning("Macro strategist LLM failed (%s); using heuristic", exc)
            return self._heuristic.analyze(battle, belief, meta_db)
