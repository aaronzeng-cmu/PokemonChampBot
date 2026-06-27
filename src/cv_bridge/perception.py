"""Computer vision perception: state detection and OCR data extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from poke_env.data import to_id_str

from src.cv_bridge.action_executor import load_ui_coordinates
from src.cv_bridge.ocr_utils import (
    get_hp_percentage_from_bar,
    parse_hp_percent,
    parse_hp_text,
    read_text_lines,
)
from src.cv_bridge.sprite_matcher import SpriteMatcher
from src.cv_bridge.template_bootstrap import ensure_templates

GameState = Literal[
    "UNKNOWN",
    "IDLE",
    "LOADING",
    "COMMUNICATING",
    "TEAM_PREVIEW",
    "TURN_DECISION",
    "MOVE_SELECTION",
    "TARGET_SELECTION",
    "FORCE_SWITCH",
    "ANIMATION",
    "RESULTS",
]

_HP_FRACTION_RE = re.compile(r"(\d{1,3})\s*[/I|]\s*(\d{1,3})")

BattleFormatHint = Literal["singles", "doubles", "unknown"]

_DEFAULT_TEMPLATES = Path(__file__).resolve().parent / "templates"

# Search ROI keys in ui_coordinates shared.perception_regions (optional speedup).
_TEMPLATE_SEARCH_REGIONS: dict[str, str | None] = {
    "fight_button": "command_menu_fight_button",
    "pokemon_button": "command_menu_fight_button",
    "move_panel_anchor": "move_panel",
    "target_overlay_close": "target_overlay_panel",
    "teampreview_header": "teampreview_header",
    "results_continue": "results_bottom_buttons",
    "lobby_battle": None,
    "communicating_banner": "communicating_banner",
}

_TEMPLATE_STATE_MAP: dict[str, GameState] = {
    "target_overlay_close": "TARGET_SELECTION",
    "move_panel_anchor": "MOVE_SELECTION",
    "fight_button": "TURN_DECISION",
    "pokemon_button": "TURN_DECISION",
    "teampreview_header": "TEAM_PREVIEW",
    "results_continue": "RESULTS",
    "lobby_battle": "IDLE",
    "communicating_banner": "COMMUNICATING",
}

_STATE_CHECK_ORDER: tuple[str, ...] = (
    "communicating_banner",
    "target_overlay_close",
    "move_panel_anchor",
    "fight_button",
    "pokemon_button",
    "teampreview_header",
    "results_continue",
    "lobby_battle",
)

# Composite states: any template in group may trigger (e.g. Fight OR Pokemon menu).
_COMPOSITE_STATE_GROUPS: dict[GameState, tuple[str, ...]] = {
    "TURN_DECISION": ("fight_button", "pokemon_button"),
}

_NAME_REGION_EXPAND_UP = 44


@dataclass(frozen=True)
class TemplateMatch:
    name: str
    confidence: float
    state: GameState


@dataclass
class PerceptionResult:
    state: GameState
    state_confidence: float
    battle_format: BattleFormatHint
    template_match: str
    ocr: dict[str, Any] = field(default_factory=dict)
    raw_matches: list[TemplateMatch] = field(default_factory=list)


def _clean_species_text(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9\-'. ]+", " ", text).strip()
    tokens = [tok for tok in text.split() if len(tok) >= 3 and tok.lower() not in {"the", "max", "hp"}]
    if not tokens:
        return ""
    return tokens[0]


class PerceptionModule:
    """Detect UI state via template matching and extract battle text via OCR."""

    def __init__(
        self,
        *,
        coordinates: dict[str, Any] | None = None,
        templates_dir: Path | str | None = None,
        confidence_threshold: float = 0.85,
        ocr_enabled: bool = True,
        sprite_matcher: SpriteMatcher | None = None,
    ):
        self.coords = coordinates or load_ui_coordinates()
        self.regions: dict[str, list[int]] = self.coords["shared"]["perception_regions"]
        self.teampreview_regions: dict[str, Any] = self.coords.get("teampreview", {})
        self.templates_dir = Path(templates_dir or _DEFAULT_TEMPLATES)
        ensure_templates(self.templates_dir)
        self.confidence_threshold = confidence_threshold
        self.ocr_enabled = ocr_enabled
        self.sprite_matcher = sprite_matcher or SpriteMatcher()
        self._templates = self._load_templates()
        self._ocr_reader: Any | None = None

    def _load_templates(self) -> dict[str, np.ndarray]:
        templates: dict[str, np.ndarray] = {}
        for path in sorted(self.templates_dir.glob("*.png")):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is not None and image.size > 0:
                templates[path.stem] = image
        return templates

    def _get_ocr_reader(self) -> Any:
        if self._ocr_reader is None:
            import easyocr

            self._ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        return self._ocr_reader

    def _crop_region(self, frame: np.ndarray, region_key: str) -> np.ndarray | None:
        spec = self.regions.get(region_key)
        if not spec or len(spec) != 4:
            return None
        return self._crop_box(frame, spec)

    @staticmethod
    def _crop_box(frame: np.ndarray, spec: list[int] | tuple[int, ...]) -> np.ndarray | None:
        if len(spec) != 4:
            return None
        x, y, w, h = (int(v) for v in spec)
        if w <= 0 or h <= 0:
            return None
        fh, fw = frame.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(fw, x + w), min(fh, y + h)
        if x1 <= x0 or y1 <= y0:
            return None
        return frame[y0:y1, x0:x1]

    def _crop_teampreview_slot(
        self,
        frame: np.ndarray,
        group_key: str,
        slot_index: int,
    ) -> np.ndarray | None:
        group = self.teampreview_regions.get(group_key, {})
        if not isinstance(group, dict):
            return None
        spec = group.get(f"slot_{slot_index}")
        if not spec:
            return None
        return self._crop_box(frame, spec)

    @staticmethod
    def _map_ocr_to_species_id(text: str) -> str:
        cleaned = _clean_species_text(text)
        if not cleaned:
            return "unknown"
        try:
            return to_id_str(cleaned)
        except Exception:
            return "unknown"

    def _identify_sprite_crop(self, crop: np.ndarray | None) -> str:
        if crop is None or crop.size == 0:
            return "unknown"
        try:
            if not self.sprite_matcher.ready:
                self.sprite_matcher.build_index()
            return self.sprite_matcher.identify_sprite(crop)
        except (FileNotFoundError, RuntimeError):
            return "unknown"

    def parse_team_preview(self, frame: np.ndarray) -> dict[str, list[str]]:
        """Parse ally and enemy teams from team-preview sprites (OCR fallback for ally)."""
        ally_team: list[str] = []
        enemy_team: list[str] = []

        for slot in range(1, 7):
            ally_crop = self._crop_teampreview_slot(frame, "ally_sprite_slots", slot)
            if ally_crop is not None and ally_crop.size > 0:
                ally_id = self._identify_sprite_crop(ally_crop)
            else:
                ally_crop = self._crop_teampreview_slot(frame, "ally_name_slots", slot)
                if ally_crop is not None and ally_crop.size > 0:
                    ally_text = self._ocr_crop(ally_crop)
                    ally_id = self._map_ocr_to_species_id(ally_text)
                else:
                    ally_id = "unknown"
            ally_team.append(ally_id)

            enemy_crop = self._crop_teampreview_slot(frame, "enemy_sprite_slots", slot)
            if enemy_crop is not None and enemy_crop.size > 0:
                enemy_team.append(self._identify_sprite_crop(enemy_crop))
            else:
                enemy_team.append("unknown")

        return {"ally_team": ally_team, "enemy_team": enemy_team}

    def _name_region_from_hp(self, hp_region_key: str) -> list[int]:
        spec = self.regions.get(hp_region_key)
        if not spec or len(spec) != 4:
            return []
        x, y, w, h = (int(v) for v in spec)
        name_h = min(_NAME_REGION_EXPAND_UP, y)
        return [x, max(0, y - name_h), w, name_h + h]

    def _search_roi(self, frame: np.ndarray, template_name: str) -> np.ndarray:
        region_key = _TEMPLATE_SEARCH_REGIONS.get(template_name)
        if region_key:
            crop = self._crop_region(frame, region_key)
            if crop is not None:
                return crop
        return frame

    def match_template(
        self,
        frame: np.ndarray,
        template_name: str,
        *,
        roi: np.ndarray | None = None,
    ) -> float:
        template = self._templates.get(template_name)
        if template is None:
            return 0.0
        haystack = roi if roi is not None else self._search_roi(frame, template_name)
        if haystack.shape[0] < template.shape[0] or haystack.shape[1] < template.shape[1]:
            return 0.0
        gray = cv2.cvtColor(haystack, cv2.COLOR_BGR2GRAY) if haystack.ndim == 3 else haystack
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return float(max_val)

    def _detect_turn_decision(self, frame: np.ndarray, confidence_by_name: dict[str, float]) -> bool:
        if confidence_by_name.get("target_overlay_close", 0.0) >= 0.88:
            return False
        if confidence_by_name.get("move_panel_anchor", 0.0) >= 0.70:
            return False
        fight = confidence_by_name.get("fight_button", 0.0)
        pokemon = confidence_by_name.get("pokemon_button", 0.0)
        # A genuine command menu scores ~1.0 on Fight. The team-preview screen only
        # weakly resembles it (~0.45), so require a strong match to avoid firing on
        # preview / other screens that happen to show a timer.
        if max(fight, pokemon) < 0.55:
            return False
        timer = self._crop_region(frame, "decision_state_move_timer")
        if timer is None:
            return max(fight, pokemon) >= 0.55
        return float(np.mean(timer)) > 30.0

    def _detect_team_preview(self, frame: np.ndarray) -> bool:
        """OCR the centre prompt ("Select N Pokemon to send into battle").

        The teampreview_header *template* matches only dark background and never
        fires, so detection is done by reading the distinctive prompt text. This
        is robust to the singles/doubles pick count.
        """
        if not self.ocr_enabled:
            return False
        crop = self._crop_region(frame, "teampreview_prompt")
        if crop is None or crop.size == 0:
            return False
        text = (read_text_lines(crop, self._get_ocr_reader()) or "").lower()
        if not text:
            return False
        return "send into battle" in text or ("select" in text and "pok" in text)

    def _detect_move_selection(self, frame: np.ndarray) -> bool:
        """OCR the "Move Info" button, which appears only on the move list.

        The move_panel_anchor template is too weak (~0.5) on this screen, so it
        would otherwise fall through to ANIMATION and the loop could never act on
        (or recover from) an open move list.
        """
        if not self.ocr_enabled:
            return False
        crop = self._crop_region(frame, "move_select_marker")
        if crop is None or crop.size == 0:
            return False
        text = (read_text_lines(crop, self._get_ocr_reader()) or "").lower()
        # OCR often renders "Move Info" as "Move Into"; accept either.
        return "move" in text and ("info" in text or "into" in text)

    def _detect_force_switch(self, frame: np.ndarray) -> bool:
        """Forced replacement screen after a faint.

        The defining feature is the full party list down the left side (the same
        panel as team preview, but mid-battle and without the "send into battle"
        prompt). Normal battle screens only ever show the single active mon's HP,
        so >=2 stacked per-row ``cur/max`` fractions reliably marks the party
        screen. Reading whole-column at once loses the slash in OCR, so we read
        the always-present top rows individually. Team preview is checked earlier
        in ``perceive`` and won't reach here.
        """
        if not self.ocr_enabled:
            return False
        return len(self.read_party_slots(frame, max_slots=3)) >= 2

    def read_party_slots(self, frame: np.ndarray, *, max_slots: int = 6) -> list[dict[str, Any]]:
        """Per-slot party HP for force-switch selection (slot index is 1-based).

        Returns one entry per readable bench row with ``slot``, ``hp``,
        ``max_hp`` and ``alive``. Rows that don't OCR into a fraction are skipped.
        """
        slots: list[dict[str, Any]] = []
        if not self.ocr_enabled:
            return slots
        reader = self._get_ocr_reader()
        for slot in range(1, max(1, min(max_slots, 6)) + 1):
            crop = self._crop_region(frame, f"force_switch_hp_{slot}")
            if crop is None or crop.size == 0:
                continue
            text = read_text_lines(crop, reader) or ""
            match = _HP_FRACTION_RE.search(text)
            if not match:
                continue
            try:
                cur, mx = int(match.group(1)), int(match.group(2))
            except ValueError:
                continue
            slots.append(
                {"slot": slot, "hp": cur, "max_hp": mx, "alive": cur > 0, "hp_text": f"{cur}/{mx}"}
            )
        return slots

    def _detect_loading(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        mean_brightness = float(np.mean(gray))
        if mean_brightness < 12.0:
            return True
        indicator = self._crop_region(frame, "loading_indicator")
        if indicator is not None and float(np.mean(indicator)) < 20.0:
            return True
        return False

    def _infer_battle_format(self, ocr: dict[str, Any]) -> BattleFormatHint:
        has_b = bool(
            ocr.get("player_slot_b", {}).get("species_id")
            or ocr.get("player_slot_b", {}).get("name")
            or ocr.get("player_slot_b", {}).get("hp_text")
        )
        has_ob = bool(
            ocr.get("opp_slot_b", {}).get("species_id")
            or ocr.get("opp_slot_b", {}).get("name")
            or ocr.get("opp_slot_b", {}).get("hp_text")
        )
        if has_b or has_ob:
            return "doubles"
        has_a = bool(
            ocr.get("player_slot_a", {}).get("species_id")
            or ocr.get("player_slot_a", {}).get("name")
            or ocr.get("player_slot_a", {}).get("hp_text")
        )
        if has_a:
            return "singles"
        return "unknown"

    def get_current_state(self, frame: np.ndarray) -> GameState:
        return self.perceive(frame).state

    def perceive(self, frame: np.ndarray) -> PerceptionResult:
        matches: list[TemplateMatch] = []
        confidence_by_name: dict[str, float] = {}

        for name in _STATE_CHECK_ORDER:
            confidence = self.match_template(frame, name)
            state = _TEMPLATE_STATE_MAP.get(name, "UNKNOWN")
            matches.append(TemplateMatch(name=name, confidence=confidence, state=state))
            confidence_by_name[name] = confidence

            if confidence >= self.confidence_threshold:
                if (
                    name == "target_overlay_close"
                    and confidence < 0.88
                    and confidence_by_name.get("move_panel_anchor", 0.0) < 0.70
                ):
                    continue
                resolved_state = state
                for group_state, members in _COMPOSITE_STATE_GROUPS.items():
                    if name in members:
                        resolved_state = group_state
                        break
                ocr = (
                    self.extract_battle_data(frame)
                    if self._should_ocr_for_state(resolved_state)
                    else {}
                )
                return PerceptionResult(
                    state=resolved_state,
                    state_confidence=confidence,
                    battle_format=self._infer_battle_format(ocr),
                    template_match=name,
                    ocr=ocr,
                    raw_matches=matches,
                )

        for group_state, members in _COMPOSITE_STATE_GROUPS.items():
            best_member = max(members, key=lambda n: confidence_by_name.get(n, 0.0))
            best_conf = confidence_by_name.get(best_member, 0.0)
            if best_conf >= self.confidence_threshold:
                ocr = (
                    self.extract_battle_data(frame)
                    if self._should_ocr_for_state(group_state)
                    else {}
                )
                return PerceptionResult(
                    state=group_state,
                    state_confidence=best_conf,
                    battle_format=self._infer_battle_format(ocr),
                    template_match=best_member,
                    ocr=ocr,
                    raw_matches=matches,
                )

        # Team preview before the turn-decision heuristic: the preview screen has a
        # timer and weakly matches the Fight button, so it would otherwise be
        # misread as TURN_DECISION.
        if self._detect_team_preview(frame):
            return PerceptionResult(
                state="TEAM_PREVIEW",
                state_confidence=0.99,
                battle_format="unknown",
                template_match="teampreview_ocr",
                raw_matches=matches,
            )

        # Move list (Fight already tapped). Detect before the turn heuristic so a
        # lingering move list isn't mislabeled ANIMATION.
        if self._detect_move_selection(frame):
            ocr = self.extract_battle_data(frame)
            return PerceptionResult(
                state="MOVE_SELECTION",
                state_confidence=0.95,
                battle_format=self._infer_battle_format(ocr),
                template_match="move_select_ocr",
                ocr=ocr,
                raw_matches=matches,
            )

        # Forced replacement after a faint: the party list is open. Detect before
        # the turn heuristic so it isn't mislabeled ANIMATION (which left the bot
        # stuck, unable to pick a replacement).
        if self._detect_force_switch(frame):
            return PerceptionResult(
                state="FORCE_SWITCH",
                state_confidence=0.97,
                battle_format="unknown",
                template_match="force_switch_ocr",
                raw_matches=matches,
            )

        if self._detect_turn_decision(frame, confidence_by_name):
            ocr = self.extract_battle_data(frame)
            best_member = max(
                _COMPOSITE_STATE_GROUPS["TURN_DECISION"],
                key=lambda n: confidence_by_name.get(n, 0.0),
            )
            return PerceptionResult(
                state="TURN_DECISION",
                state_confidence=confidence_by_name.get(best_member, 0.0),
                battle_format=self._infer_battle_format(ocr),
                template_match=f"{best_member}_heuristic",
                ocr=ocr,
                raw_matches=matches,
            )

        best = max(matches, key=lambda m: m.confidence) if matches else None

        if self._detect_loading(frame):
            return PerceptionResult(
                state="LOADING",
                state_confidence=1.0,
                battle_format="unknown",
                template_match="brightness",
                raw_matches=matches,
            )

        ocr = self.extract_battle_data(frame)
        battle_format = self._infer_battle_format(ocr)
        if ocr and any(slot.get("hp_text") for slot in ocr.values() if isinstance(slot, dict)):
            return PerceptionResult(
                state="ANIMATION",
                state_confidence=0.5,
                battle_format=battle_format,
                template_match="hp_presence",
                ocr=ocr,
                raw_matches=matches,
            )

        return PerceptionResult(
            state="UNKNOWN",
            state_confidence=best.confidence if best else 0.0,
            battle_format=battle_format,
            template_match=best.name if best else "",
            ocr=ocr,
            raw_matches=matches,
        )

    @staticmethod
    def _should_ocr_for_state(state: GameState) -> bool:
        return state in {
            "TURN_DECISION",
            "MOVE_SELECTION",
            "TARGET_SELECTION",
            "ANIMATION",
            "TEAM_PREVIEW",
        }

    def preprocess_for_ocr(self, crop: np.ndarray) -> np.ndarray:
        """Grayscale + light blur; avoids color-background bias before OCR."""
        if crop.ndim == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()
        scale = 2 if max(gray.shape[:2]) < 120 else 1
        if scale > 1:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return gray

    def read_battle_log(self, frame: np.ndarray) -> str | None:
        """OCR the bottom battle-log text box (move/stat/faint/weather messages).

        Returns the recognized text, or ``None`` when OCR is disabled or the box
        is empty. Uses the ocr_utils whiteness pipeline (white text on busy bg).
        """
        if not self.ocr_enabled:
            return None
        crop = self._crop_region(frame, "battle_action_log")
        if crop is None or crop.size == 0:
            return None
        text = read_text_lines(crop, self._get_ocr_reader())
        return text or None

    def read_ability_item_popups(self, frame: np.ndarray) -> list[str]:
        """OCR the left/right mid-screen ability & item activation banners.

        These appear separately from the bottom log box, e.g. "Volcarona's
        Leftovers" (item) or "Garchomp's Rough Skin" (ability). Returns the
        non-empty banner texts (0-2 of them).
        """
        if not self.ocr_enabled:
            return []
        texts: list[str] = []
        for key in ("ability_item_popup_left", "ability_item_popup_right"):
            crop = self._crop_region(frame, key)
            if crop is None or crop.size == 0:
                continue
            text = read_text_lines(crop, self._get_ocr_reader())
            if text:
                texts.append(text)
        return texts

    def _ocr_crop(self, crop: np.ndarray) -> str:
        if not self.ocr_enabled or crop.size == 0:
            return ""
        gray = self.preprocess_for_ocr(crop)
        reader = self._get_ocr_reader()
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        lines = reader.readtext(rgb, detail=0, paragraph=True)
        return " ".join(str(line) for line in lines).strip()

    def _read_ally_hp(self, hp_crop: np.ndarray | None, known_max: int | None) -> dict[str, Any]:
        """Ally slots show ``current/max`` numerals; parse the exact fraction."""
        if hp_crop is None or hp_crop.size == 0 or not self.ocr_enabled:
            return {"hp": None, "max_hp": None, "hp_percent": None, "hp_text": ""}
        parsed = parse_hp_text(hp_crop, self._get_ocr_reader(), known_max=known_max)
        if parsed is None:
            return {"hp": None, "max_hp": None, "hp_percent": None, "hp_text": ""}
        cur, mx = parsed
        pct = (100.0 * cur / mx) if mx > 0 else None
        return {"hp": cur, "max_hp": mx, "hp_percent": pct, "hp_text": f"{cur}/{mx}"}

    def _read_enemy_hp(self, hp_crop: np.ndarray | None) -> dict[str, Any]:
        """Enemy slots show a ``NN%`` readout; fall back to bar colour masking."""
        if hp_crop is None or hp_crop.size == 0:
            return {"hp": None, "max_hp": None, "hp_percent": None, "hp_text": ""}
        pct = parse_hp_percent(hp_crop, self._get_ocr_reader()) if self.ocr_enabled else None
        if pct is not None:
            return {"hp": None, "max_hp": None, "hp_percent": pct, "hp_text": f"{int(pct)}%"}
        bar_pct = get_hp_percentage_from_bar(hp_crop) * 100.0
        return {
            "hp": None,
            "max_hp": None,
            "hp_percent": bar_pct,
            "hp_text": f"~{int(round(bar_pct))}%",
        }

    def _extract_slot(
        self,
        frame: np.ndarray,
        hp_key: str,
        sprite_key: str,
        *,
        is_enemy: bool = False,
        known_max: int | None = None,
    ) -> dict[str, Any]:
        hp_crop = self._crop_region(frame, hp_key)
        sprite_crop = self._crop_region(frame, sprite_key)

        if is_enemy:
            hp = self._read_enemy_hp(hp_crop)
        else:
            hp = self._read_ally_hp(hp_crop, known_max)

        species_id = self._identify_sprite_crop(sprite_crop)

        return {
            "species_id": species_id,
            "hp_text": hp["hp_text"],
            "hp": hp["hp"],
            "max_hp": hp["max_hp"],
            "hp_percent": hp["hp_percent"],
        }

    def extract_battle_data(self, frame: np.ndarray) -> dict[str, Any]:
        """Read active battle slots: species via sprite match, HP via OCR."""
        data: dict[str, Any] = {
            "player_slot_a": self._extract_slot(
                frame, "player_active_hp_slot_a", "player_active_sprite_slot_a"
            ),
            "player_slot_b": self._extract_slot(
                frame, "player_active_hp_slot_b", "player_active_sprite_slot_b"
            ),
            "opp_slot_a": self._extract_slot(
                frame, "opp_active_hp_slot_a", "opp_active_sprite_slot_a", is_enemy=True
            ),
            "opp_slot_b": self._extract_slot(
                frame, "opp_active_hp_slot_b", "opp_active_sprite_slot_b", is_enemy=True
            ),
        }

        preview_counter = self._crop_region(frame, "teampreview_selection_counter")
        if preview_counter is not None:
            text = self._ocr_crop(preview_counter)
            pick_match = re.search(r"(\d+)\s*/\s*(\d+)", text)
            if pick_match:
                data["teampreview"] = {
                    "selected": int(pick_match.group(1)),
                    "required": int(pick_match.group(2)),
                    "text": text,
                }

        move_timer = self._crop_region(frame, "decision_state_move_timer")
        if move_timer is not None:
            timer_text = self._ocr_crop(move_timer)
            timer_match = re.search(r"(\d+)", timer_text)
            if timer_match:
                data["move_timer"] = int(timer_match.group(1))

        return data
