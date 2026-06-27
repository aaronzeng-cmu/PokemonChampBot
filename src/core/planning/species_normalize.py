"""Map Showdown battle species names to Pikalytics lookup keys."""

from __future__ import annotations

import re

from poke_env.data import to_id_str

_GENDER_SUFFIX_RE = re.compile(r"\s+\([MF]\)$")

# item id -> Pikalytics species slug (Champions megas / formes)
_ITEM_SPECIES: dict[str, str] = {
    "charizarditey": "Charizard-Mega-Y",
    "charizarditex": "Charizard-Mega-X",
    "venusaurite": "Venusaur-Mega",
    "blastoisinite": "Blastoise-Mega",
    "gengarite": "Gengar-Mega",
    "aerodactylite": "Aerodactyl-Mega",
    "kangaskhanite": "Kangaskhan-Mega",
    "tyranitarite": "Tyranitar-Mega",
    "scizorite": "Scizor-Mega",
    "gardevoirite": "Gardevoir-Mega",
    "lopunnite": "Lopunny-Mega",
    "dragoninite": "Dragonite-Mega",
    "froslassite": "Froslass-Mega",
    "scovillainite": "Scovillain-Mega",
    "floettite": "Floette-Mega",
    "meganiumite": "Meganium-Mega",
    "delphoxite": "Delphox-Mega",
    "glimmoranite": "Glimmora-Mega",
    "excadrillite": "Excadrill",  # Excadrite item, species stays Excadrill
    "drampanite": "Drampa-Mega",
    "cameruptite": "Camerupt-Mega",
    "skarmorite": "Skarmory-Mega",
    "steelixite": "Steelix-Mega",
    "abomasite": "Abomasnow-Mega",
    "chandelurite": "Chandelure-Mega",
}

# Pikalytics mega form -> held item display name (for belief priors)
_MEGA_FORM_ITEMS: dict[str, str] = {
    "Charizard-Mega-Y": "Charizardite Y",
    "Charizard-Mega-X": "Charizardite X",
    "Venusaur-Mega": "Venusaurite",
    "Blastoise-Mega": "Blastoisinite",
    "Gengar-Mega": "Gengarite",
    "Aerodactyl-Mega": "Aerodactylite",
    "Kangaskhan-Mega": "Kangaskhanite",
    "Tyranitar-Mega": "Tyranitarite",
    "Scizor-Mega": "Scizorite",
    "Gardevoir-Mega": "Gardevoirite",
    "Lopunny-Mega": "Lopunnite",
    "Dragonite-Mega": "Dragoninite",
    "Froslass-Mega": "Froslassite",
    "Scovillain-Mega": "Scovillainite",
    "Floette-Mega": "Floettite",
    "Meganium-Mega": "Meganiumite",
    "Delphox-Mega": "Delphoxite",
    "Glimmora-Mega": "Glimmoranite",
    "Drampa-Mega": "Drampanite",
    "Camerupt-Mega": "Cameruptite",
    "Skarmory-Mega": "Skarmorite",
    "Steelix-Mega": "Steelixite",
    "Abomasnow-Mega": "Abomasite",
    "Chandelure-Mega": "Chandelurite",
}

_MEGA_FORM_RE = re.compile(r"-Mega(?:-[YX])?$")


def _mega_form_base(form: str) -> str:
    return _MEGA_FORM_RE.sub("", form)


# base Showdown species -> preferred Pikalytics slug(s)
_BASE_ALIASES: dict[str, list[str]] = {
    "Charizard": ["Charizard-Mega-Y", "Charizard-Mega-X", "Charizard"],
    "Venusaur": ["Venusaur-Mega", "Venusaur"],
    "Blastoise": ["Blastoise-Mega", "Blastoise"],
    "Garchomp": ["Garchomp"],
    "Basculegion": ["Basculegion"],
    "Floette-Eternal": ["Floette-Eternal", "Floette-Mega"],
    "Rotom": ["Rotom-Wash", "Rotom-Heat", "Rotom-Frost", "Rotom-Mow", "Rotom-Fan"],
    "Indeedee": ["Indeedee", "Indeedee-F"],
    "Ogerpon": ["Ogerpon", "Ogerpon-Wellspring", "Ogerpon-Hearthflame", "Ogerpon-Cornerstone"],
    "Urshifu": ["Urshifu-Rapid-Strike", "Urshifu"],
    "Tauros": ["Tauros-Paldea-Combat", "Tauros-Paldea-Blaze", "Tauros-Paldea-Aqua"],
    "Sinistcha-Masterpiece": ["Sinistcha"],
    "Maushold-Four": ["Maushold"],
    "Slowbro": ["Slowbro-Galar", "Slowbro"],
    "Mr. Rime": ["Mr-Rime", "Mr. Rime"],
    "Meowstic-F": ["Meowstic", "Meowstic-F"],
    "Meowstic-M": ["Meowstic", "Meowstic-M"],
    "Lycanroc-Midnight": ["Lycanroc", "Lycanroc-Midnight"],
    "Lycanroc-Dusk": ["Lycanroc", "Lycanroc-Dusk"],
    "Polteageist-Antique": ["Polteageist", "Polteageist-Antique"],
    "Vivillon-Fancy": ["Vivillon", "Vivillon-Fancy"],
    "Vivillon-Pokeball": ["Vivillon", "Vivillon-Pokeball"],
}


def _build_mega_families() -> dict[str, list[str]]:
    families: dict[str, list[str]] = {}
    for form in set(_ITEM_SPECIES.values()):
        if "-Mega" in form:
            base = _mega_form_base(form)
            families.setdefault(base, [])
            if form not in families[base]:
                families[base].append(form)
    for base, aliases in _BASE_ALIASES.items():
        for alias in aliases:
            if "-Mega" in alias and alias not in families.get(base, []):
                families.setdefault(base, []).append(alias)
    return families


_MEGA_FAMILIES: dict[str, list[str]] = _build_mega_families()


def clean_species_name(species: str) -> str:
    """Strip gender suffixes and whitespace from paste/battle names."""
    text = (species or "").strip()
    text = _GENDER_SUFFIX_RE.sub("", text)
    return text.strip()


def pikalytics_slug(species: str) -> str:
    """URL-safe Pikalytics species slug (no spaces)."""
    return clean_species_name(species).replace(" ", "-")


def mega_family_keys(species: str) -> list[str] | None:
    """Ordered Pikalytics keys for a mega-capable family, else None."""
    name = clean_species_name(species)
    for base, forms in _MEGA_FAMILIES.items():
        if name == base or name in forms:
            ordered: list[str] = []
            if base in _BASE_ALIASES:
                for alias in _BASE_ALIASES[base]:
                    if alias not in ordered:
                        ordered.append(alias)
            else:
                ordered.extend(forms)
                if base not in ordered:
                    ordered.append(base)
            return ordered
    return None


def opponent_belief_key(mon) -> str:
    """Stable belief / snapshot key for an opponent Pokemon."""
    if mon is None:
        return ""
    return str(getattr(mon, "species", None) or getattr(mon, "base_species", None) or "")


def _showdown_ids_for_form(form: str) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        key = to_id_str(candidate)
        if key and key not in seen:
            seen.add(key)
            ids.append(key)

    add(form)
    if "-Mega-Y" in form:
        add(form.split("-Mega-Y")[0] + "-Mega-Y")
        add(to_id_str(form.split("-Mega-Y")[0]) + "megay")
    elif "-Mega-X" in form:
        add(form.split("-Mega-X")[0] + "-Mega-X")
        add(to_id_str(form.split("-Mega-X")[0]) + "megax")
    elif form.endswith("-Mega"):
        base = form[: -len("-Mega")]
        add(base + "-Mega")
        add(to_id_str(base) + "mega")
    return ids


def mega_form_for_ability(base_species: str, ability: str) -> str | None:
    """Map observed mega ability to a Pikalytics mega form slug."""
    from poke_env.data import GenData

    ability_id = to_id_str(ability or "")
    if not ability_id:
        return None
    for form in mega_family_keys(base_species) or []:
        if "-Mega" not in form:
            continue
        for dex_id in _showdown_ids_for_form(form):
            try:
                entry = GenData.from_gen(9).pokedex[dex_id]
            except KeyError:
                continue
            if to_id_str(entry["abilities"]["0"]) == ability_id:
                return form
    return None


def _parse_mega_form_from_details(details: str) -> str | None:
    head = (details or "").split(",")[0].strip()
    if "Mega" not in head:
        return None
    return head.replace(" ", "-")


def infer_mega_stone(mon) -> str:
    """Infer held mega stone display name from battle state."""
    if mon is None:
        return ""
    base = clean_species_name(getattr(mon, "base_species", None) or mon.species or "")
    item = str(getattr(mon, "item", "") or "")
    if item:
        item_id = to_id_str(item)
        if item_id in _ITEM_SPECIES:
            return _MEGA_FORM_ITEMS.get(_ITEM_SPECIES[item_id], item)
        for stone in _MEGA_FORM_ITEMS.values():
            if to_id_str(stone) == item_id:
                return stone
        if item_id.endswith("ite") or item_id.endswith("itey") or item_id.endswith("itez"):
            return item

    details = getattr(mon, "_last_details", "") or ""
    form = _parse_mega_form_from_details(details)
    if not form:
        form = mega_form_for_ability(base, str(mon.ability or ""))
    if form:
        return _MEGA_FORM_ITEMS.get(form, "")
    return ""


def is_mega_evolved(mon) -> bool:
    """True when a mega-capable Pokemon has activated its mega form."""
    if mon is None:
        return False
    if not getattr(mon, "forme_change_ability", None):
        return False
    base = clean_species_name(getattr(mon, "base_species", None) or mon.species or "")
    return mega_family_keys(base) is not None


def expand_mega_form_targets(names: list[str]) -> list[str]:
    """Add mega form pages for any mega-capable base in the fetch list."""
    seen: set[str] = set()
    ordered: list[str] = []

    def add(name: str) -> None:
        clean = clean_species_name(name)
        if clean and clean not in seen:
            seen.add(clean)
            ordered.append(clean)

    for name in names:
        add(name)
        family = mega_family_keys(name)
        if family:
            for form in family:
                add(form)
    return ordered


def resolve_pikalytics_candidates(
    species: str,
    *,
    item: str = "",
    ability: str = "",
) -> list[str]:
    """Return Pikalytics species keys to try, best match first."""
    base = clean_species_name(species)
    candidates: list[str] = []

    item_id = to_id_str(item or "")
    item_mapped = _ITEM_SPECIES.get(item_id)
    if item_mapped:
        candidates.append(item_mapped)
    if item_id.endswith("ite") or item_id.endswith("itey") or item_id.endswith("itez"):
        # Generic mega stone: try Base-Mega patterns
        for suffix in ("-Mega", "-Mega-Y", "-Mega-X"):
            candidates.append(f"{base}{suffix}")

    if base in _BASE_ALIASES:
        for alias in _BASE_ALIASES[base]:
            if item_mapped and alias == base and alias != item_mapped:
                continue
            candidates.append(alias)
    elif not item_mapped:
        candidates.append(base)

    # Formes written with parenthetical gender on paste lines
    if base.endswith("-Eternal"):
        candidates.append("Floette-Eternal")

    seen: set[str] = set()
    ordered: list[str] = []
    for name in candidates:
        if name and name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered
