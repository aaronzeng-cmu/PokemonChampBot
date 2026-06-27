"""Load format-specific MetaDatabase instances."""

from __future__ import annotations

from config.settings import PIKALYTICS_META_PATH, SINGLES_META_DATABASE_PATH
from src.doubles.planning.meta_database import MetaDatabase


def load_meta_database(*, format: str = "doubles", live_fetch: bool = False) -> MetaDatabase:
    if format == "singles":
        return MetaDatabase(
            pikalytics_path=SINGLES_META_DATABASE_PATH,
            live_fetch=live_fetch,
        )
    return MetaDatabase(pikalytics_path=PIKALYTICS_META_PATH, live_fetch=live_fetch)
