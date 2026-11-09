"""Load config/sources.yaml into the registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from sfrag.storage import repo
from sfrag.storage.db import Database


def load_source_configs(path: Path) -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(path.read_text()) or {}
    return {k: dict(v) for k, v in (data.get("sources") or {}).items()}


def sync_sources(db: Database, path: Path) -> list[str]:
    configs = load_source_configs(path)
    for sid, cfg in configs.items():
        repo.upsert_source(db, sid, cfg)
    return sorted(configs)
