"""JSON file persistence for SkillMapState. No database in v0."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from skillmap.types import SkillMapState


class PersistenceBackend(Protocol):
    """Minimal contract: load returns current state, save writes it."""

    def load(self) -> SkillMapState: ...
    def save(self, state: SkillMapState) -> None: ...


class JSONPersistence:
    """Single-file JSON backend.

    On load: if the file does not exist, returns a fresh empty state for
    the configured user_id. On save: atomically overwrites the file.
    """

    def __init__(self, path: str | Path, user_id: str) -> None:
        self.path = Path(path)
        self.user_id = user_id

    def load(self) -> SkillMapState:
        if not self.path.exists():
            return SkillMapState(user_id=self.user_id)
        raw = self.path.read_text(encoding="utf-8")
        return SkillMapState.model_validate_json(raw)

    def save(self, state: SkillMapState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(self.path)
