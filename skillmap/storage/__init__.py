"""Storage layer: SkillMap core component and JSON persistence."""

from skillmap.storage.skill_map import SkillMap
from skillmap.storage.persistence import PersistenceBackend, JSONPersistence

__all__ = ["SkillMap", "PersistenceBackend", "JSONPersistence"]
