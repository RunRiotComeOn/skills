"""Pydantic data models for SkillMap (flat experience library design)."""

from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


class CorrectionSummary(BaseModel):
    """Stage A output. One record per correction extracted from a task."""
    id: str
    task_id: str
    created_at: datetime
    triggering_situation: str   # abstracted context, no task-specific details
    what_was_wrong: str         # what the assistant did on first attempt
    what_user_wanted: str       # the positive desired behavior
    correction_quote: str       # short verbatim excerpt from user correction


class Skill(BaseModel):
    """Stage B output. Flat — no parent/child relationships."""
    id: str
    title: str                          # ≤ 10 words
    catalog_trigger: str                # one sentence: "when X"
    guidance: str                       # concrete do/don't instruction
    support_count: int
    supporting_summary_ids: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class CatalogEntry(BaseModel):
    """Lightweight index entry shown to the Stage C selector LLM."""
    id: str
    title: str
    catalog_trigger: str


class SkillMapState(BaseModel):
    """Top-level persistent state."""
    user_id: str
    skills: dict[str, Skill] = Field(default_factory=dict)
    summary_buffer: list[CorrectionSummary] = Field(default_factory=list)
    catalog: list[CatalogEntry] = Field(default_factory=list)


class SkillMapError(Exception):
    pass


class UnknownSkillError(SkillMapError):
    pass
