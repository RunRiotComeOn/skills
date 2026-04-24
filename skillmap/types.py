"""Pydantic data models for SkillMap (flat experience library design).

Two-axis skill model
--------------------
Every CorrectionSummary and Skill carries an `axis` tag that splits user
feedback into two distinct learning channels:

  preference  — user pushed back on style/format/approach. The user IS the
                ground truth; no external verification possible. Extracted
                skills should reduce user-interruption count over time.

  correctness — user caught a real bug (failed test, edge-case crash, wrong
                output). Eligible for extraction ONLY when the conversation
                contains evidence the correction was right (assistant
                adopted the fix, tests later passed, user accepted the
                revision) AND the underlying mistake generalizes to a
                FAMILY of bugs (e.g. "ignores empty input" — not "wrong
                formula for THIS specific problem"). Extracted skills
                should lift test-case pass rate over time.

Skills of different axes are NEVER merged together; reconciliation and
compaction operate within an axis.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


SkillAxis = Literal["preference", "correctness"]


class CorrectionSummary(BaseModel):
    """Stage A output. One record per correction extracted from a task."""
    id: str
    task_id: str
    created_at: datetime
    triggering_situation: str   # abstracted context, no task-specific details
    what_was_wrong: str         # what the assistant did on first attempt
    what_user_wanted: str       # the positive desired behavior
    correction_quote: str       # short verbatim excerpt from user correction
    # See module docstring for the axis semantics.
    correction_type: SkillAxis = "preference"
    # For correctness summaries only: short note on what makes this correction
    # verifiable in the trajectory (e.g. "assistant adopted fix on next turn",
    # "subsequent test_results section showed pass"). Empty for preference
    # summaries. Required to gate correctness extraction.
    verification_evidence: str = ""


class Skill(BaseModel):
    """Stage B output. Flat — no parent/child relationships."""
    id: str
    title: str                          # ≤ 10 words
    catalog_trigger: str                # one sentence: "when X"
    guidance: str                       # concrete do/don't instruction
    # Which user-feedback channel produced this skill. Determines:
    #   • which buffer it consolidates from (Stage B, axis-segregated)
    #   • which catalog the selector pulls from at task start
    #   • which eval metric it is expected to move
    axis: SkillAxis = "preference"
    support_count: int
    supporting_summary_ids: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class CatalogEntry(BaseModel):
    """Lightweight index entry shown to the Stage C selector LLM."""
    id: str
    title: str
    catalog_trigger: str
    axis: SkillAxis = "preference"


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
