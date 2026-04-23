"""EpisodeRecorder: package a completed trajectory into an Episode.

Does NOT detect corrections - that is CorrectionDetector's job. Leaves
`correction_points` empty; they are filled in during finalize.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from skillmap.types import Episode, EpisodeOutcome, Turn


class EpisodeRecorder:
    def run(
        self,
        trajectory: list[Turn],
        task_category: str,
        outcome: EpisodeOutcome = "success",
        retrieved_skills_at_start: list[str] | None = None,
    ) -> Episode:
        return Episode(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            task_category=task_category,
            trajectory=trajectory,
            correction_points=[],
            outcome=outcome,
            retrieved_skills_at_start=retrieved_skills_at_start or [],
        )
