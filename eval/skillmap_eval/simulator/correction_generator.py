"""Utilities for rendering preference-violation corrections.

The actual wording is LLM-generated (see user_simulator.py). This module
holds pure helpers: rendering the profile + conversation for the prompt
and formatting the expected-correction-trigger as a style guide.
"""

from __future__ import annotations

import random

from skillmap_eval.types import Preference, PreferenceProfile, SimulatedTurn

# Style variation pool — one hint is chosen per turn to nudge phrasing variety
_STYLE_POOL = [
    "Tone hint: be brief and blunt this turn — one short sentence.\n\n",
    "Tone hint: be mildly frustrated, as if this keeps coming up.\n\n",
    "Tone hint: phrase your correction as a question (e.g. 'Can you...?' / 'What about...?').\n\n",
    "Tone hint: be polite and matter-of-fact this turn.\n\n",
    "Tone hint: start with an informal acknowledgment then redirect (e.g. 'Ok but...' / 'Sure, though...').\n\n",
    "Tone hint: be very terse — two words or a sentence fragment is fine.\n\n",
    "Tone hint: be explicit and detailed about exactly what you want changed.\n\n",
    "Tone hint: sound mildly confused rather than directive (e.g. 'I thought I mentioned...').\n\n",
    "Tone hint: be friendly but firm, as if gently reminding a colleague.\n\n",
    "Tone hint: express mild exasperation without being rude.\n\n",
]


def render_profile(profile: PreferenceProfile) -> str:
    lines: list[str] = []
    for p in sorted(profile.preferences, key=lambda x: x.priority):
        lines.append(
            f"- [{p.id}] ({p.category}, priority {p.priority}) {p.description}\n"
            f"    example phrasing: {p.expected_correction_trigger}"
        )
    return "\n".join(lines)


def render_conversation(turns: list[SimulatedTurn]) -> str:
    if not turns:
        return "(no messages yet)"
    return "\n".join(f"{t.role.upper()}: {t.content}" for t in turns)


def sample_correction_style_hint(rng: random.Random | None = None) -> str:
    """Return a random per-turn tone nudge to vary correction phrasing."""
    pool = rng or random
    return pool.choice(_STYLE_POOL)


def count_pref_violations(turns: list[SimulatedTurn]) -> dict[str, int]:
    """How many times each pref_id has appeared in user turns so far."""
    tally: dict[str, int] = {}
    for t in turns:
        if t.role != "user":
            continue
        for pid in t.triggered_preferences:
            tally[pid] = tally.get(pid, 0) + 1
    return tally
