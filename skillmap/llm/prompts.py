"""All prompt templates. Use Python .format() placeholders."""

# ---------------------------------------------------------------------------
# Stage A – Correction Summarizer
# ---------------------------------------------------------------------------
# Placeholders: {conversation}
CORRECTION_SUMMARIZER_PROMPT = """\
You extract behavioral correction summaries from a conversation between a user and a coding assistant.

A correction is a user turn where they redirect, correct, or supplement the assistant's response.
For each correction, extract one summary. Return 0 to 5 summaries.

Rules:
- Abstract away ALL task-specific details: no problem names, variable names, code fragments, \
or specific numbers.
- Each summary must describe a FAMILY of situations, not just this one task.
- correction_quote: verbatim excerpt from the user correction turn, max 30 words.

Return ONLY a JSON object:
{{
  "summaries": [
    {{
      "triggering_situation": "when solving an algorithmic problem from scratch",
      "what_was_wrong": "assistant jumped directly to code without analyzing complexity first",
      "what_user_wanted": "state time and space complexity before writing any code",
      "correction_quote": "Hold on—state the exact time and space complexity first"
    }}
  ]
}}

Conversation:
{conversation}
"""


# ---------------------------------------------------------------------------
# Stage B Step 1 – Candidate Extraction
# ---------------------------------------------------------------------------
# Placeholders: {summaries}, {min_support}
SKILL_CANDIDATE_EXTRACTION_PROMPT = """\
You consolidate behavioral correction patterns into reusable skills.

A skill is valid ONLY if at least {min_support} summaries in the list show the SAME behavior pattern.
Each skill must pass this test: "What concrete assistant mistake does this prevent?"
Do NOT create skills tied to a specific task.

Return ONLY a JSON object:
{{
  "candidates": [
    {{
      "title": "State Complexity Before Writing Code",
      "catalog_trigger": "when solving any algorithmic or implementation problem",
      "guidance": "Before writing code, state time and space complexity in Big-O with a one-sentence justification.",
      "supporting_summary_ids": ["id1", "id3", "id7"]
    }}
  ]
}}

Each supporting_summary_id MUST be an exact ID from the summaries below.

Correction summaries:
{summaries}
"""


# ---------------------------------------------------------------------------
# Stage B Step 1b – Intra-batch deduplication
# ---------------------------------------------------------------------------
# Placeholders: {candidates}
SKILL_DEDUP_PROMPT = """\
You are given a list of candidate behavioral skills extracted from the same set of corrections.
Some candidates may address essentially the same assistant mistake — just worded differently.

Merge near-duplicate candidates into one richer skill. Two candidates are near-duplicates if:
- Their guidance would prevent the SAME concrete assistant mistake, OR
- One is a narrower special case of the other.

For each merged group: write the best unified title, trigger, and guidance.
Include ALL candidates, whether merged or kept alone.

Return ONLY a JSON object:
{{
  "merged": [
    {{
      "title": "...",
      "catalog_trigger": "...",
      "guidance": "...",
      "source_indices": [0, 2]
    }}
  ]
}}

Each integer in source_indices refers to an index in the input list below. Every input index
must appear in exactly one source_indices list.

Candidates (indexed from 0):
{candidates}
"""


# ---------------------------------------------------------------------------
# Stage B Step 2 – Reconciliation
# ---------------------------------------------------------------------------
# Placeholders: {existing_catalog}, {proposed_skills}
SKILL_RECONCILIATION_PROMPT = """\
You reconcile newly proposed skills against an existing skill library.
DEFAULT BIAS: discard unless there is a clear reason to add or update.

For each proposed skill, choose exactly one action:
- "discard"  : addresses the SAME behavior as an existing skill — even if titled or worded
               differently — OR adds only minor detail not worth a separate entry
- "update"   : overlaps with an existing skill AND contributes meaningfully different guidance;
               provide existing_skill_id and updated_guidance (merged best-of-both version)
- "replace"  : CONTRADICTS an existing skill (proposes opposite behavior for the same
               situation); provide existing_skill_id and updated_guidance (coherent synthesis)
- "add"      : covers genuinely new behavior not addressed by ANY existing skill

Two skills address the same behavior if their guidance would prevent the same concrete
assistant mistake — regardless of how their titles or triggers are phrased.

Return ONLY a JSON object:
{{
  "decisions": [
    {{
      "proposed_index": 0,
      "action": "discard",
      "existing_skill_id": "uuid-of-existing-or-null",
      "updated_guidance": "merged guidance text or null"
    }}
  ]
}}

Existing skills (id | title | trigger | guidance):
{existing_catalog}

Proposed skills (indexed from 0):
{proposed_skills}
"""


# ---------------------------------------------------------------------------
# Stage B Step 3 – Catalog Compaction  (run when catalog exceeds threshold)
# ---------------------------------------------------------------------------
# Placeholders: {skills}
SKILL_COMPACTION_PROMPT = """\
You are cleaning up a behavioral skill library for an AI coding assistant.
The library may contain duplicate, overlapping, or contradictory skills.

Group skills that address the same core behavior. For each group:
- Pick the skill with the highest support_count as the primary (use its id as keep_id)
- Write a unified title, catalog_trigger, and guidance that covers the whole group
- List the ids of all OTHER skills in the group as discard_ids

Skills that are genuinely distinct should appear as singleton groups (discard_ids = []).

Rules:
- Every skill id must appear exactly once across all keep_id and discard_ids fields combined.
- If two skills contradict each other, resolve the contradiction in the merged guidance.
- Prefer broader, reusable guidance over task-specific wording.

Return ONLY a JSON object:
{{
  "groups": [
    {{
      "keep_id": "uuid-of-primary-skill",
      "title": "unified title",
      "catalog_trigger": "unified catalog trigger",
      "guidance": "unified guidance",
      "discard_ids": ["uuid-of-absorbed-skill", ...]
    }}
  ]
}}

Current skill library:
{skills}
"""


# ---------------------------------------------------------------------------
# Stage C – Catalog Selector
# ---------------------------------------------------------------------------
# Placeholders: {task}, {catalog}
CATALOG_SELECTOR_PROMPT = """\
You select behavioral guidelines for a coding assistant to apply to the current task.

Select AT MOST 2 guidelines that are clearly relevant to this task.
Only select a guideline if its trigger directly matches the situation described in the task.
If none apply, return an empty list.

Return ONLY a JSON object:
{{"selected_ids": ["id1", "id2"]}}

Task:
{task}

Available guidelines:
{catalog}
"""


# ---------------------------------------------------------------------------
# Runtime – Constraint injection into LLM-B system prompt
# ---------------------------------------------------------------------------
# Placeholders: {skills_block}
CONSTRAINT_INJECTION_TEMPLATE = """\
Before responding, apply these behavioral guidelines learned from prior interactions \
with this user:

{skills_block}

Follow each guideline that applies to the current task. Do not mention the guidelines \
to the user unless asked.
"""
