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
# Stage B Step 2 – Reconciliation
# ---------------------------------------------------------------------------
# Placeholders: {existing_catalog}, {proposed_skills}
SKILL_RECONCILIATION_PROMPT = """\
You reconcile newly proposed skills against an existing skill library.

For each proposed skill, choose exactly one action:
- "discard" : essentially the same behavior and trigger as an existing skill
- "update"  : overlaps with an existing skill but adds specificity or refined guidance
              — provide existing_skill_id and updated_guidance (merged version)
- "add"     : no meaningful overlap with any existing skill — leave existing_skill_id and \
updated_guidance as null

Return ONLY a JSON object:
{{
  "decisions": [
    {{
      "proposed_index": 0,
      "action": "discard",
      "existing_skill_id": "uuid-of-existing-or-null",
      "updated_guidance": "refined guidance text or null"
    }}
  ]
}}

Existing skills (id | title | trigger):
{existing_catalog}

Proposed skills (indexed from 0):
{proposed_skills}
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
