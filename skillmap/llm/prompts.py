"""All prompt templates. Use Python .format() placeholders.

Two-axis skill model
====================
SkillMap learns from two distinct kinds of user feedback. The prompts in
this file enforce that distinction at every stage:

  preference  — user pushed back on style / format / approach. The user is
                the ground truth; success metric is "fewer interruptions".
                Always eligible for extraction once the same pattern repeats.

  correctness — user caught a real bug (failed test, edge-case crash,
                wrong output). Success metric is "test-case pass rate
                lifts over the stream". Eligible for extraction ONLY when:
                  (V) the conversation contains evidence the correction
                      was actually correct (assistant adopted the fix on
                      the next turn, the next test_results section showed
                      a pass that previously failed, or the user accepted
                      the revised code), AND
                  (G) the underlying mistake generalizes — i.e. it names a
                      family of bugs ("forgets empty input", "off-by-one
                      in sliding window", "ignores constraint magnitude
                      when picking algorithm"), not a one-off algorithmic
                      slip on this specific problem.

Skills of different axes are extracted, reconciled, and compacted in
separate buckets. They are NEVER merged into one another.
"""

# ---------------------------------------------------------------------------
# Stage A – Correction Summarizer
# ---------------------------------------------------------------------------
# Placeholders: {conversation}
CORRECTION_SUMMARIZER_PROMPT = """\
You extract behavioral correction summaries from a conversation between a user
and a coding assistant. A correction is a user turn where they redirect,
correct, or supplement the assistant's response.

Return 0 to 5 summaries. Each summary belongs to exactly ONE axis:

──────────────────────────────────────────────────────────────────────────────
AXIS 1 — "preference"
──────────────────────────────────────────────────────────────────────────────
The user objected to STYLE / FORMAT / APPROACH / COMMUNICATION HABIT. There
is no objective "right answer" — the user IS the ground truth.
  Examples: "state complexity first", "no recursion", "type-annotate the
  signature", "stop apologizing", "show the test you ran, not just the code".

Always eligible for extraction (the consolidator's MIN_SUPPORT gate handles
the "is this a real pattern" question downstream).

Set verification_evidence = "" for preference summaries.

──────────────────────────────────────────────────────────────────────────────
AXIS 2 — "correctness"
──────────────────────────────────────────────────────────────────────────────
The user caught the assistant producing WRONG CODE — a failed test, a
crash on an edge case, a wrong output, an obviously wrong formula.

Extract a correctness summary ONLY when BOTH of these hold:

  (V) Verification — the conversation contains EVIDENCE that the user's
      correction was actually right. Acceptable evidence:
        • a test_results section after the assistant's revised code shows
          a case that previously failed now passing
        • the assistant adopted the fix on the next turn without disputing it
        • the user accepted the revised code at the end of the trajectory
        • the assistant's reply explicitly agrees with the bug diagnosis
      If none of these is present (e.g. the user objected but the assistant
      pushed back, or the trajectory ended in give_up), DO NOT extract.

  (G) Generalizable lesson — the underlying mistake names a FAMILY of bugs,
      not the algorithmic punchline of THIS problem. Apply this test:
      "Could a different problem in the same domain trigger the same
      mistake?" If yes, extract. If no, skip.

      EXTRACT (general):
        ✓ "Treats the empty array as if it must contain at least one element"
        ✓ "Uses inclusive bounds in a sliding window built for exclusive bounds"
        ✓ "Picks an O(n²) approach when constraints imply n ≥ 10^5"
        ✓ "Forgets that integer division in Python rounds toward negative infinity"

      SKIP (one-off):
        ✗ "Computed sum instead of product for this specific recurrence"
        ✗ "Used the wrong DP transition for the houses-on-a-circle subproblem"
        ✗ "Returned 3 instead of 2 because the modulo was applied too early
           in this particular formula"

For every correctness summary, fill verification_evidence with a SHORT note
(≤ 20 words) naming the concrete evidence — e.g. "tests went 0/2 → 2/2 after
fix", or "assistant accepted bug diagnosis on next turn".

──────────────────────────────────────────────────────────────────────────────
GENERAL RULES (both axes)
──────────────────────────────────────────────────────────────────────────────
• Abstract away ALL task-specific details: no problem names, variable names,
  literal code fragments, or specific numbers from this task.
• Each summary describes a FAMILY of situations, not just this one task.
• correction_quote: verbatim excerpt from the user correction turn, ≤ 30 words.

Return ONLY a JSON object:
{{
  "summaries": [
    {{
      "triggering_situation": "when solving an algorithmic problem from scratch",
      "what_was_wrong": "assistant jumped directly to code without analyzing complexity first",
      "what_user_wanted": "state time and space complexity before writing any code",
      "correction_quote": "Hold on—state the exact time and space complexity first",
      "correction_type": "preference",
      "verification_evidence": ""
    }},
    {{
      "triggering_situation": "when implementing a search/scan that may be called on empty input",
      "what_was_wrong": "indexed into the array without first handling the empty case",
      "what_user_wanted": "guard the empty-input branch before any indexing",
      "correction_quote": "this crashes the moment you give it []",
      "correction_type": "correctness",
      "verification_evidence": "tests went 1/3 → 3/3 after the fix; assistant kept the guard on the next turn"
    }}
  ]
}}

Conversation:
{conversation}
"""


# ---------------------------------------------------------------------------
# Stage B Step 1 – Candidate Extraction (per axis)
# ---------------------------------------------------------------------------
# The consolidator runs this prompt ONCE PER AXIS, passing only summaries of
# that axis. The LLM never sees both axes in the same call.
# Placeholders: {axis}, {axis_guidance}, {summaries}, {min_support}
SKILL_CANDIDATE_EXTRACTION_PROMPT = """\
You consolidate behavioral {axis} corrections into reusable skills.

{axis_guidance}

A skill is valid ONLY if at least {min_support} summaries in the list show
the SAME underlying behavior pattern.

Each skill must pass: "What concrete assistant mistake does this prevent?"
Do NOT create skills that are tied to a single task. Do NOT create skills
that mix this axis with the other axis (you will only see one axis here).

──────────────────────────────────────────────────────────────────────────────
GRANULARITY RULES — EACH SKILL = ONE HABIT (HARD CONSTRAINTS)
──────────────────────────────────────────────────────────────────────────────
A skill is the smallest checkable habit you can describe in one sentence.
NEVER bundle multiple distinct habits into one skill, even when they all
appeared in the same correction turn. If summaries cover several habits,
emit ONE candidate per habit; reuse the same supporting_summary_id across
multiple candidates whenever a single summary evidences multiple habits.

Treat the following as DIFFERENT habits and emit them as SEPARATE skills:
  - stating time/space complexity upfront
  - naming the algorithmic pattern / core insight
  - explaining why the brute-force / naive approach fails
  - tracing through a concrete example with variable states
  - using descriptive variable names
  - extracting magic numbers into named constants
  - explaining library / data-structure choices
  - addressing edge cases / boundary values
  - preferring iteration over recursion
  - any specific verification check (e.g. re-run trace, test on empty input)

HARD CAPS (a candidate violating either is invalid and must be split):
  - title ≤ 8 words.
  - guidance ≤ 60 words, ONE sentence ideally, two short sentences max.
  - guidance must describe ONE habit. If it contains conjunctions like
    "and also", "plus", "as well as", or enumerations of independent
    habits, SPLIT it.

If you find yourself writing "do X, and Y, and Z", that is THREE skills,
not one — emit three candidates.

Return ONLY a JSON object:
{{
  "candidates": [
    {{
      "title": "State Complexity Upfront",
      "catalog_trigger": "when presenting any algorithmic solution",
      "guidance": "Before writing code, state time and space complexity in Big-O with a one-sentence justification.",
      "supporting_summary_ids": ["id1", "id3", "id7"]
    }},
    {{
      "title": "Name the Algorithmic Pattern",
      "catalog_trigger": "when presenting any algorithmic solution",
      "guidance": "Explicitly name the core algorithmic pattern (e.g. sliding window, DP, greedy) before showing code.",
      "supporting_summary_ids": ["id1", "id4"]
    }}
  ]
}}

Each supporting_summary_id MUST be an exact ID from the summaries below.
A single summary id MAY appear under multiple candidates when that summary
genuinely evidences multiple distinct habits.

Correction summaries (all axis = {axis}):
{summaries}
"""


# Axis-specific guidance blocks plugged into {axis_guidance} above.
AXIS_GUIDANCE_PREFERENCE = """\
These corrections are about STYLE / FORMAT / APPROACH. Each resulting skill
should describe a habit the assistant must adopt across many tasks. The
metric these skills move is "fewer user interruptions per task" — write
guidance that is concrete enough to be checkable in a single response.
"""

AXIS_GUIDANCE_CORRECTNESS = """\
These corrections are about BUGS the user caught and verified. Each
resulting skill should describe a CLASS OF BUG to watch for, plus the
concrete check or guard that prevents it. The metric these skills move is
"test-case pass rate". Write guidance that the assistant can apply as a
checklist BEFORE returning code (e.g. "before returning, walk through what
this function does on empty input / single-element input / max-size input").
Do NOT propose skills that are really one-off algorithmic recipes for a
specific problem family — those won't generalize.
"""


# ---------------------------------------------------------------------------
# Stage B Step 1b – Intra-batch deduplication (per axis)
# ---------------------------------------------------------------------------
# Placeholders: {axis}, {candidates}
SKILL_DEDUP_PROMPT = """\
You are given a list of candidate behavioral skills, all of axis "{axis}",
extracted from the same set of corrections. Some candidates may address
essentially the same assistant mistake — just worded differently.

DEFAULT BIAS: KEEP CANDIDATES SEPARATE. Each skill represents ONE habit.
Merge ONLY when two candidates target the SAME habit phrased differently
(e.g. "name the pattern first" vs "identify the algorithmic approach
upfront"). Do NOT merge candidates that cover DIFFERENT habits even if:
  • they share supporting summaries (one correction can evidence many habits),
  • they often co-occur in the same task,
  • they would both fire in the same situation,
  • merging would produce shorter output.

The following are DIFFERENT habits and MUST stay separate:
  - complexity statement vs algorithmic-pattern naming
  - brute-force-failure explanation vs example trace
  - descriptive variable names vs extracting magic numbers
  - edge-case enumeration vs verification of a specific edge case
  - any pair of items from the granularity list in the extraction prompt.

For genuinely-merged groups: write the best unified title, trigger, and
guidance. The merged guidance MUST still satisfy the same hard caps as the
extraction prompt (title ≤ 8 words, guidance ≤ 60 words, ONE habit). If a
merge would force the output above 60 words or to cover multiple habits,
DO NOT MERGE.

Include ALL candidates, whether merged or kept alone (singletons are fine
and preferred when in doubt).

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

Each integer in source_indices refers to an index in the input list below.
Every input index must appear in exactly one source_indices list.

Candidates (axis = {axis}, indexed from 0):
{candidates}
"""


# ---------------------------------------------------------------------------
# Stage B Step 2 – Reconciliation (per axis)
# ---------------------------------------------------------------------------
# Placeholders: {axis}, {existing_catalog}, {proposed_skills}
SKILL_RECONCILIATION_PROMPT = """\
You reconcile newly proposed skills (axis = "{axis}") against the existing
skill library OF THE SAME AXIS. Skills of a different axis are NOT shown
and MUST NOT be considered as candidates for merge.

DEFAULT BIAS: discard ONLY when the existing library covers the SAME single
habit. Lean toward "add" whenever the proposed skill names a habit not
already in the library — coverage matters more than catalog size, since
each skill is small (≤ 60 words, ONE habit) and the selector picks ≤ 2
per task.

For each proposed skill, choose exactly one action:
  "discard"  : addresses the SAME single habit as an existing skill (NOT
               just the same broad area) — e.g. both say "state complexity
               first" — OR adds only minor wording differences.
  "update"   : addresses the SAME single habit AND contributes a meaningfully
               sharper or more general phrasing; provide existing_skill_id
               and updated_guidance (≤ 60 words, ONE habit, best-of-both).
               Do NOT use "update" to merge two different habits into a
               longer combined guidance — keep the existing one and "add"
               the new one instead.
  "replace"  : CONTRADICTS an existing skill (proposes opposite behavior
               for the same situation); provide existing_skill_id and
               updated_guidance (coherent synthesis, ≤ 60 words).
  "add"      : covers a habit not addressed by any existing skill in this
               axis. Two habits are different even if their catalog_triggers
               overlap — what matters is whether the *guidance* describes
               the same single check.

If you are tempted to "update" because the existing skill is "close" but
covers a different habit (e.g. existing = "state complexity first",
proposed = "name the algorithmic pattern"), choose "add" instead. Long
combined guidance is BANNED — every stored skill must remain ≤ 60 words
and one habit.

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

Existing skills, axis = {axis} (id | title | trigger | guidance):
{existing_catalog}

Proposed skills, axis = {axis} (indexed from 0):
{proposed_skills}
"""


# ---------------------------------------------------------------------------
# Stage B Step 3 – Catalog Compaction (per axis, when catalog grows)
# ---------------------------------------------------------------------------
# Placeholders: {axis}, {skills}
SKILL_COMPACTION_PROMPT = """\
You are cleaning up the {axis} half of a behavioral skill library for an AI
coding assistant. The library may contain duplicate, overlapping, or
contradictory skills WITHIN this axis. You will not see, and must not
produce, any skills of the other axis.

Group skills that target the SAME single habit (NOT just the same broad
area). For each group:
  • Pick the skill with the highest support_count as the primary
    (use its id as keep_id)
  • Write a unified title (≤ 8 words), catalog_trigger, and guidance
    (≤ 60 words, ONE habit) that covers the whole group
  • List the ids of all OTHER skills in the group as discard_ids

Skills that target DIFFERENT habits MUST appear as singleton groups
(discard_ids = []) — even if their triggers or titles look similar.
"Different habit" means a reader following one skill could still violate
the other (e.g. naming the pattern doesn't satisfy "state complexity
first"; using descriptive names doesn't satisfy "extract magic numbers").

Rules:
  • Every skill id must appear exactly once across all keep_id and
    discard_ids fields combined.
  • If two skills contradict each other on the SAME habit, resolve the
    contradiction in the merged guidance.
  • Prefer broader, reusable guidance over task-specific wording.
  • The merged guidance MUST stay ≤ 60 words and cover ONE habit. If a
    merge would force a longer or multi-habit guidance, KEEP the skills
    separate (singleton groups).

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

Current {axis}-axis skill library:
{skills}
"""


# ---------------------------------------------------------------------------
# Stage C – Catalog Selector
# ---------------------------------------------------------------------------
# The catalog is rendered with axis labels so the selector can pick a
# balanced set (typically at most one preference + one correctness skill).
# Placeholders: {task}, {catalog}, {max_skills}
CATALOG_SELECTOR_PROMPT = """\
You select behavioral guidelines for a coding assistant to apply to the
current task. The catalog contains TWO kinds of guideline:

  [preference] — habits about HOW to communicate / format / structure the
                 response. Apply when this task offers an opportunity to
                 violate the habit.

  [correctness] — bug-class checklists describing common mistakes. Apply
                  when this task plausibly contains the kind of input or
                  structure that would trigger the bug class.

Select AT MOST {max_skills} guidelines that are clearly relevant to this task.
When several preference guidelines describe complementary response-structure
habits for the same task, select all of the relevant ones up to the limit.
Prefer a correctness guideline only when it adds a distinct bug-prevention
check; do not select near-duplicate guidelines that ask for the same behavior.
Only select a guideline if its trigger directly matches the situation described
in the task. If none apply, return an empty list.

Return ONLY a JSON object:
{{"selected_ids": ["id1", "id2"]}}

Task:
{task}

Available guidelines (each line is "[axis] [id] title \\n  trigger: ..."):
{catalog}
"""


# ---------------------------------------------------------------------------
# Runtime – Constraint injection into LLM-B system prompt
# ---------------------------------------------------------------------------
# Placeholders: {skills_block}
CONSTRAINT_INJECTION_TEMPLATE = """\
Before responding, apply these guidelines learned from prior interactions \
with this user. Each guideline is tagged [preference] (a habit to follow) \
or [correctness] (a bug class to actively check for):

{skills_block}

Follow each guideline that applies to the current task. For [correctness] \
guidelines, run the check explicitly before returning code. Do not mention \
the guidelines to the user unless asked.
"""
