# SkillMap

Training-free, per-user procedural memory for LLM assistants.

SkillMap induces reusable behavioral guidelines from in-task user corrections,
stores them in a flat experience library, and retrieves a small set of relevant
guidelines at runtime to constrain assistant generation.

## Two-axis skill model

User feedback splits into two qualitatively different signals, and SkillMap
treats them as **two independent learning channels** end-to-end. The two
channels share the same Stage A → B → C pipeline but are stored, reconciled,
selected, and evaluated separately.

| Axis | Source signal | Extraction gate | Skill effect | Primary eval metric | Secondary eval metric |
|---|---|---|---|---|---|
| **preference** | User pushed back on style / format / approach. The user IS the ground truth — there is no objective "right answer". | Same behavior pattern repeats in ≥ `MIN_SUPPORT_BY_AXIS["preference"]` summaries. | Reduces user-interruption count over the stream. | `CorrectionRateCurve.preference_per_task` (rolling-mean decay) | `PreferenceRecoveryResult.recovery_rate` (LLM judge: did SkillMap induce a skill that captures each ground-truth preference?) |
| **correctness** | User caught a real bug — failed test, edge-case crash, wrong output, obviously wrong formula. | (V) The conversation contains evidence the correction was actually right (assistant adopted the fix; a test that previously failed now passes; user accepted the revision). (G) The underlying mistake names a FAMILY of bugs ("forgets empty input", "off-by-one in sliding window", "ignores constraint magnitude"), not the algorithmic punchline of this specific problem. | Lifts test-case pass rate over the stream. | `CorrectnessTrajectory.avg_pass_rate_late − avg_pass_rate_early` (early/late split on the stream) plus `avg_pass_rate_held_out` for generalization. | `CorrectionRateCurve.correctness_per_task` (correction-side decay) |

**Hard invariant:** preference skills and correctness skills are NEVER merged
into one another. Stage B (extraction → dedup → reconciliation) and Stage B-3
(catalog compaction) all run **once per axis**, with each axis's prompt seeing
only its own summaries / catalog. Tags propagate through `Skill.axis`,
`CatalogEntry.axis`, and `SimulatedTurn.correction_axes`.

**Why this matters for the design.** The previous version of the consolidator
dropped every correctness summary on the floor because "correctness corrections
are task-specific and don't generalize". That conflated two things: a one-off
algorithmic slip (genuinely task-specific, correctly excluded) and a recurring
bug class like "forgets empty input" (generalizable, was being thrown away).
The verification + generalization gates in Stage A's prompt let the second
class through while still rejecting the first.

## Pipeline

| Stage | When | Purpose |
|-------|------|---------|
| A | After each task | Summarize user corrections into buffered `CorrectionSummary` records, tagged with `correction_type ∈ {preference, correctness}`. Correctness summaries must carry `verification_evidence`. |
| B | When the buffer reaches `CONSOLIDATION_TRIGGER` | Per-axis: extract candidates → intra-batch dedup → reconcile against the existing same-axis catalog → compact when the same-axis catalog grows. |
| C | At the start of each task | Select up to `MAX_SKILLS_PER_TASK` skills from the catalog; the prompt encourages a balanced pick (typically ≤ 1 preference + ≤ 1 correctness). |

## Core data model

`skillmap/types.py`:

| Type | Purpose |
|------|---------|
| `CorrectionSummary` | One abstracted correction; carries `correction_type` and (for correctness) `verification_evidence`. |
| `Skill` | A reusable guideline backed by ≥ `MIN_SUPPORT_BY_AXIS[axis]` summaries; carries `axis`. |
| `CatalogEntry` | Lightweight selector index; carries `axis` so the selector can balance picks. |
| `SkillMapState` | Persisted state: skills, pending summaries, rebuilt catalog. |

## Evaluation

`eval/skillmap_eval/` runs three conditions (`stateless`, `declarative_memory`,
`skillmap`) over a stream of LiveCodeBench tasks plus a held-out set. The
report (`EvalReport`) ships **four metric families**, mapped to the two-axis
model:

| Metric (in `EvalReport`) | Axis it measures | Reads as |
|---|---|---|
| `correction_curves[*].preference_per_task` (+ rolling mean) | preference (primary) | SkillMap should bend down faster than the baselines. |
| `preference_recovery[*].recovery_rate` | preference (secondary) | Fraction of ground-truth preferences captured by an induced preference-axis skill. |
| `correctness_trajectory[*]` (early vs late vs held-out) | correctness (primary) | SkillMap should show late > early; held-out close to late means generalization, held-out close to early means in-stream memorization. |
| `correction_curves[*].correctness_per_task` (+ rolling mean) | correctness (secondary) | Correctness corrections per task should also decay if correctness skills are firing. |
| `generalization[*].held_out_avg_{preference,correctness}_corrections` | both | Per-axis held-out correction rate, used as the "would the curve continue at this level on unseen tasks?" check. |

A `SimulatedTurn` issued during a `correct` decision is tagged with
`correction_axes`; an interaction's `preference_correction_count` and
`correctness_correction_count` are the per-axis sums (a single turn may count
toward both). `correction_count` is the number of user turns with any axis
tag — i.e. the total number of correction turns, not the sum of the two
counts.

## Layout

```text
skillmap/
  types.py             # Two-axis Pydantic data models
  storage/             # JSON persistence and SkillMap state operations
  induction/           # Per-axis correction summarization and consolidation
  retrieval/           # Catalog-based selector (axis-aware)
  runtime/             # Assistant wrapper (renders skills with axis labels)
  llm/                 # Client and prompt templates (axis-aware throughout)
  orchestrator.py      # Wires handle_query / finalize_task
eval/
  skillmap_eval/       # Evaluation harness and three conditions
tests/
scripts/run_demo.py    # End-to-end demo on synthetic stream
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CONSOLIDATION_TRIGGER` | 20 | Run Stage B after this many new summaries (across both axes). |
| `MIN_SUPPORT_BY_AXIS["preference"]` | 3 | Minimum same-axis summaries to write a preference skill. |
| `MIN_SUPPORT_BY_AXIS["correctness"]` | 3 | Same, for correctness. Tighten this if false-positive bug-class skills appear in eval. |
| `MAX_SKILLS_PER_TASK` | 2 | Maximum skills injected into a task. Selector aims for at most one per axis. |
| `MAX_CATALOG_SIZE` | 50 | Maximum catalog entries shown to the selector. |
| Stage A model | Haiku | Structured correction summarization. |
| Stage B model | Haiku | Skill consolidation and reconciliation. |
| Stage C model | Haiku | Catalog lookup and selection. |

## Quickstart

```bash
pip install -e .
python scripts/run_demo.py
```
