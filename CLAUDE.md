# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SkillMap is a training-free, per-user procedural memory system for LLM assistants. It learns from in-task user corrections and stores them as reusable behavioral guidelines. At runtime, it selects relevant guidelines to inject into the system prompt.

The core innovation is a **two-axis skill model**:
- **Preference axis**: User pushed back on style/format/approach (ground truth = user)
- **Correctness axis**: User caught a real bug (must carry verification evidence; false positives are high-cost)

These axes are **never merged** — they run through separate pipelines and are tracked independently.

## Commands

```bash
# Install
pip install -e .

# Run tests
pytest tests/

# Run a single test file
pytest tests/test_induction.py

# Run demo
python scripts/run_demo.py

# Evaluation pipeline
python eval/scripts/elicit_preferences.py --task-type python_coding
python eval/scripts/run_eval.py --profile coding_v1
python eval/scripts/make_figures.py --report results/eval_report_coding_v1.json
```

**Environment**: Requires `BEDROCK_API_KEY` set in `.env` (see `.env.example`). The LLM client also accepts a CSV fallback path for API keys.

## Architecture: Three Stages

**Stage A — Correction Summarization** (`skillmap/induction/summarizer.py`): After each task, extract 0–5 `CorrectionSummary` records from the conversation trajectory, tagged by axis. Correctness summaries are dropped if they lack verification evidence.

**Stage B — Consolidation** (`skillmap/induction/consolidator.py`): When pending summaries reach `CONSOLIDATION_TRIGGER` (20), run per-axis: extract skill candidates → dedup → reconcile against existing catalog. Reconcile handles containment/merging inline — no separate compaction step. The two axes are processed completely separately.

**Stage C — Selection & Retrieval** (`skillmap/retrieval/selector.py`): At task start, select up to `MAX_SKILLS_PER_TASK` (2, overridable via `SKILLMAP_TOP_K` env var) skills from the catalog using an LLM-based selector. Only `active` skills appear in the catalog. Skills are injected with axis labels so the model can balance preference vs. correctness guidance.

The pipeline is wired together in `skillmap/orchestrator.py`.

## Key Data Model (`skillmap/types.py`)

- **`CorrectionSummary`**: One record per correction; carries `correction_type` (preference/correctness) and `verification_evidence` (required for correctness axis)
- **`Skill`**: Reusable guideline backed by ≥ `MIN_SUPPORT` summaries; includes `axis`, `title`, `guidance`, `support_count`, and `status` (`"active"` | `"past"`)
- **`CatalogEntry`**: Lightweight index entry used by the selector
- **`SkillMapState`**: Persisted JSON state — skills, pending summaries, and rebuilt catalog

## LLM Client (`skillmap/llm/client.py`)

Wraps Amazon Bedrock Converse REST API. Default model is MiniMax M2.5; internal SkillMap stages (summarizer, consolidator, selector) use Claude Haiku 4.5. Supports structured output (JSON schema) for relevant models. Retries with exponential backoff (3 attempts, 2s base).

## Evaluation Framework (`eval/`)

Tests whether SkillMap accumulates and reuses preferences across LiveCodeBench/AIME task streams. Three conditions: `stateless`, `declarative_memory`, `skillmap`. Key metrics: correction-rate decay curves, preference recovery, generalization, and test-case pass rates. The eval has its own `pyproject.toml` and is installed separately.

## Skill Operations (`skillmap/storage/skill_map.py`)

| Operation | When used |
|-----------|-----------|
| `insert_skill` | Reconcile decides `"add"` (new habit, support ≥ MIN_SUPPORT) |
| `update_skill` | Reconcile decides `"update"` — same habit, containment, or correctness override |
| `deprecate_skill` | Reconcile decides `"conflict"` (preference axis only) — marks old skill `"past"`, removed from catalog |
| `delete_skill` | Manual/admin use only; not called by the pipeline |

**Reconcile actions** (four total, no `"replace"`): `"discard"` / `"update"` / `"conflict"` / `"add"`. Containment (A ⊆ B or A ⊇ B) is handled via `"update"` at reconcile time — there is no separate compaction pass.

**Conflict semantics differ by axis**:
- Preference conflict → `deprecate_skill` (archive as `"past"`) + `insert_skill` (new active skill). Old preference is kept for audit but never selected again.
- Correctness conflict → `update_skill` in place. One of the two was objectively wrong; no audit value in archiving it.

## Critical Invariants

- Preference and correctness skills are **never merged across axes** — each axis is fully independent through Stages B and C
- Correctness summaries **without verification evidence are dropped** — false-positive correctness skills can nudge the model toward incorrect fixes
- The catalog only contains `active` skills; `past` skills exist in storage for audit but are invisible to the selector
- The catalog is flat (no DAG, no hierarchy); skills are sorted by support count
- `MIN_SUPPORT_BY_AXIS` defaults to 3 for both axes (set in `consolidator.py`)
