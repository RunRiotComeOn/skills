# SkillMap

Training-free, per-user procedural memory for LLM assistants.

SkillMap induces reusable behavioral guidelines from in-task user corrections,
stores them in a flat experience library, and retrieves a small set of relevant
guidelines at runtime to constrain assistant generation.

## Design: Flat Experience Library

SkillMap no longer uses a prerequisite DAG, tree traversal, or BFS retrieval.
The flat design avoids ambiguous parent-child edges, broad parent nodes blocking
specific skills, and pending split states preventing useful guidance from being
retrieved.

The system follows a three-stage pipeline:

| Stage | When | Purpose |
|-------|------|---------|
| A | After each task | Summarize user corrections into buffered `CorrectionSummary` records |
| B | When the buffer reaches a threshold | Consolidate recurring corrections into supported `Skill` entries |
| C | At the start of each task | Select at most two relevant skills from the catalog |

## Core Data Model

`skillmap/types.py` defines the persisted state:

| Type | Purpose |
|------|---------|
| `CorrectionSummary` | One abstracted behavioral correction extracted from a task transcript |
| `Skill` | A reusable guideline backed by multiple correction summaries |
| `CatalogEntry` | Lightweight selector index containing skill ID, title, and trigger |
| `SkillMapState` | Persisted state containing skills, pending summaries, and the rebuilt catalog |

Old DAG concepts such as skill statuses, prerequisite edges, episodes,
correction points, and traversal errors are intentionally removed.

## Runtime Flow

At task start, `CatalogSelector` receives the task description and the catalog.
It selects up to `MAX_SKILLS_PER_TASK` matching skills, or none if no guideline
applies. The orchestrator passes the selected full `Skill` objects directly to
the assistant.

At task finalization, `CorrectionSummarizer` scans the task trajectory for user
corrections and appends summaries to the buffer. Once the buffer reaches
`CONSOLIDATION_TRIGGER`, `Consolidator` extracts recurring patterns, keeps only
candidates with at least `MIN_SUPPORT` backing summaries, reconciles them with
existing skills, rebuilds the catalog, and persists the updated state.

## Layout

```text
skillmap/
  types.py             # Flat Pydantic data models
  storage/             # JSON persistence and SkillMap state operations
  induction/           # Correction summarization and batch consolidation
  retrieval/           # Catalog-based skill selection
  runtime/             # Assistant wrapper
  llm/                 # Client abstraction and prompt templates
  orchestrator.py      # Wires handle_query / finalize_task
eval/
  skillmap_eval/       # Evaluation harness and SkillMap condition
tests/
scripts/run_demo.py    # End-to-end demo on synthetic stream
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CONSOLIDATION_TRIGGER` | 20 | Run consolidation after this many new summaries |
| `MIN_SUPPORT` | 3 | Minimum summaries required before writing a skill |
| `MAX_SKILLS_PER_TASK` | 2 | Maximum skills injected into a task |
| `MAX_CATALOG_SIZE` | 50 | Maximum catalog entries shown to the selector |
| Stage A model | Haiku | Structured correction summarization |
| Stage B model | Haiku | Skill consolidation and reconciliation |
| Stage C model | Haiku | Catalog lookup and selection |

## Quickstart

```bash
pip install -e .
python scripts/run_demo.py
```

## Migration Note

Old DAG-based `skill_map.json` data is incompatible with the flat library
format. Delete old evaluation run data before the first flat-library run:

```bash
rm -rf eval/data/skillmap_day_runs/
```
