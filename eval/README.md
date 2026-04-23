# SkillMap Evaluation

Evaluation harness that tests whether SkillMap accumulates and reuses user
preferences across a stream of LiveCodeBench coding tasks.

**Key principle:** LiveCodeBench provides tasks; an LLM-simulated user
(LLM-A), driven by a pre-generated preference profile, provides
corrections. Test-case pass rate is a sanity check, not the eval signal.

## Primary signals

1. **Correction-rate decay** — corrections per task across the stream.
   SkillMap should curve downward faster than baselines.
2. **Preference recovery** — for SkillMap only: how many of the N
   ground-truth preferences are captured as induced skills.
3. **Generalization** — held-out tasks measure whether SkillMap's late
   stream behavior transfers.
4. **Correctness sanity** — test-case pass rate across conditions should
   be approximately equal.

## Layout

```
skillmap_eval/
  preferences/      # LLM-A preference profile elicitation
  tasks/            # LiveCodeBench loader
  simulator/        # LLM-A as user
  conditions/       # stateless / declarative_memory / skillmap
  runner/           # per-task loop, per-stream runner
  metrics/          # four metric modules
  analysis/         # aggregate + plot
tests/
scripts/
  elicit_preferences.py
  run_eval.py
  make_figures.py
results/
```

## Phases

See Section 5 of the implementation spec. Gates:

| # | Scope                         | Gate                                                   |
|---|-------------------------------|--------------------------------------------------------|
| 1 | Preference elicitation        | Human review: >=8/N preferences are opinionated        |
| 2 | Task loader                   | sample_stream: disjoint sets, valid difficulty mix     |
| 3 | User simulator                | 5 scripted responses -> decisions match expectations   |
| 4 | Stateless + interaction loop  | 5-task stream produces valid StreamRun JSON            |
| 5 | Declarative memory baseline   | Late-stream retrieval returns early-stream facts       |
| 6 | SkillMap condition            | Skill map grows; retrieval non-empty                   |
| 7 | Metrics                       | EvalReport populated with all four sections            |
| 8 | Full run                      | 3 conditions x 40 tasks; 4 figures                     |

## Quickstart

```bash
# Stage 1: generate a preference profile (requires Bedrock API key in .env).
# The script grounds LLM-A with 3 sampled LiveCodeBench task examples.
python scripts/elicit_preferences.py --task-type python_coding

# Stage 2: run the full eval.
python scripts/run_eval.py --profile coding_v1

# Stage 3: figures.
python scripts/make_figures.py --report results/eval_report_coding_v1.json
```

## Depends on

The parent `skillmap/` package. Import-only: the eval never modifies
`skillmap/`. If a capability is missing, surface it as a gap rather than
monkey-patching.
