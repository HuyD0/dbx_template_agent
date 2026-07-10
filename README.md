# dbx_template_agent

A tool-calling agent on Databricks with a real evaluation lifecycle — scaffolded
from [databricks-agents-template](https://github.com/HuyD0/databricks-agents-template).
Everything you answered at `bundle init` is already baked in
(`src/dbx_template_agent/config.py` is the single source of truth); nothing needs
editing before notebook 01 runs.

## The lifecycle you'll learn

```
build → trace → experiment → evaluate → improve evals → gate/deploy → monitor
  ↑__________________________________________________________________|
```

| # | Notebook | Time | Level | You'll walk away with |
|---|---|---|---|---|
| 01 | `01_setup_and_first_trace` | ~15 min | 🟢 easy | endpoints verified, prompt registered, first trace |
| 02 | `02_build_the_agent` | ~25 min | 🟢 easy | the span tree read as data (`search_traces`), cost per trace |
| 03 | `03_experiments` | ~25 min | 🟡 moderate | two prompt versions compared as proper MLflow runs |
| 04 | `04_design_your_evals` | ~30 min | 🟡 moderate | judged eval run, dataset lineage, threshold floors |
| 05 | `05_traces_to_better_evals` | ~40 min | 🟠 harder | 👎-mined eval rows, deterministic scorers, judge vs humans |
| 06 | `06_gate_and_deploy` | ~30 min | 🟠 harder | eval gate, UC model, champion/challenger aliases |
| 07 | `07_monitor_cost_and_quality` | ~30 min | 🟠 harder | quality + spend time series vs the daily budget |
| 08 | `08_agent_bricks_challenger` | ~45 min | 🔴 challenge | your eval set judging a managed rival |

Each notebook opens with a "you are here" map and ends with a ✅ **checkpoint
cell** that verifies you actually succeeded, plus 🏋️ exercises. Terms are
defined in [GLOSSARY.md](GLOSSARY.md); when something breaks, start at
[TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Quickstart

```bash
make install         # uv venv + deps + git hooks
make test            # offline unit tests — no workspace needed
databricks bundle validate
databricks bundle deploy
```

Then open `notebooks/01_setup_and_first_trace.py` in your workspace
(deployed under the bundle's dev path) and run it top to bottom.

## Repo map

```
src/dbx_template_agent/   config.py (all settings) · agent.py (the loop) ·
                         tools.py · cost.py · scorers.py
evals/                   eval_set.jsonl · thresholds.yaml · run_agent_eval.py (the gate)
notebooks/               the 8-phase curriculum
resources/               the eval-gate job + (commented) serving endpoint
tests/                   offline unit tests (mocked model client)
```

## Costs, plainly

Only model inference costs meaningful money here. The starter eval set is
~10 rows; a full gate run is typically a few cents. Guardrails you get for
free: a daily budget (`DAILY_BUDGET_USD` in config.py) the gate **refuses** to
exceed, a per-answer cost ceiling in `thresholds.yaml`, and cost-per-trace on
every span tree. The pricing table in `src/dbx_template_agent/cost.py` holds
estimates — update it to your account's rates.

## Production practices demonstrated

- prompts in the **MLflow Prompt Registry**, promoted by alias (never hardcoded versions)
- eval sets logged as **run inputs** (lineage), versioned via `dataset_version`
- **tags vs params vs metrics** discipline (`config.run_tags`)
- deterministic scorers before LLM judges; judges **calibrated against humans**
- champion/challenger promotion behind a **threshold gate** (quality AND cost)
- CI split: hermetic checks on every PR (`validate.yml`); anything that spends
  money is **manual dispatch** (`deploy.yml`)
- cost control that **fails closed**
