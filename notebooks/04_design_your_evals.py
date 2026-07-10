# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Design your evals
# MAGIC
# MAGIC ⏱ ~30 min · 🟡 moderate
# MAGIC
# MAGIC **You are here:**
# MAGIC `setup → build → experiment → [EVALUATE]* → improve evals → gate/deploy → monitor`
# MAGIC
# MAGIC An eval is not a score. An eval is **(dataset, scorers, thresholds)** — all
# MAGIC three versioned, so any score can be reproduced. You'll run the starter eval
# MAGIC set through `mlflow.genai.evaluate` with two built-in LLM judges, read judge
# MAGIC *rationales* (not just numbers), and see the dataset linked as a run input.

# COMMAND ----------

# MAGIC %pip install -q mlflow>=3.14.0 databricks-sdk>=0.38.0 openai>=1.40 pandas pyyaml
# MAGIC %restart_python

# COMMAND ----------

import sys, os, json
from pathlib import Path

sys.path.insert(0, os.path.abspath("../src"))

import mlflow
import pandas as pd

from dbx_template_agent import config
from dbx_template_agent.agent import run_agent

config.setup_mlflow("evals")

EVAL_SET = Path("../evals/eval_set.jsonl")
lines = [line for line in EVAL_SET.read_text().splitlines() if line.strip()]
rows = pd.DataFrame([json.loads(line) for line in lines])
print(f"{len(rows)} eval rows")
rows.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC Each row: `inputs` (the question) + `expectations` (`expected_response` for the
# MAGIC Correctness judge, `expected_tool` for the deterministic scorer in phase 05).
# MAGIC Rows where `expected_tool` is `null` assert the agent should NOT call a tool —
# MAGIC tool overuse is a real failure mode (it costs money and latency).

# COMMAND ----------

from mlflow.genai.scorers import Correctness, RelevanceToQuery

# The judge endpoint is a DIFFERENT model family than the agent — a judge from
# the same family over-scores its sibling (self-enhancement bias).
judges = [
    Correctness(model=config.JUDGE_MODEL_URI),
    RelevanceToQuery(model=config.JUDGE_MODEL_URI),
]


def predict_fn(question: str) -> dict:
    result = run_agent(question)
    return {"answer": result["answer"], "cost_usd": result["cost_usd"]}

# COMMAND ----------

with mlflow.start_run(run_name=f"{config.CHAT_ENDPOINT}-eval-design") as run:
    mlflow.set_tags(config.run_tags(dataset_version="1", task="eval-design"))
    mlflow.log_params({"judge_endpoint": config.JUDGE_ENDPOINT, "n_rows": len(rows)})
    # Dataset LINEAGE — the run <-> dataset edge in the Datasets tab:
    mlflow.log_input(
        mlflow.data.from_pandas(rows, name="eval_set", source=str(EVAL_SET)),
        context="evaluation",
    )
    results = mlflow.genai.evaluate(
        data=rows, predict_fn=lambda question: predict_fn(question), scorers=judges
    )

print({k: v for k, v in results.metrics.items()})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read the rationales, not just the means
# MAGIC Open the run in the UI → **Evaluations**. Every judged row carries the judge's
# MAGIC *reasoning*. Read the failures first: a judge that fails a row for a bad
# MAGIC reason is a judge problem (phase 05 fixes those); a fair failure is an agent
# MAGIC problem (fix the prompt/tooling, re-run, compare).

# COMMAND ----------

# The same result table, as data:
result_rows = results.tables["eval_results"]
cols = [c for c in result_rows.columns if "rationale" in c or "value" in c or "question" in c]
result_rows.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set the floors
# MAGIC `evals/thresholds.yaml` holds the gate's floors — including the
# MAGIC **cost-per-answer ceiling**. They ship loose; now that you have a measured
# MAGIC baseline, tighten them to just below it. A floor you've never measured
# MAGIC against is a guess, not a gate.

# COMMAND ----------

# ✅ CHECKPOINT
assert results.metrics, "evaluate() returned no metrics"
assert any("correctness" in k for k in results.metrics), "Correctness judge missing"
run_data = mlflow.get_run(run.info.run_id)
assert run_data.inputs.dataset_inputs, "Dataset was not linked as a run input!"
print("✅ Phase 04 complete — judged eval run, dataset linked, floors understood.")
print("   Next: notebooks/05_traces_to_better_evals.py")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🏋️ Try this
# MAGIC 1. Add 2 rows for YOUR domain to `eval_set.jsonl` (and bump dataset_version).
# MAGIC 2. Run the same eval with only `RelevanceToQuery` vs only `Correctness` — two
# MAGIC    runs, two framings of "good". Which rows flip? That's why an eval is
# MAGIC    (dataset, scorers, thresholds), not a single number.
