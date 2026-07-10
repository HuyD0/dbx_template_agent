# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Monitor cost & quality over time
# MAGIC
# MAGIC ⏱ ~30 min · 🟠 harder
# MAGIC
# MAGIC **You are here:**
# MAGIC `setup → build → experiment → evaluate → improve evals → gate/deploy → [MONITOR]*`
# MAGIC
# MAGIC The gate protects a *release*; monitoring protects the *running system*.
# MAGIC Live traffic has no `expected_response`, so monitoring uses **reference-free**
# MAGIC judges + the deterministic scorers, on a SAMPLE of recent traces, logged as a
# MAGIC time series — plus spend aggregated against the daily budget.

# COMMAND ----------

# MAGIC %pip install -q mlflow>=3.14.0 databricks-sdk>=0.38.0 openai>=1.40 pandas
# MAGIC %restart_python

# COMMAND ----------

import sys, os

sys.path.insert(0, os.path.abspath("../src"))

import mlflow
import pandas as pd

from dbx_template_agent import config
from dbx_template_agent.cost import trace_cost_usd

# Read traces from the agent experiment; write the series to /monitoring.
config.setup_mlflow("agent")

# COMMAND ----------

# Sample recent traces (cap the sample — monitoring must never out-spend the
# system it watches):
SAMPLE = 10
traces_df = mlflow.search_traces(max_results=200)
sample_ids = traces_df["trace_id"].head(SAMPLE).tolist()
print(f"{len(traces_df)} traces available, scoring {len(sample_ids)}")

# COMMAND ----------

# Deterministic pass: cost + latency + tool usage per trace. Free.
from dbx_template_agent.scorers import tool_calls_from_trace

rows = []
for trace_id in sample_ids:
    trace = mlflow.get_trace(trace_id)
    rows.append(
        {
            "trace_id": trace_id,
            "cost_usd": trace_cost_usd(trace, config.CHAT_ENDPOINT),
            "latency_ms": trace.info.execution_time_ms,
            "n_tool_calls": len(tool_calls_from_trace(trace)),
        }
    )
window = pd.DataFrame(rows)
window.describe()

# COMMAND ----------

# Reference-free judge pass: RelevanceToQuery needs no ground truth, so it can
# score live traffic. (Groundedness would join it in a RAG agent.)
from mlflow.genai.scorers import RelevanceToQuery

judge = RelevanceToQuery(model=config.JUDGE_MODEL_URI)
scores = []
for trace_id in sample_ids:
    trace = mlflow.get_trace(trace_id)
    try:
        feedback = judge(trace=trace)
        scores.append(1.0 if str(feedback.value).lower() in ("yes", "true", "pass") else 0.0)
    except Exception as exc:  # a judge outage must not kill monitoring
        print(f"judge skipped {trace_id}: {exc}")
window["relevance"] = pd.Series(scores)

# COMMAND ----------

# Log the window as one run in the /monitoring series — run it daily (the
# bundle job pattern) and the experiment BECOMES your quality/cost dashboard.
config.setup_mlflow("monitoring")

spend_today = float(window["cost_usd"].sum())
with mlflow.start_run(run_name=f"{config.CHAT_ENDPOINT}-monitor"):
    mlflow.set_tags(config.run_tags(task="monitoring"))
    mlflow.log_params({"sample_size": len(window)})
    mlflow.log_metrics(
        {
            "monitored_traces": len(window),  # heartbeat: 0 still logs a run
            "cost_usd_sample": spend_today,
            "cost_usd_per_trace": float(window["cost_usd"].mean() or 0),
            "latency_ms_p95": float(window["latency_ms"].quantile(0.95)),
            "relevance_mean": float(window["relevance"].mean() or 0),
            "tool_calls_per_trace": float(window["n_tool_calls"].mean() or 0),
            "budget_used_frac": spend_today / config.DAILY_BUDGET_USD,
        }
    )
print(f"sample spend ${spend_today:.4f} of ${config.DAILY_BUDGET_USD} daily budget")

# COMMAND ----------

# MAGIC %md
# MAGIC > **Why a heartbeat** — an empty window still logs `monitored_traces=0`.
# MAGIC > A gap in the series is indistinguishable from broken monitoring; a zero is
# MAGIC > a fact. Alerting keys off the metric, never off "did the job run".
# MAGIC
# MAGIC To productionize: wrap this notebook in a scheduled bundle job (copy the
# MAGIC eval-gate job in `resources/`), keep the schedule PAUSED until you trust it,
# MAGIC and add `--floor relevance_mean=0.7`-style breach checks that fail the job
# MAGIC so its notification fires.

# COMMAND ----------

# ✅ CHECKPOINT
runs = mlflow.search_runs(filter_string="tags.task = 'monitoring'")
assert len(runs) >= 1, "No monitoring run logged"
latest = runs.iloc[0]
assert "metrics.cost_usd_per_trace" in runs.columns, "Cost series missing"
assert latest["metrics.budget_used_frac"] < 1.0, "Sample alone blew the daily budget!"
print("✅ Phase 07 complete — quality + cost series logged; the loop is closed.")
print("   Optional capstone: notebooks/08_agent_bricks_challenger.py")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🏋️ Try this
# MAGIC 1. Re-run this notebook after generating 20 fresh traces (phase 02) — two
# MAGIC    points make a series. Chart `relevance_mean` and `cost_usd_per_trace` in
# MAGIC    the experiment UI.
# MAGIC 2. What sample size keeps monitoring under 1% of the daily budget? Work it
# MAGIC    out from `cost_usd_per_trace` — that's a real capacity-planning exercise.
