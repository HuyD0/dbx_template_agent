# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Build the agent (and read its traces)
# MAGIC
# MAGIC ⏱ ~25 min · 🟢 easy
# MAGIC
# MAGIC **You are here:**
# MAGIC `setup → [BUILD]* → experiment → evaluate → improve evals → gate/deploy → monitor`
# MAGIC
# MAGIC This phase walks the agent's source (`src/dbx_template_agent/agent.py` +
# MAGIC `tools.py`), then works with traces as DATA: the span tree, span attributes,
# MAGIC and `mlflow.search_traces` queries. The span structure you learn here is the
# MAGIC **trace contract** — phase 05's deterministic scorers consume exactly it.
# MAGIC
# MAGIC > **Why this matters** — "your trace contract is your eval surface." An agent
# MAGIC > with disciplined spans can be evaluated with cheap exact checks; an agent
# MAGIC > with mushy traces forces you to pay an LLM judge for everything.

# COMMAND ----------

# MAGIC %pip install -q mlflow>=3.14.0 databricks-sdk>=0.38.0 openai>=1.40
# MAGIC %restart_python

# COMMAND ----------

import sys, os

sys.path.insert(0, os.path.abspath("../src"))

from dbx_template_agent import config

config.setup_mlflow("agent")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The loop, in one screen
# MAGIC Open `src/dbx_template_agent/agent.py` side by side. The whole agent is:
# MAGIC call the model → if it asked for tools, run them inside TOOL spans and append
# MAGIC results → repeat until it answers in prose (or the iteration budget runs out).
# MAGIC No framework; every span in the trace maps to a line you can point at.

# COMMAND ----------

from dbx_template_agent.agent import run_agent

# A question that needs BOTH tools:
result = run_agent(
    "Look up what an MLflow trace is, then compute how many spans 3 requests "
    "produce if each has 4 spans."
)
print(result["answer"])
print("tools used:", result["tools_used"])
print(f"cost: ${result['cost_usd']:.5f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Traces as data — not just pictures
# MAGIC The UI is for reading one trace; `mlflow.search_traces` is for asking
# MAGIC questions across MANY traces. That's the observability muscle.

# COMMAND ----------

import mlflow

traces = mlflow.search_traces(max_results=50)
print(f"{len(traces)} traces in the agent experiment")
traces[["trace_id", "state", "execution_duration"]].head()

# COMMAND ----------

# Which traces called the calculator? (span-level filtering, in plain pandas)
from dbx_template_agent.scorers import tool_calls_from_trace

rows = []
for trace_id in traces["trace_id"].head(20):
    trace = mlflow.get_trace(trace_id)
    rows.append({"trace_id": trace_id, "tools": tool_calls_from_trace(trace)})

import pandas as pd

tool_usage = pd.DataFrame(rows)
print(tool_usage.to_string(index=False))

# COMMAND ----------

# Token usage + cost, straight off the CHAT_MODEL spans (the trace contract):
from dbx_template_agent.cost import trace_cost_usd

trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
for span in trace.data.spans:
    if span.span_type == "CHAT_MODEL":
        print(span.name, (span.attributes or {}).get("usage"))
print(f"whole-trace cost: ${trace_cost_usd(trace, config.CHAT_ENDPOINT):.5f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add your own tool (the extension point)
# MAGIC In `src/dbx_template_agent/tools.py`: write the function → add a spec to
# MAGIC `TOOL_SPECS` → register it in `_REGISTRY`. Three edits, and the loop picks it
# MAGIC up with zero changes to `agent.py`. **Rule of the house:** a new tool ships
# MAGIC with an eval row that exercises it (you'll add rows in phase 04).

# COMMAND ----------

# ✅ CHECKPOINT
trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
types = [s.span_type for s in trace.data.spans]
assert types.count("CHAT_MODEL") >= 2, "Expected at least 2 model calls in the two-tool question"
assert "TOOL" in types, "Expected TOOL spans"
assert trace_cost_usd(trace, config.CHAT_ENDPOINT) > 0, "Cost should be positive"
print("✅ Phase 02 complete — you can read a span tree and query traces as data.")
print("   Next: notebooks/03_experiments.py")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🏋️ Try this
# MAGIC 1. Add a `today()` tool returning the current date; ask "what day is it?" and
# MAGIC    confirm the TOOL span appears.
# MAGIC 2. Break `run_tool` on purpose (return an error string) and watch how the
# MAGIC    model recovers — errors-as-strings is why the loop survives bad tools.
