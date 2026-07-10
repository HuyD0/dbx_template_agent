# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Setup & your first trace
# MAGIC
# MAGIC ⏱ ~15 min · 🟢 easy
# MAGIC
# MAGIC **You are here:**
# MAGIC `[SETUP]* → build → experiment → evaluate → improve evals → gate/deploy → monitor`
# MAGIC
# MAGIC By the end of this notebook you will have: verified your endpoints work,
# MAGIC registered your agent's system prompt in the **MLflow Prompt Registry**, and
# MAGIC produced **one trace** you can open in the MLflow UI — with its token usage.
# MAGIC
# MAGIC > **Why this matters** — everything later (evals, judges, cost, monitoring)
# MAGIC > reads from traces. A team that can't see its traces is debugging blind.

# COMMAND ----------

# MAGIC %pip install -q mlflow>=3.14.0 databricks-sdk>=0.38.0 openai>=1.40
# MAGIC %restart_python

# COMMAND ----------

import sys
import os

sys.path.insert(0, os.path.abspath("../src"))

from dbx_template_agent import config

print(f"catalog.schema : {config.UC_PREFIX}")
print(f"agent endpoint : {config.CHAT_ENDPOINT}")
print(f"judge endpoint : {config.JUDGE_ENDPOINT}")
print(f"experiments    : {config.EXPERIMENT_BASE}/<task>")
print(f"daily budget   : ${config.DAILY_BUDGET_USD}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preflight
# MAGIC One cheap call to each endpoint. If this cell fails, fix it NOW — everything
# MAGIC else depends on it. See `TROUBLESHOOTING.md` for the usual suspects.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

client = WorkspaceClient().serving_endpoints.get_open_ai_client()

for label, endpoint in [("agent", config.CHAT_ENDPOINT), ("judge", config.JUDGE_ENDPOINT)]:
    response = client.chat.completions.create(
        model=endpoint,
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        max_tokens=10,
    )
    print(f"✅ {label} endpoint {endpoint}: {response.choices[0].message.content!r}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the system prompt
# MAGIC
# MAGIC > **Why registry, not a string literal** — a registered prompt has a version
# MAGIC > and an alias. Every eval score can then say *exactly which prompt text* it
# MAGIC > measured, and promoting a better prompt is an alias move — no code deploy.

# COMMAND ----------

import mlflow

from dbx_template_agent.agent import DEFAULT_SYSTEM_PROMPT

mlflow.set_registry_uri("databricks-uc")
prompt = mlflow.genai.register_prompt(
    name=config.PROMPT_NAME,
    template=DEFAULT_SYSTEM_PROMPT,
    commit_message="Initial system prompt from template",
)
mlflow.genai.set_prompt_alias(config.PROMPT_NAME, alias="production", version=prompt.version)
print(f"Registered {config.PROMPT_NAME} v{prompt.version} and set @production")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Your first trace

# COMMAND ----------

config.setup_mlflow("agent")

from dbx_template_agent.agent import run_agent

result = run_agent("What is 17 * 43 + 5?")
print("answer :", result["answer"])
print("tools  :", result["tools_used"])
print("tokens :", result["usage"])
print(f"cost   : ${result['cost_usd']:.5f}")

# COMMAND ----------

# MAGIC %md
# MAGIC Open the **Traces** tab of the `dbx_template_agent/agent` experiment (Experiments
# MAGIC in the left nav). You should see one trace whose span tree looks like:
# MAGIC
# MAGIC ```
# MAGIC run_agent (AGENT)
# MAGIC ├── chat_model (CHAT_MODEL)   ← usage + cost_usd attributes live here
# MAGIC ├── calculator (TOOL)         ← inputs: the expression; outputs: 736
# MAGIC └── chat_model (CHAT_MODEL)
# MAGIC ```
# MAGIC Click into a `chat_model` span and find the `usage` attribute — that's the
# MAGIC raw material for every cost number in this project.

# COMMAND ----------

# ✅ CHECKPOINT — run this cell; it verifies phase 01 end to end.
trace_id = mlflow.get_last_active_trace_id()
assert trace_id, "No trace found — did the run_agent cell above succeed?"
trace = mlflow.get_trace(trace_id)
span_types = {getattr(s, "span_type", None) for s in trace.data.spans}
assert "TOOL" in span_types, f"Expected a TOOL span, got: {span_types}"
loaded = mlflow.genai.load_prompt(f"prompts:/{config.PROMPT_NAME}@production")
assert loaded.template, "Prompt registry lookup failed"
print("✅ Phase 01 complete — endpoints verified, prompt registered, first trace captured.")
print("   Next: notebooks/02_build_the_agent.py")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🏋️ Try this
# MAGIC 1. Ask a question that needs **no** tool ("Say hello.") and compare the span
# MAGIC    tree — why is tool overuse a cost bug and not just a style issue?
# MAGIC 2. Change `temperature` in `run_agent` and re-run. Where would you record that
# MAGIC    knob so runs stay comparable? (Hint: phase 03 answers this.)
