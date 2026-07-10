# Databricks notebook source
# MAGIC %md
# MAGIC # 08 — Capstone: an Agent Bricks challenger (optional)
# MAGIC
# MAGIC ⏱ ~45 min · 🔴 challenge
# MAGIC
# MAGIC **You are here:** past the loop — now you test it against a rival.
# MAGIC
# MAGIC Databricks **Agent Bricks** builds managed agents (Knowledge Assistant,
# MAGIC Genie, multi-agent Supervisor) from configuration — no agent code. The
# MAGIC capstone question: *is the managed agent better than yours?* Feelings don't
# MAGIC answer that. **Your eval set does.** This is champion/challenger against a
# MAGIC system you didn't code — the exact discipline you'd use to evaluate any
# MAGIC vendor claim.
# MAGIC
# MAGIC > Agent Bricks availability varies by workspace/region (it's a gated
# MAGIC > product surface). If it's not enabled for you, run this notebook with any
# MAGIC > OTHER serving endpoint as the challenger — a bigger foundation model is a
# MAGIC > perfectly good rival; the method is identical.

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
from dbx_template_agent.cost import usage_to_usd

config.setup_mlflow("evals")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — stand up the challenger
# MAGIC In the workspace UI: **Agents** → create a Knowledge Assistant (point it at
# MAGIC a small corpus, e.g. these notebooks' markdown) or a Supervisor. Note its
# MAGIC serving endpoint name and paste it below.
# MAGIC
# MAGIC No Agent Bricks in your workspace? Use any chat endpoint as the challenger.

# COMMAND ----------

CHALLENGER_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"  # ← EDIT ME

from databricks.sdk import WorkspaceClient

client = WorkspaceClient().serving_endpoints.get_open_ai_client()


def challenger_answer(question: str) -> dict:
    """The challenger, wrapped in the SAME trace + cost contract as our agent."""
    with mlflow.start_span(name="challenger", span_type="AGENT") as span:
        span.set_inputs({"question": question})
        response = client.chat.completions.create(
            model=CHALLENGER_ENDPOINT,
            messages=[{"role": "user", "content": question}],
            max_tokens=512,
        )
        usage = response.usage
        usage_dict = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
        answer = response.choices[0].message.content or ""
        span.set_attributes({"usage": usage_dict})
        span.set_outputs({"answer": answer})
    return {"answer": answer, "cost_usd": usage_to_usd(usage_dict, CHALLENGER_ENDPOINT)}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — same eval set, same judges, both agents
# MAGIC The ONLY thing that changes between the two runs is `predict_fn`. Same data,
# MAGIC same scorers, same judge — otherwise the comparison is theater.

# COMMAND ----------

from mlflow.genai.scorers import Correctness, RelevanceToQuery

lines = [line for line in Path("../evals/eval_set.jsonl").read_text().splitlines() if line.strip()]
rows = pd.DataFrame([json.loads(line) for line in lines])
judges = [Correctness(model=config.JUDGE_MODEL_URI), RelevanceToQuery(model=config.JUDGE_MODEL_URI)]

outcomes = {}
for name, fn in [
    ("champion", lambda q: {"answer": run_agent(q)["answer"]}),
    ("challenger", lambda q: {"answer": challenger_answer(q)["answer"]}),
]:
    with mlflow.start_run(run_name=f"{name}-capstone"):
        mlflow.set_tags(config.run_tags(task="capstone", contender=name))
        mlflow.log_input(
            mlflow.data.from_pandas(rows, name="eval_set", source="evals/eval_set.jsonl"),
            context="evaluation",
        )
        results = mlflow.genai.evaluate(data=rows, predict_fn=fn, scorers=judges)
        outcomes[name] = dict(results.metrics)

# COMMAND ----------

comparison = pd.DataFrame(outcomes)
print(comparison.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — decide like the gate would
# MAGIC Quality means AND cost per answer. A challenger that wins by 2 points but
# MAGIC costs 6× is a judgment call — that's why thresholds carry a cost ceiling,
# MAGIC and why the decision is a rule you wrote BEFORE seeing the numbers.

# COMMAND ----------

# ✅ CHECKPOINT
assert set(outcomes) == {"champion", "challenger"}, "Both contenders must be evaluated"
shared = set(outcomes["champion"]) & set(outcomes["challenger"])
assert any("correctness" in metric for metric in shared), "Comparable correctness metrics missing"
print("✅ Capstone complete — you evaluated a system you didn't build, with YOUR eval set.")
print("   That's the whole skill. Point it at your real project next.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🏋️ Where to go from here
# MAGIC - Swap the toy tools for your real ones; the loop, gate, and monitoring don't change.
# MAGIC - Wire `databricks bundle run dbx_template_agent_eval_gate` into your CI as a
# MAGIC   manual-dispatch promotion gate.
# MAGIC - Rows from real 👎s (phase 05) beat invented rows — keep mining.
