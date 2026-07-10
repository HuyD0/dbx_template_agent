# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Experiments: change one knob, compare honestly
# MAGIC
# MAGIC ⏱ ~25 min · 🟡 moderate
# MAGIC
# MAGIC **You are here:**
# MAGIC `setup → build → [EXPERIMENT]* → evaluate → improve evals → gate/deploy → monitor`
# MAGIC
# MAGIC You'll register a **v2 prompt**, run the agent under v1 and v2 on the same
# MAGIC questions, and log each as a comparable MLflow **run** — with the
# MAGIC tags-vs-params discipline that keeps runs queryable months later.
# MAGIC
# MAGIC > **Why this matters** — "we tried a new prompt and it felt better" is not an
# MAGIC > experiment. Two runs, same inputs, one changed knob, logged lineage — that
# MAGIC > is. The UI's run-compare view only works if you log knobs consistently.
# MAGIC
# MAGIC **The discipline (enforced by `config.run_tags`):**
# MAGIC | Where | What | Examples |
# MAGIC |---|---|---|
# MAGIC | tags | what you FILTER runs by | git commit, agent, dataset_version |
# MAGIC | params | the KNOBS this run turned | prompt_version, temperature |
# MAGIC | metrics | what happened | mean cost, latency, scores |

# COMMAND ----------

# MAGIC %pip install -q mlflow>=3.14.0 databricks-sdk>=0.38.0 openai>=1.40 pandas
# MAGIC %restart_python

# COMMAND ----------

import sys, os

sys.path.insert(0, os.path.abspath("../src"))

import mlflow

from dbx_template_agent import config
from dbx_template_agent.agent import run_agent

config.setup_mlflow("experiments")
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# Register a v2 prompt: same job, tighter style contract.
V2_TEMPLATE = (
    "You are a precise assistant. Use the calculator tool for ANY arithmetic "
    "and the lookup tool for definitions instead of answering from memory. "
    "Answer in ONE sentence, lead with the result, no preamble."
)
v2 = mlflow.genai.register_prompt(
    name=config.PROMPT_NAME,
    template=V2_TEMPLATE,
    commit_message="v2: one-sentence, result-first style",
)
print(f"registered v{v2.version}")

# COMMAND ----------

QUESTIONS = [
    "What is 17 * 43 + 5?",
    "Define 'LLM judge' for me.",
    "If I run 250 evals at $0.004 each, what does it cost?",
]


def run_variant(prompt_version: int, template: str, temperature: float) -> None:
    """One variant = one MLflow run: tags for identity, params for knobs."""
    with mlflow.start_run(run_name=f"{config.CHAT_ENDPOINT}-prompt-v{prompt_version}"):
        mlflow.set_tags(config.run_tags(task="prompt-comparison"))
        mlflow.log_params(
            {
                "prompt_version": prompt_version,
                "temperature": temperature,
                "n_questions": len(QUESTIONS),
            }
        )
        total_cost, total_chars = 0.0, 0
        for question in QUESTIONS:
            result = run_agent(question, system_prompt=template, temperature=temperature)
            total_cost += result["cost_usd"]
            total_chars += len(result["answer"])
        mlflow.log_metrics(
            {
                "cost_usd_total": total_cost,
                "cost_usd_per_q": total_cost / len(QUESTIONS),
                "answer_chars_mean": total_chars / len(QUESTIONS),
            }
        )

# COMMAND ----------

v1_template = mlflow.genai.load_prompt(f"prompts:/{config.PROMPT_NAME}/1").template
run_variant(1, v1_template, temperature=0.1)
run_variant(v2.version, V2_TEMPLATE, temperature=0.1)

# COMMAND ----------

# Compare runs AS DATA (the UI's compare view shows the same thing):
runs = mlflow.search_runs(filter_string="tags.task = 'prompt-comparison'")
runs[
    ["run_id", "params.prompt_version", "metrics.cost_usd_per_q", "metrics.answer_chars_mean"]
].head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Don't promote yet
# MAGIC v2 probably produced shorter (cheaper) answers — but shorter isn't *better*.
# MAGIC "Better" is what the **eval gate** decides in phases 04–06, with judges and
# MAGIC floors, not eyeballs. When it passes there, promotion is one alias move:
# MAGIC
# MAGIC ```python
# MAGIC mlflow.genai.set_prompt_alias(config.PROMPT_NAME, alias="production", version=2)
# MAGIC ```
# MAGIC
# MAGIC > **Why alias, not version number** — downstream code loads `@production`
# MAGIC > and never edits a hardcoded version. Rollback is also one alias move.

# COMMAND ----------

# ✅ CHECKPOINT
runs = mlflow.search_runs(filter_string="tags.task = 'prompt-comparison'")
versions = set(runs["params.prompt_version"])
assert len(runs) >= 2, "Expected at least two comparison runs"
assert len(versions) >= 2, f"Runs should span two prompt versions, got {versions}"
assert runs["metrics.cost_usd_per_q"].notna().all(), "Every run must log its cost"
print("✅ Phase 03 complete — two prompt versions, comparable runs, costs logged.")
print("   Next: notebooks/04_design_your_evals.py")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🏋️ Try this
# MAGIC 1. Add a `temperature=0.9` variant. One knob per run — why does changing two
# MAGIC    knobs at once destroy the comparison?
# MAGIC 2. Filter runs by `tags.git_commit`. That's why commit goes in tags: "which
# MAGIC    code produced this number?" should always be answerable.
