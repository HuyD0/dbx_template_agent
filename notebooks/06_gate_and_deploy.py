# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Gate & deploy
# MAGIC
# MAGIC ⏱ ~30 min · 🟠 harder
# MAGIC
# MAGIC **You are here:**
# MAGIC `setup → build → experiment → evaluate → improve evals → [GATE/DEPLOY]* → monitor`
# MAGIC
# MAGIC The gate is the whole point of the previous phases: **no candidate reaches
# MAGIC production without clearing the eval floors** — quality AND cost. You'll run
# MAGIC the full gate, register the agent in Unity Catalog, do a champion/challenger
# MAGIC comparison, and promote by moving aliases.

# COMMAND ----------

# MAGIC %pip install -q mlflow>=3.14.0 databricks-sdk>=0.38.0 openai>=1.40 pandas pyyaml
# MAGIC %restart_python

# COMMAND ----------

import sys, os

sys.path.insert(0, os.path.abspath("../src"))

from dbx_template_agent import config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the gate
# MAGIC The gate lives in `evals/run_agent_eval.py` — a plain script, so the SAME
# MAGIC code runs here, locally, or as the bundle job:
# MAGIC
# MAGIC ```
# MAGIC databricks bundle deploy
# MAGIC databricks bundle run dbx_template_agent_eval_gate
# MAGIC ```
# MAGIC
# MAGIC It refuses to start if the estimated cost exceeds what's left of today's
# MAGIC budget (cost control **fails closed**), links the dataset as a run input,
# MAGIC runs judges + deterministic scorers, and exits non-zero on any floor breach.

# COMMAND ----------

import subprocess

proc = subprocess.run(
    [sys.executable, "../evals/run_agent_eval.py"], capture_output=True, text=True
)
print(proc.stdout[-3000:])
print(proc.stderr[-1000:])
GATE_PASSED = proc.returncode == 0
print(f"gate exit code: {proc.returncode} → {'PASS' if GATE_PASSED else 'FAIL'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the agent as a UC model
# MAGIC Models-from-code: MLflow logs the agent's source, pinned deps, and the
# MAGIC prompt version in play — the full recipe, not a pickle.

# COMMAND ----------

import mlflow

config.setup_mlflow("evals")
mlflow.set_registry_uri("databricks-uc")

from dbx_template_agent.agent import run_agent


class AgentModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        questions = (
            model_input["question"].tolist()
            if hasattr(model_input, "columns")
            else list(model_input)
        )
        return [run_agent(q)["answer"] for q in questions]


with mlflow.start_run(run_name=f"{config.CHAT_ENDPOINT}-register"):
    mlflow.set_tags(config.run_tags(task="register"))
    info = mlflow.pyfunc.log_model(
        name="agent",
        python_model=AgentModel(),
        registered_model_name=config.MODEL_NAME,
        pip_requirements=["mlflow>=3.14.0", "databricks-sdk>=0.38.0", "openai>=1.40"],
    )
print(f"registered {config.MODEL_NAME} v{info.registered_model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Champion / challenger
# MAGIC Every candidate gets `@challenger`. It takes `@champion` ONLY if the gate
# MAGIC passed. Rollback is moving the alias back — no redeploy, no code change.

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()
version = info.registered_model_version

client.set_registered_model_alias(config.MODEL_NAME, "challenger", version)
print(f"@challenger → v{version}")

if GATE_PASSED:
    client.set_registered_model_alias(config.MODEL_NAME, "champion", version)
    print(f"@champion   → v{version}  (gate passed)")
else:
    print("Gate failed — champion unchanged. Fix, re-run the gate, then promote.")

# Resolve BY ALIAS downstream — never hardcode a version:
champion = client.get_model_version_by_alias(config.MODEL_NAME, "champion")
print(f"current champion: v{champion.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serving (when you're ready)
# MAGIC `resources/dbx_template_agent_agent_serving.yml` ships commented out because a
# MAGIC serving endpoint needs a registered model version — which now exists.
# MAGIC Uncomment it and `databricks bundle deploy` to stand up the endpoint.
# MAGIC
# MAGIC The prompt promotes the same way: when a NEW prompt version clears the gate,
# MAGIC `mlflow.genai.set_prompt_alias(config.PROMPT_NAME, "production", <version>)`
# MAGIC — the running agent picks it up on next load, no image rebuild.

# COMMAND ----------

# ✅ CHECKPOINT
champion = client.get_model_version_by_alias(config.MODEL_NAME, "champion") if GATE_PASSED else None
assert version, "Model was not registered"
if GATE_PASSED:
    assert champion.version == str(version), "Champion alias should point at the new version"
    print(f"✅ Phase 06 complete — gate PASSED, {config.MODEL_NAME} v{version} is champion.")
else:
    print("⚠️ Phase 06 ran end to end but the gate FAILED — that's the gate doing its job.")
    print("   Read the breach report above, fix, and re-run before promoting.")
print("   Next: notebooks/07_monitor_cost_and_quality.py")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🏋️ Try this
# MAGIC 1. Break the gate on purpose: set `correctness/mean: 0.99` in thresholds.yaml
# MAGIC    and re-run. Read the breach report. Put it back.
# MAGIC 2. Register a "bad" variant (temperature=1.5) as challenger and confirm the
# MAGIC    gate refuses it the champion alias.
