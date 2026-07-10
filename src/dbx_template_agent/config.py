"""Single source of truth for this project's configuration.

Every value below was filled in when you ran `databricks bundle init` — this is
the only file that knows your catalog, endpoints, and budget. Notebooks, the
agent, and the eval harness all import from here, so changing an endpoint or
budget is a one-line edit in one place.

Two helpers enforce the MLflow tracking discipline this project teaches:

- ``setup_mlflow(task)`` — the ONLY way code here selects an experiment.
  One experiment per concern lives under ``EXPERIMENT_BASE``:
  ``/agent`` (live traces), ``/experiments`` (knob comparisons),
  ``/evals`` (gate runs), ``/monitoring`` (quality/cost series).
- ``run_tags()`` — things you filter and group runs by (git commit, agent,
  dataset version) belong in TAGS. Scalar knobs (temperature, max_tokens)
  belong in PARAMS. Never smuggle metadata into a run name.
"""

from __future__ import annotations

import subprocess

# --- Filled in from your `databricks bundle init` answers -------------------
CATALOG = "main"
SCHEMA = "dbx_template_agent"
CHAT_ENDPOINT = "databricks-claude-sonnet-4-5"
JUDGE_ENDPOINT = "databricks-gpt-5-mini"
EXPERIMENT_BASE = "/Shared/dbx_template_agent"
DAILY_BUDGET_USD = float("5")
PROJECT = "dbx_template_agent"

# --- Derived names (UC three-level namespace, snake_case) -------------------
UC_PREFIX = f"{CATALOG}.{SCHEMA}"
PROMPT_NAME = f"{UC_PREFIX}.{PROJECT}_system"  # MLflow Prompt Registry entry
MODEL_NAME = f"{UC_PREFIX}.{PROJECT}_model"  # UC-registered agent model
EVAL_DATASET_NAME = f"{UC_PREFIX}.{PROJECT}_eval_set"

# The judge is addressed as a model URI by mlflow.genai judges/scorers.
JUDGE_MODEL_URI = f"endpoints:/{JUDGE_ENDPOINT}"

# Tasks -> experiments. One experiment per concern; nothing else creates one.
TASKS = ("agent", "experiments", "evals", "monitoring")


def setup_mlflow(task: str) -> str:
    """Point MLflow at the right tracking server + experiment for ``task``.

    This is the single entry point for experiment selection — call it before
    logging anything, and never call ``mlflow.set_experiment`` directly.
    """
    import mlflow

    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {TASKS}")
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    experiment_name = f"{EXPERIMENT_BASE}/{task}"
    mlflow.set_experiment(experiment_name)
    return experiment_name


def git_commit() -> str:
    """Current git commit SHA, or 'unknown' outside a checkout (e.g. a job run)."""
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def run_tags(**extra: str) -> dict[str, str]:
    """Standard tags every run should carry (filterable in the UI/API).

    Tags answer "which runs am I looking at?" — agent, commit, dataset version.
    Knobs like temperature go in params, not here.
    """
    tags = {
        "agent": PROJECT,
        "git_commit": git_commit(),
        "chat_endpoint": CHAT_ENDPOINT,
    }
    tags.update(extra)
    return tags
