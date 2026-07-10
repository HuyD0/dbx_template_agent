"""The eval gate: judges + deterministic scorers + threshold floors + cost.

Run locally:            make eval        (or: python evals/run_agent_eval.py)
Run as a Databricks job: databricks bundle run dbx_template_agent_eval_gate

Exit code 0 = every floor met; 1 = gate breached. That exit code is the
integration point: a CI system dispatching this job can gate a promotion on it.

What "properly logged" means here (this file is the worked example):
- the eval set is logged as a run INPUT (dataset lineage, not just a tag),
- git commit / agent / dataset_version are TAGS (filterable),
- knobs (endpoint, temperature) are PARAMS,
- every scorer emits METRICS, plus cost + token totals for the run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make src/ importable when run as a plain script or a spark_python_task
# (bundle-deployed files keep this repo layout in the workspace). A
# spark_python_task exec()s this file, so `__file__` may be undefined; fall
# back to argv[0], then to the cwd, and locate the repo root by finding the
# ancestor that actually contains src/.
def _repo_root() -> Path:
    candidates = []
    try:
        candidates.append(Path(__file__).resolve())
    except NameError:
        pass
    if sys.argv and sys.argv[0]:
        candidates.append(Path(sys.argv[0]).resolve())
    for start in candidates:
        for parent in start.parents:
            if (parent / "src").is_dir():
                return parent
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / "src").is_dir():
            return parent
    return Path.cwd()


ROOT = _repo_root()
sys.path.insert(0, str(ROOT / "src"))

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from dbx_template_agent import config  # noqa: E402
from dbx_template_agent.agent import run_agent  # noqa: E402
from dbx_template_agent.cost import budget_remaining_usd  # noqa: E402
from dbx_template_agent.scorers import (  # noqa: E402
    format_ok,
    make_cost_scorer,
    make_latency_scorer,
    tool_call_correct,
)

EVAL_SET = ROOT / "evals" / "eval_set.jsonl"
THRESHOLDS = ROOT / "evals" / "thresholds.yaml"

# Bump when eval_set.jsonl rows change — it's how two runs on different data
# stay distinguishable in the UI.
DATASET_VERSION = "1"

# Pessimistic per-row estimate used for the fail-closed budget precheck.
EST_COST_PER_ROW_USD = 0.02


def load_rows() -> pd.DataFrame:
    rows = [json.loads(line) for line in EVAL_SET.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def predict_fn(question: str) -> dict:
    """One eval row -> one traced agent call. Stateless and reproducible."""
    result = run_agent(question)
    return {"answer": result["answer"], "cost_usd": result["cost_usd"]}


def spent_today_usd() -> float:
    """Best-effort spend already logged today (0.0 when unavailable).

    Fail-open on the LOOKUP (missing data must not block the run) but the
    budget check itself fails CLOSED: an over-budget estimate refuses to run.
    """
    try:
        traces = mlflow.search_traces(max_results=500)
        if "request_time" in traces.columns:
            today = pd.Timestamp.utcnow().normalize()
            traces = traces[pd.to_datetime(traces["request_time"], utc=True) >= today]
        return float(traces.get("cost_usd", pd.Series(dtype=float)).fillna(0).sum())
    except Exception:
        return 0.0


def check_gate(metrics: dict, thresholds: dict) -> list[str]:
    """Return a list of human-readable breaches (empty = gate passes)."""
    breaches = []
    floors = dict(thresholds.get("floors", {}))
    floors.update({k: v for k, v in thresholds.get("cost", {}).items() if k.endswith("/mean")})
    for metric, floor in floors.items():
        value = metrics.get(metric)
        if value is None:
            breaches.append(f"metric {metric!r} missing from results (floor {floor})")
        elif value < floor:
            breaches.append(f"{metric} = {value:.3f} below floor {floor}")
    return breaches


def main() -> int:
    thresholds = yaml.safe_load(THRESHOLDS.read_text())
    rows = load_rows()

    config.setup_mlflow("evals")

    # Cost control fails closed: refuse a run the budget can't cover.
    estimated = len(rows) * EST_COST_PER_ROW_USD
    remaining = budget_remaining_usd(spent_today_usd())
    if estimated > remaining:
        print(
            f"REFUSED: estimated run cost ${estimated:.2f} exceeds remaining "
            f"daily budget ${remaining:.2f} (DAILY_BUDGET_USD="
            f"{config.DAILY_BUDGET_USD}). Raise the budget in config.py or "
            "shrink the eval set."
        )
        return 1

    from mlflow.genai.scorers import Correctness, RelevanceToQuery

    scorers = [
        Correctness(model=config.JUDGE_MODEL_URI),
        RelevanceToQuery(model=config.JUDGE_MODEL_URI),
        tool_call_correct,
        format_ok,
        make_latency_scorer(thresholds["latency"]["per_answer_ms"]),
        make_cost_scorer(thresholds["cost"]["per_answer_usd"], config.CHAT_ENDPOINT),
    ]

    with mlflow.start_run(run_name=f"{config.CHAT_ENDPOINT}-eval-gate") as run:
        mlflow.set_tags(config.run_tags(dataset_version=DATASET_VERSION, task="eval-gate"))
        mlflow.log_params(
            {
                "chat_endpoint": config.CHAT_ENDPOINT,
                "judge_endpoint": config.JUDGE_ENDPOINT,
                "n_rows": len(rows),
            }
        )
        # Dataset LINEAGE: the run <-> dataset edge shows up in the UI's
        # Datasets tab. A tag alone would not be traversable.
        mlflow.log_input(
            mlflow.data.from_pandas(rows, name="eval_set", source=str(EVAL_SET)),
            context="evaluation",
        )

        results = mlflow.genai.evaluate(
            data=rows,
            predict_fn=lambda question: predict_fn(question),
            scorers=scorers,
        )
        metrics = dict(results.metrics)
        print("\n=== metrics ===")
        for key in sorted(metrics):
            print(f"  {key}: {metrics[key]}")

        breaches = check_gate(metrics, thresholds)
        mlflow.log_metric("gate_passed", 0.0 if breaches else 1.0)

    if breaches:
        print("\n❌ GATE FAILED:")
        for breach in breaches:
            print(f"  - {breach}")
        return 1
    print(f"\n✅ Gate passed. Run: {run.info.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
