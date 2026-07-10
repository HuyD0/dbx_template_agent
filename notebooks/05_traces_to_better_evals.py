# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Traces → better evals (the improvement loop)
# MAGIC
# MAGIC ⏱ ~40 min · 🟠 harder
# MAGIC
# MAGIC **You are here:**
# MAGIC `setup → build → experiment → evaluate → [IMPROVE EVALS]* → gate/deploy → monitor`
# MAGIC
# MAGIC Your eval suite should get better the same way your agent does: from
# MAGIC production evidence. Three moves, in increasing sophistication:
# MAGIC
# MAGIC 1. **Mine traces into eval rows** — human 👍/👎 feedback on real traces
# MAGIC    becomes new `eval_set.jsonl` rows (real failures, not invented ones).
# MAGIC 2. **Deterministic scorers from trace structure** — exact, free checks read
# MAGIC    from the span tree. *Don't use an LLM judge where a regex works.*
# MAGIC 3. **A custom judge, calibrated against humans** — built with `make_judge`,
# MAGIC    validated with `judge.align()` before it's allowed to gate anything.

# COMMAND ----------

# MAGIC %pip install -q mlflow>=3.14.0 databricks-sdk>=0.38.0 openai>=1.40 pandas
# MAGIC %restart_python

# COMMAND ----------

import sys, os

sys.path.insert(0, os.path.abspath("../src"))

import mlflow

from dbx_template_agent import config
from dbx_template_agent.agent import run_agent

config.setup_mlflow("agent")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Move 1 — feedback on traces, then mine the 👎s
# MAGIC In production, feedback comes from a UI control or a labeling session. Here
# MAGIC we simulate a session: run a few questions, then attach human assessments to
# MAGIC the exact traces with `mlflow.log_feedback`.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from mlflow.entities import AssessmentSource, AssessmentSourceType

ME = WorkspaceClient().current_user.me().user_name

SESSION = [
    ("What is 15% of 80?", True),           # good
    ("Define 'unity catalog' for me.", True),  # good
    ("What is the meaning of life?", False),   # 👎 — no tool covers this; agent waffled
]

labeled = []
for question, thumbs_up in SESSION:
    run_agent(question)
    trace_id = mlflow.get_last_active_trace_id()
    mlflow.log_feedback(
        trace_id=trace_id,
        name="user_feedback",
        value=thumbs_up,
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id=ME),
        rationale=None if thumbs_up else "Agent should say it can't answer, briefly.",
    )
    labeled.append((trace_id, question, thumbs_up))
print(f"labeled {len(labeled)} traces")

# COMMAND ----------

# Mine the 👎 traces into candidate eval rows:
import json

candidates = []
for trace_id, question, thumbs_up in labeled:
    if thumbs_up:
        continue
    trace = mlflow.get_trace(trace_id)
    feedback = next(a for a in trace.info.assessments if a.name == "user_feedback")
    candidates.append(
        {
            "inputs": {"question": question},
            "expectations": {
                # The human's rationale seeds the expectation — a REVIEWER
                # finalizes it before it merges into eval_set.jsonl.
                "expected_response": feedback.rationale,
                "expected_tool": None,
            },
        }
    )
print(json.dumps(candidates, indent=2))
# Append reviewed candidates to evals/eval_set.jsonl and bump DATASET_VERSION.

# COMMAND ----------

# MAGIC %md
# MAGIC > **Why this matters** — eval rows mined from real 👎s track how users
# MAGIC > actually break your agent. Invented rows track how you imagine they might.
# MAGIC > The promotion into `eval_set.jsonl` stays a HUMAN edit (a PR, reviewed) —
# MAGIC > pipelines should propose, people should promote.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Move 2 — deterministic scorers (already wired into the gate)
# MAGIC `src/dbx_template_agent/scorers.py` reads the span tree you learned in
# MAGIC phase 02: `tool_call_correct` (right tool per row), `format_ok`,
# MAGIC latency + cost ceilings. Exact, instant, $0 per row.

# COMMAND ----------

from dbx_template_agent.scorers import check_expected_tool, tool_calls_from_trace

trace = mlflow.get_trace(labeled[0][0])  # the "15% of 80" trace
called = tool_calls_from_trace(trace)
print("tools called:", called, "→ expected calculator:", check_expected_tool(called, "calculator"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Move 3 — a custom judge you can trust
# MAGIC `make_judge` builds it; `align()` calibrates it against your human labels.
# MAGIC **Agreement with humans on a holdout is the judge's eval** — a judge that
# MAGIC hasn't been measured against people has no business gating a release.

# COMMAND ----------

from dbx_template_agent.scorers import make_domain_judge

judge = make_domain_judge()

# Score the labeled traces with the uncalibrated judge and measure agreement:
agree = 0
for trace_id, _question, human_label in labeled:
    trace = mlflow.get_trace(trace_id)
    feedback = judge(trace=trace)
    judge_label = str(feedback.value).lower() in ("yes", "true", "pass")
    agree += judge_label == human_label
print(f"pre-alignment agreement: {agree}/{len(labeled)}")

# COMMAND ----------

# MAGIC %md
# MAGIC With ~10+ labeled traces, run alignment (needs `dspy`; keep it out of the
# MAGIC default env — it's a tooling dep, not an agent dep):
# MAGIC
# MAGIC ```python
# MAGIC # %pip install dspy
# MAGIC traces_df = mlflow.search_traces()  # traces carrying user_feedback
# MAGIC aligned = judge.align(traces_df)    # optimizes instructions against labels
# MAGIC # Then: re-measure agreement on a HOLDOUT you did not align on,
# MAGIC # and register the aligned judge only on non-regression.
# MAGIC ```
# MAGIC
# MAGIC Name discipline: keep the human label `user_feedback` and the judge's output
# MAGIC under a different name (e.g. `predicted_user_feedback`) so LLM assessments
# MAGIC can never contaminate the human labels you align against next time.

# COMMAND ----------

# ✅ CHECKPOINT
trace = mlflow.get_trace(labeled[-1][0])
assessment_names = [a.name for a in trace.info.assessments]
assert "user_feedback" in assessment_names, "Feedback assessment missing from trace"
assert candidates, "No candidate eval rows mined from the 👎"
print("✅ Phase 05 complete — feedback on traces, mined rows, judge measured vs humans.")
print("   Next: notebooks/06_gate_and_deploy.py")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🏋️ Try this
# MAGIC 1. Write a deterministic scorer for "answer under 300 chars" and add it to the
# MAGIC    gate — did it need a judge? (No. That's the point.)
# MAGIC 2. Label 10 traces yourself, hold 3 out, align on 7, and report agreement on
# MAGIC    the 3. Would you let this judge gate a release yet?
