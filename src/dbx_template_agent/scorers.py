"""Scorers: deterministic checks + a custom LLM judge factory.

The rule this module teaches: **don't use an LLM judge where a regex works.**
Deterministic scorers are exact, free, and fast — reach for a judge only when
the criterion genuinely needs language understanding (and then calibrate it
against human labels before it gates anything; see notebook 05).

Each scorer has a plain, offline-testable core function (prefixed ``check_``)
and a thin ``@scorer``-decorated wrapper that adapts it to
``mlflow.genai.evaluate``. Tests target the cores; evaluate uses the wrappers.
"""

from __future__ import annotations

import re
from typing import Any

from mlflow.genai.scorers import scorer

from .config import JUDGE_MODEL_URI
from .cost import trace_cost_usd

# --- deterministic cores (plain functions, unit-tested offline) ---------------


def tool_calls_from_trace(trace: Any) -> list[str]:
    """Names of TOOL spans in call order — the trace contract in action."""
    calls = []
    for span in getattr(trace.data, "spans", []) or []:
        if getattr(span, "span_type", None) == "TOOL":
            calls.append(span.name)
    return calls


def check_expected_tool(called: list[str], expected: str | None) -> bool:
    """Did the agent call the tool this eval row expects?

    ``expected`` of None means the row expects NO tool call (the model should
    answer directly) — that's a real failure mode too: tool overuse costs
    money and latency.
    """
    if expected is None:
        return len(called) == 0
    return expected in called


_APOLOGY = re.compile(r"\b(i apologi[sz]e|i'?m sorry)\b", re.IGNORECASE)


def check_format(answer: str, max_chars: int = 2000) -> bool:
    """Cheap output hygiene: non-empty, bounded length, no filler apology."""
    text = (answer or "").strip()
    return bool(text) and len(text) <= max_chars and not _APOLOGY.search(text)


def check_latency(execution_ms: float | None, ceiling_ms: float) -> bool:
    """Trace-level latency under the ceiling (None = unknown = fail loud)."""
    return execution_ms is not None and execution_ms <= ceiling_ms


def check_cost(cost_usd: float, ceiling_usd: float) -> bool:
    """Per-answer cost under the ceiling."""
    return cost_usd <= ceiling_usd


# --- mlflow.genai.evaluate adapters -------------------------------------------
# Signature contract for custom scorers:
#   inputs / outputs / expectations come from the eval row + predict_fn,
#   trace is the MLflow trace produced while answering that row.


@scorer
def tool_call_correct(inputs: dict, outputs: dict, expectations: dict, trace: Any) -> bool:
    """Deterministic: agent used the tool the row expects (or none, if none)."""
    expected = (expectations or {}).get("expected_tool")
    return check_expected_tool(tool_calls_from_trace(trace), expected)


@scorer
def format_ok(outputs: dict) -> bool:
    """Deterministic: answer is non-empty, bounded, and not an apology."""
    answer = outputs.get("answer", "") if isinstance(outputs, dict) else str(outputs)
    return check_format(answer)


def make_latency_scorer(ceiling_ms: float):
    """A latency ceiling scorer bound to a threshold from thresholds.yaml."""

    @scorer(name="latency_ok")
    def latency_ok(trace: Any) -> bool:
        info = getattr(trace, "info", None)
        return check_latency(getattr(info, "execution_time_ms", None), ceiling_ms)

    return latency_ok


def make_cost_scorer(ceiling_usd: float, endpoint: str):
    """A cost-per-answer ceiling scorer — the budget discipline, per row."""

    @scorer(name="cost_ok")
    def cost_ok(trace: Any) -> bool:
        return check_cost(trace_cost_usd(trace, endpoint), ceiling_usd)

    return cost_ok


# --- custom LLM judge ----------------------------------------------------------


def make_domain_judge(name: str = "helpfulness_judge"):
    """A custom LLM judge built with ``mlflow.genai.judges.make_judge``.

    Notebook 05 aligns this judge against your human 👍/👎 labels with
    ``judge.align()`` and measures pre/post agreement on a holdout — a judge
    that hasn't been validated against humans must not gate a release.
    """
    from mlflow.genai.judges import make_judge

    return make_judge(
        name=name,
        instructions=(
            "Evaluate whether the response in {{ outputs }} correctly and "
            "directly answers the question in {{ inputs }}. A good response "
            "answers the actual question, uses tool results rather than "
            "guessing at arithmetic, and does not pad with disclaimers. "
            "Answer 'yes' or 'no'."
        ),
        model=JUDGE_MODEL_URI,
    )
