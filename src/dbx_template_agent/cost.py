"""Token -> dollars accounting.

Cost is a first-class metric in this project: every agent call records its
token usage on the trace, the eval gate enforces a cost-per-answer ceiling,
and the monitoring notebook aggregates spend against the daily budget.

The pricing table below holds ESTIMATES in USD per 1M tokens. Rates vary by
cloud, region, and contract — check your account's actual serving rates and
edit the table; the point of the exercise is the plumbing, not these numbers.
"""

from __future__ import annotations

from typing import Any

from .config import DAILY_BUDGET_USD

# USD per 1M tokens: {"input": ..., "output": ...}. EDIT ME to your rates.
PRICING_USD_PER_1M: dict[str, dict[str, float]] = {
    "databricks-claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "databricks-claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "databricks-gpt-5-mini": {"input": 0.25, "output": 2.00},
    "databricks-gpt-5-nano": {"input": 0.05, "output": 0.40},
    "databricks-meta-llama-3-3-70b-instruct": {"input": 0.50, "output": 1.50},
}

# Used when an endpoint isn't in the table — deliberately pessimistic so an
# unknown model over-counts rather than under-counts spend.
DEFAULT_PRICING = {"input": 5.00, "output": 25.00}


def usage_to_usd(usage: dict[str, Any] | None, endpoint: str) -> float:
    """Convert a chat-completions ``usage`` block to dollars for ``endpoint``."""
    if not usage:
        return 0.0
    rates = PRICING_USD_PER_1M.get(endpoint, DEFAULT_PRICING)
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    return (prompt * rates["input"] + completion * rates["output"]) / 1_000_000


def add_usage(total: dict[str, int], usage: dict[str, Any] | None) -> dict[str, int]:
    """Accumulate a usage block into a running total (mutates and returns it)."""
    if usage:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            total[key] = total.get(key, 0) + int(usage.get(key) or 0)
    return total


def trace_cost_usd(trace: Any, endpoint: str) -> float:
    """Sum the cost of every CHAT_MODEL span in an MLflow trace.

    Works on any trace whose model spans carry a ``usage`` attribute — which
    is exactly what ``agent.run_agent`` records. This is what the phase-05
    cost scorer and the phase-07 monitoring aggregation read.
    """
    total = 0.0
    for span in getattr(trace.data, "spans", []) or []:
        if getattr(span, "span_type", None) != "CHAT_MODEL":
            continue
        usage = (span.attributes or {}).get("usage")
        total += usage_to_usd(usage, endpoint)
    return total


def budget_remaining_usd(spent_today_usd: float) -> float:
    """Dollars left under today's budget (never negative)."""
    return max(0.0, DAILY_BUDGET_USD - spent_today_usd)
