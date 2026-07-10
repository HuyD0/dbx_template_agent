"""Token -> dollars math, including the pessimistic-default and budget rules."""

from __future__ import annotations

from types import SimpleNamespace

from dbx_template_agent.config import DAILY_BUDGET_USD
from dbx_template_agent.cost import (
    DEFAULT_PRICING,
    PRICING_USD_PER_1M,
    add_usage,
    budget_remaining_usd,
    trace_cost_usd,
    usage_to_usd,
)

KNOWN = next(iter(PRICING_USD_PER_1M))


def test_usage_to_usd_known_endpoint():
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    rates = PRICING_USD_PER_1M[KNOWN]
    assert usage_to_usd(usage, KNOWN) == rates["input"] + rates["output"]


def test_unknown_endpoint_overcounts_not_undercounts():
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0}
    cost = usage_to_usd(usage, "some-endpoint-nobody-priced")
    assert cost == DEFAULT_PRICING["input"]
    assert cost >= max(r["input"] for r in PRICING_USD_PER_1M.values())


def test_usage_to_usd_edge_cases():
    assert usage_to_usd(None, KNOWN) == 0.0
    assert usage_to_usd({}, KNOWN) == 0.0
    assert usage_to_usd({"prompt_tokens": None, "completion_tokens": None}, KNOWN) == 0.0


def test_add_usage_accumulates():
    total = {}
    add_usage(total, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    add_usage(total, {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})
    assert total == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}


def test_trace_cost_sums_only_chat_model_spans():
    def span(span_type, usage=None):
        return SimpleNamespace(span_type=span_type, attributes={"usage": usage})

    trace = SimpleNamespace(
        data=SimpleNamespace(
            spans=[
                span("AGENT"),
                span("CHAT_MODEL", {"prompt_tokens": 1_000_000, "completion_tokens": 0}),
                span("TOOL"),
                span("CHAT_MODEL", {"prompt_tokens": 1_000_000, "completion_tokens": 0}),
            ]
        )
    )
    assert trace_cost_usd(trace, KNOWN) == 2 * PRICING_USD_PER_1M[KNOWN]["input"]


def test_budget_remaining_never_negative():
    assert budget_remaining_usd(0.0) == DAILY_BUDGET_USD
    assert budget_remaining_usd(DAILY_BUDGET_USD + 100) == 0.0
