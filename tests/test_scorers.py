"""Deterministic scorer cores, tested against fixture traces (no LLM, no judge)."""

from __future__ import annotations

from types import SimpleNamespace

from dbx_template_agent.scorers import (
    check_cost,
    check_expected_tool,
    check_format,
    check_latency,
    tool_calls_from_trace,
)


def fake_trace(spans):
    return SimpleNamespace(
        data=SimpleNamespace(spans=spans),
        info=SimpleNamespace(execution_time_ms=1200.0),
    )


def span(name, span_type, attributes=None):
    return SimpleNamespace(name=name, span_type=span_type, attributes=attributes or {})


def test_tool_calls_extracted_in_order():
    trace = fake_trace(
        [
            span("run_agent", "AGENT"),
            span("chat_model", "CHAT_MODEL"),
            span("calculator", "TOOL"),
            span("chat_model", "CHAT_MODEL"),
            span("lookup", "TOOL"),
        ]
    )
    assert tool_calls_from_trace(trace) == ["calculator", "lookup"]


def test_expected_tool_matching():
    assert check_expected_tool(["calculator"], "calculator")
    assert not check_expected_tool(["lookup"], "calculator")
    # expected None = the row expects NO tool call (tool overuse is a failure).
    assert check_expected_tool([], None)
    assert not check_expected_tool(["calculator"], None)


def test_format_checks():
    assert check_format("A concise answer.")
    assert not check_format("")
    assert not check_format("   ")
    assert not check_format("x" * 3000)
    assert not check_format("I apologize, but I cannot help with that.")
    assert not check_format("I'm sorry, something went wrong")


def test_latency_and_cost_ceilings():
    assert check_latency(1200.0, ceiling_ms=30000)
    assert not check_latency(45000.0, ceiling_ms=30000)
    assert not check_latency(None, ceiling_ms=30000)  # unknown fails loud
    assert check_cost(0.004, ceiling_usd=0.05)
    assert not check_cost(0.09, ceiling_usd=0.05)
