"""The tool-calling loop, exercised offline against a scripted fake client."""

from __future__ import annotations

from dbx_template_agent.agent import DEFAULT_SYSTEM_PROMPT, run_agent
from dbx_template_agent.tools import calculator, lookup, run_tool


def test_direct_answer_no_tools(fake_client_factory, fake_message_factory):
    client = fake_client_factory([fake_message_factory(content="Hello!")])
    result = run_agent("Say hello.", client=client)
    assert result["answer"] == "Hello!"
    assert result["tools_used"] == []
    assert result["usage"]["total_tokens"] > 0
    assert result["cost_usd"] > 0


def test_tool_call_roundtrip(fake_client_factory, fake_message_factory, tool_call_factory):
    client = fake_client_factory(
        [
            fake_message_factory(
                tool_calls=[tool_call_factory("calculator", {"expression": "17 * 43 + 5"})]
            ),
            fake_message_factory(content="17 * 43 + 5 = 736"),
        ]
    )
    result = run_agent("What is 17 * 43 + 5?", client=client)
    assert result["tools_used"] == ["calculator"]
    assert "736" in result["answer"]
    # The tool RESULT must have been sent back to the model on the second call.
    tool_messages = [m for m in client.requests[1]["messages"] if m.get("role") == "tool"]
    assert any(m.get("content") == "736" for m in tool_messages)


def test_system_prompt_reaches_model(fake_client_factory, fake_message_factory):
    client = fake_client_factory([fake_message_factory(content="ok")])
    run_agent("q", client=client, system_prompt="CUSTOM PROMPT")
    assert client.requests[0]["messages"][0] == {"role": "system", "content": "CUSTOM PROMPT"}


def test_prompt_fallback_used_offline(fake_client_factory, fake_message_factory):
    # No registry is reachable in tests, so load_system_prompt falls back.
    client = fake_client_factory([fake_message_factory(content="ok")])
    run_agent("q", client=client)
    assert client.requests[0]["messages"][0]["content"] == DEFAULT_SYSTEM_PROMPT


def test_iteration_budget_exhausted_returns_honestly(
    fake_client_factory, fake_message_factory, tool_call_factory
):
    # The model asks for a tool every turn and never answers.
    loop_forever = [
        fake_message_factory(tool_calls=[tool_call_factory("lookup", {"topic": "mlflow trace"})])
        for _ in range(3)
    ]
    client = fake_client_factory(loop_forever)
    result = run_agent("q", client=client, max_iters=3)
    assert "could not complete" in result["answer"]
    assert result["tools_used"] == ["lookup"] * 3


def test_calculator_exact_and_safe():
    assert calculator("17 * 43 + 5") == "736"
    assert calculator("(128 - 32) / 4") == "24"
    assert calculator("2 ** 10") == "1024"
    assert calculator("__import__('os')").startswith("error:")


def test_lookup_hits_and_misses():
    assert "tree of spans" in lookup("mlflow trace")
    assert "tree of spans" in lookup("What is an MLflow trace?")  # containment match
    assert lookup("quantum llamas").startswith("error:")


def test_run_tool_dispatch_and_bad_input():
    assert run_tool("calculator", '{"expression": "1 + 1"}') == "2"
    assert run_tool("nope", "{}").startswith("error: unknown tool")
    assert run_tool("calculator", "not json").startswith("error:")
    assert run_tool("calculator", '{"wrong_arg": 1}').startswith("error: bad arguments")
