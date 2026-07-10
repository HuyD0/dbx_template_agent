"""The agent: a hand-coded tool-calling loop you can read end to end.

The loop is deliberately explicit (no framework) so every notebook can point
at real lines: the model is called, if it asks for tools they are executed,
results are appended, and the loop continues until the model answers in prose
or ``max_iters`` is hit.

Tracing contract (the eval surface — the phase-05 scorers consume it):

- one AGENT root span per request (``run_agent`` itself),
- one CHAT_MODEL span per model call, with the raw ``usage`` block and the
  computed ``cost_usd`` as span attributes,
- one TOOL span per tool execution, with the tool name, arguments, and result.

Keep that contract intact when you extend the agent; it is what makes traces
queryable, evals cheap, and cost observable.
"""

from __future__ import annotations

from typing import Any

import mlflow
from mlflow.entities import SpanType

from .config import CHAT_ENDPOINT, PROMPT_NAME
from .cost import add_usage, usage_to_usd
from .tools import TOOL_SPECS, run_tool

# Fallback only: the registered prompt (phase 01) is the source of truth.
# If you find yourself editing this string, register a new prompt version instead.
DEFAULT_SYSTEM_PROMPT = (
    "You are a precise assistant. Use the calculator tool for ANY arithmetic "
    "and the lookup tool for definitions instead of answering from memory. "
    "Answer concisely, and say so plainly when you cannot answer."
)


def load_system_prompt() -> str:
    """Load the system prompt from the MLflow Prompt Registry (@production).

    Falls back to ``DEFAULT_SYSTEM_PROMPT`` when the registry is unreachable or
    the prompt hasn't been registered yet (fresh project, offline tests) — the
    agent should degrade, not crash, on a registry hiccup.
    """
    try:
        prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@production")
        return prompt.template
    except Exception:
        return DEFAULT_SYSTEM_PROMPT


def get_client() -> Any:
    """An OpenAI-compatible client backed by Databricks model serving."""
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient().serving_endpoints.get_open_ai_client()


def _call_model(
    client: Any,
    endpoint: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> Any:
    """One chat-completions call, wrapped in a CHAT_MODEL span carrying usage."""
    with mlflow.start_span(name="chat_model", span_type=SpanType.CHAT_MODEL) as span:
        span.set_inputs({"endpoint": endpoint, "messages": messages})
        response = client.chat.completions.create(
            model=endpoint,
            messages=messages,
            tools=TOOL_SPECS,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = getattr(response, "usage", None)
        usage_dict = (
            {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
            if usage is not None
            else None
        )
        span.set_attributes(
            {
                "usage": usage_dict,
                "cost_usd": usage_to_usd(usage_dict, endpoint),
            }
        )
        span.set_outputs({"message": response.choices[0].message.model_dump()})
    return response


def _run_tool_call(tool_call: Any) -> dict:
    """Execute one requested tool inside a TOOL span; returns the tool message."""
    name = tool_call.function.name
    arguments = tool_call.function.arguments
    with mlflow.start_span(name=name, span_type=SpanType.TOOL) as span:
        span.set_inputs({"tool": name, "arguments": arguments})
        result = run_tool(name, arguments)
        span.set_outputs({"result": result})
    return {"role": "tool", "tool_call_id": tool_call.id, "content": result}


@mlflow.trace(name="run_agent", span_type=SpanType.AGENT)
def run_agent(
    question: str,
    *,
    client: Any = None,
    endpoint: str = CHAT_ENDPOINT,
    system_prompt: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 512,
    max_iters: int = 5,
) -> dict:
    """Answer ``question`` with the tool-calling loop.

    Returns a dict with the final ``answer``, the full ``messages`` transcript,
    the names of ``tools_used`` (in call order), the summed token ``usage``,
    and the estimated ``cost_usd`` for the whole request.

    ``client`` is injectable so tests can pass a fake — the loop's logic is
    fully exercisable offline, which is what keeps the unit tests hermetic.
    """
    client = client or get_client()
    messages: list[dict] = [
        {"role": "system", "content": system_prompt or load_system_prompt()},
        {"role": "user", "content": question},
    ]
    tools_used: list[str] = []
    usage_total: dict[str, int] = {}

    for _ in range(max_iters):
        response = _call_model(client, endpoint, messages, temperature, max_tokens)
        usage = getattr(response, "usage", None)
        if usage is not None:
            add_usage(
                usage_total,
                {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                },
            )
        message = response.choices[0].message

        if not getattr(message, "tool_calls", None):
            answer = message.content or ""
            return {
                "answer": answer,
                "messages": messages + [{"role": "assistant", "content": answer}],
                "tools_used": tools_used,
                "usage": usage_total,
                "cost_usd": usage_to_usd(usage_total, endpoint),
            }

        messages.append(message.model_dump())
        for tool_call in message.tool_calls:
            tools_used.append(tool_call.function.name)
            messages.append(_run_tool_call(tool_call))

    # Iteration budget exhausted: return honestly rather than looping forever.
    return {
        "answer": "I could not complete the request within the tool-call budget.",
        "messages": messages,
        "tools_used": tools_used,
        "usage": usage_total,
        "cost_usd": usage_to_usd(usage_total, endpoint),
    }
