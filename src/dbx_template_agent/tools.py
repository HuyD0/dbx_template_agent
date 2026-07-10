"""The agent's tools.

Two deliberately small examples so the mechanics stay visible:

- ``calculator`` — exact arithmetic (LLMs are bad at it; a tool is exact).
- ``lookup`` — a tiny built-in knowledge base (stands in for the database /
  API / retrieval call a real agent would make).

Add your own tool by writing the function, adding a spec to ``TOOL_SPECS``,
and registering it in ``_REGISTRY``. The phase-02 notebook walks through this,
and the phase-05 deterministic scorers check the agent calls the right tool —
so a new tool should come with an eval row that exercises it.
"""

from __future__ import annotations

import ast
import json
import operator

# --- calculator --------------------------------------------------------------

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly (AST-walked, never eval())."""
    try:
        result = _eval_node(ast.parse(expression, mode="eval"))
    except Exception as exc:  # noqa: BLE001 - the model reads this message
        return f"error: could not evaluate {expression!r}: {exc}"
    # Render integers without a trailing .0 so answers read naturally.
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


# --- lookup -------------------------------------------------------------------

KNOWLEDGE: dict[str, str] = {
    "mlflow trace": (
        "An MLflow trace records one end-to-end request through your agent as a "
        "tree of spans (AGENT -> TOOL / CHAT_MODEL), each with inputs, outputs, "
        "timing, and attributes such as token usage."
    ),
    "llm judge": (
        "An LLM judge is a model prompted to score another model's output against "
        "criteria (e.g. correctness). Judges must be validated against human labels "
        "before they gate anything."
    ),
    "unity catalog": (
        "Unity Catalog is Databricks' governance layer; objects live in a "
        "catalog.schema.name three-level namespace, including registered models, "
        "prompts, and evaluation datasets."
    ),
    "champion challenger": (
        "Champion/challenger compares the incumbent (champion) against a candidate "
        "(challenger) on the same eval set; the candidate is promoted only if it "
        "doesn't regress. Aliases (@champion) point downstream code at the winner."
    ),
    "asset bundle": (
        "A Databricks Asset Bundle (DAB) describes workspace resources (jobs, "
        "models, experiments) as versioned YAML deployed with `databricks bundle "
        "deploy` — infrastructure-as-code for the lakehouse."
    ),
}


def lookup(topic: str) -> str:
    """Look up a topic in the built-in knowledge base."""
    key = topic.strip().lower()
    if key in KNOWLEDGE:
        return KNOWLEDGE[key]
    # Forgiving containment match so "what's an MLflow trace?" still hits.
    for name, text in KNOWLEDGE.items():
        if name in key or key in name:
            return text
    known = ", ".join(sorted(KNOWLEDGE))
    return f"error: no entry for {topic!r}. Known topics: {known}"


# --- registry -----------------------------------------------------------------

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate an arithmetic expression exactly. Use for ANY math — "
                "do not do arithmetic yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression, e.g. '(17 * 43) + 5'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": (
                "Look up a definition in the knowledge base. Use when asked what a "
                "term means rather than answering from memory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic to look up, e.g. 'mlflow trace'",
                    }
                },
                "required": ["topic"],
            },
        },
    },
]

_REGISTRY = {
    "calculator": calculator,
    "lookup": lookup,
}


def run_tool(name: str, arguments: str | dict) -> str:
    """Dispatch a tool call by name with JSON (or dict) arguments.

    Errors are returned as strings, not raised: the model should see the
    failure and get a chance to recover — a crashed loop teaches it nothing.
    """
    if name not in _REGISTRY:
        return f"error: unknown tool {name!r}"
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return f"error: arguments were not valid JSON: {exc}"
    try:
        return _REGISTRY[name](**arguments)
    except TypeError as exc:
        return f"error: bad arguments for {name}: {exc}"
