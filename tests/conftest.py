"""Offline test fixtures: no workspace, no network, no credentials.

The agent takes an injectable client, so a fake OpenAI-compatible client is
all the tests need. MLflow tracking is pointed at a throwaway local directory
so traced calls work without a Databricks workspace.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import mlflow
import pytest


@pytest.fixture(autouse=True)
def _local_mlflow(tmp_path):
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    yield


def _usage(prompt=100, completion=20):
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
    )


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self):
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in (self.tool_calls or [])
            ]
            or None,
        }


def tool_call(name: str, arguments: dict, call_id: str = "call_1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


class FakeClient:
    """Replays a scripted sequence of assistant messages, recording requests."""

    def __init__(self, script: list[FakeMessage]):
        self._script = list(script)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        message = self._script.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=_usage(),
        )


@pytest.fixture
def fake_client_factory():
    def factory(script):
        return FakeClient(script)

    return factory


@pytest.fixture
def fake_message_factory():
    return FakeMessage


@pytest.fixture
def tool_call_factory():
    return tool_call
