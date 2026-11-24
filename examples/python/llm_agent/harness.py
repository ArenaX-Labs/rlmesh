"""An LLM agent as an RLMesh model: one predict is one env interaction.

Within a step the harness may call the LLM several times; every call is logged
to ``$RLMESH_OUTPUT_DIR/transcripts/<episode_id>.jsonl``.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from typing import Any

import rlmesh
from chat_env import Message, append_jsonl, output_dir
from rlmesh.types import PredictContext

Chat = Callable[
    [list[Message], list[dict[str, Any]]], tuple[Message, dict[str, Any] | None]
]
DEFAULT_MODEL = "qwen3:4b"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MAX_TOKENS = 8192
# RLMesh abandons a predict after 300 s; one call and the retries stop before that.
STEP_BUDGET_S = 240.0


def openai_chat(model: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> Chat:
    """Chat through the OpenAI-compatible server at ``$OPENAI_BASE_URL``."""
    from openai import OpenAI, omit  # pyright: ignore[reportMissingImports]

    # Messages and tools pass through as plain JSON, not the SDK's TypedDicts.
    client: Any = OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
        timeout=STEP_BUDGET_S,
        max_retries=0,
    )

    def chat(
        messages: list[Message], tools: list[dict[str, Any]]
    ) -> tuple[Message, dict[str, Any] | None]:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            # Ollama's thinking switch: reasoning moves out of content; other servers ignore it.
            extra_body={"think": False},
            tools=tools or omit,
        )
        message = completion.choices[0].message.model_dump(exclude_none=True)
        reply: Message = {"role": "assistant", "content": message.get("content")}
        for key in ("tool_calls", "reasoning"):
            if message.get(key):
                reply[key] = message[key]
        usage = completion.usage.model_dump() if completion.usage else None
        return reply, usage

    return chat


class LLMHarness(rlmesh.Model):
    """Answer each observation with the LLM's next assistant message.

    Args:
        model: The model name sent to the chat server (default ``$OPENAI_MODEL``).
        chat: The chat function; defaults to :func:`openai_chat` for ``model``.
        retries: Extra calls allowed in one step when a reply is empty.
        max_tokens: The completion budget of each call, reasoning included
            (default ``$OPENAI_MAX_TOKENS``).
    """

    def __init__(
        self,
        model: str | None = None,
        chat: Chat | None = None,
        retries: int = 2,
        max_tokens: int | None = None,
    ) -> None:
        model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        max_tokens = max_tokens or int(
            os.environ.get("OPENAI_MAX_TOKENS") or DEFAULT_MAX_TOKENS
        )
        self.model_name = model
        self.chat = chat or openai_chat(model, max_tokens)
        self.retries = retries
        self.transcripts = output_dir() / "transcripts"
        self.transcripts.mkdir(exist_ok=True)
        self.calls: dict[str, int] = {}
        self.steps: dict[str, int] = {}
        super().__init__()

    def predict(self, observation: str, context: PredictContext) -> str:
        obs = json.loads(observation)
        tools: list[dict[str, Any]] = obs["tools"]
        messages: list[Message] = context["state"].setdefault("messages", [])
        messages.extend(obs["messages"])
        episode = context["episode_id"] or "episode"
        self.steps[episode] = context["predict_index"] + 1
        reply: Message = {"role": "assistant", "content": ""}
        step_started = time.monotonic()
        for _ in range(1 + self.retries):
            call = self.calls.get(episode, 0)
            self.calls[episode] = call + 1
            record: dict[str, Any] = {
                "step": context["predict_index"],
                "call": call,
                "request": {
                    "model": self.model_name,
                    "messages": messages,
                    "tools": tools,
                },
            }
            started = time.monotonic()
            try:
                reply, usage = self.chat(messages, tools)
                record.update(response=reply, usage=usage)
            except Exception as exc:
                record["error"] = repr(exc)
                raise
            finally:
                record["latency_ms"] = round((time.monotonic() - started) * 1000, 1)
                record["ts"] = time.time()
                append_jsonl(self.transcripts / f"{episode}.jsonl", record)
            if reply.get("content") or reply.get("tool_calls"):
                break
            now = time.monotonic()
            if now - step_started + (now - started) > STEP_BUDGET_S:
                break
        reply.pop("reasoning", None)
        messages.append(reply)
        return json.dumps(reply)

    def on_episode_end(self, episode_id: str = "") -> None:
        episode = episode_id or "episode"
        append_jsonl(
            self.transcripts / f"{episode}.jsonl",
            {
                "event": "episode_end",
                "episode_id": episode_id,
                "steps": self.steps.pop(episode, 0),
                "calls": self.calls.pop(episode, 0),
                "ts": time.time(),
            },
        )
