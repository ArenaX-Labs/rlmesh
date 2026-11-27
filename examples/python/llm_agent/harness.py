"""An LLM agent as an RLMesh model: one predict is one env interaction.

Within a step the harness may call the LLM several times; every call is logged
to ``$RLMESH_OUTPUT_DIR/transcripts/<episode_id>.jsonl``.
"""

from __future__ import annotations

import json
import os
import re
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


def resolve_llm(llm: str) -> tuple[str, str, str]:
    """Split a models.dev ``provider/model`` (``$RLMESH_LLM``) into the model id
    to send and the OpenAI-compatible base URL and key to send it to.

    The provider ends at the first ``/``; the model id may hold more. Anthropic
    is called at its OpenAI-compatible ``/v1`` with ``$ANTHROPIC_API_KEY``; every
    other provider at ``$OPENAI_BASE_URL`` with ``$OPENAI_API_KEY``, which RLMesh
    Managed sets from the run's API key connection.
    """
    provider, _, model = llm.partition("/")
    if not provider or not model:
        raise ValueError(f"llm {llm!r} is not provider/model")
    if provider == "anthropic":
        base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        return model, base.rstrip("/") + "/v1", os.environ.get("ANTHROPIC_API_KEY", "")
    return (
        model,
        os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        os.environ.get("OPENAI_API_KEY", ""),
    )


def with_backoff(
    create: Callable[[float], Any], budget_s: float = STEP_BUDGET_S
) -> Any:
    """Call ``create(timeout)``, retrying a rate limit or server error until
    ``budget_s`` runs out; a ``retry-after`` header sets the wait."""
    deadline = time.monotonic() + budget_s
    delay = 1.0
    while True:
        try:
            return create(deadline - time.monotonic())
        except Exception as exc:
            status = getattr(exc, "status_code", 0)
            if status != 429 and status < 500:
                raise
            response = getattr(exc, "response", None)
            header = (
                response.headers.get("retry-after") if response is not None else None
            )
            try:
                wait = float(header) if header else delay
            except ValueError:
                wait = delay
            if time.monotonic() + wait >= deadline:
                raise
            time.sleep(wait)
            delay = min(delay * 2, 30.0)


def openai_chat(
    model: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    base_url: str | None = None,
    api_key: str | None = None,
    provider: str = "",
) -> Chat:
    """Chat through an OpenAI-compatible server, by default ``$OPENAI_BASE_URL``.

    ``provider`` is the models.dev provider id; without one the server is taken
    to be a local Ollama.
    """
    from openai import OpenAI, omit

    # Messages and tools pass through as plain JSON, not the SDK's TypedDicts.
    client: Any = OpenAI(
        base_url=base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        api_key=api_key or os.environ.get("OPENAI_API_KEY", "ollama"),
        timeout=STEP_BUDGET_S,
        max_retries=0,
    )
    # OpenAI's reasoning models refuse max_tokens.
    budget = {
        "max_completion_tokens" if provider == "openai" else "max_tokens": max_tokens
    }
    # Ollama's thinking switch: reasoning moves out of content.
    extra = None if provider else {"think": False}

    def chat(
        messages: list[Message], tools: list[dict[str, Any]]
    ) -> tuple[Message, dict[str, Any] | None]:
        completion = with_backoff(
            lambda timeout: client.chat.completions.create(
                model=model,
                messages=messages,
                extra_body=extra,
                tools=tools or omit,
                timeout=timeout,
                **budget,
            )
        )
        message = completion.choices[0].message.model_dump(exclude_none=True)
        reply: Message = {"role": "assistant", "content": message.get("content")}
        for key in ("tool_calls", "reasoning"):
            if message.get(key):
                reply[key] = message[key]
        usage = completion.usage.model_dump() if completion.usage else None
        return reply, usage

    return chat


_DATA_URL = re.compile(r"(data:[\w/+.-]+;base64,)[A-Za-z0-9+/=]+")


def redact_images(messages: list[Message]) -> list[Message]:
    """Return ``messages`` with each image data URL cut to its MIME type and size."""
    return [
        {**message, "content": [_redact_part(part) for part in message["content"]]}
        if isinstance(message.get("content"), list)
        else message
        for message in messages
    ]


def _redact_part(part: Any) -> Any:
    image = part.get("image_url") if isinstance(part, dict) else None
    url = image.get("url", "") if isinstance(image, dict) else ""
    if not isinstance(url, str) or not url.startswith("data:"):
        return part
    header, _, payload = url.partition(",")
    mime = header.removeprefix("data:").split(";")[0]
    redacted = f"data:{mime};base64,<{len(payload)} bytes>"
    return {**part, "image_url": {**part["image_url"], "url": redacted}}


class LLMHarness(rlmesh.Model):
    """Answer each observation with the LLM's next assistant message.

    Args:
        model: The model name sent to the chat server. Without one, the
            ``provider/model`` in ``$RLMESH_LLM`` picks the model and its
            endpoint (see :func:`resolve_llm`), else ``$OPENAI_MODEL``.
        chat: The chat function; defaults to :func:`openai_chat` for ``model``.
        retries: Extra calls allowed in one step when a reply is empty.
        max_tokens: The completion budget of each call, reasoning included
            (default ``$OPENAI_MAX_TOKENS``).
    """

    concurrency = 32

    def __init__(
        self,
        model: str | None = None,
        chat: Chat | None = None,
        retries: int = 2,
        max_tokens: int | None = None,
    ) -> None:
        max_tokens = max_tokens or int(
            os.environ.get("OPENAI_MAX_TOKENS") or DEFAULT_MAX_TOKENS
        )
        llm = "" if model else os.environ.get("RLMESH_LLM", "")
        if llm:
            model, base_url, api_key = resolve_llm(llm)
            chat = chat or openai_chat(
                model, max_tokens, base_url, api_key, llm.partition("/")[0]
            )
        model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
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
                    "messages": redact_images(messages),
                    "tools": tools,
                },
            }
            started = time.monotonic()
            try:
                reply, usage = self.chat(messages, tools)
                record.update(response=reply, usage=usage)
            except Exception as exc:
                record["error"] = _DATA_URL.sub(r"\1<redacted>", repr(exc))
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
