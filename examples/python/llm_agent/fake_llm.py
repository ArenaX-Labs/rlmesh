"""A scripted stand-in for the LLM, so the harness runs without a model server."""

from __future__ import annotations

from typing import Any

from harness import Chat, Message


def scripted(*replies: str) -> Chat:
    """Return a chat function that answers with ``replies`` in order."""
    remaining = list(replies)

    def chat(
        messages: list[Message], tools: list[dict[str, Any]]
    ) -> tuple[Message, dict[str, Any] | None]:
        return {"role": "assistant", "content": remaining.pop(0)}, None

    return chat
