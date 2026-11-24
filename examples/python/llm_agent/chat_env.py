"""Serve a chat task as an RLMesh text environment.

The observation is JSON ``{"messages": [...], "tools": [...]}``: the chat
messages new since the last step and the OpenAI tool schemas on offer. The
action is JSON for one OpenAI-style assistant message. Run this file to serve
``AdditionTask``, a one-question task that needs no verifiers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import rlmesh

Message = dict[str, Any]
DEFAULT_ADDRESS = "127.0.0.1:5555"


def output_dir() -> Path:
    """Return ``$RLMESH_OUTPUT_DIR`` (default ``./runs/<timestamp>``), created."""
    path = Path(
        os.environ.get("RLMESH_OUTPUT_DIR") or f"runs/{time.strftime('%Y%m%d-%H%M%S')}"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON line to ``path``."""
    with path.open("a") as file:
        file.write(json.dumps(record, default=str) + "\n")


@dataclass
class Turn:
    """What the task says after an action: new messages, and the score once it ends."""

    messages: list[Message]
    tools: list[dict[str, Any]] = field(default_factory=list)
    rewards: dict[str, float] | None = None
    reward: float = 0.0
    success: bool | None = None
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class ChatTaskEnv:
    """Base env: one step is one assistant message; subclasses run the task."""

    observation_space = rlmesh.spaces.Text(1_000_000, min_length=0)
    action_space = rlmesh.spaces.Text(1_000_000, min_length=0)

    def __init__(self) -> None:
        self.log_dir = output_dir() / "env"
        self.log_dir.mkdir(exist_ok=True)
        self.seed = 0
        self.task_id = ""

    def start(self, index: int) -> Turn:
        """Begin task ``index`` and return its prompt; set ``self.task_id``."""
        raise NotImplementedError

    def respond(self, reply: Message) -> Turn:
        """Apply the assistant ``reply``; a ``Turn`` with ``rewards`` ends the episode."""
        raise NotImplementedError

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, Any]]:
        self.seed = seed or 0
        turn = self.start(self.seed)
        return _observation(turn), {"task_id": self.task_id}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        turn = self.respond(json.loads(action))
        if turn.rewards is None:
            return _observation(turn), 0.0, False, False, {}
        success = turn.reward >= 1.0 if turn.success is None else turn.success
        info = {
            **turn.info,
            "is_success": success,
            "reward": turn.reward,
            "rewards": turn.rewards,
            "task_id": self.task_id,
        }
        append_jsonl(
            self.log_dir / f"seed-{self.seed}.jsonl", {**info, "ts": time.time()}
        )
        return _observation(turn), turn.reward, not turn.truncated, turn.truncated, info

    def close(self) -> None:
        pass


def openai_messages(messages: list[Message]) -> list[Message]:
    """Return dumped verifiers messages with tool calls in OpenAI's nested shape."""
    return [
        {
            **message,
            "tool_calls": [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"]},
                }
                for call in message["tool_calls"]
            ],
        }
        if message.get("tool_calls")
        else message
        for message in messages
    ]


def _observation(turn: Turn) -> str:
    return json.dumps({"messages": turn.messages, "tools": turn.tools})


class AdditionTask(ChatTaskEnv):
    """Ask for one sum and score the reply by exact match."""

    def start(self, index: int) -> Turn:
        a, b = 2 + index, 3
        self.answer = str(a + b)
        self.task_id = f"addition-{index}"
        question = f"What is {a} + {b}? Reply with the number only."
        return Turn([{"role": "user", "content": question}])

    def respond(self, reply: Message) -> Turn:
        correct = float((reply.get("content") or "").strip() == self.answer)
        return Turn([], rewards={"correct": correct}, reward=correct)


def serve(env: ChatTaskEnv, address: str) -> None:
    """Serve ``env`` at ``address`` until interrupted."""
    server = rlmesh.EnvServer(env, address)
    shown = server.address
    # Launched with a host identity, the key (``#…``) is a credential and stdout a log.
    if os.environ.get("RLMESH_HOST_IDENTITY"):
        shown = re.sub(r"#[^/]*", "#…", shown)
    print(f"serving {type(env).__name__} on {shown}", flush=True)
    server.serve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the addition task.")
    parser.add_argument("--address", default=DEFAULT_ADDRESS, help="bind address")
    serve(AdditionTask(), parser.parse_args().address)


if __name__ == "__main__":
    main()
