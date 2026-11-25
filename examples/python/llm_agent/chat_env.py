"""Serve a chat task as an RLMesh text environment.

The observation is JSON ``{"messages": [...], "tools": [...]}``: the chat
messages new since the last step and the OpenAI tool schemas on offer. The
action is JSON for one OpenAI-style assistant message. Run this file to serve
``AdditionTask`` (one question) or ``TicTacToeTask`` (two seats, one model);
neither needs verifiers.
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

    observation_space = rlmesh.spaces.Text(1_000_000, min_length=1)
    action_space = rlmesh.spaces.Text(1_000_000, min_length=1)

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
        turn = self.respond(_message(action))
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

    def make(self) -> ChatTaskEnv:
        """Return a new env of this class, so ``rlmesh.serve`` and ``rlmesh._describe`` can use the class as a factory (one env per lane)."""
        return type(self)()


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


def _message(action: str) -> Message:
    """Parse an action; anything but a JSON object (a probe's random text) is an empty reply."""
    try:
        reply = json.loads(action)
    except ValueError:
        return {}
    return reply if isinstance(reply, dict) else {}


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


class TicTacToeTask(ChatTaskEnv):
    """Tic-tac-toe in which the one model plays both seats, ``X`` and ``O``.

    Each observation names the seat to move. An invalid or taken cell loses for
    that seat. ``rewards`` holds each seat's score (win 1, loss -1, draw 0);
    ``reward`` is X's.
    """

    LINES = (
        (0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
        (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6),
    )  # fmt: skip
    RULES = (
        "Tic-tac-toe. Cells are numbered 1-9, left to right, top to bottom. "
        "Three in a row wins; an invalid or taken cell loses. "
        "Reply with the cell number only, no other text."
    )

    def start(self, index: int) -> Turn:
        self.board = list("123456789")
        self.mover = "XO"[index % 2]
        self.task_id = f"tictactoe-{index}"
        return Turn([self._prompt(self.RULES)])

    def respond(self, reply: Message) -> Turn:
        content = reply.get("content")
        move = content.strip() if isinstance(content, str) else ""
        seat, other = self.mover, "O" if self.mover == "X" else "X"
        if move not in [cell for cell in self.board if cell not in "XO"]:
            return self._end(other)
        self.board[int(move) - 1] = seat
        if any(all(self.board[i] == seat for i in line) for line in self.LINES):
            return self._end(seat)
        if all(cell in "XO" for cell in self.board):
            return self._end(None)
        self.mover = other
        return Turn([self._prompt(f"{seat} took cell {move}.")])

    def _prompt(self, note: str) -> Message:
        rows = ("|".join(self.board[r : r + 3]) for r in (0, 3, 6))
        board = "\n".join(rows)
        content = (
            f"{note}\n\n{board}\n\nYou are {self.mover}. Reply with a cell number 1-9."
        )
        return {"role": "user", "content": content}

    def _end(self, winner: str | None) -> Turn:
        rewards = {
            seat: 0.0 if winner is None else 1.0 if seat == winner else -1.0
            for seat in "XO"
        }
        return Turn(
            [],
            rewards=rewards,
            reward=rewards["X"],
            success=winner == "X",
            info={"winner": winner or "draw"},
        )


TASKS: dict[str, type[ChatTaskEnv]] = {
    "addition": AdditionTask,
    "tictactoe": TicTacToeTask,
}


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
    parser = argparse.ArgumentParser(description="Serve a built-in chat task.")
    parser.add_argument("--task", choices=TASKS, default="addition", help="task")
    parser.add_argument("--address", default=DEFAULT_ADDRESS, help="bind address")
    args = parser.parse_args()
    serve(TASKS[args.task](), args.address)


if __name__ == "__main__":
    main()
