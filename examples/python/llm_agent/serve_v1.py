"""Serve a Prime verifiers v1 taskset (math500, humaneval, ...) as an RLMesh env.

Run with the v1 venv (``requirements-v1.txt``). Each episode is one task picked
by the reset seed; the first assistant reply is the answer, scored by the
task's own ``score``.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

import verifiers.v1 as vf  # pyright: ignore[reportMissingImports]
from chat_env import (
    DEFAULT_ADDRESS,
    ChatTaskEnv,
    Message,
    Turn,
    openai_messages,
    serve,
)


class TasksetEnv(ChatTaskEnv):
    """One single-turn task of a v1 taskset per episode."""

    def __init__(self, taskset: str, runtime: str | None = None) -> None:
        super().__init__()
        self.tasks: list[Any] = list(
            vf.load_taskset(vf.TasksetConfig(id=taskset)).load()
        )
        self.runtime = runtime
        self.task: Any = None
        self.prompt: list[Message] = []

    def start(self, index: int) -> Turn:
        self.task = self.tasks[index % len(self.tasks)]
        data = self.task.data
        self.task_id = str(data.name or data.id or data.idx)
        if isinstance(data.prompt, str):
            self.prompt = [{"role": "user", "content": data.prompt}]
        else:
            self.prompt = [
                m.model_dump(mode="json", exclude_none=True) for m in data.prompt or []
            ]
        if data.system_prompt:
            self.prompt.insert(0, {"role": "system", "content": data.system_prompt})
        if not self.prompt:
            raise ValueError(
                f"task {self.task_id} has neither a prompt nor a system prompt"
            )
        return Turn(openai_messages(self.prompt))

    def respond(self, reply: Message) -> Turn:
        nodes = [
            vf.MessageNode(message=m, parent=i - 1 if i else None)
            for i, m in enumerate(self.prompt)
        ]
        calls = [
            vf.ToolCall(
                id=call["id"],
                name=call["function"]["name"],
                arguments=call["function"]["arguments"],
            )
            for call in reply.get("tool_calls") or []
        ]
        nodes.append(
            vf.MessageNode(
                message=vf.AssistantMessage(
                    content=reply.get("content"), tool_calls=calls or None
                ),
                sampled=True,
                parent=len(nodes) - 1,
            )
        )
        trace = vf.Trace(
            task=vf.TraceTask(type=type(self.task).__name__, data=self.task.data),
            agent=vf.AgentInfo(config=vf.AgentConfig()),
            nodes=nodes,
        )
        asyncio.run(self._score(trace))
        rewards = {name: r.score for name, r in trace.rewards.items()}
        reward = sum(r.score * r.weight for r in trace.rewards.values())
        return Turn([], rewards=rewards, reward=reward)

    async def _score(self, trace: Any) -> None:
        if self.runtime is None:
            await self.task.score(trace)
            return
        from verifiers.v1.runtimes import (  # pyright: ignore[reportMissingImports]
            provision_runtime,
        )

        async with provision_runtime(vf.SubprocessConfig()) as runtime:
            await self.task.score(trace, runtime=runtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a v1 taskset.")
    parser.add_argument("--taskset", default="primeintellect/math500")
    parser.add_argument(
        "--runtime",
        choices=["subprocess"],
        help="provision a local runtime for reward functions that execute code",
    )
    parser.add_argument("--address", default=DEFAULT_ADDRESS, help="bind address")
    args = parser.parse_args()
    serve(TasksetEnv(args.taskset, args.runtime), args.address)


if __name__ == "__main__":
    main()
