"""Serve a Prime verifiers v1 taskset or environment as an RLMesh env.

Run with the v1 venv (``requirements-v1.txt``). Each episode is one task picked
by the reset seed. ``--taskset`` (math500, humaneval, ...) takes the first
assistant reply as the answer, scored by the task's own ``score``. ``--env``
(tau2-bench, ...) runs the environment's own episode; where its harness would
call the model, a bridge client hands the request to RLMesh as an observation
and waits for the next action.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import os
import threading
import time
from collections.abc import Mapping
from typing import Any

import verifiers.v1 as vf
import verifiers.v1.interception.server as interception_server
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

    def __init__(self, taskset: str | None = None, runtime: str | None = None) -> None:
        super().__init__()
        self.taskset = taskset or os.environ.get("TASKSET") or "primeintellect/math500"
        self.runtime = runtime or os.environ.get("RUNTIME") or None
        if self.runtime not in (None, "subprocess"):
            raise ValueError(
                f"runtime must be subprocess or unset, got {self.runtime!r}"
            )
        self.tasks: list[Any] = []
        self.task: Any = None
        self.prompt: list[Message] = []

    def start(self, index: int) -> Turn:
        if not self.tasks:
            self.tasks = list(vf.load_taskset(vf.TasksetConfig(id=self.taskset)).load())
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
        from verifiers.v1.runtimes import (
            provision_runtime,
        )

        async with provision_runtime(vf.SubprocessConfig()) as runtime:
            await self.task.score(trace, runtime=runtime)


Request = tuple[dict[str, Any], "asyncio.Future[Message]"]


class _BridgeClient(vf.Client):
    """A verifiers client whose model is whoever sends the next RLMesh action."""

    def __init__(self, env: V1EnvEnv) -> None:
        self.env = env

    async def get_response(
        self,
        dialect: Any,
        body: dict,
        sampling: Any,
        session_id: str | None = None,
        turn: Any = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        if dialect.upstream_path != "/chat/completions":
            raise ValueError(f"the bridge serves chat completions, not {dialect}")
        reply: asyncio.Future[Message] = asyncio.get_running_loop().create_future()
        self.env.request.set_result((body, reply))
        message = {**await reply, "role": "assistant"}
        raw = {
            "id": "rlmesh",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "rlmesh"),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls"
                    if message.get("tool_calls")
                    else "stop",
                }
            ],
        }
        response = dialect.parse_response(dialect.validate_response(raw))
        response.raw = raw
        return response


_bridge_lock = threading.Lock()
_bridged: V1EnvEnv | None = None


class V1EnvEnv(ChatTaskEnv):
    """One episode of a v1 environment (its own harness and scoring) per episode."""

    request: asyncio.Future[Request]

    def __init__(
        self,
        env_id: str | None = None,
        domain: str | None = None,
        max_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.env_id = env_id or os.environ.get("ENV_ID")
        self.domain = domain or os.environ.get("DOMAIN")
        raw = os.environ.get("MAX_STEPS")
        self.max_steps = (
            max_steps if max_steps is not None else int(raw) if raw else None
        )
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError(f"max_steps must be at least 1, got {self.max_steps}")
        self.steps = 0
        self.env: Any = None
        self.tasks: list[Any] = []
        self.client = _BridgeClient(self)
        self.serving = contextlib.AsyncExitStack()
        self.ctx = vf.ModelContext(model="rlmesh", client=vf.EvalClientConfig())
        self.episode: asyncio.Task[Any] | None = None
        self.reply: asyncio.Future[Message] | None = None
        self.resolve_client: Any = None

    @functools.cached_property
    def loop(self) -> asyncio.AbstractEventLoop:
        return asyncio.new_event_loop()

    def _load(self) -> None:
        if not self.env_id:
            raise ValueError("no environment: pass env_id or set ENV_ID")
        global _bridged
        with _bridge_lock:
            if _bridged is not None and _bridged is not self:
                raise RuntimeError(
                    "one V1EnvEnv per process (the interception server's client is "
                    "module-global); serve it with RLMESH_NUM_ENVS=1"
                )
            _bridged = self
            self.resolve_client = interception_server.resolve_client
            # The interception server resolves only eval/train endpoints.
            interception_server.resolve_client = lambda config: self.client
        try:
            taskset = vf.taskset_config_type(self.env_id)(
                id=self.env_id, **({"domain": self.domain} if self.domain else {})
            )
            env = vf.load_environment(vf.env_config_type(self.env_id)(taskset=taskset))
            tasks = list(env.taskset.load())
            self.loop.run_until_complete(
                self.serving.enter_async_context(env.serving())
            )
        except BaseException:
            self._release()
            raise
        self.env, self.tasks = env, tasks

    def _release(self) -> None:
        global _bridged
        with _bridge_lock:
            if _bridged is self:
                interception_server.resolve_client = self.resolve_client
                _bridged = None

    def start(self, index: int) -> Turn:
        self._abandon()
        if self.env is None:
            self._load()
        task = self.tasks[index % len(self.tasks)]
        self.task_id = str(task.data.name or task.data.id or task.data.idx)
        self.steps = 0
        episode = self.episode = self.loop.create_task(
            self.env.run_episode(task, self.ctx)
        )
        turn = self._advance(first=True)
        if turn is None:
            errors = _errors(episode.result())
            raise RuntimeError(
                f"the environment ended before its first model call: {errors}"
            )
        return turn

    def respond(self, reply: Message) -> Turn:
        assert self.reply is not None and self.episode is not None
        self.reply.set_result(reply)
        self.steps += 1
        turn = self._advance()
        if turn is not None:
            if self.max_steps is None or self.steps < self.max_steps:
                return turn
            # A simulated user that never hangs up would otherwise run for hours.
            self._abandon()
            return Turn(
                [], rewards={}, truncated=True, info={"stop_condition": "max_steps"}
            )
        episode = self.episode.result()
        self.episode = None
        errors = _errors(episode)
        info: dict[str, Any] = {}
        rewards: dict[str, float] = {}
        reward = 0.0
        truncated = episode.is_timeout
        if episode.traces:
            trace = episode.traces[-1]
            rewards = {k: r.score for k, r in trace.rewards.items() if r is not None}
            reward = trace.reward
            # tau2 reports its own step limit as tau2_max_steps, not a truncation.
            truncated = (
                truncated
                or trace.is_timeout
                or trace.is_truncated
                or trace.stop_condition.endswith("max_steps")
            )
            info["stop_condition"] = trace.stop_condition
            info["metrics"] = {k: v for k, v in trace.metrics.items() if v is not None}
        if errors:
            info["errors"] = errors
        return Turn([], rewards=rewards, reward=reward, truncated=truncated, info=info)

    def close(self) -> None:
        try:
            self._abandon()
            self.loop.run_until_complete(self.serving.aclose())
            self.loop.close()
        finally:
            self._release()

    def _advance(self, first: bool = False) -> Turn | None:
        """Run the episode until it asks the model again (a turn) or ends (None).

        The first turn carries the whole prompt; later turns carry what follows
        the last assistant message.
        """
        assert self.episode is not None
        self.request = self.loop.create_future()
        self.loop.run_until_complete(
            asyncio.wait(
                [self.request, self.episode], return_when=asyncio.FIRST_COMPLETED
            )
        )
        if not self.request.done():
            return None
        body, self.reply = self.request.result()
        messages = body["messages"]
        new = 0
        if not first:
            new = 1 + max(
                (i for i, m in enumerate(messages) if m["role"] == "assistant"),
                default=-1,
            )
        return Turn(messages[new:], body.get("tools") or [])

    def _abandon(self) -> None:
        if self.episode is not None:
            self.episode.cancel()
            self.loop.run_until_complete(
                asyncio.gather(self.episode, return_exceptions=True)
            )
            self.episode = None


def _errors(episode: Any) -> list[str]:
    errors = [*episode.errors, *(e for t in episode.traces for e in t.errors)]
    return [f"{e.type}: {e.message}" for e in errors]


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a v1 taskset or environment.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--taskset", help="taskset id (default $TASKSET, then primeintellect/math500)"
    )
    source.add_argument("--env", help="serve a v1 environment (e.g. tau2-bench)")
    parser.add_argument("--domain", help="the environment's domain (tau2-bench)")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="end an --env episode as truncated after this many replies",
    )
    parser.add_argument(
        "--runtime",
        choices=["subprocess"],
        help="provision a local runtime for reward functions that execute code",
    )
    parser.add_argument("--address", default=DEFAULT_ADDRESS, help="bind address")
    args = parser.parse_args()
    env = (
        V1EnvEnv(args.env, args.domain, args.max_steps)
        if args.env or (not args.taskset and os.environ.get("ENV_ID"))
        else TasksetEnv(args.taskset, args.runtime)
    )
    serve(env, args.address)


if __name__ == "__main__":
    main()
