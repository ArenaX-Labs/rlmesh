"""Serve a Prime verifiers v0 environment (wordle, tool-test, ...) as an RLMesh env.

Run with the v0 venv (``requirements-v0.txt``). The environment runs its own
rollout loop; where it would call the model, a bridge client hands the prompt
to RLMesh as an observation and waits for the next action.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import os
import time
from typing import Any

import verifiers as vf
from chat_env import (
    DEFAULT_ADDRESS,
    ChatTaskEnv,
    Message,
    Turn,
    openai_messages,
    serve,
)
from verifiers.types import (
    Response,
    ResponseMessage,
    ToolCall,
)

Request = tuple[list[Any], list[Any], "asyncio.Future[Message]"]


class _BridgeClient(vf.Client):
    """A verifiers client whose model is whoever sends the next RLMesh action."""

    def __init__(self, env: EnvironmentEnv) -> None:
        super().__init__(None)
        self.env = env

    async def get_response(
        self, prompt: Any, model: str, sampling_args: Any, tools: Any = None, **_: Any
    ) -> Any:
        reply: asyncio.Future[Message] = asyncio.get_running_loop().create_future()
        self.env.request.set_result((prompt, tools or [], reply))
        return _response(await reply)

    def setup_client(self, config: Any) -> None:
        return None

    async def to_native_tool(self, tool: Any) -> Any:
        return tool

    async def to_native_prompt(self, messages: Any) -> tuple[Any, dict[str, Any]]:
        return messages, {}

    async def get_native_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def raise_from_native_response(self, response: Any) -> None:
        return None

    async def from_native_response(self, response: Any) -> Any:
        return response

    async def close(self) -> None:
        return None


class EnvironmentEnv(ChatTaskEnv):
    """One rollout of a v0 environment per episode."""

    request: asyncio.Future[Request]

    def __init__(self, env_id: str | None = None, max_turns: int | None = None) -> None:
        super().__init__()
        self.env_id = env_id or os.environ.get("ENV_ID") or "wordle"
        raw = os.environ.get("MAX_TURNS")
        self.max_turns = (
            max_turns if max_turns is not None else int(raw) if raw else None
        )
        self.env: Any = None
        self.dataset: Any = None
        self.client = _BridgeClient(self)
        self.rollout: asyncio.Task[Any] | None = None
        self.reply: asyncio.Future[Message] | None = None

    @functools.cached_property
    def loop(self) -> asyncio.AbstractEventLoop:
        return asyncio.new_event_loop()

    def start(self, index: int) -> Turn:
        self._abandon()
        if self.env is None:
            env = vf.load_environment(self.env_id)
            if self.max_turns is not None:
                env.max_turns = self.max_turns
            self.dataset = env.get_eval_dataset()
            self.env = env
        row = self.dataset[index % len(self.dataset)]
        self.task_id = str(row.get("example_id", index))
        rollout = self.rollout = self.loop.create_task(self._run(row))
        turn = self._advance(first=True)
        if turn is None:
            state = rollout.result()
            error = state.get("error") if state is not None else None
            error = f": {error!r}" if error is not None else ""
            raise RuntimeError(
                f"the environment ended before its first model call{error}"
            )
        return turn

    def respond(self, reply: Message) -> Turn:
        assert self.reply is not None and self.rollout is not None
        self.reply.set_result(reply)
        turn = self._advance()
        if turn is not None:
            return turn
        state = self.rollout.result()
        self.rollout = None
        metrics = {k: float(v) for k, v in (state["metrics"] or {}).items()}
        reward = float(state["reward"] or 0.0)
        info = {"stop_condition": str(state.get("stop_condition"))}
        if state.get("error") is not None:
            info["error"] = repr(state["error"])
        return Turn(
            _dump(state.get("final_env_response") or []),
            rewards=metrics,
            reward=reward,
            success=metrics["correct_answer"] >= 1.0
            if "correct_answer" in metrics
            else None,
            truncated=state.get("stop_condition") == "max_turns_reached"
            and self.env.max_turns > 1,
            info=info,
        )

    async def _run(self, row: dict[str, Any]) -> Any:
        state = await self.env.rollout(row, self.client, "rlmesh")
        await self.env.rubric.score_rollout(state)
        return state

    def close(self) -> None:
        self._abandon()
        self.loop.close()

    def _advance(self, first: bool = False) -> Turn | None:
        """Run the rollout until it asks the model again (a turn) or ends (None).

        The first turn carries the whole prompt; later turns carry what follows
        the last assistant message.
        """
        assert self.rollout is not None
        self.request = self.loop.create_future()
        self.loop.run_until_complete(
            asyncio.wait(
                [self.request, self.rollout], return_when=asyncio.FIRST_COMPLETED
            )
        )
        if not self.request.done():
            return None
        prompt, tools, self.reply = self.request.result()
        new = 0
        if not first:
            new = 1 + max(
                (i for i, m in enumerate(prompt) if m.role == "assistant"), default=-1
            )
        return Turn(_dump(prompt[new:]), [_tool(t) for t in tools])

    def _abandon(self) -> None:
        if self.rollout is not None:
            self.rollout.cancel()
            self.loop.run_until_complete(
                asyncio.gather(self.rollout, return_exceptions=True)
            )
            self.rollout = None


def _dump(messages: list[Any]) -> list[Message]:
    return openai_messages(
        [m.model_dump(mode="json", exclude_none=True) for m in messages]
    )


def _tool(tool: Any) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _response(reply: Message) -> Any:
    calls = [
        ToolCall(
            id=call["id"],
            name=call["function"]["name"],
            arguments=call["function"]["arguments"],
        )
        for call in reply.get("tool_calls") or []
    ]
    return Response(
        id="rlmesh",
        created=int(time.time()),
        model="rlmesh",
        message=ResponseMessage(
            content=reply.get("content"),
            tool_calls=calls or None,
            finish_reason="tool_calls" if calls else "stop",
            is_truncated=False,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a v0 environment.")
    parser.add_argument(
        "--env", help="hub environment name (default $ENV_ID, then wordle)"
    )
    parser.add_argument("--max-turns", type=int, help="override the env's turn limit")
    parser.add_argument("--address", default=DEFAULT_ADDRESS, help="bind address")
    args = parser.parse_args()
    serve(EnvironmentEnv(args.env, args.max_turns), args.address)


if __name__ == "__main__":
    main()
