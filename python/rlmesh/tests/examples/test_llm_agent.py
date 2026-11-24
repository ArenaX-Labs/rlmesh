from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from test_examples_run import EXAMPLES, _serve_example


def test_harness_solves_served_addition_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RLMESH_OUTPUT_DIR", str(tmp_path))
    monkeypatch.syspath_prepend(str(EXAMPLES / "llm_agent"))
    from fake_llm import scripted
    from harness import LLMHarness

    harness = LLMHarness(model="fake", chat=scripted("", "5"))
    for address in _serve_example("llm_agent/chat_env.py"):
        result = harness.run(address, seeds=[0])

    episode = result.episodes[0]
    assert episode.success is True
    assert episode.terminated and episode.steps == 1

    (transcript,) = (tmp_path / "transcripts").glob("*.jsonl")
    lines = [json.loads(line) for line in transcript.read_text().splitlines()]
    calls = [line for line in lines if "call" in line]
    assert [call["response"]["content"] for call in calls] == ["", "5"]
    (prompt,) = calls[0]["request"]["messages"]
    assert prompt["role"] == "user" and "2 + 3" in prompt["content"]
    assert calls[-1]["response"] == {"role": "assistant", "content": "5"}
    assert lines[-1]["event"] == "episode_end" and lines[-1]["calls"] == 2
    (env_log,) = (tmp_path / "env").glob("*.jsonl")
    assert json.loads(env_log.read_text())["rewards"] == {"correct": 1.0}


class _Msg(SimpleNamespace):
    def model_dump(self, **_: Any) -> dict[str, Any]:
        return dict(vars(self))


class _FakeEnv:
    max_turns = 3

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.rubric = SimpleNamespace(score_rollout=self._score)

    def get_eval_dataset(self) -> list[dict[str, Any]]:
        return [{"example_id": 7}]

    async def _score(self, state: dict[str, Any]) -> None:
        state["reward"] = 1.0

    async def rollout(self, row: Any, client: Any, model: str) -> dict[str, Any]:
        if self.mode == "raise":
            raise ValueError("no dataset")
        state: dict[str, Any] = {}
        if self.mode == "error":
            state["error"] = ValueError("bad setup")
            return state
        call = {"id": "c1", "name": "lookup", "arguments": '{"q": "x"}'}
        prompt = [
            _Msg(role="system", content="rules"),
            _Msg(role="user", content="example"),
            _Msg(role="assistant", content="shot", tool_calls=[call]),
            _Msg(role="tool", tool_call_id="c1", content="found"),
            _Msg(role="user", content="go"),
        ]
        tool = SimpleNamespace(name="lookup", description="find", parameters={})
        reply = await client.get_response(
            prompt=prompt, model=model, tools=[tool], sampling_args={}, state=state
        )
        prompt += [
            _Msg(role="assistant", content=reply.message.content),
            _Msg(role="user", content="again"),
        ]
        await client.get_response(
            prompt=prompt, model=model, tools=None, sampling_args={}, state=state
        )
        return {**state, "metrics": {"correct_answer": 1.0}, "stop_condition": "done"}


def test_v0_bridge_turns_reset_and_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    monkeypatch.setenv("RLMESH_OUTPUT_DIR", str(tmp_path))
    monkeypatch.syspath_prepend(str(EXAMPLES / "llm_agent"))
    verifiers = ModuleType("verifiers")
    verifiers.__dict__.update(
        Client=type("Client", (), {"__init__": lambda self, config: None}),
        load_environment=_FakeEnv,
    )
    types = ModuleType("verifiers.types")
    types.__dict__.update(
        Response=SimpleNamespace,
        ResponseMessage=SimpleNamespace,
        ToolCall=SimpleNamespace,
    )
    monkeypatch.setitem(sys.modules, "verifiers", verifiers)
    monkeypatch.setitem(sys.modules, "verifiers.types", types)
    monkeypatch.delitem(sys.modules, "serve_v0", raising=False)
    request.addfinalizer(lambda: sys.modules.pop("serve_v0", None))
    from serve_v0 import EnvironmentEnv

    env = EnvironmentEnv("fake")
    obs, info = env.reset(seed=0)
    assert json.loads(obs) == {
        "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "example"},
            {
                "role": "assistant",
                "content": "shot",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q": "x"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "found"},
            {"role": "user", "content": "go"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "lookup", "description": "find", "parameters": {}},
            }
        ],
    }
    assert info == {"task_id": "7"}
    obs, _, done, _, _ = env.step(json.dumps({"role": "assistant", "content": "a"}))
    assert json.loads(obs) == {
        "messages": [{"role": "user", "content": "again"}],
        "tools": [],
    }
    assert not done

    abandoned = env.rollout
    obs, _ = env.reset(seed=0)
    assert abandoned is not None and abandoned.cancelled()
    assert json.loads(obs)["messages"][0]["role"] == "system"
    env.step(json.dumps({"role": "assistant", "content": "a"}))
    _, reward, done, truncated, info = env.step(
        json.dumps({"role": "assistant", "content": "b"})
    )
    assert (reward, done, truncated, info["is_success"]) == (1.0, True, False, True)

    env.reset(seed=0)
    env.close()
    assert env.loop.is_closed()

    failing = EnvironmentEnv("raise")
    with pytest.raises(ValueError, match="no dataset"):
        failing.reset(seed=0)
    failing.close()

    erroring = EnvironmentEnv("error")
    with pytest.raises(RuntimeError, match=re.escape(repr(ValueError("bad setup")))):
        erroring.reset(seed=0)
    erroring.close()
