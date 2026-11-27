from __future__ import annotations

import contextlib
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


def test_served_harness_calls_the_llm_for_every_session_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, session_port: int
) -> None:
    import threading
    import time

    import rlmesh
    from rlmesh import spaces

    monkeypatch.setenv("RLMESH_OUTPUT_DIR", str(tmp_path))
    monkeypatch.syspath_prepend(str(EXAMPLES / "llm_agent"))
    from harness import LLMHarness

    lock = threading.Lock()
    running = [0, 0]

    def chat(messages: list[Any], tools: list[Any]) -> tuple[Any, Any]:
        with lock:
            running[0] += 1
            running[1] = max(running)
        time.sleep(0.3)
        with lock:
            running[0] -= 1
        return {"role": "assistant", "content": messages[-1]["content"]}, {}

    class Ask:
        observation_space = spaces.Text(4096, min_length=0)
        action_space = spaces.Text(4096, min_length=0)

        def __init__(self) -> None:
            self.replies: list[str] = []

        def reset(self, *, seed: int | None = None, options: Any = None) -> Any:
            message = {"role": "user", "content": f"q{seed}"}
            return json.dumps({"messages": [message], "tools": []}), {}

        def step(self, action: str) -> Any:
            self.replies.append(json.loads(action)["content"])
            return json.dumps({"messages": [], "tools": []}), 0.0, True, False, {}

        def close(self) -> None:
            return None

    harness = LLMHarness(model="fake", chat=chat, retries=0)
    session = f"127.0.0.1:{session_port}"
    server = threading.Thread(
        target=harness.serve, args=(session,), kwargs={"name": "h"}
    )
    server.start()
    envs = [Ask() for _ in range(3)]
    try:
        clients = [
            threading.Thread(
                target=lambda env=env, seed=seed: rlmesh.Model(f"{session}/h").run(
                    env, seeds=[seed]
                )
            )
            for seed, env in enumerate(envs)
        ]
        for client in clients:
            client.start()
        for client in clients:
            client.join(30)
    finally:
        harness.close()
        server.join(10)
    assert [env.replies for env in envs] == [["q0"], ["q1"], ["q2"]]
    assert running[1] > 1, "the sessions' chat calls overlapped"


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


class _FakeV1Env:
    def __init__(self, config: Any) -> None:
        self.taskset = SimpleNamespace(
            load=lambda: [SimpleNamespace(data=SimpleNamespace(name="t7"))]
        )
        self.seen: list[Any] = []
        self.served = False
        self.calls = 2
        self.trace_errors: list[Any] = []
        self.episode_errors: list[Any] = []
        self.stop_condition = "user_completed"

    @contextlib.asynccontextmanager
    async def serving(self) -> Any:
        self.served = True
        yield
        self.served = False

    async def run_episode(self, task: Any, ctx: Any) -> Any:
        client = sys.modules["verifiers.v1.interception.server"].resolve_client(None)
        dialect = SimpleNamespace(
            upstream_path="/chat/completions",
            validate_response=lambda raw: raw,
            parse_response=lambda raw: SimpleNamespace(
                message=raw["choices"][0]["message"]
            ),
        )
        tools = [{"type": "function", "function": {"name": "lookup"}}]
        messages = [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "my phone is broken"},
        ]
        for _ in range(self.calls):
            reply = await client.get_response(
                dialect,
                {"messages": messages, "tools": tools},
                None,
                session_id="s",
                turn=None,
                headers={},
            )
            self.seen.append(reply.message)
            messages = [
                *messages,
                reply.message,
                {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            ]
        reward = SimpleNamespace(score=1.0)
        trace = SimpleNamespace(
            rewards={"tau2_reward": reward, "skipped": None},
            reward=1.0,
            metrics={"turns": 2.0},
            stop_condition=self.stop_condition,
            is_timeout=False,
            is_truncated=False,
            errors=self.trace_errors,
        )
        return SimpleNamespace(
            traces=[trace], errors=self.episode_errors, is_timeout=False
        )


def _stub_verifiers_v1(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    v1 = ModuleType("verifiers.v1")
    v1.__dict__.update(
        Client=object,
        taskset_config_type=lambda env_id: SimpleNamespace,
        env_config_type=lambda env_id: SimpleNamespace,
        load_environment=_FakeV1Env,
        ModelContext=SimpleNamespace,
        EvalClientConfig=SimpleNamespace,
    )
    interception = ModuleType("verifiers.v1.interception")
    server = ModuleType("verifiers.v1.interception.server")
    verifiers = ModuleType("verifiers")
    verifiers.__dict__["v1"] = v1
    v1.__dict__["interception"] = interception
    interception.__dict__["server"] = server
    for module in (verifiers, v1, interception, server):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.delitem(sys.modules, "serve_v1", raising=False)
    request.addfinalizer(lambda: sys.modules.pop("serve_v1", None))


def test_v1_env_bridge_turns_and_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    monkeypatch.setenv("RLMESH_OUTPUT_DIR", str(tmp_path))
    monkeypatch.syspath_prepend(str(EXAMPLES / "llm_agent"))
    _stub_verifiers_v1(monkeypatch, request)
    from serve_v1 import V1EnvEnv

    env = V1EnvEnv("tau2", "telecom")
    obs, info = env.reset(seed=0)
    assert json.loads(obs) == {
        "messages": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "my phone is broken"},
        ],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    }
    assert info == {"task_id": "t7"}
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "lookup", "arguments": "{}"},
    }
    obs, _, done, _, _ = env.step(json.dumps({"content": None, "tool_calls": [call]}))
    assert env.env.seen[0] == {
        "content": None,
        "tool_calls": [call],
        "role": "assistant",
    }
    assert json.loads(obs)["messages"] == [
        {"role": "tool", "tool_call_id": "c1", "content": "ok"}
    ]
    assert not done
    _, reward, done, truncated, info = env.step(
        json.dumps({"role": "assistant", "content": "fixed"})
    )
    assert (reward, done, truncated, info["is_success"]) == (1.0, True, False, True)
    assert info["rewards"] == {"tau2_reward": 1.0}
    assert info["stop_condition"] == "user_completed"

    env.env.episode_errors = [SimpleNamespace(type="EnvError", message="no seat")]
    env.env.trace_errors = [SimpleNamespace(type="HarnessError", message="exited 1")]
    env.env.calls = 0
    with pytest.raises(
        RuntimeError,
        match=r"first model call: \['EnvError: no seat', 'HarnessError: exited 1'\]",
    ):
        env.reset(seed=1)

    env.close()
    assert env.loop.is_closed() and not env.env.served


def test_v1_env_bridge_max_steps_truncates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    monkeypatch.setenv("RLMESH_OUTPUT_DIR", str(tmp_path))
    monkeypatch.syspath_prepend(str(EXAMPLES / "llm_agent"))
    _stub_verifiers_v1(monkeypatch, request)
    from serve_v1 import V1EnvEnv

    env = V1EnvEnv("tau2", max_steps=1)
    env.reset(seed=0)
    obs, reward, done, truncated, info = env.step(json.dumps({"content": "hi"}))
    assert (json.loads(obs)["messages"], reward, done, truncated) == (
        [],
        0.0,
        False,
        True,
    )
    assert (info["stop_condition"], info["is_success"]) == ("max_steps", False)
    assert env.episode is None and len(env.env.seen) == 1

    env.env.calls = 1
    env.env.stop_condition = "tau2_max_steps"
    env.reset(seed=0)
    _, _, done, truncated, info = env.step(json.dumps({"content": "hi"}))
    assert (done, truncated, info["stop_condition"]) == (False, True, "tau2_max_steps")
    env.close()


def test_transcript_redacts_image_data_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(EXAMPLES / "llm_agent"))
    from harness import redact_images

    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}
    remote = {"type": "image_url", "image_url": {"url": "https://x/a.png"}}
    text = {"type": "text", "text": "hi"}
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": [image, remote, text]},
    ]
    assert redact_images(messages) == [
        {"role": "system", "content": "rules"},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,<4 bytes>"},
                },
                remote,
                text,
            ],
        },
    ]
    assert image["image_url"]["url"] == "data:image/png;base64,QUJD"


def test_predict_logs_redacted_images_but_sends_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RLMESH_OUTPUT_DIR", str(tmp_path))
    monkeypatch.syspath_prepend(str(EXAMPLES / "llm_agent"))
    from harness import LLMHarness

    sent: list[Any] = []

    def chat(messages: Any, tools: Any) -> tuple[dict[str, Any], None]:
        sent.append(json.loads(json.dumps(messages)))
        raise RuntimeError(f"bad request: {messages[0]['content'][0]}")

    harness = LLMHarness(model="fake", chat=chat, retries=0)
    url = "data:image/png;base64,QUJD"
    observation = json.dumps(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": url}}],
                }
            ],
            "tools": [],
        }
    )
    context: Any = {
        "episode_id": "ep",
        "episode_seed": None,
        "predict_index": 0,
        "predict_seed": None,
        "state": {},
    }
    with pytest.raises(RuntimeError):
        harness.predict(observation, context)

    assert sent[0][0]["content"][0]["image_url"]["url"] == url
    assert context["state"]["messages"][0]["content"][0]["image_url"]["url"] == url
    (line,) = (tmp_path / "transcripts" / "ep.jsonl").read_text().splitlines()
    assert "QUJD" not in line
    record = json.loads(line)
    assert (
        record["request"]["messages"][0]["content"][0]["image_url"]["url"]
        == "data:image/png;base64,<4 bytes>"
    )
    assert "<redacted>" in record["error"]


def test_tictactoe_scores_both_seats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RLMESH_OUTPUT_DIR", str(tmp_path))
    monkeypatch.syspath_prepend(str(EXAMPLES / "llm_agent"))
    from chat_env import TicTacToeTask

    def play(*moves: str, seed: int = 0) -> tuple[list[str], dict[str, Any]]:
        env = TicTacToeTask()
        obs, _ = env.reset(seed=seed)
        prompts = [json.loads(obs)["messages"][0]["content"]]
        for move in moves:
            obs, _, done, _, info = env.step(
                json.dumps({"role": "assistant", "content": move})
            )
            if done:
                return prompts, info
            prompts.append(json.loads(obs)["messages"][0]["content"])
        raise AssertionError("game did not end")

    prompts, info = play("1", "4", "2", "5", "3")
    assert [p.split("You are ")[1][0] for p in prompts] == ["X", "O", "X", "O", "X"]
    assert info["rewards"] == {"X": 1.0, "O": -1.0}
    assert (info["is_success"], info["reward"], info["winner"]) == (True, 1.0, "X")

    _, info = play("1", "1")
    assert info["rewards"] == {"X": 1.0, "O": -1.0} and info["winner"] == "X"
    _, info = play("seven")
    assert info["rewards"] == {"X": -1.0, "O": 1.0} and info["is_success"] is False

    prompts, info = play("seven", seed=1)
    assert prompts[0].split("You are ")[1][0] == "O"
    assert info["rewards"] == {"X": 1.0, "O": -1.0} and info["reward"] == 1.0

    _, info = play(json.dumps([{"type": "text", "text": "5"}]))
    assert info["winner"] == "O"

    _, info = play("1", "2", "3", "5", "4", "6", "8", "7", "9")
    assert info["rewards"] == {"X": 0.0, "O": 0.0} and info["winner"] == "draw"
