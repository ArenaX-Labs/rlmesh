from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
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


def _stub_verifiers_v0(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
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


def test_v0_bridge_turns_reset_and_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    monkeypatch.setenv("RLMESH_OUTPUT_DIR", str(tmp_path))
    monkeypatch.syspath_prepend(str(EXAMPLES / "llm_agent"))
    _stub_verifiers_v0(monkeypatch, request)
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


class _UnservableV1Env(_FakeV1Env):
    @contextlib.asynccontextmanager
    async def serving(self) -> Any:
        raise OSError("port in use")
        yield


def _unbridged(config: Any) -> Any:
    raise AssertionError("no bridge")


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
        TasksetConfig=SimpleNamespace,
    )
    interception = ModuleType("verifiers.v1.interception")
    server = ModuleType("verifiers.v1.interception.server")
    server.__dict__["resolve_client"] = _unbridged
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


def _fail_first(load: Any, calls: list[Any]) -> Any:
    def flaky(config: Any) -> Any:
        calls.append(config)
        if len(calls) == 1:
            raise ValueError("hub down")
        return load(config)

    return flaky


def test_v0_env_reads_env_vars_and_retries_a_failed_first_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    monkeypatch.setenv("RLMESH_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("ENV_ID", "fake")
    monkeypatch.setenv("MAX_TURNS", "5")
    monkeypatch.syspath_prepend(str(EXAMPLES / "llm_agent"))
    _stub_verifiers_v0(monkeypatch, request)
    loads: list[str] = []
    monkeypatch.setattr(
        sys.modules["verifiers"],
        "load_environment",
        _fail_first(lambda _: _FakeEnv("ok"), loads),
    )
    from serve_v0 import EnvironmentEnv

    env = EnvironmentEnv()
    assert loads == []
    with pytest.raises(ValueError, match="hub down"):
        env.reset(seed=0)
    env.reset(seed=0)
    env.reset(seed=0)
    assert loads == ["fake", "fake"] and env.env.max_turns == 5
    env.close()


def test_v1_envs_read_env_vars_and_retry_a_failed_first_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    monkeypatch.setenv("RLMESH_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("TASKSET", "primeintellect/gpqa")
    monkeypatch.setenv("ENV_ID", "tau2")
    monkeypatch.setenv("DOMAIN", "airline")
    monkeypatch.setenv("MAX_STEPS", "3")
    monkeypatch.syspath_prepend(str(EXAMPLES / "llm_agent"))
    _stub_verifiers_v1(monkeypatch, request)
    tasksets: list[Any] = []
    configs: list[Any] = []
    data = SimpleNamespace(name="q0", prompt="2 + 2?", system_prompt=None)
    v1 = sys.modules["verifiers.v1"]
    monkeypatch.setattr(
        v1,
        "load_taskset",
        _fail_first(
            lambda _: SimpleNamespace(load=lambda: [SimpleNamespace(data=data)]),
            tasksets,
        ),
        raising=False,
    )
    monkeypatch.setattr(v1, "load_environment", _fail_first(_FakeV1Env, configs))
    from serve_v1 import TasksetEnv, V1EnvEnv

    for env, loads in ((TasksetEnv(), tasksets), (V1EnvEnv(), configs)):
        assert loads == []
        with pytest.raises(ValueError, match="hub down"):
            env.reset(seed=0)
        env.reset(seed=0)
        env.reset(seed=0)
        assert len(loads) == 2
        env.close()
    assert tasksets[-1].id == "primeintellect/gpqa"
    assert configs[-1].taskset.domain == "airline"
    with pytest.raises(ValueError, match="at least 1"):
        V1EnvEnv(max_steps=0)


def test_v1_env_serves_one_lane_per_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    monkeypatch.setenv("RLMESH_OUTPUT_DIR", str(tmp_path))
    monkeypatch.syspath_prepend(str(EXAMPLES / "llm_agent"))
    _stub_verifiers_v1(monkeypatch, request)
    from serve_v1 import V1EnvEnv

    server = sys.modules["verifiers.v1.interception.server"]
    v1 = sys.modules["verifiers.v1"]
    first, second = V1EnvEnv("tau2"), V1EnvEnv("tau2")
    monkeypatch.setattr(v1, "load_environment", _UnservableV1Env)
    with pytest.raises(OSError, match="port in use"):
        first.reset(seed=0)
    assert server.resolve_client is _unbridged
    monkeypatch.setattr(v1, "load_environment", _FakeV1Env)
    second.reset(seed=0)
    with pytest.raises(RuntimeError, match="RLMESH_NUM_ENVS=1"):
        first.reset(seed=0)
    second.close()
    assert server.resolve_client is _unbridged
    first.reset(seed=0)
    first.close()


def test_probe_random_text_is_answered_without_the_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RLMESH_OUTPUT_DIR", str(tmp_path))
    monkeypatch.syspath_prepend(str(EXAMPLES / "llm_agent"))
    from chat_env import AdditionTask
    from harness import LLMHarness

    env = AdditionTask()
    for action in ("\x00zq\ufffd", "[1, 2]", "42"):
        env.reset(seed=0)
        _, reward, done, _, info = env.step(action)
        assert (reward, done, info["is_success"]) == (0.0, True, False)

    def chat(messages: Any, tools: Any) -> tuple[dict[str, Any], None]:
        raise AssertionError("the LLM must not be called")

    harness = LLMHarness(chat=chat)
    context: Any = {
        "episode_id": "e",
        "episode_seed": 0,
        "predict_index": 0,
        "predict_seed": 0,
        "state": {},
    }
    assert json.loads(harness.predict("\x00zq", context)) == {
        "role": "assistant",
        "content": "",
    }
    assert json.loads(harness.predict("[1]", context))["content"] == ""


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


FAKE_DOCKER = """\
#!/bin/sh
echo "$*" >> "$DOCKER_LOG"
case "$1" in
  build)
    while [ $# -gt 1 ]; do
      [ "$1" = --iidfile ] && echo sha256:abc > "$2"
      shift
    done ;;
  run) printf '%s' "$ENVELOPE" ;;
esac
"""


def test_publish_refuses_unknown_names_and_failed_describes_and_pushes_oci(
    tmp_path: Path,
) -> None:
    docker = tmp_path / "docker"
    docker.write_text(FAKE_DOCKER)
    docker.chmod(0o755)
    log = tmp_path / "log"

    def publish(envelope: dict[str, Any], *args: str) -> Any:
        log.write_text("")
        env = {
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "DOCKER_LOG": str(log),
            "ENVELOPE": json.dumps(envelope),
        }
        script = EXAMPLES / "llm_agent/docker/publish.sh"
        result = subprocess.run(
            [script, "reg/ns", "t1", *args], env=env, capture_output=True, text=True
        )
        return result.returncode, log.read_text().splitlines()

    ok = {"kind": "env", "env_spec": {"observation_space": {}}}
    assert publish(ok, "llm-env-chat", "bogus") == (2, [])
    code, calls = publish({"env_spec": {"error": "boom"}}, "llm-env-chat")
    assert code == 1 and not any("--label" in call for call in calls)
    for flags, push in (((), "true"), (("--no-push",), "false")):
        code, calls = publish(ok, *flags, "llm-env-chat")
        (stamp,) = [call for call in calls if "--label" in call]
        assert code == 0
        assert f"name=reg/ns/llm-env-chat:t1,oci-mediatypes=true,push={push}" in stamp
