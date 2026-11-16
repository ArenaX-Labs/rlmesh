"""A coding agent that explores an RLMesh env and writes its policy as code.

Claude never sits in the control loop. Through five tools it probes the env
(``RemoteEnv``), writes ``workdir/policy.py``, and scores it with
``Model(predict).run()``. A policy that passes every evaluation episode is kept
in ``workdir/skills/``.

WARNING: evaluate() executes the code the model writes, in this process. Run it
in a container or VM, and never against hardware whose env has no safety layer.
"""

import argparse
import json
import runpy
import shutil
import time
from pathlib import Path

import anthropic
import numpy as np
from anthropic import beta_tool
from rlmesh.numpy import Model, RemoteEnv

DEFAULT_ADDRESS = "127.0.0.1:5555"
WORKDIR = Path(__file__).parent / "workdir"
POLICY = WORKDIR / "policy.py"
SKILLS = WORKDIR / "skills"
EVAL_SEEDS = range(100, 150)  # evaluation layouts the agent never explores on

PROMPT = """You control a robot through an RLMesh environment.

Goal: write a Python policy that completes the env's task on every evaluation
episode. The task is the "instruction" field of reset()'s info.{goal}

Tools:
- describe(): the observation and action spaces.
- reset(seed) / step(action): drive the env by hand to learn how it behaves.
  Do not assume action frames, scales or signs; measure them.
- write_policy(code): save policy.py. It must define `predict(obs) -> action`,
  where obs is a dict of numpy arrays. Keep any per-episode state (such as a
  phase counter) in module globals, and reset it when the observation shows a
  new episode has started.
- evaluate(episodes): run policy.py against the env and report per-episode
  results. Read the failures, probe the env again if needed, fix the policy,
  and evaluate again.

Before every tool call, write one short sentence saying what you are about to
do and why (for example "Checking which way +x moves the gripper"). These
sentences are shown to people watching the run.

Stop when evaluate reports a success rate of 1.0 (or you run out of ideas),
then summarize what you learned about the robot in a few sentences."""


def summarize(value):
    """JSON-friendly view of an obs/info value; large arrays become their shape."""
    if isinstance(value, dict):
        return {key: summarize(v) for key, v in value.items()}
    if isinstance(value, str | bool | int | float) or value is None:
        return round(value, 4) if isinstance(value, float) else value
    array = np.asarray(value)
    if array.size > 32:
        return f"<array shape={array.shape} dtype={array.dtype}>"
    return np.round(array.astype(float), 4).tolist()


def tool_label(name: str, args: dict) -> str:
    """One readable line per tool call; write_policy shows the code itself."""
    if name == "write_policy":
        return "write_policy:\n" + args.get("code", "")
    return f"{name}({', '.join(f'{k}={v}' for k, v in args.items())})"


class Session:
    """The env connection the agent explores with, plus the run's event log."""

    def __init__(self, address: str, skill: str, log_path: str | None):
        self.address = address
        self.skill = skill
        self.env: RemoteEnv | None = None
        self._log = open(log_path, "a") if log_path else None

    def log(self, kind: str, text: str) -> None:
        """Print an agent event and, with --log, append it for make_video.py."""
        print(f"[{kind}] {text}")
        if self._log is not None:
            row = {"t": time.time(), "kind": kind, "text": text}
            self._log.write(json.dumps(row) + "\n")
            self._log.flush()

    def connect(self) -> RemoteEnv:
        if self.env is None:
            self.env = RemoteEnv(self.address)
        return self.env

    def disconnect(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None


def make_tools(session: Session) -> list:
    @beta_tool
    def describe() -> str:
        """Describe the env's observation and action spaces."""
        env = session.connect()
        return f"observation_space={env.observation_space!r}\naction_space={env.action_space!r}"

    @beta_tool
    def reset(seed: int = 0) -> str:
        """Reset the env and return the first observation.

        Args:
            seed: Reset seed; the same seed gives the same initial layout.
        """
        obs, info = session.connect().reset(seed=seed)
        info = {k: v for k, v in info.items() if k != "episode_ids"}
        return json.dumps({"obs": summarize(obs), "info": summarize(info)})

    @beta_tool
    def step(action: list[float]) -> str:
        """Apply one action and return the observation, reward and flags.

        Args:
            action: One float per action dimension, within the action space bounds.
        """
        env = session.connect()
        expected = int(np.prod(env.action_space.shape))
        if len(action) != expected:
            return f"error: action must have exactly {expected} elements"
        obs, reward, terminated, truncated, info = env.step(
            np.asarray(action, dtype=np.float32)
        )
        return json.dumps(
            {
                "obs": summarize(obs),
                "reward": round(reward, 4),
                "terminated": terminated,
                "truncated": truncated,
                "success": bool(info.get("success")),
            }
        )

    @beta_tool
    def write_policy(code: str) -> str:
        """Save policy.py. It must define predict(obs) -> action.

        Args:
            code: Full Python source of policy.py.
        """
        POLICY.write_text(code)
        return f"wrote {POLICY.name} ({len(code.splitlines())} lines)"

    @beta_tool
    def evaluate(episodes: int = 10) -> str:
        """Run policy.py for several seeded episodes and report the results.

        Args:
            episodes: Number of episodes, 1 to 50.
        """
        seeds = list(EVAL_SEEDS[: max(1, episodes)])
        try:
            predict = runpy.run_path(str(POLICY))["predict"]
        except Exception as exc:
            return f"policy.py failed to load: {exc!r}"
        session.disconnect()  # the env server takes one client at a time
        try:
            # Cast so a float64 policy output doesn't fail the float32 action space.
            model = Model(lambda obs: np.asarray(predict(obs), np.float32))
            result = model.run(session.address, seeds=seeds)
        except Exception as exc:
            session.log("result", f"evaluation failed: {exc!r}")
            return f"evaluation failed: {exc!r}"
        rows = [
            f"seed={ep.seed} steps={ep.steps} success={ep.success} reward={ep.reward:.3f}"
            for ep in result.episodes
        ]
        rate = sum(bool(ep.success) for ep in result.episodes) / len(result.episodes)
        if rate == 1.0:
            SKILLS.mkdir(exist_ok=True)
            shutil.copy(POLICY, SKILLS / f"{session.skill}.py")
            rows.append(f"saved skill to skills/{session.skill}.py")
        session.log("result", f"evaluate: success rate {rate:.0%}")
        return f"success_rate={rate:.2f}\n" + "\n".join(rows)

    return [describe, reset, step, write_policy, evaluate]


def run(session: Session, model: str, max_iterations: int, goal: str | None) -> None:
    goal = f" If the env gives none, it is: {goal}" if goal else ""
    client = anthropic.Anthropic()
    runner = client.beta.messages.tool_runner(
        model=model,
        max_tokens=16000,
        output_config={"effort": "high"},
        # "updates" returns the model's short progress notes between tool calls.
        thinking={"type": "adaptive", "display": "updates"},
        # On a policy decline, the API re-runs the turn on a fallback model.
        betas=[
            "server-side-fallback-2026-07-01",
            "thinking-display-updates-2026-08-18",
        ],
        fallbacks="default",
        max_iterations=max_iterations,
        tools=make_tools(session),
        messages=[{"role": "user", "content": PROMPT.format(goal=goal)}],
    )
    for message in runner:
        for block in message.content:
            if block.type == "thinking" and block.thinking:
                session.log("note", block.thinking)
            elif block.type == "text":
                session.log("agent", block.text)
            elif block.type == "tool_use":
                session.log("tool", tool_label(block.name, block.input))
        if message.stop_reason == "refusal":
            session.log("agent", "request declined")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Let Claude solve an RLMesh env by writing its policy as code."
    )
    parser.add_argument("--address", default=DEFAULT_ADDRESS, help="env endpoint")
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--max-iterations", type=int, default=60, help="agent turns")
    parser.add_argument("--skill", default="policy", help="name for a passing policy")
    parser.add_argument("--goal", help="task text, for envs whose reset() gives none")
    parser.add_argument("--log", help="JSONL file of agent events, for make_video.py")
    args = parser.parse_args()

    WORKDIR.mkdir(exist_ok=True)
    session = Session(args.address, args.skill, args.log)
    session.log("start", f"new agent run ({args.model})")
    try:
        run(session, args.model, args.max_iterations, args.goal)
    finally:
        session.disconnect()


if __name__ == "__main__":
    main()
