"""One ``Policy``, one ``predict``, on both clocks: the robot decides.

- A robot declares its clock when it joins; a policy learns it at bind
  (``Policy.clock``), before the robot serves a step.
- On a lockstep robot ``run`` runs episodes, each from a reset seeded as a
  model run seeds it; a chunk of H actions is H steps, rewards summed.
- On a wall-clock robot the same ``predict`` drives it for a duration.
- Each clock refuses the other's settings at once, and a policy binding a
  lockstep robot among others is refused.
- A Gym env served with ``EnvServer`` is a lockstep robot: a policy binds it
  by kind.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pytest
import rlmesh
import rlmesh.numpy
import rlmesh.realtime as v2
from rlmesh._rlmesh import episode_seed
from rlmesh.realtime import actions, inputs, parts, results
from typed_results import check

ARM = {
    "arm": parts.joints(
        ["a", "b"], hz=100, fields=["pos"], command="position", safe=parts.Safe.STOP
    )
}


def reach(session: v2.Host | str, name: str = "reach") -> v2.Policy:
    """Reads the arm, commands it 4 actions at a time at 100 Hz."""
    return v2.Policy(
        hz=25,
        inputs={"q": inputs.joints("arm", fields=["pos"])},
        actions={"arm": actions.joints("arm", command="position", hz=100, chunk=4)},
        session=session,
        name=name,
    )


def predict(obs: v2.Observation) -> dict[str, Any]:
    """The one model: reach for 1 on both joints."""
    assert obs["q"].shape == (2,)
    return {"arm": np.ones((4, 2), np.float32)}


class Sim:
    """A lockstep arm: halfway to its target each step, cut at ``length`` steps."""

    def __init__(self, host: v2.Host, name: str = "sim", length: int = 10) -> None:
        self.robot = v2.Robot(name, ARM, session=host, clock=v2.Clock.LOCKSTEP)
        self.length = length
        self.q = np.zeros(2, np.float32)
        self.t = 0
        self.seeds: list[int | None] = []
        self.rewards: list[float] = []
        self.steps = 0

    def serve(self) -> None:
        self.robot.serve(self.step, self.reset, background=True)

    def step(self, commands: dict[str, Any]) -> Any:
        self.q = self.q + 0.5 * (commands["arm"] - self.q)
        self.t += 1
        self.steps += 1
        self.robot.publish("arm", pos=self.q)
        reward = -float(np.abs(1 - self.q).sum())
        self.rewards.append(reward)
        return reward, False, self.t >= self.length

    def reset(self, seed: int | None, options: Any) -> None:
        self.seeds.append(seed)
        # The seed sets where the arm starts.
        self.q = np.full(2, 0.0 if seed is None else (seed % 7) / 10, np.float32)
        self.t = 0
        self.robot.publish("arm", pos=self.q)


class Arm:
    """A wall-clock arm: publishes and plays what is due at 100 Hz."""

    def __init__(self, host: v2.Host, name: str = "ur10e") -> None:
        self.robot = v2.Robot(name, ARM, session=host)
        self.played = 0
        self._stop = threading.Event()
        self._loop = threading.Thread(target=self._run, daemon=True)
        self._loop.start()

    def _run(self) -> None:
        q = np.zeros(2, np.float32)
        rate = v2.Rate(100)
        while not self._stop.is_set():
            self.robot.publish("arm", pos=q)
            target = self.robot.command("arm")
            if target is not None:
                self.played += 1
                q = np.asarray(target, np.float32)
            rate.sleep()

    def close(self) -> None:
        self._stop.set()
        self._loop.join()
        self.robot.close()


def test_one_predict_runs_a_lockstep_robot_and_a_wall_clock_robot() -> None:
    with v2.Host() as host:
        sim = Sim(host)
        sim.serve()
        arm = Arm(host)
        # The lockstep robot armed the session before the arm declared.
        arm.robot.rearm("ur10e")
        try:
            with reach(host) as policy:
                stats = policy.bind("sim").run(predict, episodes=2, seed=3)
                assert policy.clock is v2.Clock.LOCKSTEP
                assert policy.holds and policy.events() == [] and policy.held == []
            assert stats["clock"] == "lockstep"
            check(stats, results.LockstepStats, "run()")
            assert [e["steps"] for e in stats["episodes"]] == [10, 10]
            assert [e["return"] for e in stats["episodes"]] == [
                sum(sim.rewards[:10]),
                sum(sim.rewards[10:]),
            ]
            assert all(
                e["truncated"] and not e["terminated"] for e in stats["episodes"]
            )

            with reach(host, "reach-wall") as policy:
                stats = policy.bind("ur10e").run(predict, duration=1.5)
                assert policy.clock is v2.Clock.WALL
            assert stats["clock"] == "wall"
            check(stats, results.PolicyStats, "run()")
            assert stats["completions"] > 0
            assert arm.played > 0, "the arm played the chunks"
        finally:
            arm.close()
            sim.robot.close()


def test_a_seed_runs_the_same_episodes_again() -> None:
    with v2.Host() as host:
        sim = Sim(host)
        sim.serve()
        try:
            with reach(host) as policy:
                policy.bind("sim")
                first = policy.run(predict, episodes=3, seed=11)
                first_seeds, sim.seeds = sim.seeds, []
                again = policy.run(predict, episodes=3, seed=11)
                assert again == first
                assert sim.seeds == first_seeds
                # Each episode's seed derives from the base, the robot's
                # name and the episode's number, as a model run's does.
                assert first_seeds == [episode_seed(11, "sim", i) for i in range(3)]
                assert len(set(first_seeds)) == 3
                # Unseeded resets without a seed.
                sim.seeds = []
                policy.run(predict)
                assert sim.seeds == [None]
        finally:
            sim.robot.close()


def test_a_seed_draws_the_episodes_a_model_run_draws(session_port: int) -> None:
    gym = pytest.importorskip("gymnasium")

    class Seeded(gym.Env):  # type: ignore[misc]
        """Two steps an episode; records its reset seeds."""

        observation_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

        def __init__(self) -> None:
            self.seeds: list[int | None] = []
            self.t = 0

        def reset(self, *, seed: int | None = None, options: Any = None) -> Any:
            self.seeds.append(seed)
            self.t = 0
            return np.zeros(2, np.float32), {}

        def step(self, action: Any) -> Any:
            self.t += 1
            return np.zeros(2, np.float32), 1.0, self.t >= 2, False, {}

    env = Seeded()
    base = f"127.0.0.1:{session_port}"
    server = rlmesh.EnvServer(env, f"{base}/seeded")
    server.start()
    try:
        box = {"kind": "gym/box", "fields": {"value": ["float32", [2]]}}
        with v2.Policy(
            hz=10,
            inputs={"x": {**box, "at": "obs"}},
            actions={"a": {**box, "at": "action"}},
            session=base,
            name="p",
        ) as policy:
            policy.bind("seeded").run(
                lambda obs: {"a": np.zeros((1, 2), np.float32)}, episodes=3, seed=5
            )
        by_policy, env.seeds = env.seeds, []
        model = rlmesh.numpy.Model(lambda obs: np.zeros(2, np.float32))
        [result] = model.run_routes([server], episodes=3, base_seed=5)
        assert not isinstance(result, Exception), result
        assert env.seeds == by_policy
        assert len(set(by_policy)) == 3
    finally:
        server.shutdown()


def test_a_chunk_of_four_is_four_steps_with_its_rewards_summed() -> None:
    with v2.Host() as host:
        sim = Sim(host, length=8)
        sim.serve()
        played: list[tuple[int, ...]] = []
        try:
            with reach(host) as policy:
                stats = policy.bind("sim").run(
                    predict,
                    on_step=lambda step: played.append(
                        step.actions["arm"]["pos"].shape
                    ),
                )
            assert stats["clock"] == "lockstep"
            assert stats["starts"] == stats["chunks_sent"] == 2
            assert sim.steps == 8
            assert played == [(4, 2), (4, 2)]
            [episode] = stats["episodes"]
            assert episode["steps"] == 8
            assert episode["return"] == sum(sim.rewards)
        finally:
            sim.robot.close()


def test_input_ages_are_session_time_and_never_stale() -> None:
    # A camera published every 4th step: between, its age grows by one
    # action period (10 ms at 100 Hz) a step, past its 25 ms bound, and it
    # is never stale.
    cam = {
        "cam": parts.camera(4, 4, hz=50),
        "arm": ARM["arm"],
    }
    with v2.Host() as host:
        robot = v2.Robot("eye", cam, session=host, clock=v2.Clock.LOCKSTEP)
        t = 0

        def publish() -> None:
            robot.publish("arm", pos=np.zeros(2, np.float32))
            if t % 4 == 0:
                robot.publish("cam", np.zeros((4, 4, 3), np.uint8))

        def step(commands: dict[str, Any]) -> Any:
            nonlocal t
            t += 1
            publish()
            return 0.0, False, t >= 8

        def reset(seed: int | None, options: Any) -> None:
            nonlocal t
            t = 0
            publish()

        robot.serve(step, reset, background=True)
        ages: list[float | None] = []
        stale: list[frozenset[str]] = []

        def look(obs: v2.Observation) -> dict[str, Any]:
            ages.append(obs.age_ms("cam"))
            stale.append(obs.meta.stale)
            return {"arm": np.zeros((1, 2), np.float32)}

        try:
            with v2.Policy(
                hz=100,
                inputs={"cam": inputs.camera("cam", max_age_ms=25)},
                actions={"arm": actions.joints("arm", command="position", chunk=1)},
                session=host,
            ) as policy:
                policy.bind("eye").run(look)
            assert ages == [0.0, 10.0, 20.0, 30.0] * 2
            assert all(not s for s in stale)
        finally:
            robot.close()


def test_each_clock_refuses_the_other_s_settings_at_once() -> None:
    with v2.Host() as host:
        sim = Sim(host)
        sim.serve()
        arm = Arm(host)
        arm.robot.rearm("ur10e")
        try:
            with reach(host) as policy:
                policy.bind("sim")
                for given, says in [
                    ({"duration": 1.0}, "duration= is for a wall-clock robot"),
                    (
                        {"mode": v2.RunMode.SHADOW},
                        "mode=shadow is for a wall-clock robot",
                    ),
                    ({"resume": v2.Resume.MANUAL}, "resume= is for a wall-clock robot"),
                    ({"episodes": 0}, "episodes=0"),
                ]:
                    started = time.monotonic()
                    with pytest.raises(ValueError, match=says):
                        policy.run(predict, **given)
                    assert time.monotonic() - started < 1
                assert sim.steps == 0, "nothing was stepped"
            with reach(host, "reach-wall") as policy:
                policy.bind("ur10e")
                for given, says in [
                    ({"episodes": 2}, "episodes= is for a lockstep robot"),
                    ({"seed": 1}, "seed= is for a lockstep robot"),
                ]:
                    with pytest.raises(ValueError, match=says):
                        policy.run(predict, **given)
                # Lockstep has no leases or sends to ask for.
            with reach(host, "reach-calls") as policy:
                policy.bind("sim")
                with pytest.raises(v2.SessionException, match="runs in lockstep"):
                    policy.acquire()
        finally:
            arm.close()
            sim.robot.close()


def test_a_policy_binding_a_lockstep_robot_among_others_is_refused() -> None:
    with v2.Host() as host:
        sim = Sim(host)
        sim.serve()
        board = v2.Robot("board", ARM, session=host)
        try:
            both = v2.Policy(
                hz=25,
                inputs={
                    "sim": inputs.joints("sim:arm", fields=["pos"]),
                    "board": inputs.joints("board:arm", fields=["pos"]),
                },
                session=host,
                name="both",
            )
            with both, pytest.raises(v2.BindingRefused, match="sim runs in lockstep"):
                both.bind()
        finally:
            board.close()
            sim.robot.close()


def test_a_lockstep_robot_is_known_by_its_declaration_before_it_serves() -> None:
    with v2.Host() as host:
        # Attached and declared lockstep, nothing served: nothing answers
        # its lockstep key, and its parts come with its serve.
        sim = Sim(host)
        bound: list[v2.Policy] = []
        policy = reach(host)
        binder = threading.Thread(target=lambda: bound.append(policy.bind("sim")))
        binder.start()
        time.sleep(0.3)
        assert not bound, "waiting for the robot's parts"
        sim.serve()
        binder.join(10)
        try:
            assert bound and policy.clock is v2.Clock.LOCKSTEP
            stats = policy.run(predict)
            assert stats["clock"] == "lockstep" and len(stats["episodes"]) == 1
        finally:
            policy.close()
            sim.robot.close()


def test_a_served_gym_env_is_run_by_a_policy_binding_by_kind(session_port: int) -> None:
    gym = pytest.importorskip("gymnasium")
    base = f"127.0.0.1:{session_port}"
    server = rlmesh.EnvServer(gym.make("CartPole-v1"), f"{base}/cartpole")
    server.start()
    try:
        balance = v2.Policy(
            hz=50,
            inputs={"x": {"kind": "gym/box", "fields": {"value": ["float32", [4]]}}},
            actions={
                "push": {"kind": "gym/discrete", "fields": {"value": ["int64", []]}}
            },
            session=base,
            name="balance",
        )

        def lean(obs: v2.Observation) -> dict[str, Any]:
            # Push toward the side the pole leans to.
            return {"push": np.array([int(obs["x"][2] > 0)], np.int64)}

        with balance:
            stats = balance.bind("cartpole").run(lean, episodes=2, seed=0)
            assert balance.clock is v2.Clock.LOCKSTEP
        assert stats["clock"] == "lockstep"
        assert len(stats["episodes"]) == 2
        for episode in stats["episodes"]:
            assert episode["terminated"] or episode["truncated"]
            # CartPole pays 1 a step.
            assert episode["return"] == episode["steps"] > 0
        # The participant address read as a session names an empty session.
        with (
            v2.Policy(
                hz=10,
                inputs={
                    "x": {"kind": "gym/box", "fields": {"value": ["float32", [4]]}}
                },
                session=f"{base}/cartpole",
                name="reader",
                timeout=4,
            ) as reader,
            pytest.raises(
                v2.BindingRefused, match="participant address, not a session"
            ),
        ):
            reader.bind()
    finally:
        server.shutdown()
