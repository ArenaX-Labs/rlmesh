"""Python implementation of the shared two-step counter and constant policy."""

from __future__ import annotations

import builtins
import os
import pathlib
import subprocess
import sys
import threading
import warnings

import rlmesh
from rlmesh import spaces
from rlmesh.adapters import AdapterResolutionError


class ScriptedEnv:
    """Count starts and closes."""

    def __init__(self):
        self.metadata = {rlmesh.ENV_RESET_OPTIONS_KEY: ["trial_index"]}
        self.observation_space = spaces.Discrete(3)
        self.action_space = spaces.Discrete(3)
        self.steps = self.closes = self.starts = 0
        self.seed = self.trial = None

    def reset(self, *, seed=None, options=None):
        self.starts += 1
        self.steps = 0
        self.seed, self.trial = seed, rlmesh.trial_index(options)
        return 0, {}

    def step(self, action):
        self.starts += 1
        self.steps += 1
        return self.steps, 1.0, self.steps >= 2, False, {}

    def close(self):
        self.closes += 1


class ScriptedPolicy:
    """Return action zero and count lifecycle calls independently of sessions."""

    def __init__(self):
        self.ends = self.closes = self.starts = 0
        self.failure = None

    def __call__(self, observation):
        self.starts += 1
        if self.failure:
            raise self.failure
        return 0

    def on_episode_end(self):
        self.ends += 1

    def close(self):
        self.closes += 1


def error_record(error):
    """Expose the R-21 row, preserving folded Python subclasses by isinstance."""
    classes = [
        AdapterResolutionError,
        rlmesh.RecoverableEnvironmentException,
        rlmesh.EnvironmentException,
        rlmesh.ModelException,
        rlmesh.ProtocolException,
        rlmesh.RLMeshException,
        ValueError,
        TypeError,
        RuntimeError,
        ConnectionError,
        TimeoutError,
        OSError,
        KeyboardInterrupt,
    ]
    cls = next((cls for cls in classes if isinstance(error, cls)), None)
    if cls is None:
        raise error
    return {
        "error": cls.__name__,
        "text": str(error),
        "recoverable": bool(getattr(error, "is_recoverable", False)),
        "code": str(getattr(error, "code", "")),
        "runtime_base": isinstance(error, RuntimeError),
        "os_base": isinstance(error, OSError),
        "value_base": isinstance(error, ValueError),
        "rlmesh_base": isinstance(error, rlmesh.RLMeshException),
        "environment_base": isinstance(error, rlmesh.EnvironmentException),
    }


def execute(case):
    """Execute one fixture with fresh handles and return observed values."""
    env, policy = ScriptedEnv(), ScriptedPolicy()
    model = rlmesh.Model(
        policy, on_episode_end=policy.on_episode_end, on_close=policy.close
    )
    operation = case["operation"]
    try:
        if operation == "dict_error":
            env.action_space = spaces.Dict(
                {"path": spaces.Dict({"k": spaces.Discrete(3)})}
            )
            with rlmesh.EnvServer(env, address=":0/env") as server:
                server.start()
                with rlmesh.RemoteEnv(address=server.address) as remote:
                    remote.reset(seed=0)
                    remote.step({"path": {}})
                    raise AssertionError("missing Dict key accepted")
        if operation == "reset_order":
            with model.bind(env) as binding:
                observation, info = binding.reset(seed=37, options={"trial_index": 11})
                return dict(
                    seed=env.seed,
                    trial_index=env.trial,
                    observation=observation,
                    info_empty=not info,
                )
        if operation == "config_error":

            def load(self, checkpoint):
                raise AssertionError("invalid config reached load")

            load.__qualname__ = "load"
            keyed = type(
                "Keyed",
                (rlmesh.Model,),
                {"load": load, "predict": lambda self, observation: 0},
            )
            config = (
                {} if case["variant"] == "missing" else {"checkpoint": 1, "scripted": 2}
            )
            keyed.from_config(**config)
            raise AssertionError("invalid config accepted")
        if operation == "validation":
            variant = case["variant"]
            if variant.startswith("horizon"):
                model.run(
                    env, execution_horizon=0 if variant == "horizon_zero" else 1025
                )
            else:
                rlmesh.View(
                    **{
                        variant.removeprefix("view_"): 0.0
                        if variant == "view_step_hz"
                        else 0
                    }
                )
            raise AssertionError("invalid argument accepted")
        if operation == "run":
            opts = {
                key: case[key]
                for key in ("episodes", "seeds", "close_env")
                if key in case
            }
            result = model.run(env, **opts)
            observed = {
                "episodes": result.num_episodes,
                "steps": result.total_steps,
                "reward": result.mean_reward,
                "env_closes": env.closes,
                "model_closes": policy.closes,
            }
            # Reuse the same model, including after close_env=true.
            model.run(ScriptedEnv())
            observed["reusable"] = True
            return observed
        if operation == "warning_error":
            script = pathlib.Path(__file__).resolve()
            code = f"import runpy; d=runpy.run_path({str(script)!r}); d['check_warning_error']()"
            subprocess.run(
                [sys.executable, "-W", "error", "-c", code], check=True, timeout=30
            )
            return {"error": "UserWarning", "caller_frame": True}
        if operation == "warning":
            caller = threading.get_ident()
            seen = []
            threads = []
            original = warnings.showwarning

            def handler(message, category, filename, lineno, file=None, line=None):
                threads.append(threading.get_ident())
                seen.append(category.__name__)
                if case["variant"] in ("fail", "runtime_fail", "recorder"):
                    raise TypeError("warning abort")

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    warnings.showwarning = handler
                    try:
                        variant = case["variant"]
                        if variant == "workflow":
                            os.environ.pop("RLMESH_WORKFLOW_EDITION", None)
                            from rlmesh import _editions

                            _editions._warned = False

                            class Authored(rlmesh.Model):
                                def predict(self, observation):
                                    return 0

                            authored = Authored()
                            try:
                                authored.run(env)
                            finally:
                                authored.close()
                        elif variant.startswith("runtime"):

                            class Chunk:
                                def predict(self, observation):
                                    return 0

                                def predict_chunk(self, observation, execution_horizon):
                                    return [0]

                            chunk = rlmesh.Model(Chunk())
                            try:
                                chunk.run(env, execution_horizon=2)
                            finally:
                                chunk.close()
                        elif variant == "recorder":
                            with rlmesh.Recorder() as recorder:
                                capture = recorder.capture(
                                    model="model", env="env", cameras=["absent"]
                                )
                                model.run(env, hooks=capture)
                        else:
                            model.run(env, execution_horizon=2)
                        observed = {}
                    except TypeError as error:
                        observed = error_record(error)
                observed.update(
                    category=seen[-1],
                    calling_thread=all(t == caller for t in threads),
                    warnings=len(seen),
                )
                return observed
            finally:
                warnings.showwarning = original
        binding = model.bind(env)
        try:
            if operation != "hook_error":
                binding.reset(seed=0)
            if operation == "borrow":
                binding.close()
                observed = {
                    "env_closes": env.closes,
                    "model_closes": policy.closes,
                    "episode_ends": policy.ends,
                }
                model.run(env)
                observed["reusable"] = True
                return observed
            if operation == "hook_error":

                class FailHook(rlmesh.RunHooks):
                    ends = 0

                    def on_step(self, event):
                        raise TypeError("hook abort")

                    def on_run_end(self, result):
                        self.ends += 1

                hooks = FailHook()
                binding.close()
                try:
                    model.run(env, hooks=hooks)
                except TypeError as error:
                    observed = error_record(error)
                    observed["run_ends"] = hooks.ends
                    return observed
                raise AssertionError("hook succeeded")
            if operation == "error":
                variant = case["variant"]
                if variant in ("ParamError", "MissingParamError", "UnknownParamError"):
                    from rlmesh import params

                    cls = getattr(params, variant)
                else:
                    cls = (
                        rlmesh.ModelException
                        if variant == "ModelRecoverable"
                        else AdapterResolutionError
                        if variant == "AdapterResolutionError"
                        else (
                            getattr(rlmesh, variant, None) or getattr(builtins, variant)
                        )
                    )
                policy.failure = cls(
                    "authored timeout: checkpoint shard violet"
                    if variant == "TimeoutError"
                    else "scripted failure"
                )
                if variant == "ModelRecoverable":
                    policy.failure = cls("scripted failure", is_recoverable=True)
                binding.step()
                raise AssertionError("scripted error succeeded")
            raise AssertionError(operation)
        finally:
            binding.close()
    except BaseException as error:
        if isinstance(error, (AssertionError, SystemExit)):
            raise
        return error_record(error)
    finally:
        model.close()


def check_warning_error():
    """Exercise warning frame attribution and the actual -W error interpreter."""
    model = rlmesh.Model(ScriptedPolicy())
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.run(ScriptedEnv(), execution_horizon=2)
        user = next(
            warning
            for warning in caught
            if warning.category is UserWarning
            and str(warning.message).startswith("execution_horizon=")
        )
        assert pathlib.Path(user.filename).resolve() == pathlib.Path(__file__).resolve()
        try:
            model.run(ScriptedEnv(), execution_horizon=2)
        except UserWarning:
            return
        raise AssertionError("-W error did not raise")
    finally:
        model.close()
