"""Shared helpers for Unreal primitive gym demos.

Address / dotenv use the shared identity under ``integrations/unreal/.env``.
Learned-agent helpers wrap ``rlmesh.numpy.RemoteEnv`` with Stable-Baselines3 PPO.

Press Play in Unreal with the matching env component Serving, then pass the
Serving address (``--address`` / ``RLMESH_UNREAL_ADDR``).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

_COMMON_DIR = Path(__file__).resolve().parent
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))

from dotenv import apply_dotenv
from paths import (
    DOTENV_PATH,
    EPISODES_VAR,
    UNREAL_ADDR_VAR,
    UNREAL_MAX_FPS_VAR,
)

# Soft defaults for live Unreal PIE (~1 step/frame unless ServingMaxFPS is raised).
DEFAULT_TRAIN_TIMESTEPS = 512
DEFAULT_PPO_N_STEPS = 128
DEFAULT_EVAL_EPISODES = 1


def resolve_serving_max_fps(cli_fps: float | None = None) -> float:
    """FPS hint for Unreal lockstep (``RLMESH_UNREAL_MAX_FPS`` or CLI). 0 = unchanged."""
    if cli_fps is not None and cli_fps > 0:
        return float(cli_fps)
    raw = os.environ.get(UNREAL_MAX_FPS_VAR, "").strip()
    if not raw:
        return 0.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def reset_options_with_fps(
    options: dict[str, Any] | None = None,
    *,
    serving_max_fps: float | None = None,
) -> dict[str, Any]:
    """Merge ``serving_max_fps`` into gym/rlmesh reset options when set."""
    out = dict(options or {})
    fps = resolve_serving_max_fps(serving_max_fps)
    if fps > 0:
        out.setdefault("serving_max_fps", fps)
    return out


def ensure_artifacts_dir(artifacts_dir: Path) -> Path:
    """Ensure ``artifacts_dir`` exists and return it."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


def default_model_path(artifacts_dir: Path, *, filename: str = "ppo.zip") -> Path:
    return ensure_artifacts_dir(artifacts_dir) / filename


def load_suite_dotenv() -> dict[str, str]:
    """Apply ``integrations/unreal/.env`` without overwriting existing env vars."""
    return apply_dotenv(DOTENV_PATH)


def resolve_address(cli_address: str | None = None) -> str:
    """Resolve Serving address from CLI override or ``RLMESH_UNREAL_ADDR``."""
    load_suite_dotenv()
    if cli_address and cli_address.strip():
        return cli_address.strip()
    return os.environ.get(UNREAL_ADDR_VAR, "").strip()


def require_address(cli_address: str | None = None) -> str:
    """Like :func:`resolve_address`, but raise if empty."""
    addr = resolve_address(cli_address)
    if not addr:
        raise ValueError(
            "No Serving address. Add an RLMesh env component in Unreal, Press Play, "
            "then pass --address / RLMESH_UNREAL_ADDR (see LogRLMesh Serving …), "
            "or run integrations/unreal/demos/reach_target/app.py --init-identity "
            "once for a stable .env invite."
        )
    return addr


def make_remote_env(
    address: str | None = None,
    *,
    serving_max_fps: float | None = None,
    **remote_kwargs: Any,
):
    """Connect to a serving Unreal env as ``rlmesh.numpy.RemoteEnv``.

    Press Play with the matching component first. When gymnasium is installed,
    the client is a ``gymnasium.Env``.

    ``serving_max_fps`` (or ``RLMESH_UNREAL_MAX_FPS``) is sent on each ``reset`` as
    option ``serving_max_fps`` so the Unreal plugin can raise MaxFPS for faster
    lockstep. The component Details default ``Serving Max FPS`` also applies on Play.
    """
    addr = require_address(address)
    from rlmesh.numpy import RemoteEnv

    env = RemoteEnv(addr, **remote_kwargs)
    fps = resolve_serving_max_fps(serving_max_fps)
    if fps <= 0:
        return env

    inner_reset = env.reset

    def reset_with_fps(*args: Any, **kwargs: Any):
        # gymnasium reset(*, seed=None, options=None) — options are usually kw-only.
        kwargs["options"] = reset_options_with_fps(
            kwargs.get("options") if isinstance(kwargs.get("options"), dict) else None,
            serving_max_fps=fps,
        )
        return inner_reset(*args, **kwargs)

    env.reset = reset_with_fps  # type: ignore[method-assign]
    return env


def smoke_remote_env(
    address: str | None = None,
    **remote_kwargs: Any,
) -> int:
    """Connect, ``reset``, one ``step``, ``close``. Returns process exit code."""
    try:
        env = make_remote_env(address, **remote_kwargs)
    except Exception as exc:  # noqa: BLE001 - CLI surface
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    try:
        obs, info = env.reset()
        print(f"reset obs={obs} info_keys={sorted(info)}")
        print(f"observation_space={env.observation_space}")
        print(f"action_space={env.action_space}")
        action = env.action_space.sample()
        step = env.step(action)
        print(f"step action={action} -> len={len(step)}")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: smoke failed: {exc}", file=sys.stderr)
        return 1
    finally:
        env.close()
    print("gym smoke finished successfully.")
    return 0


def run_scripted(
    predict: Callable[[object], object],
    *,
    address: str | None = None,
    episodes: int | None = None,
    seeds: str = "",
    model_label: str = "scripted",
) -> int:
    """Drive a serving env with ``rlmesh.run`` and a scripted predict callable."""
    try:
        import rlmesh
        import rlmesh.numpy
    except ImportError:
        print(
            "ERROR: rlmesh is not installed in this Python.\n"
            "From the rlmesh checkout: mise run setup:python\n"
            f"Current interpreter: {sys.executable}",
            file=sys.stderr,
        )
        return 1

    try:
        addr = require_address(address)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if "#" not in addr:
        print(
            "WARNING: address has no #keys; loopback hosts usually require the full invite.",
            file=sys.stderr,
        )

    if episodes is None:
        episodes = int(os.environ.get(EPISODES_VAR, "1") or "1")

    run_kwargs: dict[str, Any] = {"episodes": episodes}
    if seeds.strip():
        parsed = [int(part.strip()) for part in seeds.split(",") if part.strip()]
        if not parsed:
            print("ERROR: --seeds was set but no integers were parsed.", file=sys.stderr)
            return 1
        run_kwargs = {"seeds": parsed}

    model = rlmesh.numpy.Model(predict)
    print(f"Connecting to {addr} …")
    print(f"Model: {model_label}  options: {run_kwargs}")
    try:
        result = rlmesh.run(model, addr, **run_kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: run failed: {exc}", file=sys.stderr)
        print(
            "Checks:\n"
            "  - Unreal editor is open and LogRLMesh shows Serving\n"
            "  - the matching RLMesh env component is on a playing actor\n"
            "  - LogRLMesh shows Loaded RLMESH_HOST_IDENTITY (rebuild plugin if missing)\n"
            "  - RLMESH_UNREAL_ADDR in integrations/unreal/.env matches the Serving #keys\n"
            "  - plugin rlmesh_capi matches this Python's workflow edition "
            "(RLMESH_RELEASE_BUILD=1 on Windows)",
            file=sys.stderr,
        )
        return 1

    print(result)
    print("scripted run finished successfully.")
    return 0


def _require_sb3():
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover - optional suite dep
        raise ImportError(
            "stable-baselines3 is required for learned agents.\n"
            "Install suite deps: pip install -r "
            "integrations/unreal/demos/common/requirements-gym-suite.txt"
        ) from exc
    return PPO


def train_ppo(
    *,
    address: str | None = None,
    artifacts_dir: Path | None = None,
    total_timesteps: int = DEFAULT_TRAIN_TIMESTEPS,
    model_path: Path | str | None = None,
    seed: int | None = 0,
    verbose: int = 1,
    n_steps: int = DEFAULT_PPO_N_STEPS,
    **ppo_kwargs: Any,
) -> Path:
    """Train SB3 PPO on a live Unreal ``RemoteEnv``; save under ``artifacts_dir``."""
    PPO = _require_sb3()
    if model_path is not None:
        out = Path(model_path)
    elif artifacts_dir is not None:
        out = default_model_path(artifacts_dir)
    else:
        raise ValueError("pass model_path= or artifacts_dir=")
    out.parent.mkdir(parents=True, exist_ok=True)

    env = make_remote_env(address)
    save_stem = str(out.with_suffix(""))
    # Keep n_steps small so PIE smoke trains log/update before multi-minute waits.
    ppo_kwargs.setdefault("n_steps", n_steps)
    ppo_kwargs.setdefault("batch_size", min(64, n_steps))
    try:
        model = PPO("MlpPolicy", env, verbose=verbose, seed=seed, **ppo_kwargs)
        model.learn(total_timesteps=total_timesteps)
        # SB3 writes ``<stem>.zip``.
        model.save(save_stem)
    except AttributeError as exc:
        # SB3/gymnasium bridge may touch gym.__version__ during teardown after save.
        if "__version__" not in str(exc):
            raise
        print(f"warning: ignoring gym version quirk after train: {exc}", file=sys.stderr)
    finally:
        try:
            env.close()
        except Exception as close_exc:  # noqa: BLE001 — SB3/gym version quirks on close
            print(f"warning: env.close() after train: {close_exc}", file=sys.stderr)

    saved = Path(save_stem + ".zip")
    if not saved.is_file():
        raise FileNotFoundError(f"PPO save missing at {saved}")
    print(f"saved {saved}")
    return saved


def eval_ppo(
    *,
    address: str | None = None,
    artifacts_dir: Path | None = None,
    model_path: Path | str | None = None,
    episodes: int = DEFAULT_EVAL_EPISODES,
    deterministic: bool = True,
    seed: int | None = 0,
) -> list[float]:
    """Run a saved PPO zip for ``episodes``; return per-episode returns."""
    PPO = _require_sb3()
    if model_path is not None:
        path = Path(model_path)
    elif artifacts_dir is not None:
        path = default_model_path(artifacts_dir)
    else:
        raise ValueError("pass model_path= or artifacts_dir=")
    if not path.is_file():
        # Allow stem without .zip
        alt = path if path.suffix == ".zip" else Path(str(path) + ".zip")
        if alt.is_file():
            path = alt
        else:
            raise FileNotFoundError(f"no saved model at {path}")

    load_stem = str(path.with_suffix("")) if path.suffix == ".zip" else str(path)
    model = PPO.load(load_stem)

    env = make_remote_env(address)
    returns: list[float] = []
    try:
        for ep in range(episodes):
            obs, _info = env.reset(seed=None if seed is None else seed + ep)
            done = False
            ep_ret = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _info = env.step(action)
                ep_ret += float(reward)
                done = bool(terminated or truncated)
            returns.append(ep_ret)
            print(f"episode {ep + 1}/{episodes} return={ep_ret:.4f}")
    finally:
        env.close()

    if returns:
        mean = sum(returns) / len(returns)
        print(f"mean return over {len(returns)} episode(s): {mean:.4f}")
    return returns
