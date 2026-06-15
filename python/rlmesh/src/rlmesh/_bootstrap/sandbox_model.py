"""Sandbox model bootstrap entrypoint.

Mirror of ``sandbox_env``: reconstructs the policy from the recipe (the
``module:Class._rlmesh_load`` entrypoint, run in-container so weight mounts resolve
to their target paths) and either serves it as a model endpoint on ``RLMESH_ADDRESS``
or, when ``RLMESH_DRIVE_ENV_ADDRESS`` is set, drives it against that env and reports
the run.
"""

from __future__ import annotations

import json
import os
import sys
from typing import TYPE_CHECKING

from .env import (
    BootstrapUsageError,
    apply_member_params,
    expect_mapping,
    member_params_from_env,
    resolve_bootstrap_spec,
)

if TYPE_CHECKING:
    from rlmesh import ServeOptions


def _env_alias(*names: str) -> str | None:
    """First non-empty value among ``names`` (canonical name first, aliases after)."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _env_flag(*names: str) -> bool:
    value = _env_alias(*names)
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(*names: str) -> float | None:
    value = _env_alias(*names)
    return float(value) if value else None


def _serve_options() -> ServeOptions | None:
    """Build ``ServeOptions`` from the lifecycle env vars; ``None`` when all default.

    A token and a clean remote shutdown are legitimate for any gRPC server, so these
    stay deployment knobs (flat env, default off) rather than recipe properties.
    """
    allow_remote_shutdown = _env_flag(
        "RLMESH_ALLOW_REMOTE_SHUTDOWN",
        "RLMESH_MODEL_ALLOW_REMOTE_SHUTDOWN",
        "RLMESH_MODEL_WAIT_FOR_TERMINATION",
    )
    idle = _env_float(
        "RLMESH_IDLE_SHUTDOWN_SECONDS", "RLMESH_MODEL_IDLE_SHUTDOWN_SECONDS"
    )
    drain = _env_float("RLMESH_WAIT_FOR_TERMINATION")
    if not allow_remote_shutdown and idle is None and drain is None:
        return None
    from rlmesh import ServeOptions

    return ServeOptions(
        allow_remote_shutdown=allow_remote_shutdown,
        idle_timeout_seconds=idle,
        drain_timeout_seconds=drain,
    )


def main(
    argv: list[str] | None = None,
    *,
    prog: str = "python -m rlmesh._bootstrap.sandbox_model",
) -> int:
    argv = sys.argv[1:] if argv is None else argv

    try:
        spec = resolve_bootstrap_spec(argv, prog=prog)
    except BootstrapUsageError as exc:
        print(exc, file=sys.stderr)
        return 2
    document = expect_mapping(spec.get("document"), "bootstrap spec.document")

    try:
        from rlmesh._entrypoint import resolve_entrypoint
        from rlmesh.numpy import Model
        from rlmesh.recipes import Recipe
        from rlmesh.recipes._construct import apply_setup
        from rlmesh.recipes._schema import PyMake

        recipe = Recipe.from_dict(document)
        make = recipe.make
        if not isinstance(make, PyMake):
            print(
                "bootstrap failed: model recipe has no PyMake entrypoint",
                file=sys.stderr,
            )
            return 2
        # Construct-time config: apply setup.env (with declared RLMESH_PARAMS_JSON
        # overrides) to os.environ before load(), which the policy reads. The member
        # selector touches only setup.env / make.kwargs, so the entrypoint is stable.
        setup_env, kwargs = member_params_from_env()
        recipe = apply_member_params(recipe, setup_env=setup_env, kwargs=kwargs)
        apply_setup(recipe.setup)
        load = resolve_entrypoint(make.entrypoint, label="model bootstrap entrypoint")
        policy = load()
        server = Model(policy)

        # De-overloaded: RLMESH_DRIVE_ENV_ADDRESS triggers drive-mode (keep the
        # legacy RLMESH_ENV_ADDRESS as an alias); RLMESH_ADDRESS only ever means serve.
        drive_address = _env_alias("RLMESH_DRIVE_ENV_ADDRESS", "RLMESH_ENV_ADDRESS")
        if drive_address:
            from rlmesh.numpy import RemoteEnv

            # `os.environ.get(key, default)` returns "" (not the default) when the
            # var is set but empty, and int("") raises; `or` falls back cleanly. The
            # trailing `or [0, 1]` keeps an all-separator value (",") from yielding
            # zero seeds, which would silently run zero episodes.
            raw_seeds = os.environ.get("RLMESH_SEEDS") or "0,1"
            seeds = [int(s) for s in raw_seeds.split(",") if s.strip()] or [0, 1]
            result = server.run(RemoteEnv(drive_address), seeds=seeds)
            print(
                "RLMESH_RUN_RESULT "
                + json.dumps(
                    {
                        "episodes": result.num_episodes,
                        "steps": result.total_steps,
                        "mean_reward": result.mean_reward,
                    }
                ),
                flush=True,
            )
            return 0

        address = _env_alias(
            "RLMESH_ADDRESS", "RLMESH_MODEL_ADDRESS", "RLMESH_MODEL_ENDPOINT_ADDRESS"
        )
        if not address:
            port = (
                _env_alias("RLMESH_PORT", "RLMESH_MODEL_PORT", "RLMESH_ENV_PORT")
                or "50051"
            )
            address = f"0.0.0.0:{port}"
        token = _env_alias("RLMESH_TOKEN", "RLMESH_MODEL_ENDPOINT_TOKEN") or ""
        print(f"RLMesh sandbox serving {address}", flush=True)
        server.serve(address, token=token, options=_serve_options())
        return 0
    except Exception as exc:  # pragma: no cover - exercised through container runs
        print(f"bootstrap failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
