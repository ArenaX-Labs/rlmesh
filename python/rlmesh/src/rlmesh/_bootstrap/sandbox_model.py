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

from .env import (
    BootstrapUsageError,
    apply_member_params,
    expect_mapping,
    member_params_from_env,
    resolve_bootstrap_spec,
)


def _env_alias(*names: str) -> str | None:
    """First non-empty value among ``names`` (canonical name first, aliases after)."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


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
        setup_env, kwargs = member_params_from_env()
        if kwargs:
            print(
                "bootstrap failed: model recipes do not accept member-selector kwargs; "
                "select a variant via setup.env (member params) or a per-variant recipe",
                file=sys.stderr,
            )
            return 2
        recipe = apply_member_params(recipe, setup_env=setup_env)
        apply_setup(recipe.setup)
        load = resolve_entrypoint(make.entrypoint, label="model bootstrap entrypoint")
        from rlmesh.recipes.authoring.model import (
            construct_authored_model,
            is_model_recipe,
        )

        model_cls = getattr(load, "__self__", None)
        if is_model_recipe(model_cls):
            policy = construct_authored_model(
                model_cls, in_container=True, artifacts=recipe.inputs
            )
        else:
            policy = load()
        server = Model(policy)

        # De-overloaded: RLMESH_DRIVE_ENV_ADDRESS triggers drive-mode (keep the
        # legacy RLMESH_ENV_ADDRESS as an alias); RLMESH_ADDRESS only ever means serve.
        drive_address = _env_alias("RLMESH_DRIVE_ENV_ADDRESS", "RLMESH_ENV_ADDRESS")
        if drive_address:
            from rlmesh.numpy import RemoteEnv

            # "" is an explicit empty seeds=() (0 episodes); only an absent var defaults.
            raw_seeds = os.environ.get("RLMESH_SEEDS")
            seeds = (
                [0, 1]
                if raw_seeds is None
                else [int(s) for s in raw_seeds.split(",") if s.strip()]
            )
            max_episodes = os.environ.get("RLMESH_MAX_EPISODES")
            result = server.run(
                RemoteEnv(drive_address),
                seeds=seeds,
                max_episodes=int(max_episodes) if max_episodes else None,
            )
            # Emit each episode so the host reconstructs a full RunResult (per-episode,
            # success_rate), not just aggregates.
            print(
                "RLMESH_RUN_RESULT "
                + json.dumps(
                    {
                        "episodes": [
                            {
                                "index": e.index,
                                "seed": e.seed,
                                "steps": e.steps,
                                "reward": e.reward,
                                "terminated": e.terminated,
                                "truncated": e.truncated,
                            }
                            for e in result.episodes
                        ]
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
        server.serve(address, token=token, options=None)
        return 0
    except Exception as exc:  # pragma: no cover - exercised through container runs
        print(f"bootstrap failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
