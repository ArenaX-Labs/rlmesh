"""Sandbox model bootstrap entrypoint.

Mirror of ``sandbox_env``: reconstructs the policy from the recipe in the bootstrap
payload (the ``module:Class._rlmesh_load`` entrypoint, run in-container so weight
mounts resolve to their target paths) and either serves it as a model endpoint or,
when ``RLMESH_ENV_ADDRESS`` is set, drives it against that env and reports the run.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import cast

from .env import expect_mapping


def main(
    argv: list[str] | None = None,
    *,
    prog: str = "python -m rlmesh._bootstrap.sandbox_model",
) -> int:
    argv = sys.argv[1:] if argv is None else argv

    inline = os.environ.get("RLMESH_BOOTSTRAP_JSON")
    if inline is not None:
        if argv:
            print(f"usage: {prog} (set RLMESH_BOOTSTRAP_JSON, no arguments)", file=sys.stderr)
            return 2
        raw = inline
    else:
        if len(argv) != 1:
            print(f"usage: {prog} <bootstrap.json> (or set RLMESH_BOOTSTRAP_JSON)", file=sys.stderr)
            return 2
        raw = Path(argv[0]).read_text(encoding="utf-8")

    payload = expect_mapping(cast(object, json.loads(raw)), "bootstrap payload")
    spec = expect_mapping(payload.get("spec"), "bootstrap spec")
    document = expect_mapping(spec.get("document"), "bootstrap spec.document")

    try:
        from rlmesh._bootstrap.entrypoint import resolve_entrypoint
        from rlmesh.numpy import Model
        from rlmesh.recipes import Recipe
        from rlmesh.recipes._schema import PyMake

        recipe = Recipe.from_dict(document)
        if not isinstance(recipe.make, PyMake):
            print("bootstrap failed: model recipe has no PyMake entrypoint", file=sys.stderr)
            return 2
        load = resolve_entrypoint(recipe.make.entrypoint, label="model bootstrap entrypoint")
        policy = load()
        server = Model(policy)

        env_address = os.environ.get("RLMESH_ENV_ADDRESS")
        if env_address:
            from rlmesh.numpy import RemoteEnv

            seeds = [int(s) for s in os.environ.get("RLMESH_SEEDS", "0,1").split(",")]
            result = server.run(RemoteEnv(env_address), seeds=seeds)
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

        address = os.environ.get("RLMESH_MODEL_ADDRESS")
        if not address:
            port = os.environ.get("RLMESH_MODEL_PORT") or os.environ.get("RLMESH_ENV_PORT", "50051")
            address = f"0.0.0.0:{port}"
        print(f"RLMesh sandbox serving {address}", flush=True)
        server.serve(address)
        return 0
    except Exception as exc:  # pragma: no cover - exercised through container runs
        print(f"bootstrap failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
