"""Templated container entrypoint.

``python -m rlmesh.serve my_pkg:Policy`` serves a model; ``--env my_pkg:Env``
serves an environment. The target may be a :class:`ModelRecipe`/:class:`EnvRecipe`
subclass or a bare predict / make-env callable. Serves on ``RLMESH_ADDRESS``
(default ``0.0.0.0:50051``); point your Dockerfile ``ENTRYPOINT`` here instead of
hand-writing a serve loop.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from ._entrypoint import resolve_entrypoint

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``[model | --env]`` and serve it on ``--address``/``RLMESH_ADDRESS``."""
    parser = argparse.ArgumentParser(prog="python -m rlmesh.serve")
    parser.add_argument("model", nargs="?", help="module:Class for a model/policy")
    parser.add_argument("--env", help="module:Class for an environment")
    parser.add_argument(
        "--address", default=os.environ.get("RLMESH_ADDRESS", "0.0.0.0:50051")
    )
    parser.add_argument("--token", default="")
    args = parser.parse_args(argv)

    if bool(args.model) == bool(args.env):
        parser.error("provide exactly one of a model entrypoint or --env")

    if args.env:
        return _serve_env(args.env, args.address)
    return _serve_model(args.model, args.address, args.token)


def _serve_model(entrypoint: str, address: str, token: str) -> int:
    from rlmesh.numpy import Model

    obj = resolve_entrypoint(entrypoint, label="model entrypoint")
    print(f"RLMesh serving model {entrypoint} on {address}", flush=True)
    # Model(...) routes a ModelRecipe class through coerce_model; a bare callable
    # serves directly. Either way, no hand-written request builder.
    Model(obj).serve(address, token=token)
    return 0


def _serve_env(entrypoint: str, address: str) -> int:
    from rlmesh import EnvServer

    from ._bootstrap.loaders import construct_authored_env

    obj = resolve_entrypoint(entrypoint, label="env entrypoint")
    print(f"RLMesh serving env {entrypoint} on {address}", flush=True)
    if hasattr(obj, "make"):  # EnvRecipe class or instance
        env = construct_authored_env(obj)
        EnvServer(env, address, tags=getattr(obj, "tags", None)).serve()
    else:  # bare make-env callable
        EnvServer(obj(), address).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
