"""Templated container entrypoint.

``python -m rlmesh.serve my_pkg:Policy`` serves a model; ``--env my_pkg:Env``
serves an environment. The target may be a ``Model`` subclass, an :class:`EnvFactory`,
or a bare predict / make-env callable. Serves on ``RLMESH_ADDRESS``
(default ``0.0.0.0:50051``); point your Dockerfile ``ENTRYPOINT`` here instead of
hand-writing a serve loop.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

from ._entrypoint import resolve_entrypoint

if TYPE_CHECKING:
    from rlmesh._models.base import ModelBase
    from rlmesh.types import EnvLike

__all__ = ["main", "serve_env", "serve_model"]


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
        env = resolve_entrypoint(args.env, label="env entrypoint")
        print(f"RLMesh serving env {args.env} on {args.address}", flush=True)
        serve_env(env, args.address)
    else:
        model = resolve_entrypoint(args.model, label="model entrypoint")
        print(f"RLMesh serving model {args.model} on {args.address}", flush=True)
        serve_model(model, args.address, token=args.token)
    return 0


def serve_model(model_source: object, address: str, *, token: str = "") -> None:
    """Host a model on ``address`` (blocking).

    Resolves the source to a serveable ``Model``: a ``Model`` subclass class is
    instantiated, an existing ``Model`` instance is used directly, and a bare predict
    callable (or duck-typed policy object) is wrapped in a ``Model`` -- either way, no
    hand-written request builder. Heavy imports stay inside this call so importing the
    authoring base stays cheap.
    """
    _resolve_model(model_source).serve(address, token=token)


def _resolve_model(model_source: object) -> ModelBase[Any, Any]:
    """Resolve a model source to a serveable ``Model`` without double-construction.

    A ``Model`` subclass *class* is instantiated once (its ``load()`` runs); an
    existing ``Model`` instance is used as-is (it already built its worker); anything
    else (a bare predict callable or a duck-typed policy object) is wrapped in a
    framework ``Model``.
    """
    from rlmesh._models.base import ModelBase
    from rlmesh.numpy import Model

    if isinstance(model_source, ModelBase):
        return cast("ModelBase[Any, Any]", model_source)
    if isinstance(model_source, type) and issubclass(model_source, ModelBase):
        return cast("ModelBase[Any, Any]", model_source())
    return Model(cast(object, model_source))


def serve_env(env_source: object, address: str, **make_kwargs: object) -> None:
    """Host an environment on ``address`` (blocking).

    An :class:`EnvFactory` class/instance is constructed via ``prepare()`` +
    ``make(**make_kwargs)`` and its ``tags`` published; a bare make-env callable is
    invoked to produce the env. Heavy imports stay inside this call so importing the
    authoring base stays cheap.
    """
    from rlmesh import EnvServer

    from ._bootstrap.loaders import construct_authored_env

    if hasattr(env_source, "make"):  # EnvFactory class or instance
        env = construct_authored_env(env_source, **make_kwargs)
        EnvServer(env, address, tags=getattr(env_source, "tags", None)).serve()
    else:  # bare make-env callable
        make_env = cast("Callable[..., EnvLike[Any, Any]]", env_source)
        EnvServer(make_env(**make_kwargs), address).serve()


if __name__ == "__main__":
    raise SystemExit(main())
