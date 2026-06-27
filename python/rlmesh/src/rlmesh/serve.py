"""Templated container entrypoint.

``python -m rlmesh.serve my_pkg:Policy`` serves a model; ``--env my_pkg:Env``
serves an environment. The target may be a ``Model`` subclass, an :class:`EnvFactory`,
or a bare predict / make-env callable. Serves on ``RLMESH_ADDRESS``
(default ``0.0.0.0:50051``); point your Dockerfile ``ENTRYPOINT`` here instead of
hand-writing a serve loop.
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument(
        "--kwargs-json",
        type=_json_object,
        help=(
            "JSON object bound to the env factory's make(**binding) -- the env "
            "variation to serve. Defaults to RLMESH_MAKE_KWARGS; absent serves "
            "make()'s defaults. Validated against the factory's declared params "
            "before construction. Env only."
        ),
    )
    args = parser.parse_args(argv)

    if bool(args.model) == bool(args.env):
        parser.error("provide exactly one of a model entrypoint or --env")

    # Eval shape from the container env (start_prebuilt_container injects these for a
    # SandboxVectorEnv): honor RLMESH_NUM_ENVS by fanning the factory out into a
    # vector env, so a prebuilt EnvFactory image serves the requested lanes.
    num_envs = 1
    raw_num_envs = os.environ.get("RLMESH_NUM_ENVS")
    if raw_num_envs:
        try:
            num_envs = int(raw_num_envs)
        except ValueError:
            parser.error(f"RLMESH_NUM_ENVS must be an integer, got {raw_num_envs!r}")
    vectorization_mode = os.environ.get("RLMESH_VECTORIZATION_MODE") or None

    binding = args.kwargs_json
    if binding is None:
        raw = os.environ.get("RLMESH_MAKE_KWARGS")
        if raw:
            try:
                binding = _json_object(raw)
            except argparse.ArgumentTypeError as exc:
                parser.error(f"RLMESH_MAKE_KWARGS: {exc}")
        else:
            binding = {}

    if args.env:
        # num_envs / vectorization_mode are vectorization controls (their own env
        # vars), not env make() kwargs; a binding key of either name would otherwise
        # collide with serve_env's explicit args as an opaque "multiple values"
        # TypeError. Point the operator at the right knob instead.
        control_collisions = {"num_envs", "vectorization_mode"} & binding.keys()
        if control_collisions:
            parser.error(
                f"{', '.join(sorted(control_collisions))} control vectorization, not "
                "env construction; set RLMESH_NUM_ENVS / RLMESH_VECTORIZATION_MODE "
                "instead of passing them in RLMESH_MAKE_KWARGS / --kwargs-json"
            )
        env = resolve_entrypoint(args.env, label="env entrypoint")
        print(f"RLMesh serving env {args.env} on {args.address}", flush=True)
        serve_env(
            env,
            args.address,
            num_envs=num_envs,
            vectorization_mode=vectorization_mode,
            **binding,
        )
    else:
        model = resolve_entrypoint(args.model, label="model entrypoint")
        print(f"RLMesh serving model {args.model} on {args.address}", flush=True)
        serve_model(model, args.address, token=args.token, binding=binding)
    return 0


def _json_object(value: str) -> dict[str, Any]:
    """Argparse type: parse a JSON object, rejecting non-objects."""
    try:
        parsed = cast("object", json.loads(value))
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return cast("dict[str, Any]", parsed)


def serve_model(
    model_source: object,
    address: str,
    *,
    token: str = "",
    binding: dict[str, Any] | None = None,
) -> None:
    """Host a model on ``address`` (blocking).

    Resolves the source to a serveable ``Model``: a ``Model`` subclass class is
    instantiated, an existing ``Model`` instance is used directly, and a bare predict
    callable (or duck-typed policy object) is wrapped in a ``Model`` -- either way, no
    hand-written request builder. A non-empty ``binding`` validates against the
    model's declared ``params`` and is applied to ``load(**binding)`` on the
    bootstrap-authoritative path. Heavy imports stay inside this call so importing
    the authoring base stays cheap.
    """
    _resolve_model(model_source, binding).serve(address, token=token)


def _resolve_model(
    model_source: object, binding: dict[str, Any] | None = None
) -> ModelBase[Any, Any]:
    """Resolve a model source to a serveable ``Model`` without double-construction.

    A ``Model`` subclass *class* is instantiated once (its ``load()`` runs); an
    existing ``Model`` instance is used as-is (it already built its worker); anything
    else (a bare predict callable or a duck-typed policy object) is wrapped in a
    framework ``Model``. Delegates to the shared :func:`rlmesh._models.base.as_model`
    normalizer so the serve path and the run path agree on what a model source is.

    With a non-empty ``binding`` the model is built on the bootstrap-authoritative
    path: the eager auto-load is suppressed, the binding is resolved against the
    declared ``params``, and ``load(**binding)`` runs once before serving. Bindings
    require a :class:`rlmesh.Model` subclass *class* entrypoint -- the documented
    authoring path -- so there is no double-construction of a wrapped policy.
    """
    from rlmesh._models.base import ModelBase, as_model

    if isinstance(model_source, type) and issubclass(model_source, ModelBase):
        # Always resolve a Model subclass through the authored path -- even with no
        # binding -- so declared required params are enforced before weights load,
        # matching the env path (construct_authored_env always resolves).
        from ._bootstrap.loaders import construct_authored_model

        resolved_binding: dict[str, Any] = binding or {}
        return cast(
            "ModelBase[Any, Any]",
            construct_authored_model(
                cast("type[Any]", model_source), **resolved_binding
            ),
        )
    if binding:
        raise TypeError(
            "model construction params (--kwargs-json / RLMESH_MAKE_KWARGS) "
            "require a rlmesh.Model subclass entrypoint (module:Class)"
        )
    return as_model(cast("object", model_source))


def serve_env(
    env_source: object,
    address: str,
    /,
    *,
    num_envs: int = 1,
    vectorization_mode: str | None = None,
    **make_kwargs: object,
) -> None:
    """Host an environment on ``address`` (blocking).

    An :class:`EnvFactory` class/instance is constructed via ``prepare()`` +
    ``make(**make_kwargs)`` and its ``tags`` published; a bare make-env callable is
    invoked to produce the env. ``num_envs > 1`` fans the factory/callable out into
    a vector env (``EnvServer`` auto-detects and serves it via the vector server).
    Heavy imports stay inside this call so importing the authoring base stays cheap.
    """
    from rlmesh import EnvServer

    from ._bootstrap.loaders import construct_authored_env

    vectorized = num_envs > 1
    if hasattr(env_source, "make"):  # EnvFactory class or instance
        env = construct_authored_env(
            env_source,
            num_envs=num_envs,
            vectorization_mode=vectorization_mode,
            **make_kwargs,
        )
        # Adapters resolve per single-env lane and are rejected at num_envs>1, so
        # publish tags only on the scalar path -- mirroring the gym build path,
        # which serves vector envs untagged.
        tags = None if vectorized else getattr(env_source, "tags", None)
        EnvServer(env, address, tags=tags).serve()
    else:  # bare make-env callable
        make_env = cast("Callable[..., EnvLike[Any, Any]]", env_source)
        if vectorized:
            from ._bootstrap.gym_support import vectorize

            # vectorize returns a gym Sync/Async vector env (VectorServerEnvLike);
            # EnvServer auto-detects the vector shape.
            env = cast(
                "EnvLike[Any, Any]",
                vectorize(
                    lambda: make_env(**make_kwargs), num_envs, vectorization_mode
                ),
            )
        else:
            env = make_env(**make_kwargs)
        EnvServer(env, address).serve()


if __name__ == "__main__":
    raise SystemExit(main())
