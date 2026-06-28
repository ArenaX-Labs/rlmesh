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
    from rlmesh._value_conversion import ValueBridge
    from rlmesh.types import EnvLike

__all__ = ["main", "serve_env", "serve_model"]

# Frameworks whose obs/action seam carries device tensors. numpy and the default
# Auto backend have no device, so a device= is meaningless (and unsupported) there.
_DEVICE_FRAMEWORKS = ("torch", "jax")


def _framework_name(framework: object) -> object:
    # framework is a str, a ValueBridge (which carries .name), or None; normalize to
    # the name string (or None) without importing the bridge at runtime.
    return getattr(framework, "name", framework)


def _gate_device(device: object, framework: object) -> object:
    # --device defaults to RLMESH_DEVICE, so a GPU node's global default would reach
    # a numpy/gym env and make EnvServer reject it at startup. The device only types
    # the torch/jax seam; ignore it for anything else.
    if device is None:
        return None
    return device if _framework_name(framework) in _DEVICE_FRAMEWORKS else None


def _reject_vectorized_framework(vectorized: bool, framework: object) -> None:
    # gym vectorization concatenates observations with numpy, discarding the
    # framework tensors (and crashing on GPU tensors), so a torch/jax env can't be
    # fanned out that way. A natively batched env returning [N, ...] tensors is
    # still fine at num_envs=1; this only blocks the gym fan-out.
    name = _framework_name(framework)
    if vectorized and name in _DEVICE_FRAMEWORKS:
        raise NotImplementedError(
            f"serving a {name} env with num_envs>1 is not supported: gym "
            "vectorization concatenates observations with numpy, which discards the "
            "framework tensors. Serve scalar (num_envs=1) -- a natively batched env "
            "returning [N, ...] tensors works there -- or use framework='numpy'."
        )


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
        "--framework",
        default=os.environ.get("RLMESH_FRAMEWORK") or None,
        help=(
            "Array framework for the env's obs/action seam: 'torch', 'jax', or "
            "'numpy' (default). Needed only for a classless --env (a make-callable "
            "/ gym-id / hf source); an EnvFactory pins it on the class. Defaults to "
            "RLMESH_FRAMEWORK. Env only."
        ),
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("RLMESH_DEVICE") or None,
        help=(
            "Device for the incoming action (torch/jax only), e.g. 'cuda:0'. "
            "Defaults to RLMESH_DEVICE. Env only."
        ),
    )
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
            framework=args.framework,
            device=args.device,
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
    framework: str | None = None,
    device: object | None = None,
    **make_kwargs: object,
) -> None:
    """Host an environment on ``address`` (blocking).

    An :class:`EnvFactory` class/instance is constructed via ``prepare()`` +
    ``make(**make_kwargs)`` and its ``tags`` published; a bare make-env callable is
    invoked to produce the env. ``num_envs > 1`` fans the factory/callable out into
    a vector env (``EnvServer`` auto-detects and serves it via the vector server).

    ``framework`` (``"torch"``/``"jax"``/``"numpy"``) types the env's obs/action
    seam; for an :class:`EnvFactory` it defaults to the factory's pinned framework
    (``rlmesh.torch.EnvFactory`` etc.), so a classless make-callable / gym-id /
    hf source is the only case that needs it passed explicitly. ``device`` places
    the incoming action (torch/jax only). Heavy imports stay inside this call so
    importing the authoring base stays cheap.
    """
    from rlmesh import EnvServer

    from ._bootstrap.loaders import construct_authored_env

    vectorized = num_envs > 1
    if hasattr(env_source, "make"):  # EnvFactory class or instance
        # The framework rides the factory class (_bridge ClassVar); an explicit
        # framework= overrides it. Unlike tags, it survives vectorization. Resolve
        # it before constructing so a vectorized framework env is rejected up front
        # rather than after a pointless gym fan-out.
        env_framework: str | ValueBridge | None = (
            framework
            if framework is not None
            else cast("ValueBridge | None", getattr(env_source, "_bridge", None))
        )
        _reject_vectorized_framework(vectorized, env_framework)
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
        EnvServer(
            env,
            address,
            tags=tags,
            framework=env_framework,
            device=_gate_device(device, env_framework),
        ).serve()
    else:  # bare make-env callable
        # A bare callable has no class to pin a framework, so honor only the
        # explicit framework= (from --framework / RLMESH_FRAMEWORK).
        _reject_vectorized_framework(vectorized, framework)
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
        EnvServer(
            env, address, framework=framework, device=_gate_device(device, framework)
        ).serve()


if __name__ == "__main__":
    raise SystemExit(main())
