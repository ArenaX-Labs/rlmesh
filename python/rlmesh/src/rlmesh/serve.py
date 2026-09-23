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
import sys
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

from ._entrypoint import resolve_entrypoint

# Startup phase marks. Every mark is wall-clock milliseconds since the process
# started (interpreter start plus the ``rlmesh`` import land before this module
# does), so a model container's wait splits into imports, construction (weights),
# and the listen point without a profiler. Printed on the serving line and
# stamped on the handshake ``PeerInfo.extra`` as ``rlmesh.startup.<phase>_ms``;
# the runtime's first ``endpoint.total`` sample is the first predict.
_ENTRY = time.monotonic()


def _process_age_ms() -> int | None:
    """Milliseconds this process had been alive when this module loaded.

    Linux only (``/proc``); ``None`` elsewhere, and the marks then count from
    this module's import instead.
    """
    try:
        with open("/proc/self/stat", encoding="ascii") as stat:
            fields = stat.read().rsplit(")", 1)[1].split()
        with open("/proc/uptime", encoding="ascii") as uptime_file:
            uptime = float(uptime_file.read().split()[0])
        # Field 22 of /proc/[pid]/stat is starttime in clock ticks since boot;
        # after the ')' split the command name is gone, so it is index 19.
        started = int(fields[19]) / os.sysconf("SC_CLK_TCK")
        return max(0, int((uptime - started) * 1000))
    except (OSError, ValueError, IndexError, AttributeError):
        return None


_PROCESS_AGE_MS = _process_age_ms()
_marks: dict[str, int] = {}


def _mark(phase: str) -> int:
    """Record ``phase`` as reached now; returns its millisecond mark."""
    since_entry = int((time.monotonic() - _ENTRY) * 1000)
    _marks[phase] = since_entry + (_PROCESS_AGE_MS or 0)
    return _marks[phase]


def startup_marks() -> dict[str, str]:
    """The phase marks reached so far, as ``rlmesh.startup.<phase>_ms`` strings."""
    marks = {f"rlmesh.startup.{phase}_ms": str(ms) for phase, ms in _marks.items()}
    if _PROCESS_AGE_MS is not None:
        marks["rlmesh.startup.process_ms"] = str(_PROCESS_AGE_MS)
    return marks


def _stamp_startup(
    kind: str, address: str, target: object, served_env: object | None = None
) -> None:
    """Mark the listen point, put the marks and the describe on the handshake, print them."""
    from ._peer_info import register_python_peer_info
    from ._rlmesh import DESCRIBE_METADATA_KEY

    _mark("listen")
    extra = startup_marks()
    if describe := _handshake_describe(target, kind, served_env):
        extra[DESCRIBE_METADATA_KEY] = describe
    register_python_peer_info(extra=extra)
    phases = ", ".join(
        f"{phase} {ms / 1000:.1f}s"
        for phase, ms in (
            [("process", _PROCESS_AGE_MS)] if _PROCESS_AGE_MS is not None else []
        )
        + list(_marks.items())
    )
    print(f"RLMesh serving {kind} on {address} (startup: {phases})", flush=True)


def _handshake_describe(
    target: object, kind: str, served_env: object | None = None
) -> str | None:
    """The served target's describe envelope, for ``PeerInfo.extra``.

    The managed platform reads it off the handshake when an image carries no
    describe label, so a plain ``python -m rlmesh.serve`` image is enough to
    probe. Best-effort: a target ``describe()`` cannot cover still serves,
    without the envelope, and says so once on stderr.
    """
    from ._describe import describe_json

    try:
        return describe_json(target, kind=kind, served_env=served_env)
    except Exception as exc:
        print(
            f"RLMesh could not describe the served {kind} ({exc}); the handshake "
            "carries no describe envelope",
            file=sys.stderr,
            flush=True,
        )
        return None


if TYPE_CHECKING:
    from rlmesh._models.base import ModelBase
    from rlmesh._value_conversion import ValueBridge
    from rlmesh.types import EnvLike

__all__ = ["main", "serve_env", "serve_model", "startup_marks"]

# Frameworks whose obs/action seam carries device tensors. numpy and the default
# Auto backend have no device, so a device= is meaningless (and unsupported) there.
_DEVICE_FRAMEWORKS = ("torch", "jax")


def _framework_name(framework: str | ValueBridge | None) -> str | None:
    # framework is a str, a ValueBridge (which carries .name), or None; normalize to
    # the name string (or None) without importing the bridge at runtime.
    return cast("str | None", getattr(framework, "name", framework))


def _normalize_framework(framework: str | None) -> str | None:
    """Canonicalize a framework name the way ``resolve_bridge`` does.

    ``resolve_bridge`` strips and lowercases the name, so the device and
    vectorization guards must compare the same canonical value -- otherwise an
    ambient ``RLMESH_FRAMEWORK=JAX`` would slip past them while still resolving
    to the jax bridge.
    """
    if isinstance(framework, str):
        return framework.strip().lower()
    return framework


def _gate_device(
    device: object | None, framework: str | ValueBridge | None
) -> object | None:
    # --device defaults to RLMESH_DEVICE, so a GPU node's global default would reach
    # a numpy/gym env and make EnvServer reject it at startup. The device only types
    # the torch/jax seam; ignore it for anything else.
    if device is None:
        return None
    return device if _framework_name(framework) in _DEVICE_FRAMEWORKS else None


def _reject_vectorized_framework(
    vectorized: bool, framework: str | ValueBridge | None
) -> None:
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
    parser.add_argument(
        "--framework",
        default=os.environ.get("RLMESH_FRAMEWORK") or None,
        help=(
            "Array framework for the env's obs/action seam: 'torch', 'jax', or "
            "'numpy' (default). Needed only when --env resolves to a bare "
            "make-callable or env class; an EnvFactory pins it on the class. "
            "Defaults to RLMESH_FRAMEWORK. Env only."
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
        "--workflow-edition",
        default=None,
        help=(
            "Workflow edition this served peer declares -- the semantics it was "
            "authored against, kept until you bump it. Defaults to "
            "RLMESH_WORKFLOW_EDITION, then the entrypoint class's "
            "workflow_edition, then [tool.rlmesh] workflow_edition. Paste what "
            "rlmesh.current_workflow_edition() reports."
        ),
    )
    parser.add_argument(
        "--kwargs-json",
        type=_json_object,
        help=(
            "JSON object bound to the entrypoint's declared params -- the env "
            "factory's make(**binding) or the model's load(**binding), i.e. the "
            "variation to serve. Defaults to RLMESH_MAKE_KWARGS; absent serves "
            "the declared defaults. Validated before construction."
        ),
    )
    args = parser.parse_args(argv)

    if bool(args.model) == bool(args.env):
        parser.error("provide exactly one of a model entrypoint or --env")
    if args.model and (args.framework is not None or args.device is not None):
        parser.error(
            "--framework/--device apply only to an env (--env); unset them (and "
            "RLMESH_FRAMEWORK / RLMESH_DEVICE) when serving a model"
        )

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

    try:
        if args.env:
            # num_envs / vectorization_mode / framework / device are serve controls
            # (their own env vars / flags), not env make() kwargs; a binding key of any
            # of those names would otherwise collide with serve_env's explicit args as
            # an opaque "multiple values" TypeError. Point the operator at the right
            # knob instead.
            control_collisions = {
                "num_envs",
                "vectorization_mode",
                "framework",
                "device",
                "workflow_edition",
            } & binding.keys()
            if control_collisions:
                parser.error(
                    f"{', '.join(sorted(control_collisions))} control serving, not "
                    "env construction; set RLMESH_NUM_ENVS / RLMESH_VECTORIZATION_MODE "
                    "/ --framework / --device / --workflow-edition instead of passing "
                    "them in RLMESH_MAKE_KWARGS / --kwargs-json"
                )
            env = resolve_entrypoint(args.env, label="env entrypoint")
            _mark("imports")
            serve_env(
                env,
                args.address,
                num_envs=num_envs,
                vectorization_mode=vectorization_mode,
                framework=args.framework,
                device=args.device,
                workflow_edition=args.workflow_edition,
                **binding,
            )
        else:
            model = resolve_entrypoint(args.model, label="model entrypoint")
            _mark("imports")
            serve_model(
                model,
                args.address,
                binding=binding,
                workflow_edition=args.workflow_edition,
            )
    except KeyboardInterrupt:
        # Ctrl-C is how an operator stops a served container: the serve loop has
        # already drained and closed by the time the interrupt surfaces here, so
        # report the conventional interrupted-by-SIGINT status instead of a
        # traceback. Not 0: an interrupt during startup served nothing.
        return 130
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
    binding: dict[str, Any] | None = None,
    workflow_edition: str | None = None,
) -> None:
    """Host a model on ``address`` (blocking).

    Resolves the source to a serveable ``Model``: a ``Model`` subclass class is
    instantiated, an existing ``Model`` instance is used directly, and a bare predict
    callable (or duck-typed policy object) is wrapped in a ``Model`` -- either way, no
    hand-written request builder. A non-empty ``binding`` validates against the
    model's declared ``params`` and is applied to ``load(**binding)`` on the
    bootstrap-authoritative path. Heavy imports stay inside this call so importing
    the authoring base stays cheap. A one-line serving status is printed once the
    model is resolved and loaded, before the blocking serve, with the startup
    phase marks (see :func:`startup_marks`), which also ride the handshake.
    """
    from ._editions import serve_options_declaring

    model = _resolve_model(model_source, binding)
    _mark("model")
    _stamp_startup("model", address, model)
    # `--workflow-edition` is the ServeOptions rung of the precedence chain;
    # the surfaces above it (the env var) and below it (the model class, the
    # project manifest) are resolved here, once.
    model.serve(
        address,
        options=serve_options_declaring(
            option=workflow_edition,
            declared=getattr(type(model), "workflow_edition", None),
        ),
    )


def _resolve_model(
    model_source: object, binding: dict[str, Any] | None = None
) -> ModelBase[Any, Any]:
    """Resolve a model source to a serveable ``Model`` without double-construction.

    A ``Model`` subclass *class* is built by :meth:`rlmesh.Model.from_config`
    with the binding (``load(**binding)`` runs once, after the binding is
    resolved against the declared ``params``); an existing ``Model`` instance is
    used as-is (it already built its worker); anything else (a bare predict
    callable or a duck-typed policy object) is served as a NumPy framework
    ``Model``: the documented default for a policy image, where the library's
    ``run`` / ``session`` refuse to pick a framework for the caller. Bindings
    require a :class:`rlmesh.Model` subclass *class* entrypoint -- the documented
    authoring path -- so there is no double-construction of a wrapped policy.
    """
    from rlmesh._models.base import ModelBase

    if isinstance(model_source, type) and issubclass(model_source, ModelBase):
        # A Model subclass class is `from_config`'d -- even with no binding -- so
        # declared required params are enforced before weights load, matching
        # the env path (construct_authored_env always resolves).
        resolved_binding: dict[str, Any] = binding or {}
        return cast("ModelBase[Any, Any]", model_source.from_config(**resolved_binding))
    if binding:
        raise TypeError(
            "model construction params (--kwargs-json / RLMESH_MAKE_KWARGS) "
            "require a rlmesh.Model subclass entrypoint (module:Class)"
        )
    if isinstance(model_source, ModelBase):
        return cast("ModelBase[Any, Any]", model_source)
    from rlmesh.numpy import Model as NumpyModel

    return NumpyModel(cast("object", model_source))


def serve_env(
    env_source: object,
    address: str,
    /,
    *,
    num_envs: int = 1,
    vectorization_mode: str | None = None,
    framework: str | None = None,
    device: object | None = None,
    workflow_edition: str | None = None,
    **make_kwargs: object,
) -> None:
    """Host an environment on ``address`` (blocking).

    An :class:`EnvFactory` class/instance is constructed via ``prepare()`` +
    ``make(**make_kwargs)`` and its ``tags`` published; a bare make-env callable is
    invoked to produce the env. ``num_envs > 1`` makes that many instances and
    serves them as the lanes of one endpoint, each stepped independently (tags and
    framework carry through). An explicit ``vectorization_mode`` instead fans out
    into a gym vector env stepped in lockstep, served untagged.

    ``framework`` (``"torch"``/``"jax"``/``"numpy"``) types the env's obs/action
    seam; for an :class:`EnvFactory` it defaults to the factory's pinned framework
    (``rlmesh.torch.EnvFactory`` etc.), so a classless make-callable / gym-id /
    hf source is the only case that needs it passed explicitly. ``device`` places
    the incoming action (torch/jax only). ``workflow_edition`` declares the
    semantics this endpoint was authored against, above the source's own
    :attr:`EnvFactory.workflow_edition <rlmesh.EnvFactory.workflow_edition>` and
    below ``RLMESH_WORKFLOW_EDITION``. Heavy imports stay inside this call so
    importing the authoring base stays cheap. A one-line serving status is
    printed once the server has bound its address, before the blocking serve.
    """
    from rlmesh import EnvServer

    from ._bootstrap.loaders import construct_authored_env

    framework = _normalize_framework(framework)
    # Lanes are the default for num_envs > 1; the gym fan-out only on request.
    gym_vectorized = num_envs > 1 and vectorization_mode is not None
    tags: object | None = None
    if hasattr(env_source, "make"):
        # The framework rides the factory class (_bridge ClassVar); an explicit
        # framework= overrides it. Unlike tags, it survives vectorization. Resolve
        # it before constructing so a vectorized framework env is rejected up front
        # rather than after a pointless gym fan-out.
        env_framework: str | ValueBridge | None = (
            framework
            if framework is not None
            else cast("ValueBridge | None", getattr(env_source, "_bridge", None))
        )
        _reject_vectorized_framework(gym_vectorized, env_framework)
        env = construct_authored_env(
            env_source,
            num_envs=num_envs,
            vectorization_mode=vectorization_mode,
            **make_kwargs,
        )
        # Lanes are scalar envs, so tags publish per lane as on the scalar path.
        # A gym vector env's batched spaces don't match the per-lane tags, so
        # that path serves untagged. A branched factory already stamped the
        # branch's own tags inside make(); re-publishing the ClassVar here would
        # overwrite them with the default branch's contract, so leave them alone
        # and let EnvServer validate what is published. The ClassVar read stays
        # for the unstamped duck-typed source (a make-haver that is not an
        # EnvFactory).
        tags = (
            None
            if gym_vectorized or getattr(env_source, "tag_params", ())
            else getattr(env_source, "tags", None)
        )
    else:
        # A bare callable has no class to pin a framework, so honor only the
        # explicit framework= (from --framework / RLMESH_FRAMEWORK).
        env_framework = framework
        _reject_vectorized_framework(gym_vectorized, env_framework)
        make_env = cast("Callable[..., EnvLike[Any, Any]]", env_source)
        if gym_vectorized:
            from ._bootstrap.gym_support import vectorize

            # vectorize returns a gym Sync/Async vector env (VectorServerEnvLike);
            # EnvServer auto-detects the vector shape.
            env = cast(
                "EnvLike[Any, Any]",
                vectorize(
                    lambda: make_env(**make_kwargs), num_envs, vectorization_mode
                ),
            )
        elif num_envs > 1:
            env = cast(
                "EnvLike[Any, Any]",
                [make_env(**make_kwargs) for _ in range(num_envs)],
            )
        else:
            env = make_env(**make_kwargs)
    from ._editions import serve_options_declaring

    server = EnvServer(
        env,
        address,
        tags=cast("Any", tags),
        framework=env_framework,
        device=_gate_device(device, env_framework),
        options=serve_options_declaring(
            option=workflow_edition,
            declared=getattr(env_source, "workflow_edition", None),
        ),
    )
    _mark("env")
    _stamp_startup("env", server.address, env_source, env)
    server.serve()


if __name__ == "__main__":
    raise SystemExit(main())
