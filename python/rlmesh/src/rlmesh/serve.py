"""Templated container entrypoint.

``python -m rlmesh.serve my_pkg:Policy`` serves a model; ``--env my_pkg:Env``
serves an environment. The target may be a ``Model`` subclass, an :class:`EnvFactory`,
or a bare predict / make-env callable. Serves as ``RLMESH_NAME`` (default
``model``, or ``env``) in the session at ``RLMESH_ADDRESS`` (default
``0.0.0.0:7447``, hosted here); point your Dockerfile ``ENTRYPOINT`` here
instead of hand-writing a serve loop.

Every option has a command-line flag and an ``RLMESH_*`` environment variable,
so a container is configured through its environment alone; a flag wins over
its variable, and a value that does not parse stops the process at startup,
naming the variable. A served model's batching (:class:`rlmesh.BatchOptions`)
is ``--max-batch-size`` / ``RLMESH_MAX_BATCH_SIZE``, ``--batch-window-ms`` /
``RLMESH_BATCH_WINDOW_MS``, ``--batch-byte-budget`` /
``RLMESH_BATCH_BYTE_BUDGET``, ``--max-in-flight-batches`` /
``RLMESH_MAX_IN_FLIGHT_BATCHES`` and ``--batch-records`` /
``RLMESH_BATCH_RECORDS``; one left unset takes ``BatchOptions``' default.
``python -m rlmesh.serve --help`` lists them all.

Whoever launched the process ends it. SIGTERM drains: the env or model call
in progress finishes, the env is closed, and the process exits 0; a driver
in a call hears that the server went away. Ctrl-C (SIGINT) does the same and
exits 130. A second signal while the drain waits on a call that will not
return ends the process at once, by that signal (128 + its number: 143 for
SIGTERM, 130 for SIGINT), without closing the env.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from ._authoring import render_binding
from ._entrypoint import resolve_entrypoint

# Startup phase marks: wall-clock ms since process start, printed on the serving line and
# stamped on PeerInfo.extra as ``rlmesh.startup.<phase>_ms``.
_ENTRY = time.monotonic()


def _process_age_ms() -> int | None:
    """Milliseconds this process had been alive when this module loaded."""
    try:
        with open("/proc/self/stat", encoding="ascii") as stat:
            fields = stat.read().rsplit(")", 1)[1].split()
        with open("/proc/uptime", encoding="ascii") as uptime_file:
            uptime = float(uptime_file.read().split()[0])
        # Field 22 of /proc/[pid]/stat is starttime; after the ')' split the command name is gone, so it is index 19.
        started = int(fields[19]) / os.sysconf("SC_CLK_TCK")
        return max(0, int((uptime - started) * 1000))
    except (OSError, ValueError, IndexError, AttributeError):
        return None


_PROCESS_AGE_MS = _process_age_ms()
_marks: dict[str, int] = {}


def _mark(phase: str) -> int:
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
    kind: str,
    target: object,
    served_env: object | None = None,
    *,
    options: ServeOptions | None = None,
) -> None:
    from ._peer_info import register_python_peer_info
    from ._rlmesh import DESCRIBE_METADATA_KEY

    _mark("listen")
    extra = startup_marks()
    resolved = (options.workflow_edition if options is not None else None) or ""
    if describe := _handshake_describe(target, kind, served_env, resolved):
        extra[DESCRIBE_METADATA_KEY] = describe
    register_python_peer_info(extra=extra)


def _print_serving_line(ready: Mapping[str, object]) -> None:
    _mark("ready")
    phases = ", ".join(
        f"{phase} {ms / 1000:.1f}s"
        for phase, ms in (
            [("process", _PROCESS_AGE_MS)] if _PROCESS_AGE_MS is not None else []
        )
        + list(_marks.items())
    )
    print(
        f"RLMesh serving {ready['kind']} on {ready['address']} (startup: {phases})",
        flush=True,
    )


@contextlib.contextmanager
def _serving_line() -> Iterator[None]:
    from ._load_native import load_native

    set_ready_hook = load_native("set_ready_hook")
    set_ready_hook(_print_serving_line)
    try:
        yield
    finally:
        set_ready_hook(None)


def _handshake_describe(
    target: object, kind: str, served_env: object | None, workflow_edition: str
) -> str | None:
    from ._describe import describe_json

    try:
        return describe_json(
            target, kind=kind, served_env=served_env, workflow_edition=workflow_edition
        )
    except Exception as exc:
        print(
            f"RLMesh could not describe the served {kind} ({exc}); the handshake "
            "carries no describe envelope",
            file=sys.stderr,
            flush=True,
        )
        return None


if TYPE_CHECKING:
    from rlmesh._models.base import BatchOptions, ModelBase
    from rlmesh._rlmesh import ServeOptions
    from rlmesh._value_conversion import ValueBridge
    from rlmesh.types import EnvLike

__all__ = ["main", "serve_env", "serve_model", "startup_marks"]

_ADDRESS = "0.0.0.0:7447"
_HOST_IDENTITY_VAR = "RLMESH_HOST_IDENTITY"
_READY_FILE_VAR = "RLMESH_READY_FILE"


class _NoHostIdentityError(ValueError):
    """An all-interfaces address served without a launch identity."""


def _refuse_unlaunched_wide_bind(address: str) -> None:
    """Refuse ``address`` before anything loads when it cannot host here."""
    if os.environ.get(_HOST_IDENTITY_VAR):
        return
    first = address.removeprefix("rlmesh://").split("/", 1)[0].split("#", 1)[0]
    if first.startswith("["):
        host, _, port = first[1:].partition("]")
        port = port.removeprefix(":")
    else:
        host, _, port = first.rpartition(":")
    if host not in ("0.0.0.0", "::") or not port.isdigit():
        return
    import socket

    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=0.2):
            return
    except OSError:
        pass
    raise _NoHostIdentityError(
        f"serving on {first} (all interfaces) needs a host identity: set "
        f"{_HOST_IDENTITY_VAR}, through the environment, to the identity "
        "`rlmesh host-identity new --address HOST:PORT --participant NAME` makes "
        "(rlmesh.launch in Python), and dial the keyed address it prints; or set "
        f"RLMESH_ADDRESS=127.0.0.1:{port} to serve this machine only (in a "
        "container, only that container: nothing outside it can reach it)"
    )


_BATCH_INTEGERS = (
    (
        "max_batch_size",
        "RLMESH_MAX_BATCH_SIZE",
        1,
        "The most predict groups (one route's rows each) one grouped predict "
        "carries; 1 turns batching off.",
    ),
    (
        "batch_window_ms",
        "RLMESH_BATCH_WINDOW_MS",
        0,
        "The longest, in milliseconds, an idle model holds a batch open for "
        "more requests.",
    ),
    (
        "batch_byte_budget",
        "RLMESH_BATCH_BYTE_BUDGET",
        1,
        "The most observation bytes one grouped predict carries.",
    ),
    (
        "max_in_flight_batches",
        "RLMESH_MAX_IN_FLIGHT_BATCHES",
        1,
        "Batches past assembly at once: 2 assembles the next while one runs, "
        "1 assembles only while the model is idle.",
    ),
)
_BATCH_RECORDS_VAR = "RLMESH_BATCH_RECORDS"
# The native batch fields are 64-bit.
_BATCH_MAX = 2**63 - 1


def _batch_integer(raw: str, minimum: int) -> int:
    """A batch setting's integer, or ``ValueError`` saying what it must be."""
    try:
        value = int(raw.strip(), 10)
    except ValueError:
        value = None
    if value is None or not minimum <= value <= _BATCH_MAX:
        raise ValueError(f"must be an integer from {minimum} to {_BATCH_MAX}")
    return value


def _batch_flag_type(minimum: int) -> Callable[[str], int]:
    def parse(raw: str) -> int:
        try:
            return _batch_integer(raw, minimum)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{exc}, got {raw!r}") from exc

    return parse


def _add_batching_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "model batching",
        "How a served model coalesces predicts from every session it answers "
        "in (rlmesh.BatchOptions; experimental). Each flag defaults to its "
        "environment variable, then to BatchOptions' default. Model only.",
    )
    for field, var, minimum, help_text in _BATCH_INTEGERS:
        group.add_argument(
            "--" + field.replace("_", "-"),
            dest=field,
            type=_batch_flag_type(minimum),
            metavar="N",
            help=f"{help_text} Defaults to {var}.",
        )
    group.add_argument(
        "--batch-records",
        metavar="PATH",
        help=(
            "Append one JSON line per batch here (batch size, queue wait, "
            f"forward timing, outcome). Defaults to {_BATCH_RECORDS_VAR}."
        ),
    )


def _batching(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> BatchOptions | None:
    """The batch options the flags and their variables set, else ``None``."""
    from ._models import BatchOptions

    values: dict[str, int] = {}
    for field, var, minimum, _ in _BATCH_INTEGERS:
        value = cast("int | None", getattr(args, field))
        raw = os.environ.get(var)
        if value is None and raw:
            try:
                value = _batch_integer(raw, minimum)
            except ValueError as exc:
                parser.error(f"{var} {exc}, got {raw!r}")
        if value is not None:
            values[field] = value
    records = cast("str | None", args.batch_records) or os.environ.get(
        _BATCH_RECORDS_VAR
    )
    if not values and not records:
        return None
    return BatchOptions(
        max_batch_size=values.get("max_batch_size"),
        batch_window_ms=values.get("batch_window_ms"),
        batch_byte_budget=values.get("batch_byte_budget"),
        max_in_flight_batches=values.get("max_in_flight_batches"),
        batch_records=records or None,
    )


# numpy and the default Auto backend have no device.
_DEVICE_FRAMEWORKS = ("torch", "jax")


def _framework_name(framework: str | ValueBridge | None) -> str | None:
    return cast("str | None", getattr(framework, "name", framework))


def _normalize_framework(framework: str | None) -> str | None:
    """Canonicalize a framework name the way ``resolve_bridge`` does."""
    if isinstance(framework, str):
        return framework.strip().lower()
    return framework


def _gate_device(
    device: object | None, framework: str | ValueBridge | None
) -> object | None:
    # --device defaults to RLMESH_DEVICE; a numpy/gym env would make EnvServer reject it, so ignore it there.
    if device is None:
        return None
    return device if _framework_name(framework) in _DEVICE_FRAMEWORKS else None


def _reject_vectorized_framework(
    vectorized: bool, framework: str | ValueBridge | None
) -> None:
    # gym vectorization concatenates with numpy and breaks framework tensors; a natively batched env at num_envs=1 is fine.
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
    parser = argparse.ArgumentParser(
        prog="python -m rlmesh.serve",
        epilog=(
            "For orchestrators: serving on every interface (the default "
            f"{_ADDRESS}) takes a host identity in {_HOST_IDENTITY_VAR}, passed "
            "through the environment, never argv: `rlmesh host-identity new "
            "--address HOST:PORT --participant NAME` makes one and prints the "
            "keyed address peers dial (a sandbox does this for you). "
            f"{_READY_FILE_VAR}=PATH writes PATH, a small JSON file, once the "
            "env or model answers, and removes it when it stops: wait on it "
            "(HEALTHCHECK CMD test -f PATH) rather than on the log."
        ),
    )
    parser.add_argument(
        "model",
        nargs="?",
        help=(
            "module:Class for a model/policy. RLMESH_SERVE_TARGET overrides it, so "
            "one image serves every checkpoint whose package entry names its own "
            "target."
        ),
    )
    parser.add_argument("--env", help="module:Class for an environment")
    parser.add_argument(
        "--address",
        default=os.environ.get("RLMESH_ADDRESS") or None,
        help=(
            f"Where to serve (default {_ADDRESS}, every interface; 127.0.0.1:PORT "
            "for this machine only). Defaults to RLMESH_ADDRESS."
        ),
    )
    parser.add_argument(
        "--name",
        default=os.environ.get("RLMESH_NAME") or None,
        help=(
            "The participant name in the session, reached at '<address>/<name>'. "
            "Defaults to RLMESH_NAME, then 'model' (or 'env' with --env)."
        ),
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
    _add_batching_arguments(parser)
    args = parser.parse_args(argv)
    if not args.env and (target := os.environ.get("RLMESH_SERVE_TARGET")):
        args.model = target

    if bool(args.model) == bool(args.env):
        parser.error("provide exactly one of a model entrypoint or --env")
    if args.model and (args.framework is not None or args.device is not None):
        parser.error(
            "--framework/--device apply only to an env (--env); unset them (and "
            "RLMESH_FRAMEWORK / RLMESH_DEVICE) when serving a model"
        )
    batching = _batching(args, parser)
    if args.env and batching is not None:
        parser.error(
            "the batching options apply only to a model; unset them (and "
            "RLMESH_MAX_BATCH_SIZE, RLMESH_BATCH_WINDOW_MS, "
            "RLMESH_BATCH_BYTE_BUDGET, RLMESH_MAX_IN_FLIGHT_BATCHES, "
            "RLMESH_BATCH_RECORDS) when serving an env"
        )

    # Honor RLMESH_NUM_ENVS (set by start_prebuilt_container) by fanning the factory out into a vector env.
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
            # These are serve controls, not make() kwargs; a binding key of the same name would collide as an opaque
            # "multiple values" TypeError.
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
            binding = render_binding(env, _render_size(parser)) | binding
            serve_env(
                env,
                f"{(args.address or _ADDRESS).rstrip('/')}/{args.name or 'env'}",
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
                args.address or _ADDRESS,
                name=args.name or "model",
                binding=binding,
                workflow_edition=args.workflow_edition,
                batching=batching,
            )
    except _NoHostIdentityError as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        # The serve loop has already drained by now; report the SIGINT exit status, not 0
        # (an interrupt during startup served nothing).
        return 130
    return 0


def _render_size(parser: argparse.ArgumentParser) -> tuple[int, int] | None:
    raw = os.environ.get("RLMESH_RENDER_SIZE")
    if not raw:
        return None
    height, sep, width = raw.partition("x")
    if not (sep and height.isdigit() and width.isdigit()):
        parser.error(f"RLMESH_RENDER_SIZE must be <height>x<width>, got {raw!r}")
    return int(height), int(width)


def _json_object(value: str) -> dict[str, Any]:
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
    name: str = "model",
    binding: dict[str, Any] | None = None,
    workflow_edition: str | None = None,
    batching: BatchOptions | None = None,
) -> None:
    """Serve a model in the session at ``address`` as ``name`` (blocking).

    Resolves the source to a serveable ``Model``: a ``Model`` subclass class is
    instantiated, an existing ``Model`` instance is used directly, and a bare predict
    callable (or duck-typed policy object) is wrapped in a ``Model`` -- either way, no
    hand-written request builder. A non-empty ``binding`` validates against the
    model's declared ``params`` and is applied to ``load(**binding)`` on the
    bootstrap-authoritative path. Heavy imports stay inside this call so importing
    the authoring base stays cheap. A one-line serving status is printed once the
    model answers, with the address a peer dials and the startup phase marks
    (see :func:`startup_marks`), which also ride the handshake.
    ``batching`` coalesces its predicts (:class:`rlmesh.BatchOptions`' defaults
    when ``None``).
    """
    from ._editions import serve_options_declaring

    _refuse_unlaunched_wide_bind(address)
    model = _resolve_model(model_source, binding)
    _mark("model")
    _warn_without_spec(model)
    options = serve_options_declaring(
        option=workflow_edition,
        declared=model._class_workflow_edition(),  # pyright: ignore[reportPrivateUsage]
        origin=model._edition_origin(),  # pyright: ignore[reportPrivateUsage]
    )
    _stamp_startup("model", model, options=options)
    with _serving_line():
        model.serve(address, name=name, options=options, batching=batching)


def _warn_without_spec(model: object) -> None:
    from .adapters.specs import ModelSpec

    if isinstance(getattr(model, "spec", None), ModelSpec):
        return
    print(
        "RLMesh: the served model declares no spec, so it cannot be tested with "
        "generated inputs (rlmesh doctor, rlmesh check) without a reference env. "
        "Set `spec = ModelSpec(...)` on the class.",
        file=sys.stderr,
        flush=True,
    )


def _resolve_model(
    model_source: object, binding: dict[str, Any] | None = None
) -> ModelBase[Any, Any]:
    from rlmesh._models.base import ModelBase

    if isinstance(model_source, type) and issubclass(model_source, ModelBase):
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
    """Serve an environment at the participant address ``address`` (blocking).

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
    printed once the env answers, with the address a peer dials.
    """
    from rlmesh import EnvServer

    from ._bootstrap.loaders import construct_authored_env

    _refuse_unlaunched_wide_bind(address)
    framework = _normalize_framework(framework)
    gym_vectorized = num_envs > 1 and vectorization_mode is not None
    tags: object | None = None
    if hasattr(env_source, "make"):
        # The framework rides the factory class; resolve it before constructing so a vectorized framework env is rejected up front.
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
        # A gym vector env's batched spaces don't match per-lane tags, so that path serves untagged;
        # a branched factory's own stamped tags must not be overwritten by the ClassVar.
        tags = (
            None
            if gym_vectorized or getattr(env_source, "tag_params", ())
            else getattr(env_source, "tags", None)
        )
    else:
        env_framework = framework
        _reject_vectorized_framework(gym_vectorized, env_framework)
        make_env = cast("Callable[..., EnvLike[Any, Any]]", env_source)
        if gym_vectorized:
            from ._bootstrap.gym_support import vectorize

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

    options = serve_options_declaring(
        option=workflow_edition,
        declared=getattr(env_source, "workflow_edition", None),
        origin=env_source,
    )
    server = EnvServer(
        env,
        address,
        tags=cast("Any", tags),
        framework=env_framework,
        device=_gate_device(device, env_framework),
        options=options,
    )
    _mark("env")
    _stamp_startup("env", env_source, env, options=options)
    with _serving_line():
        server.serve()


if __name__ == "__main__":
    raise SystemExit(main())
