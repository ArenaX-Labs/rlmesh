"""Python-backed environment serving for the Rust RLMesh CLI."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from rlmesh._client._endpoint import redact_address
from rlmesh._serving import load_env as _serving_load_env
from rlmesh._serving import load_env_entrypoint as _serving_load_env_entrypoint

if TYPE_CHECKING:
    from rlmesh._server import EnvLike, VectorServerEnvLike

    ServedEnv = EnvLike[Any, Any] | VectorServerEnvLike

__all__ = [
    "ServeArgs",
    "add_arguments",
    "create_parser",
    "load_env_entrypoint",
    "load_environment",
    "main",
    "serve_args_from_namespace",
    "serve_from_args",
    "write_ready_fd",
]


@dataclass
class ServeArgs:
    """Parsed arguments for the ``env serve`` command."""

    env: str | None
    entrypoint: str | None
    address: str | None
    num_envs: int
    vectorization_mode: str | None
    package: list[str]
    verbose: bool
    kwargs: dict[str, Any] | None = None
    ready_fd: int | None = None
    workflow_edition: str | None = None


def write_ready_fd(fd: int, address: str) -> None:
    """Write the bound server address to ``fd`` and close it.

    This is the machine-readable readiness signal: supervisors pass a writable
    file descriptor via ``--ready-fd`` and block reading it until a single line
    (the resolved bind address, e.g. ``127.0.0.1:54321/env``) arrives and the
    descriptor is closed. The write happens only after the listener is bound, so
    a successful read means the server is accepting connections. Closing the
    descriptor signals end-of-file so the reader unblocks deterministically.

    Errors are surfaced to the caller (a bad descriptor is a supervisor
    misconfiguration worth failing loudly on) rather than swallowed.
    """
    payload = f"{address}\n".encode()
    written = 0
    while written < len(payload):
        written += os.write(fd, payload[written:])
    os.close(fd)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register env serve arguments on an existing parser."""
    source = parser.add_mutually_exclusive_group(required=True)
    _ = source.add_argument("--env", help="Environment ID (e.g., CartPole-v1)")
    _ = source.add_argument(
        "--entrypoint",
        help="Environment factory entrypoint in module:callable form",
    )
    _ = parser.add_argument(
        "--address",
        help="Session participant address (default :0/env)",
    )
    _ = parser.add_argument(
        "--num-envs",
        type=_env_count,
        default=1,
        help="Number of vectorized Gym/Gymnasium environments for --env (>= 1)",
    )
    _ = parser.add_argument(
        "--vectorization-mode",
        choices=["sync", "async"],
        help="Preferred Gym/Gymnasium vectorization mode when --num-envs > 1",
    )
    _ = parser.add_argument(
        "--package",
        action="append",
        default=[],
        help="Import a package before loading the environment. Repeat as needed.",
    )
    _ = parser.add_argument(
        "--kwargs-json",
        type=_json_object,
        help="JSON object passed as keyword arguments to the environment loader",
    )
    _ = parser.add_argument(
        "--ready-fd",
        type=int,
        help=(
            "File descriptor to write the bound address to (one line) and close "
            "once the server is accepting connections. Lets supervisors wait for "
            "readiness without grepping stdout."
        ),
    )
    _ = parser.add_argument(
        "--workflow-edition",
        help=(
            "Workflow edition this endpoint declares -- the semantics it was "
            "authored against, kept until you bump it. Defaults to "
            "RLMESH_WORKFLOW_EDITION, then the entrypoint class's "
            "workflow_edition, then [tool.rlmesh] workflow_edition. Paste what "
            "rlmesh.current_workflow_edition() reports."
        ),
    )
    _ = parser.add_argument("--verbose", action="store_true", help="Verbose output")


def create_parser() -> argparse.ArgumentParser:
    """Create the standalone parser used by the Rust CLI bridge."""
    parser = argparse.ArgumentParser(
        prog="python -m rlmesh._cli.serve_env",
        description="Serve a Python environment through RLMesh",
    )
    add_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Serve an environment using the Python RLMesh bindings."""
    parser = create_parser()
    args = serve_args_from_namespace(parser.parse_args(argv))
    return serve_from_args(args)


def serve_from_args(args: ServeArgs) -> int:
    """Handle `env serve` command arguments.

    The server is always shut down on the way out (including Ctrl+C and serve
    errors) so the served env's ``close()`` runs instead of being skipped.
    """
    try:
        from rlmesh import EnvServer
        from rlmesh._editions import serve_options_declaring

        if args.entrypoint is not None:
            if args.num_envs != 1 or args.vectorization_mode is not None:
                raise ValueError(
                    "--num-envs and --vectorization-mode are only supported with --env"
                )
            env = load_env_entrypoint(args.entrypoint, args.package, args.kwargs)
        else:
            assert args.env is not None
            env = load_environment(
                args.env,
                args.package,
                args.num_envs,
                args.vectorization_mode,
                args.kwargs,
            )

        served_num_envs = _served_num_envs(env, fallback=args.num_envs)
        # The served env declares an edition on every handshake: --workflow-edition
        # if given, else the process / class / project declaration.
        options = serve_options_declaring(
            option=args.workflow_edition,
            # The entrypoint's factory is gone by now (it returned the env), so
            # only an env class that declares the edition on itself is visible
            # here; RLMESH_WORKFLOW_EDITION / [tool.rlmesh] / --workflow-edition
            # cover the rest.
            declared=getattr(type(env), "workflow_edition", None),
            origin=type(env),
        )
        # EnvServer auto-detects the vectorized shape from the env, so there is one
        # construction path for both scalar and vector envs.
        server = EnvServer(env, args.address, options=options)

        if args.entrypoint is not None:
            print(f"✓ Environment entrypoint: {args.entrypoint}")
        else:
            print(f"✓ Environment: {args.env}")
        # The keyed address is the invite for whoever runs this at a
        # terminal; a process its launcher started under a host identity
        # already gave the launcher the key, and its stdout is a log.
        shown = (
            redact_address(server.address)
            if os.environ.get("RLMESH_HOST_IDENTITY")
            else server.address
        )
        print(f"✓ Server address: {shown}")
        if args.entrypoint is None or served_num_envs > 1:
            print(f"✓ Num envs: {served_num_envs}")
        print()
        print("Waiting for client connection...")
        print("Press Ctrl+C to stop", flush=True)

        try:
            if args.ready_fd is not None:
                write_ready_fd(args.ready_fd, server.address)

            server.serve()
        finally:
            server.shutdown()
        print("\nClient disconnected")
        return 0

    except KeyboardInterrupt:
        print("\nShutting down server")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def load_environment(
    env_id: str,
    package_names: list[str],
    num_envs: int,
    vectorization_mode: str | None = None,
    kwargs: dict[str, Any] | None = None,
) -> ServedEnv:
    """Compatibility wrapper delegating to the public ``rlmesh._serving`` loader."""
    return _serving_load_env(
        env_id,
        packages=package_names,
        num_envs=num_envs,
        vectorization_mode=vectorization_mode,
        kwargs=kwargs,
    )


def load_env_entrypoint(
    entrypoint: str,
    package_names: list[str],
    kwargs: dict[str, Any] | None = None,
) -> ServedEnv:
    """Compatibility wrapper delegating to the public ``rlmesh._serving`` loader."""
    return _serving_load_env_entrypoint(
        entrypoint,
        packages=package_names,
        kwargs=kwargs,
    )


def _served_num_envs(env: object, *, fallback: int) -> int:
    num_envs = getattr(env, "num_envs", None)
    if num_envs is None:
        return fallback
    try:
        return int(num_envs)
    except (TypeError, ValueError):
        return fallback


def serve_args_from_namespace(args: argparse.Namespace) -> ServeArgs:
    """Convert an argparse namespace into typed serve arguments.

    Every field's type is pinned by the parser (``type=int``,
    ``action="store_true"``/``"append"``, ``choices=...``, ``type=_json_object``),
    so the namespace attributes are read directly without re-validation.
    """
    return ServeArgs(
        env=args.env,
        entrypoint=args.entrypoint,
        address=args.address,
        num_envs=args.num_envs,
        vectorization_mode=args.vectorization_mode,
        package=args.package,
        verbose=args.verbose,
        kwargs=args.kwargs_json,
        ready_fd=args.ready_fd,
        workflow_edition=args.workflow_edition,
    )


def _env_count(value: str) -> int:
    """Parse ``--num-envs``: an int >= 1, rejected in the parser otherwise.

    Without this, 0 or a negative slips through and silently serves one env.
    """
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"--num-envs must be >= 1, got {parsed}")
    return parsed


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--kwargs-json must be a JSON object")
    return cast(dict[str, Any], parsed)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
