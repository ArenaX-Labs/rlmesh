"""Python-backed environment serving for the Rust RLMesh CLI."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from rlmesh._bootstrap.env import (
    import_packages,
    is_env_lookup_error,
)
from rlmesh.serving import load_env as _serving_load_env
from rlmesh.serving import load_env_entrypoint as _serving_load_env_entrypoint

if TYPE_CHECKING:
    from rlmesh.server import EnvLike

__all__ = [
    "ServeArgs",
    "add_arguments",
    "create_parser",
    "import_packages",
    "is_env_lookup_error",
    "load_env_entrypoint",
    "load_environment",
    "main",
    "serve_args_from_namespace",
    "serve_from_args",
]


@dataclass
class ServeArgs:
    env: str | None
    entrypoint: str | None
    transport: str
    address: str | None
    num_envs: int
    vectorization_mode: str | None
    package: list[str]
    verbose: bool
    kwargs: dict[str, Any] | None = None


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register env serve arguments on an existing parser."""
    source = parser.add_mutually_exclusive_group(required=True)
    _ = source.add_argument("--env", help="Environment ID (e.g., CartPole-v1)")
    _ = source.add_argument(
        "--entrypoint",
        help="Environment factory entrypoint in module:callable form",
    )
    _ = parser.add_argument(
        "--transport",
        default="tcp",
        choices=["unix", "tcp"],
        help="Transport type (default: tcp)",
    )
    _ = parser.add_argument(
        "--address",
        help="Socket path (unix) or host:port/tcp://host:port (tcp)",
    )
    _ = parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of vectorized Gym/Gymnasium environments for --env",
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
    """Handle `env serve` command arguments."""
    try:
        from rlmesh import EnvServer

        if args.transport == "unix" and os.name == "nt":
            raise ValueError(
                "unix sockets are not supported on Windows; use --transport tcp"
            )

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

        if args.transport == "unix":
            path = args.address
            if path is None:
                source_name = args.env if args.env is not None else args.entrypoint
                assert source_name is not None
                path = _default_unix_socket_path(source_name)
            server = EnvServer(env, path=path, transport="unix")
        else:
            server = (
                EnvServer(env) if args.address is None else EnvServer(env, args.address)
            )

        if args.entrypoint is not None:
            print(f"✓ Environment entrypoint: {args.entrypoint}")
        else:
            print(f"✓ Environment: {args.env}")
        print(f"✓ Server address: {server.address}")
        print(f"✓ Transport: {args.transport}")
        if args.entrypoint is None:
            print(f"✓ Num envs: {args.num_envs}")
        print()
        print("Waiting for client connection...")
        print("Press Ctrl+C to stop")

        server.serve()
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
) -> EnvLike:
    """Compatibility wrapper delegating to the public ``rlmesh.serving`` loader."""
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
) -> EnvLike:
    """Compatibility wrapper delegating to the public ``rlmesh.serving`` loader."""
    return _serving_load_env_entrypoint(
        entrypoint,
        packages=package_names,
        kwargs=kwargs,
    )


def serve_args_from_namespace(args: argparse.Namespace) -> ServeArgs:
    """Convert an argparse namespace into typed serve arguments."""
    return ServeArgs(
        env=_namespace_str_or_none(args, "env"),
        entrypoint=_namespace_str_or_none(args, "entrypoint"),
        transport=_namespace_str(args, "transport"),
        address=_namespace_str_or_none(args, "address"),
        num_envs=_namespace_int(args, "num_envs"),
        vectorization_mode=_namespace_str_or_none(args, "vectorization_mode"),
        package=_namespace_str_list(args, "package"),
        verbose=_namespace_bool(args, "verbose"),
        kwargs=_namespace_dict_or_none(args, "kwargs_json"),
    )


def _namespace_str(args: argparse.Namespace, name: str) -> str:
    value: object = vars(args).get(name)
    if not isinstance(value, str):
        raise TypeError(f"expected argparse field {name!r} to be a str")
    return value


def _namespace_str_or_none(args: argparse.Namespace, name: str) -> str | None:
    value: object = vars(args).get(name)
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"expected argparse field {name!r} to be a str | None")


def _namespace_int(args: argparse.Namespace, name: str) -> int:
    value: object = vars(args).get(name)
    if not isinstance(value, int):
        raise TypeError(f"expected argparse field {name!r} to be an int")
    return value


def _namespace_bool(args: argparse.Namespace, name: str) -> bool:
    value: object = vars(args).get(name)
    if not isinstance(value, bool):
        raise TypeError(f"expected argparse field {name!r} to be a bool")
    return value


def _namespace_str_list(args: argparse.Namespace, name: str) -> list[str]:
    value: object = vars(args).get(name)
    if value is None:
        return []
    if isinstance(value, list):
        items = cast(list[object], value)
        if all(isinstance(item, str) for item in items):
            return cast(list[str], items)
    raise TypeError(f"expected argparse field {name!r} to be a list[str]")


def _namespace_dict_or_none(
    args: argparse.Namespace, name: str
) -> dict[str, Any] | None:
    value: object = vars(args).get(name)
    if value is None or isinstance(value, dict):
        return cast(dict[str, Any] | None, value)
    raise TypeError(f"expected argparse field {name!r} to be a JSON object")


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--kwargs-json must be a JSON object")
    return cast(dict[str, Any], parsed)


def _socket_label(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "_", value.lower()).strip("_") or "env"


def _default_unix_socket_path(source_name: str) -> str:
    """Return a per-user private default path for the unix socket.

    A predictable world-readable name in shared ``/tmp`` lets another local
    user squat the socket or pre-bind it. Prefer ``$XDG_RUNTIME_DIR`` (already
    per-user and 0700) and otherwise create a private ``0700`` temp directory,
    so the default socket is not reachable or hijackable by other users.
    """
    import tempfile

    filename = f"rlmesh-{_socket_label(source_name)}.sock"

    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir and os.path.isdir(runtime_dir):
        base = os.path.join(runtime_dir, "rlmesh")
        try:
            os.makedirs(base, mode=0o700, exist_ok=True)
            return os.path.join(base, filename)
        except OSError:
            pass

    private_dir = tempfile.mkdtemp(prefix="rlmesh-")
    return os.path.join(private_dir, filename)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
