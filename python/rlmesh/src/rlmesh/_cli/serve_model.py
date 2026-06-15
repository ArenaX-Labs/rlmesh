"""Python-backed model worker for the Rust RLMesh CLI."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from rlmesh._bootstrap.loaders import load_predict
from rlmesh._entrypoint import parse_entrypoint

__all__ = [
    "ServeModelArgs",
    "create_parser",
    "load_predict",
    "main",
    "parse_entrypoint",
    "serve_from_args",
]


@dataclass
class ServeModelArgs:
    model: str
    address: str
    token: str
    verbose: bool


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rlmesh._cli.serve_model",
        description="Serve a Python model callable as an RLMesh model endpoint",
    )
    _ = parser.add_argument(
        "--model", required=True, help="Model entrypoint in module:callable form"
    )
    _ = parser.add_argument(
        "--address",
        required=True,
        help="Model endpoint bind address (host:port or tcp://...)",
    )
    _ = parser.add_argument("--token", default="", help="Session token")
    _ = parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    ns = parser.parse_args(argv)
    args = ServeModelArgs(
        model=_namespace_str(ns, "model"),
        address=_namespace_str(ns, "address"),
        token=_namespace_str(ns, "token"),
        verbose=_namespace_bool(ns, "verbose"),
    )
    return serve_from_args(args)


def serve_from_args(args: ServeModelArgs) -> int:
    try:
        from rlmesh.numpy import Model

        predict_fn = load_predict(args.model)
        worker = Model(predict_fn)
        print(f"✓ Model entrypoint: {args.model}")
        print(f"✓ Model endpoint: {args.address}")
        print("Serving model endpoint...")
        worker.serve(args.address, token=args.token)
        print("Model endpoint stopped")
        return 0
    except KeyboardInterrupt:
        print("\nStopping model bridge")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def _namespace_str(args: argparse.Namespace, name: str) -> str:
    value: object = vars(args).get(name)
    if not isinstance(value, str):
        raise TypeError(f"expected argparse field {name!r} to be a str")
    return value


def _namespace_bool(args: argparse.Namespace, name: str) -> bool:
    value: object = vars(args).get(name)
    if not isinstance(value, bool):
        raise TypeError(f"expected argparse field {name!r} to be a bool")
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
