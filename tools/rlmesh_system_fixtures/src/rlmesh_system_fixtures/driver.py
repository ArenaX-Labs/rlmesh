from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rlmesh_system_fixtures.registry import resolve_model
from rlmesh_system_fixtures.trace import canonical_info, fingerprint


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RLMesh system fixture drivers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    trace_parser = subparsers.add_parser("trace", help="run a traced env/model loop")
    trace_parser.add_argument("--scenario", required=True)
    trace_parser.add_argument("--address", required=True)
    trace_parser.add_argument(
        "--client", choices=["native", "numpy", "torch"], required=True
    )
    trace_parser.add_argument(
        "--model", help="Registered fixture model to run in-process"
    )
    trace_parser.add_argument(
        "--model-address",
        help="Drive the loop through a model served at this address instead of --model",
    )
    trace_parser.add_argument("--seed", type=int)
    trace_parser.add_argument("--steps", type=int, required=True)
    trace_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "trace":
        return run_trace(args)
    raise AssertionError(f"unhandled command {args.command!r}")


def run_trace(args: argparse.Namespace) -> int:
    if (args.model is None) == (args.model_address is None):
        raise SystemExit("pass exactly one of --model and --model-address")
    remote = remote_env(args.client, args.address)
    # A served model drives the same loop through its session, so the
    # cross-version matrix's model leg records the one trace shape too.
    session = model_session(args.model_address, remote) if args.model_address else None
    loop = remote if session is None else session
    model = resolve_model(args.model) if session is None else session.predict
    trace: dict[str, Any] = {
        "schema_version": 1,
        "scenario": args.scenario,
        "client": args.client,
        "seed": args.seed,
        "steps": [],
    }
    try:
        observation, info = loop.reset(seed=args.seed)
        trace["reset"] = {
            "observation": fingerprint(observation),
            "info": canonical_info(info),
        }
        for index in range(args.steps):
            action = model(observation)
            observation, reward, terminated, truncated, info = loop.step(action)
            trace["steps"].append(
                {
                    "index": index,
                    "action": fingerprint(action),
                    "observation": fingerprint(observation),
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": canonical_info(info),
                }
            )
            if terminated or truncated:
                break
        if session is None:
            remote.shutdown(f"fixture trace {args.scenario} complete")
    finally:
        if session is not None:
            session.close()
        remote.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
    print(f"trace={args.output}")
    return 0


def model_session(address: str, remote: Any) -> Any:
    import rlmesh

    return rlmesh.RemoteModel(address).session(remote)


def remote_env(client: str, address: str) -> Any:
    if client == "native":
        import rlmesh

        return rlmesh.RemoteEnv(address)
    if client == "numpy":
        from rlmesh import numpy as rlmesh_numpy

        return rlmesh_numpy.RemoteEnv(address)
    if client == "torch":
        from rlmesh import torch as rlmesh_torch

        return rlmesh_torch.RemoteEnv(address)
    raise ValueError(f"unknown client {client!r}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
