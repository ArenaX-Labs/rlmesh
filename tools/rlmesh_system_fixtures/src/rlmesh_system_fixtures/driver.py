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
    trace_parser.add_argument("--model", required=True)
    trace_parser.add_argument("--seed", type=int)
    trace_parser.add_argument("--steps", type=int, required=True)
    trace_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "trace":
        return run_trace(args)
    raise AssertionError(f"unhandled command {args.command!r}")


def run_trace(args: argparse.Namespace) -> int:
    remote = remote_env(args.client, args.address)
    model = resolve_model(args.model)
    trace: dict[str, Any] = {
        "schema_version": 1,
        "scenario": args.scenario,
        "client": args.client,
        "seed": args.seed,
        "steps": [],
    }
    try:
        observation, info = remote.reset(seed=args.seed)
        trace["reset"] = {
            "observation": fingerprint(observation),
            "info": canonical_info(info),
        }
        for index in range(args.steps):
            action = model(observation)
            observation, reward, terminated, truncated, info = remote.step(action)
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
        remote.shutdown(f"fixture trace {args.scenario} complete")
    finally:
        remote.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
    print(f"trace={args.output}")
    return 0


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
