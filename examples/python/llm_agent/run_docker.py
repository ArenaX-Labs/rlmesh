"""Run the harness image against an env image, both as RLMesh sandboxes.

Both write under one host directory (``--output``, default
``./runs/<timestamp>``), each container mounting only its own part: the
harness ``transcripts/``, the env ``env/``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import rlmesh
from harness import LLMHarness

HERE = Path(__file__).resolve().parent


def build(dockerfile: str, tag: str, build_args: list[str]) -> None:
    """Build ``docker/<dockerfile>`` from this directory as ``tag``."""
    flags = [flag for arg in build_args for flag in ("--build-arg", arg)]
    subprocess.run(
        ["docker", "build", "-f", f"docker/{dockerfile}", "-t", tag, *flags, "."],
        cwd=HERE,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the harness image.")
    parser.add_argument("--env-image", help="env image tag (default llm-env-<build>)")
    parser.add_argument("--harness-image", default="llm-harness", help="model tag")
    parser.add_argument(
        "--build",
        choices=["v0", "v1"],
        help="build the harness and env-<v0|v1> images under those tags first",
    )
    parser.add_argument(
        "--build-arg", action="append", default=[], help="KEY=VALUE for the env build"
    )
    parser.add_argument(
        "--local-harness",
        action="store_true",
        help="run the harness here, not in its image (Docker Desktop: see README)",
    )
    parser.add_argument("-n", type=int, default=1, help="episodes (seeds 0..n-1)")
    parser.add_argument("--output", type=Path, help="host output directory")
    args = parser.parse_args()

    env_image = args.env_image or f"llm-env-{args.build or 'v0'}"
    if args.build:
        build("harness.Dockerfile", args.harness_image, [])
        build(f"env-{args.build}.Dockerfile", env_image, args.build_arg)
    output = (args.output or Path(f"runs/{time.strftime('%Y%m%d-%H%M%S')}")).resolve()
    output.mkdir(parents=True, exist_ok=True)

    def runtime(role: str, part: str) -> rlmesh.SandboxRuntime:
        (output / part).mkdir(exist_ok=True)
        return rlmesh.SandboxRuntime(
            volumes=[f"{output / part}:/rlmesh/output/{role}/{part}"], user=os.getuid()
        )

    # Loading a taskset or environment can download its dataset first.
    env = rlmesh.SandboxEnv(
        f"image://{env_image}",
        runtime=runtime("env", "env"),
        connect_timeout_seconds=600,
    )
    seeds = list(range(args.n))
    try:
        if args.local_harness:
            os.environ["RLMESH_OUTPUT_DIR"] = str(output)
            result = LLMHarness().run(env, seeds=seeds)
        else:
            with rlmesh.SandboxModel(
                f"image://{args.harness_image}", runtime=runtime("model", "transcripts")
            ) as model:
                result = model.run(env, seeds=seeds)
    finally:
        env.close()
    for episode in result.episodes:
        print(
            f"seed={episode.seed} steps={episode.steps} reward={episode.reward:.3f} "
            f"success={episode.success} truncated={episode.truncated}"
        )
    print(f"mean_reward={result.mean_reward:.3f} output={output}")


if __name__ == "__main__":
    main()
