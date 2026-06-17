"""BYO env container entrypoint: serve a Gymnasium env over the RLMesh protocol.

You write a Dockerfile (see the sibling ``Dockerfile``) and this entrypoint, and
RLMesh runs the image. The container serves the env on
``RLMESH_ADDRESS`` (default ``0.0.0.0:50051``); a client dials it with
``rlmesh.RemoteEnv(address)``, and the RLMesh Managed platform runs the same
image.
"""

from __future__ import annotations

import os

import gymnasium as gym
from rlmesh import EnvServer


def make_env() -> gym.Env:
    # Replace with your environment construction (wrappers, render backend, ...).
    return gym.make("CartPole-v1")


def main() -> None:
    address = os.environ.get("RLMESH_ADDRESS", "0.0.0.0:50051")
    print(f"RLMesh BYO env serving {address}", flush=True)
    EnvServer(make_env(), address).serve()


if __name__ == "__main__":
    main()
