"""BYO env container entrypoint: serve a Gymnasium env over the RLMesh protocol.

You write a Dockerfile (see the sibling ``Dockerfile``) and this entrypoint, and
RLMesh runs the image. The container serves the env on
``RLMESH_ADDRESS`` (default ``0.0.0.0:50051``); a client dials it with
``rlmesh.RemoteEnv(address)``, and the RLMesh Managed platform runs the same
image.

The lazy path: set the Dockerfile ENTRYPOINT to
``python -m rlmesh.serve --env my_pkg:MyEnv`` -- it runs the MyEnv factory below
(``prepare`` then ``make``, tagging with ``tags``) with no hand-written loop.
"""

from __future__ import annotations

import os

import gymnasium as gym
from rlmesh import EnvFactory, EnvServer


class MyEnv(EnvFactory):
    """Subclass EnvFactory: set ``tags`` (optional), implement make (+ prepare)."""

    # tags = EnvTags(observation=..., action=...)  # describe obs/action for adapters.

    def prepare(self) -> None:
        # Optional one-time setup before make() (download assets, start a sim, ...).
        pass

    def make(self, **kwargs: object) -> gym.Env:
        # Replace with your environment construction (wrappers, render backend, ...).
        return gym.make("CartPole-v1")


def main() -> None:
    address = os.environ.get("RLMESH_ADDRESS", "0.0.0.0:50051")
    print(f"RLMesh BYO env serving {address}", flush=True)
    # Equivalent one-liner: `python -m rlmesh.serve --env <this_module>:MyEnv`.
    recipe = MyEnv()
    recipe.prepare()
    EnvServer(recipe.make(), address, tags=recipe.tags).serve()


if __name__ == "__main__":
    main()
