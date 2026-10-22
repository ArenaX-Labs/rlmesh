"""BYO model container entrypoint: serve a policy on an RLMesh session.

You write a Dockerfile (see the sibling ``Dockerfile``) and this entrypoint, and
RLMesh runs the image. The container hosts a session on ``RLMESH_ADDRESS``
(default ``0.0.0.0:7447``) and serves the policy in it as ``RLMESH_NAME``
(default ``model``); a client runs it with
``rlmesh.Model("<host>:7447/model").run(env)`` un-managed, or
``rlmesh.SandboxModel("image://<tag>").run(env)`` managed -- and RLMesh Managed
runs the same image. ``docker push`` the tag to a registry it can reach.

The lazy path: skip this file entirely and set the Dockerfile ENTRYPOINT to
``python -m rlmesh.serve my_pkg:MyPolicy`` -- it serves the MyPolicy model below
with no hand-written serve loop.
"""

from __future__ import annotations

import os

from rlmesh.numpy import Model


class MyPolicy(Model):
    """Subclass ``rlmesh.numpy.Model``: set ``spec`` (optional), implement load + predict."""

    # spec = ModelSpec(...)  # declare inputs/outputs to get automatic adapters.

    def load(self) -> None:
        # Load weights INTO self (a baked path or mount, from_pretrained, ...).
        # Keep heavy imports inside here, not at module top.
        self.bias = 0

    def predict(self, observation: object) -> int:
        return self.bias  # always push the cart left


def main() -> None:
    address = os.environ.get("RLMESH_ADDRESS", "0.0.0.0:7447")
    name = os.environ.get("RLMESH_NAME", "model")
    print(f"RLMesh BYO model serving {address}/{name}", flush=True)
    # Equivalent one-liner: `python -m rlmesh.serve <this_module>:MyPolicy`.
    MyPolicy().serve(address, name=name)


if __name__ == "__main__":
    main()
