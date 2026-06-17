"""BYO model container entrypoint: serve a policy as an RLMesh model endpoint.

You write a Dockerfile (see the sibling ``Dockerfile``) and this entrypoint, and
RLMesh runs the image. The container serves the policy on
``RLMESH_ADDRESS`` (default ``0.0.0.0:50051``); a client drives it with the
symmetric loop -- ``rlmesh.RemoteModel(address).against(env)`` un-managed, or
``rlmesh.SandboxModel("image://<tag>").against(env)`` managed -- and RLMesh
Managed runs the same image. ``docker push`` the tag to a registry it can reach.
"""

from __future__ import annotations

import os

from rlmesh.numpy import Model


def load_policy():
    # Replace with your real policy load (weights via a baked path or mount, the
    # framework's from_pretrained, etc.). Keep heavy imports inside here.
    def predict(observation: object) -> int:
        return 0  # always push the cart left

    return predict


def main() -> None:
    address = os.environ.get("RLMESH_ADDRESS", "0.0.0.0:50051")
    print(f"RLMesh BYO model serving {address}", flush=True)
    Model(load_policy()).serve(address)


if __name__ == "__main__":
    main()
