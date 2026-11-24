from __future__ import annotations

import os
import sys

from rlmesh.numpy import Model
from rlmesh.serve import serve_model

MARKER = "rlmesh-docker-gate: predict failed on purpose"


class Raising(Model):
    def predict(self, observation: object) -> int:
        print(MARKER, file=sys.stderr, flush=True)
        raise AssertionError


if __name__ == "__main__":
    serve_model(
        Raising,
        os.environ.get("RLMESH_ADDRESS", "0.0.0.0:7447"),
        name=os.environ.get("RLMESH_NAME", "model"),
    )
