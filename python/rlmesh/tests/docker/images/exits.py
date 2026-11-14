"""An env container that exits during startup, as one whose make() fails does."""

from __future__ import annotations

import os

from rlmesh import EnvFactory

MARKER = "rlmesh-docker-gate: make() failed on purpose"


class Failing(EnvFactory):
    def make(self, **kwargs: object) -> object:
        raise RuntimeError(MARKER)


if __name__ == "__main__":
    Failing().serve(f"{os.environ['RLMESH_ADDRESS']}/{os.environ['RLMESH_NAME']}")
