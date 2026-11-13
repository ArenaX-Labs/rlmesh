"""A local env served as a lockstep robot for session.test.mjs's lockstep test.

Two steps an episode, a reward of 1 a step. It declares the ``trial_index``
reset option and prints one JSON line per reset (its seed and trial index),
after a first line saying it serves. It serves at the participant address
given as its argument until its stdin closes.
"""

from __future__ import annotations

import json
import sys
from typing import Any, ClassVar

import numpy as np
import rlmesh
from rlmesh import spaces


class Trials:
    """Two steps an episode; prints each reset's seed and ``trial_index``."""

    metadata: ClassVar[dict[str, Any]] = {rlmesh.ENV_RESET_OPTIONS_KEY: ["trial_index"]}

    def __init__(self) -> None:
        self.observation_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.t = 0

    def reset(self, *, seed: int | None = None, options: Any = None) -> Any:
        """Print this reset's seed and trial index, then start an episode."""
        trial = rlmesh.trial_index(options)
        print(json.dumps({"seed": seed, "trial": trial}), flush=True)
        self.t = 0
        return np.zeros(2, np.float32), {}

    def step(self, action: Any) -> Any:
        """One step: a reward of 1; the second ends the episode."""
        self.t += 1
        return np.zeros(2, np.float32), 1.0, self.t >= 2, False, {}

    def close(self) -> None:
        """Nothing to release."""


def main() -> None:
    """Serve at ``argv[1]`` until stdin closes."""
    server = rlmesh.EnvServer(Trials(), sys.argv[1])
    server.start()
    print(json.dumps({"serving": True}), flush=True)
    try:
        sys.stdin.read()
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
