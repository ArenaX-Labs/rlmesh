from __future__ import annotations


def pendulum_zero_numpy(observation: object) -> object:
    _ = observation
    import numpy as np

    return np.zeros((1,), dtype=np.float32)
