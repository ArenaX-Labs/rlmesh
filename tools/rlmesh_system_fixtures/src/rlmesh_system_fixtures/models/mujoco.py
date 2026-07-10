from __future__ import annotations


def halfcheetah_zero_numpy(observation: object) -> object:
    _ = observation
    import numpy as np

    return np.zeros((6,), dtype=np.float32)
