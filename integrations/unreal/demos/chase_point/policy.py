"""Scripted toward-goal policy for the Unreal Chase Point env (XY)."""

from __future__ import annotations


def gap_xy(obs: object):
    """Remaining Goal - Agent as a length-2 float32 vector."""
    import numpy as np

    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"expected shape (2,) observation, got {getattr(obs, 'shape', obs)!r}")
    return arr[:2]


def toward_goal(obs: object):
    """Unit XY direction from Agent toward Goal."""
    import numpy as np

    gap = gap_xy(obs)
    norm = float(np.linalg.norm(gap))
    if norm < 1e-6:
        return np.zeros(2, dtype=np.float32)
    return (gap / norm).astype(np.float32)
