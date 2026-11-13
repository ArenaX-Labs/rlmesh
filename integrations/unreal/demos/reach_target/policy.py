"""Scripted toward-target policy for the Unreal Reach Target env."""

from __future__ import annotations


def gap_xyz(obs: object):
    """Remaining Target - Agent as a length-3 float32 vector."""
    import numpy as np

    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        raise ValueError(f"expected shape (3,) observation, got {getattr(obs, 'shape', obs)!r}")
    return arr[:3]


def toward_target(obs: object):
    """Unit direction from Agent toward Target (XYZ)."""
    import numpy as np

    gap = gap_xyz(obs)
    norm = float(np.linalg.norm(gap))
    if norm < 1e-6:
        return np.zeros(3, dtype=np.float32)
    return (gap / norm).astype(np.float32)
