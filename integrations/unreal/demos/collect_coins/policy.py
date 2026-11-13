"""Scripted policy for Collect Coins: always move toward nearest remaining coin."""

from __future__ import annotations


def parse_obs(obs: object):
    """Return (nearest_coin_gap[2], coins_left_norm)."""
    import numpy as np

    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        raise ValueError(
            f"expected shape (3,) observation [nearest_xy, coins_left_norm], "
            f"got {getattr(obs, 'shape', obs)!r}"
        )
    return arr[:2], float(arr[2])


def toward_nearest_coin(obs: object):
    """Unit XY toward the nearest remaining coin (zeros when none left)."""
    import numpy as np

    gap, _coins_left = parse_obs(obs)
    n = float(np.linalg.norm(gap))
    if n < 1e-6:
        return np.zeros(2, dtype=np.float32)
    return (gap / n).astype(np.float32)
