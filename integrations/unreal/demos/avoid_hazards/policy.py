"""Scripted policy for Avoid Hazards: toward goal with hazard repulsion."""

from __future__ import annotations


def parse_obs(obs: object, *, num_hazards: int = 4):
    """Return (goal_gap[2], hazard_gaps[N,2])."""
    import numpy as np

    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    need = 2 + 2 * num_hazards
    if arr.size < need:
        # Infer hazard count from obs length when possible.
        if arr.size >= 2 and (arr.size - 2) % 2 == 0:
            num_hazards = (arr.size - 2) // 2
            need = arr.size
        else:
            raise ValueError(
                f"expected shape (2+2N,) observation, got {getattr(obs, 'shape', obs)!r}"
            )
    goal = arr[:2]
    hazards = arr[2:need].reshape(num_hazards, 2)
    return goal, hazards


def toward_goal_avoid(obs: object, *, num_hazards: int = 4, avoid_radius: float = 220.0):
    """Unit XY: attraction to goal + repulsion from nearby hazards."""
    import numpy as np

    goal, hazards = parse_obs(obs, num_hazards=num_hazards)
    pull = goal.astype(np.float64)
    pull_norm = float(np.linalg.norm(pull))
    if pull_norm > 1e-6:
        pull = pull / pull_norm
    else:
        pull = np.zeros(2, dtype=np.float64)

    push = np.zeros(2, dtype=np.float64)
    for gap in hazards:
        dist = float(np.linalg.norm(gap))
        if dist < 1e-6 or dist > avoid_radius:
            continue
        # Away from hazard (gap is hazard - agent, so -gap is away).
        strength = (avoid_radius - dist) / avoid_radius
        push -= (gap / dist) * strength

    push_norm = float(np.linalg.norm(push))
    if push_norm > 1e-6:
        push = push / push_norm

    combined = pull + 1.25 * push
    cnorm = float(np.linalg.norm(combined))
    if cnorm < 1e-6:
        return np.zeros(2, dtype=np.float32)
    return (combined / cnorm).astype(np.float32)
