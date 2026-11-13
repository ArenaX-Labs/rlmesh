"""Scripted Demo Arena policy: collect every coin, then the goal last."""

from __future__ import annotations

import math

# Defaults must match URLMeshDemoArenaComponent (NumWallRays / WallSenseRange).
DEFAULT_NUM_COINS = 3
# Level may have 0 obstacles; obs still pads to NumObstacles slots (often zeros).
DEFAULT_NUM_OBSTACLES = 1
DEFAULT_NUM_WALL_RAYS = 8
DEFAULT_WALL_SENSE_RANGE = 350.0

# Must exceed env StepSize (50) so a "clear" gap is actually traversable.
CLEAR = 100.0

_hug = None
_hug_steps: int = 0
_open_streak: int = 0
_last_target_dist: float | None = None


def reset_arena_policy_state() -> None:
    """Clear episodic policy state."""
    global _hug, _hug_steps, _open_streak, _last_target_dist
    _hug = None
    _hug_steps = 0
    _open_streak = 0
    _last_target_dist = None


def parse_obs(
    obs: object,
    *,
    num_coins: int = DEFAULT_NUM_COINS,
    num_obstacles: int = DEFAULT_NUM_OBSTACLES,
    num_wall_rays: int = DEFAULT_NUM_WALL_RAYS,
):
    """Return (goal[2], coins[N,2], obstacles[M,2], walls[W]).

    Goal/coin gaps are zeros once collected.
    """
    import numpy as np

    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    need = 2 + 2 * num_coins + 2 * num_obstacles + num_wall_rays
    if arr.size != need:
        rest = arr.size - 2
        if arr.size >= 2 and rest >= 0:
            body = 2 * num_coins + 2 * num_obstacles
            if rest >= body:
                num_wall_rays = rest - body
            elif rest % 2 == 0:
                pairs = rest // 2
                if pairs >= num_coins:
                    num_obstacles = min(num_obstacles, pairs - num_coins)
                    num_wall_rays = rest - 2 * num_coins - 2 * num_obstacles
                else:
                    num_coins = pairs
                    num_obstacles = 0
                    num_wall_rays = 0
            else:
                raise ValueError(
                    f"expected shape (2+2C+2O+W,), got {getattr(obs, 'shape', obs)!r}"
                )
            need = arr.size
        else:
            raise ValueError(
                f"expected shape (2+2C+2O+W,), got {getattr(obs, 'shape', obs)!r}"
            )
    goal = arr[:2]
    coins = arr[2 : 2 + 2 * num_coins].reshape(num_coins, 2)
    base_o = 2 + 2 * num_coins
    obstacles = arr[base_o : base_o + 2 * num_obstacles].reshape(num_obstacles, 2)
    base_w = base_o + 2 * num_obstacles
    walls = arr[base_w : base_w + num_wall_rays].astype(np.float64)
    return goal, coins, obstacles, walls


def nearest_target_gap(goal, coins):
    """Nearest remaining coin; goal only after every coin is collected."""
    import numpy as np

    best = None
    best_d = 1e18
    for gap in coins:
        dist = float(np.linalg.norm(gap))
        if dist < 1e-6:
            continue
        if dist < best_d:
            best_d = dist
            best = gap
    if best is not None:
        return best.astype(np.float64), best_d
    gdist = float(np.linalg.norm(goal))
    if gdist > 1e-6:
        return goal.astype(np.float64), gdist
    return np.zeros(2, dtype=np.float64), 0.0


def _ray_dirs(n: int):
    import numpy as np

    if n <= 0:
        return np.zeros((0, 2), dtype=np.float64)
    angles = (2.0 * math.pi) * np.arange(n, dtype=np.float64) / float(n)
    return np.stack((np.cos(angles), np.sin(angles)), axis=1)


def _clearance_along(direction, walls) -> float:
    import numpy as np

    n = int(walls.shape[0])
    if n == 0:
        return float("inf")
    dn = float(np.linalg.norm(direction))
    if dn < 1e-6:
        return float(np.min(walls))
    ang = math.atan2(float(direction[1]), float(direction[0]))
    if ang < 0.0:
        ang += 2.0 * math.pi
    slot = ang / (2.0 * math.pi) * n
    i0 = int(math.floor(slot)) % n
    i1 = (i0 + 1) % n
    t = slot - math.floor(slot)
    return float((1.0 - t) * walls[i0] + t * walls[i1])


def _unit(v):
    import numpy as np

    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.zeros(2, dtype=np.float64)
    return np.asarray(v, dtype=np.float64) / n


def _blocking_normal(pull, walls, ray_dirs, clear: float):
    import numpy as np

    best_i = None
    best_d = 1e9
    for i, dist in enumerate(walls):
        d = float(dist)
        if d >= clear:
            continue
        if float(np.dot(ray_dirs[i], pull)) < 0.15:
            continue
        if d < best_d:
            best_d = d
            best_i = i
    if best_i is None and walls.size:
        best_i = int(np.argmin(walls))
    if best_i is None:
        return None
    return ray_dirs[int(best_i)]


def _hug_tangent(normal, pull, walls, obstacles=None, avoid_radius: float = 180.0):
    import numpy as np

    left = np.array([-normal[1], normal[0]], dtype=np.float64)
    right = -left
    left_c = _clearance_along(left, walls)
    right_c = _clearance_along(right, walls)
    left_score = left_c + 80.0 * float(np.dot(left, pull))
    right_score = right_c + 80.0 * float(np.dot(right, pull))

    def _obstacle_risk(direction) -> float:
        risk = 0.0
        if obstacles is None:
            return risk
        for gap in obstacles:
            dist = float(np.linalg.norm(gap))
            if dist < 1e-6 or dist > avoid_radius:
                continue
            toward = float(np.dot(direction, _unit(gap)))
            if toward > 0.3:
                risk += toward * ((avoid_radius - dist) / avoid_radius) * 120.0
        return risk

    left_score -= _obstacle_risk(left)
    right_score -= _obstacle_risk(right)
    return left if left_score >= right_score else right


def toward_goal_arena(
    obs: object,
    *,
    num_coins: int = DEFAULT_NUM_COINS,
    num_obstacles: int = DEFAULT_NUM_OBSTACLES,
    num_wall_rays: int = DEFAULT_NUM_WALL_RAYS,
    wall_sense_range: float = DEFAULT_WALL_SENSE_RANGE,
    avoid_radius: float = 160.0,
    clear: float = CLEAR,
):
    """Unit XY toward the nearest remaining coin/goal; hug walls when blocked."""
    import numpy as np

    global _hug, _hug_steps, _open_streak, _last_target_dist
    _ = wall_sense_range  # reserved for policies that scale CLEAR by sense range

    goal, coins, obstacles, walls = parse_obs(
        obs,
        num_coins=num_coins,
        num_obstacles=num_obstacles,
        num_wall_rays=num_wall_rays,
    )
    ray_dirs = _ray_dirs(int(walls.shape[0]))
    target, td = nearest_target_gap(goal, coins)
    if td < 1e-6:
        return np.zeros(2, dtype=np.float32)

    pull = _unit(target)

    if _last_target_dist is not None and abs(td - _last_target_dist) > 400.0:
        _hug = None
        _hug_steps = 0
        _open_streak = 0
    _last_target_dist = td

    target_clear = _clearance_along(pull, walls) if walls.size else float("inf")

    push = np.zeros(2, dtype=np.float64)
    if td >= 120.0:
        for gap in obstacles:
            dist = float(np.linalg.norm(gap))
            limit = min(avoid_radius, max(td * 0.4, 80.0))
            if dist < 1e-6 or dist > limit:
                continue
            push -= _unit(gap) * ((limit - dist) / limit)

    # Close + clear shot: drive straight into the collectible.
    if td < 140.0 and target_clear >= 50.0:
        _hug = None
        _hug_steps = 0
        return pull.astype(np.float32)

    if target_clear >= clear:
        _open_streak += 1
        if _open_streak >= 6:
            _hug = None
            _hug_steps = 0
        choice = pull + 0.45 * _unit(push)
        if float(np.dot(_unit(choice), pull)) < 0.3:
            choice = pull
        choice = _unit(choice)
    else:
        _open_streak = 0
        normal = _blocking_normal(pull, walls, ray_dirs, clear)
        if normal is None:
            choice = pull
        else:
            tangent = _hug_tangent(
                normal, pull, walls, obstacles=obstacles, avoid_radius=avoid_radius
            )
            if td < 200.0 or target_clear < 45.0 or _hug is None or _hug_steps >= 50:
                _hug = tangent.copy()
                _hug_steps = 0
            elif float(np.dot(_hug, tangent)) < 0.0:
                if _clearance_along(_hug, walls) < clear * 0.55:
                    _hug = tangent.copy()
                    _hug_steps = 0
            else:
                _hug = _unit(0.7 * _hug + 0.3 * tangent)
            _hug_steps += 1
            choice = _unit(_hug + 0.45 * (-normal) + 0.2 * _unit(push))

    if walls.size:
        for i, dist in enumerate(walls):
            if float(dist) >= 65.0:
                continue
            normal = ray_dirs[i]
            into = float(np.dot(choice, normal))
            if into > 0.0:
                choice = choice - into * normal
        if float(np.linalg.norm(choice)) < 0.15:
            i = int(np.argmin(walls))
            normal = ray_dirs[i]
            choice = -normal + _hug_tangent(
                normal, pull, walls, obstacles=obstacles, avoid_radius=avoid_radius
            )

    # Soft dodge only when a real obstacle gap is present (zeros = empty slot).
    for gap in obstacles:
        dist = float(np.linalg.norm(gap))
        if dist < 1e-6 or dist > 160.0:
            continue
        toward = float(np.dot(_unit(choice), _unit(gap)))
        if toward > 0.35:
            choice = choice - _unit(gap) * (1.0 + toward) * ((160.0 - dist) / 160.0)
    if float(np.linalg.norm(choice)) < 0.15:
        choice = pull

    return _unit(choice).astype(np.float32)


# Back-compat alias used by app.py / docs.
toward_nearest_collectible = toward_goal_arena
