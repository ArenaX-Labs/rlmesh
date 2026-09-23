"""Serve a one-lane vectorized env as the scalar env it is.

The contract's ``num_envs`` is the only vector signal on the wire, so a batched
simulator configured with one lane (Isaac Lab at ``num_envs=1``) cannot be
served as a vector. :class:`SingleLaneEnv` drops the batch axis instead, and the
runtime drives it like any scalar env: driver-owned resets, so per-episode
seeds, ``trial_index`` and ``max_episode_steps`` all apply.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

_AUTORESET_KEY = "autoreset_mode"


def _mode_name(mode: object) -> str:
    return str(getattr(mode, "name", mode)).upper()


class SingleLaneEnv:
    """A scalar view of a vectorized env with ``num_envs == 1``.

    Every leaf loses (observations, info) or gains (actions) its leading batch
    axis. The runtime owns resets: after an episode ends it calls ``reset()``,
    which also clears a NEXT_STEP env's pending autoreset, so the lane is never
    reset twice.
    """

    def __init__(self, env: Any):
        metadata: dict[str, Any] = dict(getattr(env, "metadata", None) or {})
        mode = metadata.get(_AUTORESET_KEY)
        if mode is not None and _mode_name(mode) == "SAME_STEP":
            raise ValueError(
                "a one-lane vector env with SAME_STEP autoreset returns the next "
                "episode's first observation in place of the terminal one; "
                "construct it with NEXT_STEP autoreset"
            )
        self.env = env
        self.observation_space = env.single_observation_space
        self.action_space = env.single_action_space
        self.metadata = {k: v for k, v in metadata.items() if k != _AUTORESET_KEY}

    def __getattr__(self, name: str) -> Any:
        # Hide the vector shape, or the server would detect a vector again.
        if name == "num_envs" or name.startswith("single_"):
            raise AttributeError(name)
        return getattr(self.env, name)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return _lane0(self.observation_space, obs), _info0(info)

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(
            _batch1(self.action_space, action)
        )
        return (
            _lane0(self.observation_space, obs),
            float(reward[0]),
            bool(terminated[0]),
            bool(truncated[0]),
            _info0(info),
        )

    def render(self) -> Any:
        frames = self.env.render()
        return frames[0] if frames is not None else None

    def close(self) -> None:
        self.env.close()


def _lane0(space: Any, value: Any) -> Any:
    """Lane 0 of a batched value, walking the per-lane ``space``'s structure."""
    children = _children(space)
    if isinstance(children, Mapping):
        return {key: _lane0(sub, value[key]) for key, sub in children.items()}
    if isinstance(children, Sequence):
        return tuple(
            _lane0(sub, part) for sub, part in zip(children, value, strict=True)
        )
    return value[0]


def _batch1(space: Any, value: Any) -> Any:
    """A one-lane batch of ``value``, walking the per-lane ``space``'s structure."""
    children = _children(space)
    if isinstance(children, Mapping):
        return {key: _batch1(sub, value[key]) for key, sub in children.items()}
    if isinstance(children, Sequence):
        return tuple(
            _batch1(sub, part) for sub, part in zip(children, value, strict=True)
        )
    return value[None] if hasattr(value, "shape") else [value]


def _children(space: Any) -> Mapping[str, Any] | Sequence[Any] | None:
    """A Dict space's named children, a Tuple space's ordered ones, or None."""
    return cast(
        "Mapping[str, Any] | Sequence[Any] | None", getattr(space, "spaces", None)
    )


def _info0(info: Mapping[str, Any]) -> dict[str, Any]:
    """Lane 0 of a vector info (gymnasium's ``key`` array plus ``_key`` mask)."""
    lane: dict[str, Any] = {}
    for key, value in info.items():
        if key.startswith("_"):
            continue
        mask = info.get(f"_{key}")
        if mask is not None and not mask[0]:
            continue
        if isinstance(value, Mapping):
            lane[key] = _info0(cast("Mapping[str, Any]", value))
        elif (
            not isinstance(value, (str, bytes))
            and hasattr(value, "__len__")
            and len(value) == 1
        ):
            lane[key] = value[0]
        else:
            lane[key] = value
    return lane
