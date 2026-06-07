from __future__ import annotations

from typing import Any

from rlmesh_system_fixtures.registry import env_fixture


class ImageGridEnv:
    def __init__(self, width: int = 4, height: int = 3, channels: int = 3) -> None:
        from rlmesh import spaces

        self.width = width
        self.height = height
        self.channels = channels
        self.step_count = 0
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(0.0, 16.0, shape=[4], dtype="float32"),
                "pixels": spaces.Box(
                    0.0,
                    255.0,
                    shape=[height, width, channels],
                    dtype="uint8",
                ),
            }
        )
        self.action_space = spaces.Discrete(2)

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[dict[str, Any], dict[str, object]]:
        self.step_count = 0
        return self._observation(), {"seed": seed, "options": options}

    def step(
        self, action: object
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, object]]:
        self.step_count += 1
        reward = 0.25 * self.step_count
        return (
            self._observation(),
            reward,
            self.step_count >= 2,
            False,
            {"action": action},
        )

    def close(self) -> None:
        pass

    def _observation(self) -> dict[str, Any]:
        import numpy as np

        pixels = np.arange(
            self.height * self.width * self.channels, dtype=np.uint8
        ).reshape(self.height, self.width, self.channels)
        pixels = (pixels + self.step_count).astype(np.uint8, copy=False)
        state = np.array(
            [self.step_count, self.width, self.height, self.channels],
            dtype=np.float32,
        )
        return {"state": state, "pixels": pixels}


@env_fixture("image-grid")
def make_image_grid(width: int = 4, height: int = 3, channels: int = 3) -> ImageGridEnv:
    return ImageGridEnv(width=width, height=height, channels=channels)
