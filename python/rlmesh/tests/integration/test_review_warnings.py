"""Native engine warnings retain their Python category and caller frame."""

import inspect
import warnings

import numpy as np
import pytest
import rlmesh
from rlmesh import adapters as adapt
from rlmesh import spaces
from rlmesh.numpy import Model


class TaggedEnv:
    observation_space = spaces.Dict({"state": spaces.Box(-1, 1, [3])})
    action_space = spaces.Box(-1, 1, [3])
    tags = adapt.EnvTags(
        observation={"state": adapt.StateTag(role=adapt.EEF_POS)},
        action=adapt.Action(adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3)),
    )

    def reset(self, *, seed=None, options=None):
        return {"state": [0, 0, 0]}, {}

    def step(self, action):
        return {"state": [0, 0, 0]}, 0.0, True, False, {}

    def close(self):
        pass


class ShortChunk(Model):
    spec = adapt.ModelSpec(
        input={"state": adapt.State(role=adapt.EEF_POS)},
        output=adapt.Action(adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3)),
    )

    def predict(self, obs):
        return np.zeros(3, np.float32)

    def predict_chunk(self, obs):
        return np.zeros((1, 3), np.float32)


@pytest.mark.parametrize("as_error", [False, True])
def test_engine_short_chunk_warning_keeps_category_and_call_site(as_error):
    server = rlmesh.EnvServer(TaggedEnv(), "127.0.0.1:0", tags=TaggedEnv.tags)
    server.start()
    model = ShortChunk()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("error" if as_error else "always", RuntimeWarning)
            if as_error:
                with pytest.raises(RuntimeWarning, match="returned 1 actions"):
                    model._run_local_for_episodes(
                        server.address, episodes=1, execution_horizon=2
                    )
            else:
                frame = inspect.currentframe()
                line = frame.f_lineno + 1
                model._run_local_for_episodes(
                    server.address, episodes=1, execution_horizon=2
                )
                warning = next(
                    w for w in caught if issubclass(w.category, RuntimeWarning)
                )
                assert warning.filename == __file__
                assert warning.lineno == line
    finally:
        model.close()
        server.shutdown()
