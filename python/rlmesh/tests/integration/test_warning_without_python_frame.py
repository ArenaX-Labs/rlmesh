"""A warning delivered on a runtime thread, which has no Python frame, must not crash."""

import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(
    """
    import sys
    import threading
    import time

    import numpy as np
    from rlmesh import adapters as adapt
    from rlmesh import spaces
    from rlmesh.numpy import Model

    # Any audit hook: `filelock`, which `transformers` imports, installs one.
    sys.addaudithook(lambda *args: None)

    out = adapt.Action(adapt.Actuator(adapt.ACTION_JOINT_POS, dim=2))
    tags = adapt.EnvTags(
        observation={"joint_pos": adapt.StateTag(adapt.JOINT_POS)}, action=out
    )


    class Env:
        observation_space = spaces.Dict(
            {"joint_pos": spaces.Box(-10.0, 10.0, (2,), np.float32)}
        )
        action_space = spaces.Box(-1000.0, 1000.0, (2,), np.float32)

        def reset(self, *, seed=None, options=None):
            self.steps = 0
            return {"joint_pos": np.zeros(2, np.float32)}, {}

        def step(self, action):
            self.steps += 1
            return {"joint_pos": np.zeros(2, np.float32)}, 1.0, self.steps >= 6, False, {}

        def close(self):
            pass


    class ShortChunk(Model):
        spec = adapt.ModelSpec(adapt.State(adapt.JOINT_POS, dim=2), out)

        def predict_chunk(self, observation, horizon):
            return np.zeros((2, 2), np.float32)


    session = "127.0.0.1:%d"
    served = ShortChunk()
    threading.Thread(
        target=served.serve, args=(session,), kwargs={"name": "policy"}, daemon=True
    ).start()
    time.sleep(1.0)
    result = Model(f"{session}/policy").run(
        adapt.tag(Env(), tags), episodes=1, execution_horizon=3
    )
    print("OK", result.total_steps, flush=True)
    """
)


def test_short_chunk_warning_from_a_served_model_survives_an_audit_hook(session_port):
    done = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", SCRIPT % session_port],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert "OK 6" in done.stdout
