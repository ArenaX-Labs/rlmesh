"""Drive a containerized env with a containerized model, end to end.

Needs Docker. ``SandboxModel.run`` is the containerized sibling of ``Model.run``:
it builds the model recipe's image and runs a one-shot container that drives the
env and reports a ``RunResult`` -- the policy executes in its own container, next
to its weights and deps, never in this process.

    uv run python examples/python/sandbox/drive_model/model_drives_env.py
"""

from cartpole_policy import CartpolePolicy
from rlmesh.numpy import SandboxEnv, SandboxModel

# rlmesh_package="local" installs this checkout's wheel in both containers; drop it
# to use the released package. The env serves CartPole; the model drives it.
with SandboxEnv("CartPole-v1", rlmesh_package="local") as env:
    result = SandboxModel(CartpolePolicy, rlmesh_package="local").run(
        env, seeds=[0, 1, 2]
    )
    print(result)  # RunResult(episodes=3, mean_reward=..., total_steps=...)
    for episode in result.episodes:
        print(f"  seed={episode.seed} steps={episode.steps} reward={episode.reward}")
