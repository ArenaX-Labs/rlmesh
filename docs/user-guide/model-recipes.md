# Model Recipes

```{note}
`rlmesh.models` is **experimental** in this beta: it may change or disappear before the stable
release. Pin versions; see {doc}`/compatibility`.
```

A model recipe is one class that is both the policy and its construction document. Subclass
`rlmesh.ModelRecipe`, set the data attributes, and define `load()`/`predict()`. It is the model-side
sibling of {doc}`env-recipes`; see {doc}`../api/model-recipes` for the full API.

## Author a `ModelRecipe`

```python
# keeps method annotations as strings; see the note below
from __future__ import annotations

import rlmesh
from rlmesh.recipes import ArtifactInput, Build, PipInstall, hf_load


class SmolVLA(rlmesh.ModelRecipe):
    name = "policy/smolvla"
    build = Build(pip=[PipInstall(["lerobot==0.4.0"])], gpu=True)
    inputs = (
        ArtifactInput(
            "weights", "/weights/smolvla", uri="hf://lerobot/smolvla_base@<sha>"
        ),
    )
    spec = None

    def load(self):
        self._policy = hf_load(
            "lerobot/smolvla_base",
            loader="lerobot:SmolVLAPolicy",
            local_dir=self.input_path("weights"),
        )

    def predict(self, observation):
        return self._policy.select_action(observation)
```

`load()` builds the model into `self`; the instance is the policy for the whole eval. Put every
heavy import inside `load()`. `predict()` maps one observation to one action. `reset()` (per
episode) and `close()` are optional. `build` shares the phase-1 vocabulary with `EnvRecipe`.

Your heavy imports live inside `load()`, so the types named in a method's annotations aren't in
scope when the class is defined. Add `from __future__ import annotations` at the top of the module
to keep annotations as strings instead of evaluating them. A signature like
`def predict(self, obs) -> torch.Tensor:` then defines fine, even though `torch` is imported later.

## Run it

A backend `Model` drives the policy against a served env. The model is a client that dials the env;
pass an env object, an object with an `address`, or a bare address string.

```python
from rlmesh.numpy import Model

result = Model(SmolVLA).run(env, seeds=[0, 1, 2])
print(result.mean_reward)
```

`run` returns a `RunResult`; `seeds` sets the episode count unless `max_episodes` is given. The
source can also be a registered name or a plain predict callable:

```python
Model("policy/smolvla").run(env, seeds=[0])
Model(lambda observation: 0).run("127.0.0.1:5555", seeds=[0])
```

## Load Hugging Face weights

Weights are a runtime mount, never baked into the image. Declare an `ArtifactInput`, then resolve
its path inside `load()` with `self.input_path(name)`. A `uri="hf://org/repo@sha"` resolves through
the rlmesh artifact cache; `local_dir=` points at a host directory instead. `hf_load` loads the
policy from that path.

Resolving an `hf://` uri needs `huggingface_hub` (the `rlmesh[hf]` extra: `pip install --pre
"rlmesh[hf]"`). In a `SandboxModel` the container resolves the uri, not the host, so a recipe with a
`uri=` input must install it in the recipe's `build` (`PipInstall(["huggingface_hub"])`). Point the
input at `local_dir=` to bind-mount weights from the host instead, which needs nothing extra in the
container.

## Register

The class form stores the projected recipe and keeps the live class:

```python
rlmesh.register(SmolVLA)  # or @rlmesh.register above the class
```

The flat form synthesizes the recipe for you. Use `hf=` for a Hugging Face policy or `load=` for a
factory; both take a `spec=`:

```python
rlmesh.register("policy/openvla", hf="org/openvla", spec=SPEC)
rlmesh.register("policy/custom", load="mod:make_policy", spec=SPEC)
```

Model keywords (`hf=`/`load=`/`spec=`) are disjoint from env keywords (`gym=`/`factory=`), so
`register` routes by kind. A flat registration is in-process only; subclass `ModelRecipe` to run in
a container.

## Run in a container

`SandboxModel` is the containerized sibling of `Model`. It builds the recipe to an image and runs
the policy in its own container, so the model's dependencies never enter your process. Like `Model`,
construction is inert: `SandboxModel(source)` resolves the recipe but starts nothing. From there it
either runs a one-shot eval or serves a long-lived endpoint.

```{important}
`SandboxModel` builds a fresh image and imports your recipe inside the container by reference, as
`module:Class`. Define the class in an importable module, not a `__main__` script: a file run as
`__main__` cannot be imported, and the build rejects it. The image also needs the module's source,
so add `project=ProjectInstall(src=".")` to `build`, with a `pyproject.toml` beside it so the folder
installs. Skip the `project` step and the image still builds, but the container cannot import your
class at startup.
```

### One-shot eval

`SandboxModel(source).run(env)` builds the image, runs a single `--rm` container that drives `env`,
and returns the same `RunResult` as `Model.run`; `seeds` and `max_episodes` carry over. Pair it with
a `SandboxEnv` and both sides are isolated at once: the env runs its dependencies in one container,
the model runs its own in another, and they meet over the host network.

The recipe class lives in its own module so the container can import it:

```python
# greedy_cartpole.py
from __future__ import annotations

import rlmesh
from rlmesh.models import DELEGATED
from rlmesh.recipes import Build, PipInstall, ProjectInstall


class GreedyCartPole(rlmesh.ModelRecipe):
    name = "policy/greedy-cartpole"
    # project=ProjectInstall stages this folder so the container can import the class
    build = Build(pip=[PipInstall(["numpy==2.1.3"])], project=ProjectInstall(src="."))
    spec = DELEGATED  # adapts its own observations; no adapter is resolved

    def load(self):
        import numpy as np

        self._np = np

    def predict(self, observation):
        # Nudge toward the side the pole is leaning: 1 if it tilts right, else 0.
        return int(self._np.asarray(observation)[2] > 0)
```

A second file imports the class and drives the env:

```python
# run.py
from greedy_cartpole import GreedyCartPole
from rlmesh.numpy import SandboxEnv, SandboxModel

with SandboxEnv(
    "CartPole-v1", packages=["gymnasium==1.3.0"], imports=["gymnasium"]
) as env:
    result = SandboxModel(GreedyCartPole).run(env, seeds=[0, 1, 2, 3, 4])

print(
    f"{result.num_episodes} episodes, success {result.success_rate:.0%}, "
    f"mean reward {result.mean_reward:.1f}"
)
```

`run.py` and `greedy_cartpole.py` are two files in one folder. `ProjectInstall(src=".")` installs
that folder into the image, so it needs a `pyproject.toml` beside them;
`examples/python/sandbox/drive_model/` has the complete layout to copy.

`env` is the dialed party, exactly as with `Model.run`: pass an env object (a `SandboxEnv` exposes
its `.sandbox.address`), an object with an `.address`, or a bare address string. The container
reaches that address over the host network, drives the env for the given `seeds` (or
`max_episodes`), reports the run, and exits. Nothing stays running.

### Served endpoint

As a context manager (or via `serve()`), `SandboxModel` serves the policy as a long-lived model
endpoint instead. It exposes `.address` and `.container_id` (both raise until it is serving), and
`.shutdown()` stops the container on exit:

```python
with SandboxModel(GreedyCartPole) as model:
    print(model.address)  # a model endpoint, not an env address
```

`model.address` is a _model_ endpoint, not an env address, so don't pass it to
`Model(...).run(...)`, which dials its argument as an env. Serving needs a spec-less or `DELEGATED`
model: a `ModelSpec` model resolves its adapter from the env contract, which a served endpoint does
not yet receive on dial-in, so drive a spec'd model with `run(env)` instead.

## spec: None, DELEGATED, or a ModelSpec

`spec` declares how the model's observations relate to the env's tags.

| `spec` value     | Meaning                                                                                                         |
| ---------------- | --------------------------------------------------------------------------------------------------------------- |
| `None`           | No adaptation. Runs against an untagged env; fails loud against a tagged one.                                   |
| `DELEGATED`      | The model adapts its own observations. No adapter is resolved.                                                  |
| `ModelSpec(...)` | The adapter is resolved from the env's tags and this spec, so `predict` sees the model's declared input format. |
