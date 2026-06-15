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
from __future__ import annotations  # keeps method annotations as strings; see the note below

import rlmesh
from rlmesh.recipes import ArtifactInput, Build, PipInstall, hf_load


class SmolVLA(rlmesh.ModelRecipe):
    name = "policy/smolvla"
    build = Build(pip=[PipInstall(["lerobot==0.4.0"])], gpu=True)
    inputs = (ArtifactInput("weights", "/weights/smolvla",
                            uri="hf://lerobot/smolvla_base@<sha>"),)
    spec = None

    def load(self):
        self._policy = hf_load("lerobot/smolvla_base", loader="lerobot:SmolVLAPolicy",
                               local_dir=self.input_path("weights"))

    def predict(self, observation):
        return self._policy.select_action(observation)
```

`load()` builds the model into `self`; the instance is the policy for the whole eval. Put every
heavy import inside `load()`. `predict()` maps one observation to one action. `reset()` (per
episode) and `close()` are optional. `build` shares the phase-1 vocabulary with `EnvRecipe`.

Your heavy imports live inside `load()`, so the types named in a method's annotations aren't in
scope when the class is defined. Add `from __future__ import annotations` at the top of the module
to keep annotations as strings instead of evaluating them. A signature like `def predict(self, obs)
-> torch.Tensor:` then defines fine, even though `torch` is imported later.

## Run it

A backend `Model` drives the policy against a served env. The model is a client that dials the
env; pass an env object, an object with an `address`, or a bare address string.

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

Resolving an `hf://` uri on the host needs the optional extra: `pip install --pre "rlmesh[hf]"`. In
a `SandboxModel` the container fetches it, so the host doesn't need the extra.

## Register

The class form stores the projected recipe and keeps the live class:

```python
rlmesh.register(SmolVLA)          # or @rlmesh.register above the class
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

`SandboxModel` builds the recipe to an image and starts the policy in its own container. It exposes
`.address` and `.container_id`, has `.shutdown()`, and is a context manager:

```python
from rlmesh.numpy import SandboxModel

with SandboxModel(SmolVLA) as model:
    Model("policy/smolvla").run(model.address, seeds=[0])
```

## spec: None, DELEGATED, or a ModelSpec

`spec` declares how the model's observations relate to the env's tags.

| `spec` value | Meaning |
| --- | --- |
| `None` | No adaptation. Runs against an untagged env; fails loud against a tagged one. |
| `DELEGATED` | The model adapts its own observations. No adapter is resolved. |
| `ModelSpec(...)` | The adapter is resolved from the env's tags and this spec, so `predict` sees the model's declared input format. |
