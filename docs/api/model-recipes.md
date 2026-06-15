# Model Recipes

```{note}
`rlmesh.models` is **experimental** in this beta: it may change or disappear before the stable
release. Pin versions; see {doc}`/compatibility`.
```

A model recipe is one class that is both the policy and an inert, JSON-serializable construction
document. `ModelRecipe.to_recipe()` projects the class to a `kind='model'` `Recipe` without
importing the model's dependencies. It is the model-side sibling of {doc}`env-recipes`. See
{doc}`../user-guide/model-recipes` for a task-oriented guide.

## Authoring: `ModelRecipe`

Subclass `ModelRecipe`: set `name`/`build`/`spec`/`inputs`, then define `load()` and `predict()`,
with optional `reset()`/`close()`. Exported at the top level as `rlmesh.ModelRecipe`. Start the
module with `from __future__ import annotations`. It keeps method annotations as strings instead of
evaluating them at import time, so you can annotate a method with a type that you import inside
`load()`, such as `torch.Tensor`, without hitting a `NameError`.

```{eval-rst}
.. autoclass:: rlmesh.recipes.ModelRecipe
   :members: load, predict, reset, close, input_path, to_recipe, check
   :show-inheritance:
```

## Running

`Model(source).run(env, seeds=[...])` drives the policy against a served env and returns a
`RunResult`; `source` is a predict callable, a `ModelRecipe`, a `kind='model'` `Recipe`, or a
registered name. The backend `Model` (`rlmesh.numpy.Model`, `rlmesh.torch.Model`) is documented in
{doc}`models`. A spec'd model recipe runs with `run`; `Model.serve()` hosts only a spec-less or
`spec=DELEGATED` model as an endpoint (serving a spec'd model is not yet implemented). The run loop
returns these typed results.

```{eval-rst}
.. autoclass:: rlmesh.models.RunResult
   :members:
```

```{eval-rst}
.. autoclass:: rlmesh.models.EpisodeResult
   :members:
```

## Artifacts

Weights are a runtime mount, never baked into the image. Declare an `ArtifactInput`, resolve its
path inside `load()` with `self.input_path(name)`, and load a Hugging Face policy with `hf_load`.
Resolving an `hf://` uri (or calling `hf_load` without `local_dir`) needs the `rlmesh[hf]` extra
(`huggingface_hub`). In a `SandboxModel` the container resolves the uri, so a recipe with a `uri=`
input must install it in the recipe's `build`; `local_dir=` bind-mounts from the host and needs no
extra.

```{eval-rst}
.. autoclass:: rlmesh.recipes.ArtifactInput
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autofunction:: rlmesh.recipes.hf_load
```

```{eval-rst}
.. autofunction:: rlmesh.recipes.input_path
```

## Registry

`register` takes a `ModelRecipe` subclass, or a name with the flat `hf=`/`load=` sugar. Exported at
the top level as `rlmesh.register`, which routes to the model registry when the source is a
`ModelRecipe` or any of `hf=`/`load=`/`spec=` is set.

```{eval-rst}
.. autofunction:: rlmesh.models.register
```

The self-adapting-model sentinel for `spec`:

```{eval-rst}
.. autodata:: rlmesh.models.DELEGATED
   :annotation:
```

## Sandbox

`SandboxModel` is the containerized sibling of the backend `Model`, exported per backend as
`rlmesh.numpy.SandboxModel`. It builds a model recipe to an image and runs the policy in its own
container, keeping the model's dependencies isolated from the caller. Construction is inert; it
resolves the recipe but builds and runs nothing until you call `run` or `serve`.

`run(env, ...)` builds the image, runs a one-shot `--rm` container that drives `env`, and returns
the same `RunResult` as `Model.run` (only `seeds`/`max_episodes` overlap); `env` resolves to a
`SandboxEnv`'s `.sandbox.address`, an object's `.address`, or a bare address string. Used as a
context manager (or via `serve()`), it instead serves the policy from a long-lived container as a
model endpoint, exposing `.address` and `.container_id` (both raise until serving) with
`.shutdown()` to stop it. `serve()` covers a spec-less or `spec=DELEGATED` model only; a spec'd
model is drive-only via `run`. See {doc}`../user-guide/model-recipes` for usage.

```{eval-rst}
.. autoclass:: rlmesh.numpy.SandboxModel
   :members: run, serve, address, container_id, shutdown
```

## Errors

Model construction and resolution raise the shared recipe and adapter errors:
{exc}`~rlmesh.recipes.RecipeValidationError` (see {doc}`env-recipes`) and
{exc}`~rlmesh.adapters.AdapterResolutionError` (see {doc}`adapters`).
