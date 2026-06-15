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

`SandboxModel` builds a model recipe to an image and runs the policy in its own container. It
exposes `.address` and `.container_id`, has `.shutdown()`, and is a context manager. Exported per
backend as `rlmesh.numpy.SandboxModel`.

```{eval-rst}
.. autoclass:: rlmesh.numpy.SandboxModel
   :members: address, container_id, shutdown
```

## Errors

Model construction and resolution raise the shared recipe and adapter errors:
{exc}`~rlmesh.recipes.RecipeValidationError` (see {doc}`env-recipes`) and
{exc}`~rlmesh.adapters.AdapterResolutionError` (see {doc}`adapters`).
