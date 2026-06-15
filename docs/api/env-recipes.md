# Environment Recipes

```{note}
`rlmesh.recipes` is **experimental** in this beta: it may change or disappear before the stable release. Pin versions; see {doc}`/compatibility`.
```

A recipe is an inert, JSON-serializable description of how to construct an environment, with three
phases: `make` (the factory), `build` (the image), and `setup` (construct-time data). See
{doc}`../user-guide/env-recipes` for a task-oriented guide. The inert `Recipe` form and the shared
`Build`/`Setup` phases documented here are the same ones a {doc}`model recipe <model-recipes>`
lowers to.

```{note}
Structured builds assume a Debian/Ubuntu base and `apt`. For another distro, use `build.dockerfile`.
```

## Authoring: `EnvRecipe`

Subclass `EnvRecipe` to co-locate the build/setup data and the factory. It projects to an inert
`Recipe` and is exported at the top level as `rlmesh.EnvRecipe`. Like a
{doc}`model recipe <model-recipes>`, it can declare runtime `inputs` (`ArtifactInput` asset mounts,
documented there) and resolve them inside `make()`/`prepare()` with `self.input_path(name)`.

```{eval-rst}
.. autoclass:: rlmesh.recipes.EnvRecipe
   :members: make, prepare, input_path, to_recipe, check
   :show-inheritance:
```

## Construction & Registry

`rlmesh.make` and `rlmesh.register` re-export `rlmesh.recipes.make` and `rlmesh.recipes.register`.

```{eval-rst}
.. autofunction:: rlmesh.recipes.make
```

```{eval-rst}
.. autofunction:: rlmesh.recipes.register
```

```{eval-rst}
.. autofunction:: rlmesh.recipes.check
```

```{eval-rst}
.. autofunction:: rlmesh.recipes.build
```

```{eval-rst}
.. autofunction:: rlmesh.recipes.resolve
```

```{eval-rst}
.. autofunction:: rlmesh.recipes.resolve_from_recipe
```

```{eval-rst}
.. autofunction:: rlmesh.recipes.registered_names
```

```{eval-rst}
.. autofunction:: rlmesh.recipes.unregister
```

## The Recipe

```{eval-rst}
.. autoclass:: rlmesh.recipes.Recipe
   :members:
   :show-inheritance:
```

## Build Phases

```{tip}
The phase numbers below are execution order. The sections list the phases by authoring relevance:
`make` first, then `build`, then `setup`.
```

### Make (phase 3)

```{eval-rst}
.. autoclass:: rlmesh.recipes.GymMake
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: rlmesh.recipes.PyMake
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: rlmesh.recipes.HfMake
   :members:
   :show-inheritance:
```

### Build (phase 1)

```{eval-rst}
.. autoclass:: rlmesh.recipes.Build
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: rlmesh.recipes.PipInstall
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: rlmesh.recipes.Fetch
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: rlmesh.recipes.ProjectInstall
   :members:
   :show-inheritance:
```

### Setup (phase 2)

```{eval-rst}
.. autoclass:: rlmesh.recipes.Setup
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: rlmesh.recipes.FileWrite
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: rlmesh.recipes.Requires
   :members:
   :show-inheritance:
```

## Migration

One-shot tooling in the `rlmesh.recipes.scaffold` submodule.

```{eval-rst}
.. autofunction:: rlmesh.recipes.scaffold.scaffold_from_pyproject
```

```{eval-rst}
.. autofunction:: rlmesh.recipes.scaffold.scaffold_recipe
```

```{eval-rst}
.. autoclass:: rlmesh.recipes.scaffold.ScaffoldResult
   :members:
   :show-inheritance:
```

## Errors

```{eval-rst}
.. autoclass:: rlmesh.recipes.RecipeValidationError
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: rlmesh.recipes.RecipeConstructionError
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: rlmesh.recipes.RecipeNotFoundError
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: rlmesh.recipes.UnsupportedRecipeError
   :show-inheritance:
```
