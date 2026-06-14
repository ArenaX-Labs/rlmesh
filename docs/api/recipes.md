# Recipes

Recipe APIs are experimental in this beta. A recipe is an inert, JSON-serializable description of
how to construct an environment, with three phases: `make` (the factory), `build` (the image), and
`setup` (construct-time data). Structured builds assume a Debian/Ubuntu base and `apt`; for another
distro use `build.dockerfile`. See {doc}`../user-guide/recipes` for a task-oriented guide.

## Authoring -- `EnvRecipe`

The headline authoring surface: subclass `EnvRecipe` to co-locate the build/setup data and the
factory, projecting to an inert `Recipe`. Exported at the top level as `rlmesh.EnvRecipe`.

```{eval-rst}
.. autoclass:: rlmesh.recipes.EnvRecipe
   :members: make, prepare, to_recipe, check
   :show-inheritance:
```

## Construction

`rlmesh.make` and `rlmesh.register` are the top-level pair; they are the same functions as
`rlmesh.recipes.make` / `rlmesh.recipes.register`.

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

## The Recipe

```{eval-rst}
.. autoclass:: rlmesh.recipes.Recipe
   :members:
   :show-inheritance:
```

## Make (phase 3)

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

## Build (phase 1)

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

## Setup (phase 2)

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

## Registry

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
