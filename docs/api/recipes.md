# Recipes

Recipe APIs are experimental in this beta. A recipe is an inert, JSON-serializable description of
how to construct an environment, with three phases: `make` (the factory), `build` (the image), and
`setup` (construct-time data). See {doc}`../user-guide/recipes` for a task-oriented guide.

## Construction

`rlmesh.make` is the top-level entry point; it is the same function as `rlmesh.recipes.make`.

```{eval-rst}
.. autofunction:: rlmesh.recipes.make
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
.. autofunction:: rlmesh.recipes.register
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

## Migration

```{eval-rst}
.. autofunction:: rlmesh.recipes.scaffold_recipe
```

```{eval-rst}
.. autofunction:: rlmesh.recipes.scaffold_from_pyproject
```

```{eval-rst}
.. autoclass:: rlmesh.recipes.ScaffoldResult
   :members:
   :show-inheritance:
```

## Errors

```{eval-rst}
.. autoclass:: rlmesh.recipes.RecipeValidationError
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
