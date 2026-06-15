# API Reference

Use this when you need exact imports, methods, and backend behavior.

## Behavior

- {doc}`core`: top-level package exports.
- {doc}`env-server`: serving Gymnasium-compatible environments.
- {doc}`serving`: experimental helpers for loading environments to serve.
- {doc}`remote-envs`: remote client base classes and endpoint behavior.
- {doc}`contracts`: environment contracts, space specs, tensors, and serve options.
- {doc}`models`: model worker wrappers.
- {doc}`adapters`: experimental declarative env-to-model IO adapters.
- {doc}`sandbox`: experimental Docker-backed sandbox sessions.
- {doc}`recipes`: experimental environment recipes and the registry.
- {doc}`model-recipes`: experimental model construction — ModelRecipe authoring, the runtime Model, hf_load, and ArtifactInput.
- {doc}`spaces`: RLMesh space wrappers and conversion helpers.
- {doc}`types`: structural protocols and shared value aliases.

## Framework Backends

- {doc}`numpy`: NumPy-backed clients, models, and tensor helpers.
- {doc}`torch`: experimental Torch-backed clients, models, and tensor helpers.
- {doc}`jax`: experimental JAX-backed clients, models, and tensor helpers.

```{toctree}
:maxdepth: 2

core
env-server
serving
remote-envs
contracts
models
adapters
numpy
torch
jax
spaces
types
sandbox
recipes
model-recipes
```
