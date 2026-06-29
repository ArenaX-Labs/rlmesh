# API Reference

The exact imports, method signatures, and backend behavior. Reach for these pages when you already
know what you want to call and need the precise surface; the user-guide pages cover when and why.

## Behavior

- {doc}`core`: top-level package exports.
- {doc}`env-server`: serving Gymnasium-compatible environments.
- {doc}`serving`: experimental helpers for loading an environment to serve.
- {doc}`remote-envs`: remote client base classes and endpoint behavior.
- {doc}`contracts`: environment contracts, space specs, tensors, and serve options.
- {doc}`models`: model worker wrappers.
- {doc}`adapters`: declarative model-to-environment IO layer for mapping a model's observations and actions to an environment's spaces.
- {doc}`sandbox`: experimental Docker-backed sandbox sessions.
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
sandbox
spaces
types
numpy
torch
jax
```
