# API Reference

The exact imports, method signatures, and backend behavior. Reach for these pages when you already
know what you want to call and need the precise surface; the user-guide pages cover when and why.

## Behavior

- {doc}`core`: top-level package exports, structural protocols, and shared value aliases.
- {doc}`contracts`: environment contracts, space specs, tensors, and serve options.
- {doc}`spaces`: RLMesh space wrappers and conversion helpers.
- {doc}`models`: model worker wrappers.
- {doc}`adapters`: declarative model-to-environment IO layer for mapping a model's observations and actions to an environment's spaces.
- {doc}`serving`: the env server, the serving helpers, and the remote client classes.
- {doc}`sandbox`: experimental Docker-backed sandbox sessions.

## Framework Backends

- {doc}`backends`: NumPy, Torch (experimental), and JAX (experimental) clients, models, and tensor helpers.

```{toctree}
:maxdepth: 2

core
contracts
spaces
models
adapters
serving
sandbox
backends
```
