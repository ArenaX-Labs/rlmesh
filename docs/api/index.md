# API Reference

Use this when you need exact imports, methods, and adapter behavior.

## Behavior

- {doc}`core`: top-level package exports.
- {doc}`env-server`: serving Gymnasium-compatible environments.
- {doc}`remote-envs`: remote client base classes and endpoint behavior.
- {doc}`contracts`: environment contracts, space specs, tensors, and serve options.
- {doc}`models`: model worker wrappers.
- {doc}`sandbox`: experimental Docker-backed sandbox sessions.
- {doc}`spaces`: RLMesh space wrappers and conversion helpers.
- {doc}`types`: structural protocols and shared value aliases.

## Adapters

- {doc}`numpy`: NumPy-backed clients, models, and tensor helpers.
- {doc}`torch`: experimental Torch-backed clients, models, and tensor helpers.

```{toctree}
:maxdepth: 2

core
env-server
remote-envs
contracts
models
numpy
torch
spaces
types
sandbox
```
