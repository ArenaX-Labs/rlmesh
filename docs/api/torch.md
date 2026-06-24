# Torch

The Torch backend is experimental.

## What This Backend Changes

`rlmesh.torch` keeps the same environment, model, and sandbox behavior as the shared RLMesh client APIs, but decodes tensor leaves to Torch tensors. Space wrappers returned from Torch clients also sample Torch-compatible values.

Install it with:

```bash
pip install "rlmesh[torch]"
```

| Concrete API                    | Shared behavior                        | Backend-specific behavior                               |
| ------------------------------- | -------------------------------------- | ------------------------------------------------------- |
| `rlmesh.torch.RemoteEnv`        | {doc}`remote-envs` single clients      | Observations, actions, and render frames use tensors.   |
| `rlmesh.torch.RemoteVectorEnv`  | {doc}`remote-envs` vector clients      | Batched values use Torch-compatible containers.         |
| `rlmesh.torch.Model`            | {doc}`models`                          | `predict_fn` receives Torch-decoded observations.       |
| `rlmesh.torch.SandboxEnv`       | {doc}`sandbox` single sandbox sessions | Owned sandbox client is `rlmesh.torch.RemoteEnv`.       |
| `rlmesh.torch.SandboxVectorEnv` | {doc}`sandbox` vector sandbox sessions | Owned sandbox client is `rlmesh.torch.RemoteVectorEnv`. |

## Memory Sharing and Mutation

Decoded observations are owned, writable copies, so `predict_fn` can normalize in place (`img.div_(255)`) without corrupting the wire buffer. `as_tensor(tensor)` is the zero-copy opt-in: the Torch tensor shares memory with the RLMesh tensor over DLPack.

```{warning}
A zero-copy `as_tensor(tensor)` view shares memory. RLMesh flags the export read-only, but Torch,
like most DLPack consumers, does not enforce that flag, so an in-place write corrupts the RLMesh
tensor for every other view of the same data, including NumPy views in the same process. Treat a
shared view as read-only, or pass `copy=True`.
```

Conversion details:

- Decode uses `torch.utils.dlpack.from_dlpack`; `bool` tensors fall back to a buffer copy on Torch older than 2.2 (no bool DLPack support there).
- `uint16`, `uint32`, and `uint64` dtypes require Torch 2.3 or newer.
- Encode (`from_tensor`) detaches, moves to CPU, and exports over DLPack; NumPy is not required.

## Value Helpers

```{eval-rst}
.. autofunction:: rlmesh.torch.ensure_available
```

```{eval-rst}
.. autofunction:: rlmesh.torch.as_tensor
```

```{eval-rst}
.. autofunction:: rlmesh.torch.from_tensor
```

```{eval-rst}
.. autofunction:: rlmesh.torch.space_from_spec
```

## RemoteEnv

```{eval-rst}
.. autoclass:: rlmesh.torch.RemoteEnv
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```

## RemoteVectorEnv

```{eval-rst}
.. autoclass:: rlmesh.torch.RemoteVectorEnv
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```

## Model

```{eval-rst}
.. autoclass:: rlmesh.torch.Model
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```

## Sandbox

```{eval-rst}
.. autoclass:: rlmesh.torch.SandboxEnv
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: rlmesh.torch.SandboxVectorEnv
   :class-doc-from: class
   :exclude-members: __init__, __new__
   :show-inheritance:
```
