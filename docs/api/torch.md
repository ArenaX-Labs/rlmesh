# Torch

The Torch adapter is experimental in this beta.

## What This Adapter Changes

`rlmesh.torch` keeps the same environment, model, and sandbox behavior as the shared RLMesh client
APIs, but decodes tensor leaves to Torch tensors. Space wrappers returned from Torch clients also
sample Torch-compatible values.

Install it with:

```bash
pip install --pre "rlmesh[torch]"
```

| Concrete API                    | Shared behavior                        | Adapter-specific behavior                               |
| ------------------------------- | -------------------------------------- | ------------------------------------------------------- |
| `rlmesh.torch.RemoteEnv`        | {doc}`remote-envs` single clients      | Observations, actions, and render frames use tensors.   |
| `rlmesh.torch.RemoteVectorEnv`  | {doc}`remote-envs` vector clients      | Batched values use Torch-compatible containers.         |
| `rlmesh.torch.Model`            | {doc}`models`                          | `predict_fn` receives Torch-decoded observations.       |
| `rlmesh.torch.SandboxEnv`       | {doc}`sandbox` single sandbox sessions | Owned sandbox client is `rlmesh.torch.RemoteEnv`.       |
| `rlmesh.torch.SandboxVectorEnv` | {doc}`sandbox` vector sandbox sessions | Owned sandbox client is `rlmesh.torch.RemoteVectorEnv`. |

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
