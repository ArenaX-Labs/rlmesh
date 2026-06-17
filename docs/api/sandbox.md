# Sandbox

Sandbox APIs are experimental. A sandbox session owns a Docker-backed environment process, connects
a remote client to it, and stops the container when the session closes. See
{doc}`../examples/sandboxes` for runnable examples.

## Session Info

```{eval-rst}
.. autoclass:: rlmesh._sandbox.SandboxInfo
   :members:
   :show-inheritance:
```

## Base Sessions

```{eval-rst}
.. autoclass:: rlmesh._sandbox.SandboxSessionBase
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: rlmesh._sandbox.SandboxEnvBase
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: rlmesh._sandbox.SandboxVectorEnvBase
   :members:
   :show-inheritance:
```

## Backend Sessions

Concrete sandbox classes inherit the base session behavior and only choose the remote client used
inside the owned sandbox session.

| Class                           | Remote client                  | Value decoding               |
| ------------------------------- | ------------------------------ | ---------------------------- |
| `rlmesh.numpy.SandboxEnv`       | `rlmesh.numpy.RemoteEnv`       | NumPy arrays and primitives  |
| `rlmesh.numpy.SandboxVectorEnv` | `rlmesh.numpy.RemoteVectorEnv` | NumPy arrays and primitives  |
| `rlmesh.torch.SandboxEnv`       | `rlmesh.torch.RemoteEnv`       | Torch tensors and primitives |
| `rlmesh.torch.SandboxVectorEnv` | `rlmesh.torch.RemoteVectorEnv` | Torch tensors and primitives |

See {doc}`numpy` and {doc}`torch` for backend-specific class entries and helper functions.
