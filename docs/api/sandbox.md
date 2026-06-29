# Sandbox

```{note}
This is the autodoc API reference. For the guide see {doc}`../user-guide/sandbox`; for runnable files
see {doc}`../examples/sandboxes`.
```

Sandbox APIs are experimental. A sandbox session owns a Docker-backed environment process, connects a remote client to it, and stops the container when the session closes. Reach for one when an environment needs its own dependencies and process: the session handles build, connect, and teardown behind the normal remote client.

## Session Info

```{eval-rst}
.. autoclass:: rlmesh._sandbox.SandboxInfo
   :members:
   :show-inheritance:
```

## Base Sessions

```{eval-rst}
.. autoclass:: rlmesh._sandbox.session.SandboxLifecycle
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

Concrete sandbox classes inherit the base session behavior and only choose the remote client used inside the owned sandbox session.

| Class                           | Remote client                  | Value decoding               |
| ------------------------------- | ------------------------------ | ---------------------------- |
| `rlmesh.numpy.SandboxEnv`       | `rlmesh.numpy.RemoteEnv`       | NumPy arrays and primitives  |
| `rlmesh.numpy.SandboxVectorEnv` | `rlmesh.numpy.RemoteVectorEnv` | NumPy arrays and primitives  |
| `rlmesh.torch.SandboxEnv`       | `rlmesh.torch.RemoteEnv`       | Torch tensors and primitives |
| `rlmesh.torch.SandboxVectorEnv` | `rlmesh.torch.RemoteVectorEnv` | Torch tensors and primitives |

See {doc}`backends` for backend-specific class entries and helper functions.
