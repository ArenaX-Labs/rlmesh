# Remote Environments

Remote clients connect to an environment endpoint and expose the same core calls as a local
Gymnasium environment: `reset`, `step`, `render`, and `close`.

Use the concrete backend modules in application code:

- `rlmesh.RemoteEnv` and `rlmesh.RemoteVectorEnv` preserve RLMesh-native values.
- `rlmesh.numpy.RemoteEnv` and `rlmesh.numpy.RemoteVectorEnv` decode tensor leaves as NumPy arrays.
- `rlmesh.torch.RemoteEnv` and `rlmesh.torch.RemoteVectorEnv` decode tensor leaves as Torch tensors.

Every remote client keeps the endpoint handshake in `env_contract`. The `spec` property is an alias
for that same contract. `observation_space` and `action_space` are client-side wrappers built from
the contract's `SpaceSpec` values.

## Render Viewer

Every remote client (and the experimental sandbox sessions) inherits two viewer methods from a
shared mixin:

- `open_viewer(*, env_index=0, fps="env")` opens a local render window and streams frames after each
  `reset`, `step`, and `render`. `env_index` selects which sub-environment of a vector client to
  view. `fps` accepts `"env"` (read `render_fps` from environment metadata), a positive number for
  an explicit limit, or `None` to disable pacing.
- `close_viewer()` closes the window if one is open. It is also called on client `close()`.

The viewer is best-effort and experimental. It launches a separate GUI process via
`python -m rlmesh viewer`, so it needs a desktop host. The environment must produce frames,
typically `render_mode="rgb_array"`. Frame pushes are dropped instead of blocking the step loop when
the viewer stalls.

## Single Environment Base

```{eval-rst}
.. autoclass:: rlmesh._client.RemoteEnvBase
   :members:
   :show-inheritance:
```

## Vector Environment Base

```{eval-rst}
.. autoclass:: rlmesh._client.RemoteVectorEnvBase
   :members:
   :show-inheritance:
```
