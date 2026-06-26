# Core Exports

The top-level `rlmesh` package re-exports the common entry points: environment serving and clients, model running, sandboxing, and the `spaces`, `types`, and `adapters` subpackages. The common imports are listed below.

The top-level client and model classes are dependency-free wrappers around RLMesh-native values. For most user code, prefer the backend-specific modules when you want decoded NumPy arrays or Torch tensors.

| Import                    | Description                                                           |
| ------------------------- | --------------------------------------------------------------------- |
| `rlmesh.EnvServer`        | Serve a Gymnasium-compatible environment endpoint (scalar or vector). |
| `rlmesh.RemoteEnv`        | Connect to one environment and preserve RLMesh-native values.         |
| `rlmesh.RemoteVectorEnv`  | Connect to a vector endpoint and preserve RLMesh-native values.       |
| `rlmesh.SandboxEnv`       | Build an env image and own the container behind a single client.      |
| `rlmesh.SandboxVectorEnv` | Build an env image and own the container behind a vector client.      |
| `rlmesh.Model`            | Wrap a Python prediction function as a native-value model worker.     |
| `rlmesh.RemoteModel`      | Connect to an already-served model and drive it against an env.       |
| `rlmesh.SandboxModel`     | Run a model policy in its own container (experimental).               |
| `rlmesh.ServeOptions`     | Native serve lifecycle options.                                       |
| `rlmesh.Tensor`           | Native tensor value used by dependency-free clients.                  |
| `rlmesh.adapters`         | Observation/action adapters and contract-based resolution.            |
| `rlmesh.spaces`           | Space wrappers and Gymnasium conversion helpers.                      |
| `rlmesh.types`            | Structural protocols and value aliases.                               |

The detailed pages below describe the shared behavior:

- {doc}`env-server`
- {doc}`remote-envs`
- {doc}`models`
- {doc}`contracts`
