# Core Exports

The top-level `rlmesh` package exports `EnvServer`, `RemoteEnv`, `RemoteVectorEnv`, `Model`,
`ServeOptions`, `Tensor`, `make`, `recipes`, `spaces`, and `types`.

The top-level client and model classes are dependency-free wrappers around RLMesh-native values. For
most user code, prefer the backend-specific modules when you want decoded NumPy arrays or Torch
tensors.

| Import                   | Description                                                           |
| ------------------------ | --------------------------------------------------------------------- |
| `rlmesh.EnvServer`       | Serve a Gymnasium-compatible environment endpoint.                    |
| `rlmesh.RemoteEnv`       | Connect to one environment and preserve RLMesh-native values.         |
| `rlmesh.RemoteVectorEnv` | Connect to a vector endpoint and preserve RLMesh-native values.       |
| `rlmesh.Model`           | Wrap a Python prediction function as a native-value model worker.     |
| `rlmesh.ServeOptions`    | Native serve lifecycle options.                                       |
| `rlmesh.Tensor`          | Native tensor value used by dependency-free clients.                  |
| `rlmesh.make`            | Construct an environment from a recipe name, a `Recipe`, or a gym id. |
| `rlmesh.recipes`         | Environment recipes, the registry, and migration helpers.             |
| `rlmesh.spaces`          | Space wrappers and Gymnasium conversion helpers.                      |
| `rlmesh.types`           | Structural protocols and value aliases.                               |

The detailed pages below describe the shared behavior:

- {doc}`env-server`
- {doc}`remote-envs`
- {doc}`models`
- {doc}`contracts`
- {doc}`recipes`
