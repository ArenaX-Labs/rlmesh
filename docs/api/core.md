# Core Exports

The top-level `rlmesh` package re-exports the common entry points: environment serving and clients,
model running, recipe authoring, sandboxing, and the `spaces`, `types`, `adapters`, and `serving`
subpackages. The full table is below.

The top-level client and model classes are dependency-free wrappers around RLMesh-native values. For
most user code, prefer the backend-specific modules when you want decoded NumPy arrays or Torch
tensors.

| Import                    | Description                                                             |
| ------------------------- | ----------------------------------------------------------------------- |
| `rlmesh.EnvServer`        | Serve a Gymnasium-compatible environment endpoint.                      |
| `rlmesh.RemoteEnv`        | Connect to one environment and preserve RLMesh-native values.           |
| `rlmesh.RemoteVectorEnv`  | Connect to a vector endpoint and preserve RLMesh-native values.         |
| `rlmesh.SandboxEnv`       | Build an env image and own the container behind a single client.        |
| `rlmesh.SandboxVectorEnv` | Build an env image and own the container behind a vector client.        |
| `rlmesh.Model`            | Wrap a Python prediction function as a native-value model worker.       |
| `rlmesh.SandboxModel`     | Run a `ModelRecipe` policy in its own container (experimental).         |
| `rlmesh.ServeOptions`     | Native serve lifecycle options.                                         |
| `rlmesh.Tensor`           | Native tensor value used by dependency-free clients.                    |
| `rlmesh.make`             | Construct an environment from a name, `Recipe`, `EnvRecipe`, or gym id. |
| `rlmesh.register`         | Register a recipe (or the flat `gym=`/`factory=` sugar) by name.        |
| `rlmesh.EnvRecipe`        | Base class for class-style environment authoring.                       |
| `rlmesh.ModelRecipe`      | Base class for class-style model (policy) authoring (experimental).     |
| `rlmesh.Recipe`           | The inert recipe document your authoring lowers to.                     |
| `rlmesh.recipes`          | Environment recipes, the registry, and migration helpers.               |
| `rlmesh.models`           | Model recipes, the eval loop, and the model registry (experimental).    |
| `rlmesh.adapters`         | Observation/action adapters and contract-based resolution.              |
| `rlmesh.export`           | Build a sandbox image and return without starting a container.          |
| `rlmesh.ExportResult`     | Result of `export`: the content-addressed image and optional alias.     |
| `rlmesh.serving`          | Helpers for loading and hosting an env inside a server process.         |
| `rlmesh.spaces`           | Space wrappers and Gymnasium conversion helpers.                        |
| `rlmesh.types`            | Structural protocols and value aliases.                                 |

The detailed pages below describe the shared behavior:

- {doc}`env-server`
- {doc}`remote-envs`
- {doc}`models`
- {doc}`contracts`
- {doc}`env-recipes`
- {doc}`model-recipes`
