# Core Exports

```{note}
This is the autodoc API reference. For authoring guides see {doc}`../user-guide/environments` and
{doc}`../user-guide/models`.
```

The top-level `rlmesh` package re-exports the common entry points: environment serving and clients, model running, sandboxing, and the `spaces`, `types`, and `adapters` subpackages.

The top-level client and model classes are dependency-free wrappers around RLMesh-native values. Reach for them when you want native values and no framework dependency; use a backend module ({doc}`backends`) when you want tensor leaves decoded to NumPy arrays or Torch tensors.

| Import                    | Description                                                           |
| ------------------------- | --------------------------------------------------------------------- |
| `rlmesh.EnvServer`        | Serve a Gymnasium-compatible environment endpoint (scalar or vector). |
| `rlmesh.RemoteEnv`        | Connect to one environment and preserve RLMesh-native values.         |
| `rlmesh.RemoteVectorEnv`  | Connect to a vector endpoint and preserve RLMesh-native values.       |
| `rlmesh.SandboxEnv`       | Build an env image and own the container behind a single client.      |
| `rlmesh.SandboxVectorEnv` | Build an env image and own the container behind a vector client.      |
| `rlmesh.Model`            | Wrap a Python prediction function as a native-value model.            |
| `rlmesh.RemoteModel`      | Connect to an already-served model and drive it against an env.       |
| `rlmesh.SandboxModel`     | Run a model policy in its own container (experimental).               |
| `rlmesh.ServeOptions`     | Native serve lifecycle options.                                       |
| `rlmesh.Tensor`           | Native tensor value used by dependency-free clients.                  |
| `rlmesh.adapters`         | Observation/action adapters and contract-based resolution.            |
| `rlmesh.spaces`           | Space wrappers and Gymnasium conversion helpers.                      |
| `rlmesh.types`            | Structural protocols and value aliases.                               |

The detailed pages below describe the shared behavior:

- {doc}`serving`
- {doc}`models`
- {doc}`contracts`

## Run and Evaluate

The eval surface binds a model to an environment. {func}`~rlmesh.run` pumps whole episodes and returns a {class}`~rlmesh.RunResult`; {func}`~rlmesh.session` returns a {class}`~rlmesh.Session` you drive by hand with `reset` / `predict` / `step`.

```{eval-rst}
.. autofunction:: rlmesh.run
```

```{eval-rst}
.. autofunction:: rlmesh.session
```

```{eval-rst}
.. autoclass:: rlmesh.Session
   :members:
```

```{eval-rst}
.. autoclass:: rlmesh.RunResult
   :members:
```

```{eval-rst}
.. autoclass:: rlmesh.EpisodeResult
   :members:
```

```{eval-rst}
.. autoclass:: rlmesh.RunHooks
   :members:
```

```{eval-rst}
.. autoclass:: rlmesh.StepEvent
   :members:
```

```{eval-rst}
.. autoclass:: rlmesh.View
   :members:
```

```{eval-rst}
.. autofunction:: rlmesh.sanitize_metadata
```

### Sentinels

Two module-level sentinels change what `run` and `session` do:

- `rlmesh.NO_ADAPTER`: pass as a model's `spec` to explicitly skip adapter resolution; the model handles raw env observations and actions itself.
- `rlmesh.RANDOM_SAMPLE`: pass as the model to `rlmesh.run` / `rlmesh.session` to sample the env's action space each step, a random baseline with no spec or adapter involved.

## Environment Authoring

The authoring surface declares an environment before anything is built: a factory with its construction params, the catalog of named variants, and the describe envelope tooling. The guide is {doc}`../user-guide/environments`.

```{eval-rst}
.. autoclass:: rlmesh.EnvFactory
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: rlmesh.Param
   :members:
```

```{eval-rst}
.. autoclass:: rlmesh.ParamSpec
   :members:
```

```{eval-rst}
.. autoclass:: rlmesh.Vector
   :members:
```

```{eval-rst}
.. autoclass:: rlmesh.Variant
   :members:
```

```{eval-rst}
.. autofunction:: rlmesh.describe
```

```{eval-rst}
.. autofunction:: rlmesh.describe_json
```

## Types

The `rlmesh.types` module defines the structural protocols that {py:class}`~rlmesh.EnvServer` accepts and the shared value aliases used by dependency-free clients. The protocols are structural, so any object with the right methods satisfies them; you do not subclass anything. Use them to type-annotate an environment or a value, or to check what `EnvServer` expects. For authoring an environment against these protocols see {doc}`../user-guide/environments`.

```{eval-rst}
.. automodule:: rlmesh.types
   :members:
   :show-inheritance:
```
