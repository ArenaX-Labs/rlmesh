# Contracts and Specs

Contracts and specs describe what a remote endpoint serves.

## EnvContract

```{py:class} rlmesh.specs.EnvContract

Immutable description of an environment endpoint.
```

`EnvContract` is returned by:

- `EnvServer.env_contract`
- `EnvServer.spec`
- `RemoteEnv.env_contract`
- `RemoteEnv.spec`
- `RemoteVectorEnv.env_contract`
- `RemoteVectorEnv.spec`
- sandbox session `env_contract` and `spec` properties

The server builds the contract when it wraps the Python environment. For Gymnasium environments, `id` comes from `env.spec.id` when available and falls back to `"UnknownEnv-v1"`. Spaces are parsed from `observation_space` and `action_space` for single environments, or from `single_observation_space` and `single_action_space` for vector environments.

| Attribute           | Type                        | Meaning                                                            |
| ------------------- | --------------------------- | ------------------------------------------------------------------ |
| `id`                | `str`                       | Environment id reported by the wrapped environment.                |
| `env`               | `EnvContract`               | Alias returning the same contract shape.                           |
| `spec`              | `EnvContract`               | Alias returning the same contract shape.                           |
| `render_mode`       | `str \| None`               | Configured render mode, or `None` when no render mode is reported. |
| `num_envs`          | `int`                       | Number of environment instances served by the endpoint.            |
| `metadata`          | `dict[str, object] \| None` | Optional environment metadata.                                     |
| `observation_space` | `SpaceSpec`                 | Native spec for one observation.                                   |
| `action_space`      | `SpaceSpec`                 | Native spec for one action.                                        |

`to_dict()` returns a serializable dictionary containing the same fields, with nested space specs converted to dictionaries.

## SpaceSpec

```{py:class} rlmesh.specs.SpaceSpec

Immutable native description of a space.
```

`SpaceSpec` is the wire-safe description of a space. A `Space` wrapper is the Python object that can sample, seed, and validate values from that spec.

| Attribute or method | Type                   | Meaning                                                             |
| ------------------- | ---------------------- | ------------------------------------------------------------------- |
| `kind`              | `str`                  | Space family such as `"box"`, `"discrete"`, `"dict"`, or `"tuple"`. |
| `shape`             | `list[int]`            | Tensor-like shape for spaces where shape is meaningful.             |
| `dtype`             | `str`                  | Element dtype reported by the native space spec.                    |
| `_details()`        | `object`               | Native detail payload for kind-specific fields.                     |
| `_to_dict()`        | `dict[str, object]`    | Dictionary form used by higher-level wrappers.                      |
| `to_space()`        | `rlmesh._rlmesh.Space` | Native sampler and validator for the spec.                          |
| `to_gym_space()`    | `object`               | Best-effort conversion to a Gymnasium space.                        |

Use `rlmesh.spaces.space_from_spec(spec)` for a Python wrapper around a `SpaceSpec`. Use `rlmesh.spaces.to_gymnasium_space(spec)` when code expects a Gymnasium space object.

## ServeOptions

```{py:class} rlmesh.ServeOptions

Native options controlling endpoint lifecycle behavior.
```

| Constructor argument    | Type            | Meaning                                                   |
| ----------------------- | --------------- | --------------------------------------------------------- |
| `allow_remote_shutdown` | `bool`          | Whether clients may request endpoint shutdown.            |
| `idle_timeout_seconds`  | `float \| None` | Optional idle timeout before the server exits.            |
| `drain_timeout_seconds` | `float \| None` | Optional grace period for in-flight work during shutdown. |
| `close_timeout_seconds` | `float \| None` | Optional timeout for closing the wrapped environment.     |

Pass options to `EnvServer(..., options=options)` or model-serving APIs that accept serve options.

## Tensor

```{py:class} rlmesh.Tensor

Native tensor value used at the dependency-free RLMesh value boundary.
```

`Tensor` is a validated transport container: immutable element bytes plus shape, dtype, and stride metadata, with DLPack and buffer-protocol edges. It is not an ndarray. Compute, slicing, and broadcasting belong to the frameworks. The NumPy, Torch, and JAX backends convert tensor leaves to backend arrays or tensors.

| Attribute or method   | Type         | Meaning                                                       |
| --------------------- | ------------ | ------------------------------------------------------------- |
| `shape`               | `list[int]`  | Tensor dimensions.                                            |
| `dtype`               | `str`        | Element dtype name (for example `"float32"`).                 |
| `ndim`                | `int`        | Number of dimensions.                                         |
| `size`                | `int`        | Number of elements.                                           |
| `nbytes`              | `int`        | Element data size in bytes.                                   |
| `strides`             | `list[int]`  | Byte strides per dimension, C order.                          |
| `device`              | `str`        | Device holding the data; currently always `"cpu"`.            |
| `is_contiguous()`     | `bool`       | Whether elements are laid out C-contiguously.                 |
| `reshape(shape)`      | `Tensor`     | Same elements, new shape; zero-copy view when contiguous.     |
| `copy()`              | `Tensor`     | Deep copy backed by fresh storage.                            |
| `buffer`              | `memoryview` | Read-only N-D typed memory view over the elements.            |
| `tobytes()`           | `bytes`      | Copy the element bytes (C order) into Python bytes.           |
| `__dlpack__(...)`     | capsule      | DLPack export; negotiates legacy or v1.0 capsules.            |
| `__dlpack_device__()` | `(int, int)` | DLPack device tuple, `(kDLCPU, 0)`.                           |
| `from_dlpack(obj)`    | `Tensor`     | Static method; imports (and copies) from any DLPack producer. |

Use `rlmesh.numpy.asarray(tensor)` to get a writable NumPy copy of an RLMesh tensor (`numpy.from_dlpack(tensor)` for a zero-copy, read-only view), `rlmesh.torch.as_tensor(tensor)` to view or copy it as a Torch tensor, and `rlmesh.jax.asarray(tensor)` to import it as a JAX array.
