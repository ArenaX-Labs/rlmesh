# Gymnasium Compatibility

RLMesh is built around the Gymnasium environment shape:

```python
env = gym.make("CartPole-v1")
EnvServer(env, "127.0.0.1:5555").serve()
```

The server reads `observation_space`, `action_space`, `reset`, `step`, and `close`. Common Gymnasium
wrappers can stay in place.

## Spaces

RLMesh supports these Gymnasium spaces. The **Stability** column matches the API surface policy in
`api_metadata.json`: `Stable` spaces follow the compatibility guarantees in {doc}`compatibility`,
while `Experimental` spaces may still change before the stable release.

### Fundamental Spaces

| Gymnasium space | RLMesh space    | Stability    | Notes                                                                     |
| --------------- | --------------- | ------------ | ------------------------------------------------------------------------- |
| `Box`           | `Box`           | Stable       | Uniform and array bounds are accepted.                                    |
| `Discrete`      | `Discrete`      | Stable       | `start` is preserved.                                                     |
| `MultiBinary`   | `MultiBinary`   | Experimental | Integer and shaped forms are accepted.                                    |
| `MultiDiscrete` | `MultiDiscrete` | Experimental | One- and two-dimensional `nvec` are supported.                            |
| `Text`          | `Text`          | Experimental | Custom charsets are preserved; default charsets are treated as unbounded. |

For `Text`, RLMesh still preserves `min_length` and `max_length`. Only the default Gymnasium charset
is treated as unrestricted, so punctuation and whitespace are not rejected just because the source
space used Gymnasium's default alphanumeric charset.

### Composite Spaces

| Gymnasium space | RLMesh space | Stability    | Notes                            |
| --------------- | ------------ | ------------ | -------------------------------- |
| `Tuple`         | `Tuple`      | Experimental | Supported when child spaces are. |
| `Dict`          | `Dict`       | Stable       | Supported when child spaces are. |

Not supported in this beta:

- `Graph`
- `Sequence`
- `OneOf`

Unsupported spaces fail directly instead of silently changing the environment contract.

## Conversion Helpers

Use these helpers when you need to convert spaces yourself:

```python
from rlmesh import spaces

rlmesh_space = spaces.from_gymnasium_space(gym_space)
gym_space = spaces.to_gymnasium_space(rlmesh_space)
```
