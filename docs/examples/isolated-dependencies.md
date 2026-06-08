# Isolated Dependencies

Some environments need packages you do not want in the evaluator process. The optional SAI examples
keep those dependencies in their own folders.

## SAI Pygame

```bash
cd examples/python/sai-pygame
mise run sync
mise run serve
```

This serves the Gymnasium registration:

```python
ENV_ID = "sai_pygame:SquidHunt-v0"
env = gym.make(ENV_ID)
EnvServer(env, "127.0.0.1:5555").serve()
```

In another terminal from the same folder:

```bash
mise run eval
```

## SAI MuJoCo

```bash
cd examples/python/sai-mujoco
mise run sync
mise run serve
```

This serves:

```python
ENV_ID = "sai_mujoco:So101IkColorSortPickPlace-v0"
env = gym.make(ENV_ID, render_mode="rgb_array")
```

The evaluator command stays the same because it only connects to the RLMesh endpoint.
