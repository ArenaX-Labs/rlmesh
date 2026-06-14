# Adapters quickstart

The smallest end-to-end `rlmesh.adapters` example: an environment tags its spaces, a model declares
its format, and they are paired automatically — no per-environment glue in the model.

```sh
cd examples/python/adapters_quickstart
uv run serve_and_run.py
```

It runs one process:

1. `CubePickEnv` exposes a wrist camera, end-effector pose, and gripper, plus a 7-dim delta action.
2. `EnvServer(env, tags=ENV_TAGS)` serves it and publishes the tags in the contract (validated
   against the env's spaces first).
3. `MODEL_SPEC` declares a checkpoint that wants a 224x224 image, a flat `rot6d` proprio list, and a
   10-dim `rot6d` action — conventions that do not match the env.
4. `Model(predict, spec=MODEL_SPEC).run(client)` resolves the adapter from the env's contract and
   runs an episode. `predict` only ever sees the model's own format; the env only ever sees its own.

The script first prints `resolve_from_contract(...).describe()`, the exact transformations chosen:
the image is resized, `quat_xyzw -> rot6d` is applied to the rotation, the instruction key is
remapped (`goal -> task`), and on the way back the 10-dim `rot6d` action is converted
`rot6d -> axis_angle`, sliced, and clipped into the env's 7-dim action.

See [`../vla_adapters`](../vla_adapters) for the multi-model, multi-env project layout, the
custom-adapter escape hatches, and offline dry runs.
