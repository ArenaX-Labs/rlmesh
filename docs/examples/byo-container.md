# Bring-Your-Own Container

A bring-your-own container is a Docker image you build yourself: you write the Dockerfile and a small entrypoint, and RLMesh runs the image. The same image works locally and on the managed platform. Sandbox helpers like `SandboxModel` are experimental.

The runnable files live in `examples/python/byo_container`. There are two images: `env/` serves a Gymnasium environment, and `model/` serves a policy. Both serve on `RLMESH_ADDRESS` (default `0.0.0.0:50051`).

## Environment container

The env image installs `rlmesh` and the environment's dependencies, then runs an entrypoint that serves a Gymnasium environment with `EnvServer`:

```python
import os

import gymnasium as gym
from rlmesh import EnvServer


def make_env():
    return gym.make("CartPole-v1")


address = os.environ.get("RLMESH_ADDRESS", "0.0.0.0:50051")
EnvServer(make_env(), address).serve()
```

Build the image and run it, then dial it with `rlmesh.RemoteEnv`:

```bash
docker build -t my-env:latest examples/python/byo_container/env
docker run --rm -p 50051:50051 my-env:latest
```

```python
import rlmesh

env = rlmesh.RemoteEnv("127.0.0.1:50051")
obs, info = env.reset(seed=0)
```

The Dockerfile is {source}`examples/python/byo_container/env/Dockerfile <examples/python/byo_container/env/Dockerfile>` and the entrypoint is {source}`examples/python/byo_container/env/entrypoint.py <examples/python/byo_container/env/entrypoint.py>`.

## Model container

The model image serves a policy. Its entrypoint wraps a `predict` function in `Model` and serves it on the same address:

```python
import os

from rlmesh.numpy import Model


def load_policy():
    def predict(observation):
        return 0  # always push the cart left

    return predict


address = os.environ.get("RLMESH_ADDRESS", "0.0.0.0:50051")
Model(load_policy()).serve(address)
```

Build the tag, then drive it against an environment. `SandboxModel` runs a prebuilt `image://` tag directly, with no build step, and opens a route from the environment's contract:

```bash
docker build -t my-model:latest examples/python/byo_container/model
```

```python
import rlmesh

env = rlmesh.RemoteEnv("127.0.0.1:50051")
sess = rlmesh.session(rlmesh.SandboxModel("image://my-model:latest"), env)

obs, _ = sess.reset()
while not sess.done:
    action = sess.predict(obs)
    obs, reward, terminated, truncated, _ = sess.step(action)
```

`rlmesh.session(rlmesh.SandboxModel("image://<tag>"), env)` starts the model container, opens a route configured from the environment's contract, and returns a {class}`~rlmesh.Session` you drive with `reset` / `predict` / `step`. The identical loop drives a model that is already running: swap the construction for `rlmesh.session(rlmesh.RemoteModel("127.0.0.1:50052"), env)`, on a distinct port since the environment already holds `50051`. A prebuilt `image://` tag runs from its own baked configuration, so `SandboxModel` does not inject a bootstrap payload.

The Dockerfile is {source}`examples/python/byo_container/model/Dockerfile <examples/python/byo_container/model/Dockerfile>` and the entrypoint is {source}`examples/python/byo_container/model/entrypoint.py <examples/python/byo_container/model/entrypoint.py>`.

## Both sides in a sandbox

The same drive loop runs when RLMesh owns both containers. A `SandboxEnv` builds the environment container from a Gymnasium or Hugging Face source, and a `SandboxModel` runs your prebuilt `image://` tag. Closing the session stops the model container it started; closing the env stops the environment container:

```python
import rlmesh

env = rlmesh.SandboxEnv(
    "CartPole-v1",
    build=rlmesh.SandboxBuild(packages=["gymnasium==1.3.0"], imports=["gymnasium"]),
)
try:
    with rlmesh.session(rlmesh.SandboxModel("image://my-model:latest"), env) as sess:
        obs, _ = sess.reset()
        while not sess.done:
            action = sess.predict(obs)
            obs, reward, terminated, truncated, _ = sess.step(action)
finally:
    env.close()
```

A sandboxed environment is built from a source, so it takes a Gymnasium id or `gym://`/`hf://` reference rather than an `image://` tag; the bring-your-own `image://` path is for the model. See {doc}`sandboxes` for the environment side.

## Version pinning

The protocol handshake pins the workflow edition and fails closed. Released builds advertise the sealed `2026.06` edition; source builds use exact `dev.<git>` cohorts that only interoperate with the same build. Pin the same `rlmesh` version in your Dockerfile as the host that drives it. To run on the managed platform, `docker push` the tag to a registry the platform can reach; it runs the identical image.

## Where next

- {doc}`sandboxes`: let RLMesh build and own the environment container.
- {doc}`../user-guide/sandbox`: the sandbox helpers in depth.
- {doc}`../user-guide/serving-environments`: addresses, readiness, and health.
- {doc}`/compatibility`: version and edition pinning rules.
