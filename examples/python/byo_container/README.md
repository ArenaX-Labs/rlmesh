# Bring-your-own container (env + model)

Hand-written Dockerfiles that produce RLMesh-compatible env and model images — the replacement for recipe authoring. You own the Dockerfile and a small entrypoint; RLMesh runs the image. The same image works locally and on RLMesh Managed.

- `env/` — hosts an RLMesh session on port 7447 and serves a Gymnasium env in it as the participant `env`.
- `model/` — hosts an RLMesh session on port 7447 and serves a policy in it as the participant `model`.

A session host listens on four consecutive ports (7447-7450), one per traffic class.

## Quick start

```bash
docker build -t my-env:latest env
docker build -t my-model:latest model
```

```python
import rlmesh

# Each sandbox starts its container and publishes the session's ports on loopback.
with rlmesh.SandboxEnv("image://my-env:latest") as env:
    obs, _ = env.reset()
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())

model = rlmesh.SandboxModel("image://my-model:latest")
with model:
    result = model.run(MyLocalEnv(), episodes=1)  # the env is served in the model's session
    with model.bind(MyLocalEnv()) as binding:  # or step by hand
        obs, _ = binding.reset()
        while not binding.done:
            obs, reward, terminated, truncated, _ = binding.step()
```

`SandboxModel("image://<tag>")` runs your prebuilt tag directly — no recipe, no build. Its `run` and `bind` drive the model as `rlmesh.Model(address)` does, serving a local env in the session the container hosts, which it can reach. A model served on a session you started yourself is reached by its participant address:

```python
result = rlmesh.Model("127.0.0.1:7447/model").run(env, episodes=1)
```

## The bare protocol contract

Both containers host a session on `RLMESH_ADDRESS` (default `0.0.0.0:7447`) and serve in it as `RLMESH_NAME` (`env` or `model`). A sandbox picks four free ports, hosts the session on them and publishes them to the same numbers on loopback, so a session address means the same inside the container and out. A participant that joins a session host another process runs joins it as a Zenoh client, which the host routes for, so the published ports are all the machine outside needs to reach.

An image must be built with rlmesh 0.2: one built with rlmesh 0.1 serves the v1 wire, and a sandbox refuses it, naming the image. Pin the same `rlmesh` version in your Dockerfile as the host that drives it.
