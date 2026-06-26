# Bring-your-own container (env + model)

Hand-written Dockerfiles that produce RLMesh-compatible env and model images — the v0.1 replacement for recipe authoring. You own the Dockerfile and a small entrypoint; RLMesh runs the image. The same image works locally and on RLMesh Managed.

- `env/` — serves a Gymnasium env over the RLMesh protocol on port 50051. Dial it with `rlmesh.RemoteEnv(address)`.
- `model/` — serves a policy as an RLMesh model endpoint on port 50051.

## Quick start

```bash
# Env: build and serve
docker build -t my-env:latest env
docker run --rm -p 50051:50051 my-env:latest

# Model: build, then drive it against the env with the symmetric loop
docker build -t my-model:latest model
```

```python
import rlmesh

env = rlmesh.RemoteEnv("127.0.0.1:50051")
sess = rlmesh.session(rlmesh.SandboxModel("image://my-model:latest"), env)  # starts the container
obs, _ = sess.reset()
while not sess.done:
    action = sess.predict(obs)
    obs, reward, terminated, truncated, _ = sess.step(action)
```

`rlmesh.session(rlmesh.SandboxModel("image://<tag>"), env)` runs your prebuilt tag directly — no recipe, no build — and opens a route configured from the env's contract. The identical loop drives the un-managed pair by swapping the model construction:

```python
env = rlmesh.RemoteEnv(env_address)
sess = rlmesh.session(rlmesh.RemoteModel(model_address), env)
```

## The bare protocol contract

Both containers serve on `RLMESH_ADDRESS` (default `0.0.0.0:50051`): the env serves the environment, the model serves the policy. A client drives them with the loop above.

**Beta footgun:** the protocol handshake pins a provisional edition and fails closed — a hand-built container's `rlmesh` build must be compatible with the host's until the edition seals at GA. Pin the same `rlmesh` version in your Dockerfile as the host that drives it.
