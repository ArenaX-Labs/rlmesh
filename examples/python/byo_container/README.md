# Bring-your-own container (env + model)

Hand-written Dockerfiles that produce RLMesh env and model images. You own the Dockerfile and a small entrypoint; RLMesh runs the image. The same image works locally and on RLMesh Managed.

- `env/`: serves a Gymnasium env.
- `model/`: serves a policy.

## Quick start

Build both images, then run them as sandboxes. A sandbox starts the container and connects to it for you:

```bash
docker build -t my-env:latest env
docker build -t my-model:latest model
```

```python
import rlmesh

with rlmesh.SandboxEnv("image://my-env:latest") as env:
    obs, _ = env.reset()
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())

model = rlmesh.SandboxModel("image://my-model:latest")
with model:
    result = model.run(MyLocalEnv(), episodes=1)  # the env is served beside the model
    with model.bind(MyLocalEnv()) as binding:  # or step by hand
        obs, _ = binding.reset()
        while not binding.done:
            obs, reward, terminated, truncated, _ = binding.step()
```

`SandboxModel("image://<tag>")` runs your prebuilt tag directly: no recipe, no build. Its `run` and `bind` drive the model as `rlmesh.Model(address)` does.

## Running the container yourself

Without a sandbox, a container serving on every interface needs a key its launcher makes: run `rlmesh host-identity new --address HOST:7447 --participant env`, pass the identity it prints to the container as `RLMESH_HOST_IDENTITY`, and connect to the address it prints (`rlmesh.RemoteEnv(address)`, `rlmesh.Model(address)`).

Each image sets `RLMESH_READY_FILE`, so `docker ps` shows it healthy once the env or model answers. A launcher can wait on the same file (`test -f /tmp/rlmesh-ready.json`) instead of reading the log. The file holds `{"kind", "participant", "address", "port", "pid"}`, the address without its key, and goes away when the server stops.

For a model, with the identity in a file:

```bash
rlmesh host-identity new --out model.identity --address 127.0.0.1:7447 --participant model
# prints key=... and address=127.0.0.1:7447#<key>/model
RLMESH_HOST_IDENTITY="$(cat model.identity)" \
  docker run --rm -e RLMESH_HOST_IDENTITY -p 7447-7450:7447-7450 my-model:latest
```

```python
result = rlmesh.Model("127.0.0.1:7447#<key>/model").run(MyLocalEnv(), episodes=1)
```

`run` and `bind` serve a local env in a session of its own on the model's host, through the ports the container publishes, so the model reaches it inside its own container. From another machine, dial the same key at the address that reaches those ports (`10.0.0.5:7447#<key>/model`). An env given by address instead is joined by the model, which must reach that address from inside its container.

## The bare protocol contract

For orchestrators and custom launchers. A session host listens on four consecutive ports (7447-7450), one per traffic class. Both containers host a session on `RLMESH_ADDRESS` (default `0.0.0.0:7447`) and serve in it as `RLMESH_NAME` (`env` or `model`). A sandbox picks four free ports, hosts the session on them and publishes them to the same numbers on loopback, so a session address means the same inside the container and out. A participant that joins a session host another process runs joins it as a Zenoh client, which the host routes for, so the published ports are all the machine outside needs to reach.

Listening beyond loopback, as `0.0.0.0` in a container does, the session host starts only under a host identity its launcher passes in `RLMESH_HOST_IDENTITY` (a secret: pass it through the environment, never argv), and its participants dial the keyed address that identity's key forms (`127.0.0.1:7447#{key}/env`), from this machine or another one that reaches the published ports (`10.0.0.5:7447#{key}/env`): the key is the only credential. A sandbox does this for you; any other launcher makes the identity and address with `rlmesh.launch` in Python or `rlmesh host-identity new --address HOST:PORT --participant NAME` from a shell.

An image must be built with rlmesh 0.2: one built with rlmesh 0.1 serves the v1 wire, and a sandbox refuses it, naming the image. Pin the same `rlmesh` version in your Dockerfile as the host that drives it.
