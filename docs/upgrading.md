# Upgrading from 0.1

0.2 replaces 0.1's gRPC transport with real-time sessions. A 0.1 peer and a 0.2 peer cannot talk to each other, so you move both sides of a connection together: the env, the model and whatever drives them. This page lists what a 0.1 project has to change. Most 0.1 names that 0.2 dropped still import and raise an error naming their replacement, so running your code on 0.2 finds most of these places for you.

0.2 is an alpha: pip needs `--pre` to install it (`pip install --pre rlmesh`), and anything may still change before 0.2.0.

## Staying on 0.1

Move when you can move every side. Until then, pin the 0.1 line:

```bash
pip install "rlmesh<0.2"
```

`<0.2` excludes the 0.2 pre-releases too. In Rust, `rlmesh = "0.1"` in `Cargo.toml` stays on 0.1.x. A 0.1 install keeps running the sealed `2026.06` workflow edition, which 0.2 does not offer.

## Rebuild what was built on 0.1

Anything that carries rlmesh 0.1 inside it speaks the old protocol: env and model container images, wheels and lock files that pin `rlmesh` 0.1. Rebuild each one with 0.2. 0.2 refuses a 0.1 image instead of hanging on it: `SandboxEnv` and `SandboxModel` fail with `... was built with rlmesh 0.1; rebuild it with rlmesh 0.2`, and `rlmesh check-image` fails it.

A session listens on port 7447 by default, where 0.1 served on 50051: 50051 is gRPC's customary port, and 7447 is Zenoh's, which 0.2's sessions run on. A container that serves with `python -m rlmesh.serve` changes like this:

```dockerfile
# 0.1
RUN pip install --no-cache-dir "rlmesh"
EXPOSE 50051
ENV RLMESH_ADDRESS=0.0.0.0:50051
```

```dockerfile
# 0.2
RUN pip install --no-cache-dir "rlmesh==0.2.0a1"
EXPOSE 7447
ENV RLMESH_ADDRESS=0.0.0.0:7447
ENV RLMESH_READY_FILE=/tmp/rlmesh-ready.json
HEALTHCHECK --interval=2s --start-period=120s CMD test -f /tmp/rlmesh-ready.json
```

The ready file and `HEALTHCHECK` are optional; see [Serving from the command line](#serving-from-the-command-line).

## Addresses

An env or model is a named participant in a session: `host:port/name`. Where an env is served or dialled, a bare `host:port` is short for `host:port/env`, so the 0.1 quickstart's addresses still meet:

```python
import rlmesh

server = rlmesh.EnvServer(CounterEnv(), "127.0.0.1:5555")  # serves 127.0.0.1:5555/env
server.start()

env = rlmesh.RemoteEnv("127.0.0.1:5555")
obs, info = env.reset(seed=0)
env.close()

result = rlmesh.Model(lambda observation: 0).run("127.0.0.1:5555", episodes=2)
```

An env takes one driver at a time: close a `RemoteEnv` before a model runs against the same env.

On one machine, as the same user, the address is all you need. From another machine or a container, use the full address the server reports (`server.address`): it carries a key after `#` that the server makes when it starts.

A served model has no default name. Serve it under one and dial it by that name:

```python
# 0.1
model.serve("127.0.0.1:50051")
remote = rlmesh.RemoteModel("127.0.0.1:50051")
```

```python
# 0.2
model.serve("127.0.0.1:7447", name="policy")
remote = rlmesh.Model("127.0.0.1:7447/policy")
```

`Model.serve(address)` without `name=` raises `TypeError`, and `Model("host:port")` without a name raises `ValueError` with an example of the full form. Serving on every interface (`0.0.0.0`), in process or from the command line, also needs a host identity; see [Serving from the command line](#serving-from-the-command-line).

0.2 refuses 0.1's other endpoint forms:

| 0.1 address                                              | In 0.2                                                                                       |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `tcp://…`, `unix://…`, `https://…`                       | refused: `v1 endpoints need RLMesh 0.1`. Serve the env at a session address and pass that.   |
| `5555` (a bare port)                                     | refused, naming the fix: `":5555"` for this machine's port, or `"host:5555"`.                |
| `RemoteEnv(host=..., port=..., path=..., transport=...)` | the keywords are gone (`TypeError`); pass the address string.                                |
| an open `RemoteEnv` passed to `run` or `bind`            | refused (`TypeError`): the open `RemoteEnv` holds the env, so close it and pass the address. |

Dialling a server that still runs 0.1 fails within about a second with `ProtocolException`:

```text
the server at 127.0.0.1:5555 doesn't speak the RLMesh session protocol: it answers HTTP/2,
as an RLMesh 0.1 (gRPC) server does. RLMesh 0.2 can't connect to 0.1; serve it with RLMesh 0.2
```

## Workflow editions

0.2 runs workflow edition `2026.10`, provisional until 0.2.0 seals it, and refuses `2026.06` wherever it is declared. Serving refuses it, and so does a local run or binding: `model.run(MyEnvFactory())` or `model.bind(env)` with an env or factory that declares `2026.06` fails before the factory makes an env:

```text
ValueError: workflow edition "2026.06" is unknown to this rlmesh build, which knows 2026.10; [...]
```

0.2 reads an edition from two places only: `RLMESH_WORKFLOW_EDITION`, then `workflow_edition` on the `EnvFactory` or `Model` class. Remove the rest:

- `RLMESH_WORKFLOW_EDITION=2026.06` in a Dockerfile or launch script, and `workflow_edition = "2026.06"` on a class
- `python -m rlmesh.serve --workflow-edition ...` (the flag is gone)
- `run(..., workflow_edition=...)`, `bind(..., workflow_edition=...)` and `RemoteEnv(..., workflow_edition=...)` (the keywords are gone)
- `[tool.rlmesh] workflow_edition` in `pyproject.toml` (no longer read)

A participant that declares nothing runs at its build's current edition. Declare one when you need an evaluation to mean the same thing after you upgrade rlmesh, with the value `rlmesh.current_workflow_edition()` prints (`2026.10` on every 0.2 build):

```python
class MyEnv(rlmesh.EnvFactory):
    workflow_edition = "2026.10"

    def make(self): ...
```

An edition is declared locally and never negotiated; see [workflow editions](editions/index.md).

## Removed and renamed Python APIs

| 0.1                                                                                            | 0.2                                                                                     |
| ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `rlmesh.Session`, `rlmesh.session(model, env)`                                                 | `model.bind(target)` returns a `Binding` you step by hand                               |
| `model.session(env)`                                                                           | `model.bind(env)`                                                                       |
| `SandboxModel(...).session(env)`                                                               | `SandboxModel(...).bind(env)`                                                           |
| `rlmesh.run(SandboxModel(...), env)`                                                           | `SandboxModel(...).run(env)`                                                            |
| `RemoteModel(address)`                                                                         | `Model("host:port/name")`                                                               |
| `model.serve(address)`                                                                         | `model.serve(address, name="policy")`                                                   |
| `Recorder.capture(..., session=...)`                                                           | `Recorder.capture(..., binding=...)`                                                    |
| `rlmesh.__build__`                                                                             | `rlmesh.build_info().workflow_edition`                                                  |
| `rlmesh.platform`, `PlatformError`                                                             | `rlmesh.managed`, `ManagedError` (no alias: `rlmesh.platform` fails to import)          |
| `ServeOptions(allow_remote_shutdown=..., idle_timeout_seconds=..., drain_timeout_seconds=...)` | gone (`TypeError`); `ServeOptions` keeps `close_timeout_seconds` and `workflow_edition` |

Each of the first eight raises an error that names its replacement, such as:

```text
TypeError: Model.session(env) is gone in rlmesh 0.2: model.bind(target) returns a Binding
you step by hand (reset(), step(), step(action))
```

A `Binding` has no separate `predict`: `step()` asks the model for its action and steps the env with it, and `step(action)` steps with yours. To run whole episodes, call `model.run(target)` instead of `Session.run()`.

```python
# 0.1
with model.session("127.0.0.1:5555") as session:
    obs, info = session.reset(seed=0)
    action = session.predict(obs)
    obs, reward, terminated, truncated, info = session.step(action)
```

```python
# 0.2
with model.bind("127.0.0.1:5555") as binding:
    obs, info = binding.reset(seed=0)
    while not binding.done:
        obs, reward, terminated, truncated, info = binding.step()
```

Closing a binding leaves the model open for the next `bind` or `run`.

The [changelog](../CHANGELOG.md)'s Breaking section lists the behaviour changes that need no code change but can change what you see, such as the exception types env errors raise and how `base_seed` derives each episode's seed.

## Serving from the command line

`python -m rlmesh.serve` takes the same entrypoints as in 0.1. What a launch script may notice:

- It serves on `0.0.0.0:7447` by default (it was `0.0.0.0:50051`), as participant `model`, or `env` with `--env`. `--name` or `RLMESH_NAME` sets another name; dial it at `host:port/<name>`.
- Serving on every interface needs a host identity in `RLMESH_HOST_IDENTITY`. Without one it exits with status 2 before it loads the model or env, saying how to make one. A sandbox passes one; `RLMESH_ADDRESS=127.0.0.1:7447` serves this machine only and needs none.
- SIGTERM drains the call in progress and exits 0; Ctrl-C exits 130.
- `RLMESH_READY_FILE=PATH` writes a small JSON file once the env or model answers, for a launcher or `HEALTHCHECK` to wait on.

The `rlmesh` CLI keeps its 0.1 commands and flags. `rlmesh check-image` now checks that an image exposes 7447 rather than 50051, and fails an image built with 0.1.

## Rust, C and C++

The `rlmesh` Rust crate is rebuilt around the same pieces as Python: envs implement `Env` or `VectorEnv` and are served with `EnvServer`, models implement `Policy` (or are a closure) wrapped in `Model`, and `Model::bind` and `run` replace the 0.1 worker and runtime types. `rlmesh-grpc`, `rlmesh-runtime`, `rlmesh-sandbox` and `rlmesh-viewer` are gone. The [changelog](../CHANGELOG.md) lists each break, and the [crate README](../crates/rlmesh/README.md) shows the 0.2 API.

The C API moved from ABI generation 4 to development generation 6, so a 0.1 C or C++ program does not build or load against it unchanged. 0.2 does not publish the C API; it will ship as ABI 7. See the [C API README](../crates/rlmesh-capi/README.md).

## Checklist

- [ ] Every side of each connection (envs, models, drivers, images) moves to 0.2 together, or stays pinned to `rlmesh<0.2`.
- [ ] Env and model images are rebuilt with 0.2 and expose port 7447.
- [ ] `tcp://`, `unix://`, `https://` and bare-port addresses are now `host:port` or `host:port/name`.
- [ ] Every served model has a `name=`, and its clients dial `host:port/<name>`.
- [ ] No `2026.06` declaration is left in code, `pyproject.toml`, Dockerfiles or launch scripts (`rlmesh check module:Class`).
- [ ] Participants that need reproducible evaluation declare `workflow_edition = "2026.10"` in source.
- [ ] `Session`, `session()`, `RemoteModel` and the other removed names are replaced; the code runs on 0.2 without a `... is gone in rlmesh 0.2` error.
- [ ] Launch scripts set `RLMESH_HOST_IDENTITY` or a loopback `RLMESH_ADDRESS`, and expect exit 0 on SIGTERM.
