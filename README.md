<div align="center">

<img src="docs/assets/brand-mark.svg" alt="RLMesh" width="96" height="96">

# RLMesh

**Plug any model into any Gymnasium environment, locally or over the network.**

[![CI](https://github.com/ArenaX-Labs/rlmesh/actions/workflows/ci.yml/badge.svg)](https://github.com/ArenaX-Labs/rlmesh/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/rlmesh.svg)](https://pypi.org/project/rlmesh/) [![crates.io](https://img.shields.io/crates/v/rlmesh.svg)](https://crates.io/crates/rlmesh) [![Docs](https://img.shields.io/badge/docs-rlmesh.dev-blue.svg)](https://rlmesh.dev/docs/)

</div>

Serve an environment in one process, drive it from another. Same `reset` / `step` / `render` / `close` you already know, just over the wire. Python on top, Rust underneath.

## Install

```bash
pip install "rlmesh[gymnasium,numpy]"
```

## Quickstart

Serve an env:

```python
import gymnasium as gym
import rlmesh

rlmesh.EnvServer(gym.make("CartPole-v1"), "127.0.0.1:5555/cartpole").serve()
```

Use it from somewhere else:

```python
from rlmesh.numpy import RemoteEnv

env = RemoteEnv("127.0.0.1:5555/cartpole")
obs, info = env.reset(seed=0)

done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated

env.close()
```

More in [`examples/python`](examples/python): adapters, Docker sandboxes, bring-your-own containers.

### Endpoint authentication

A Rust, C or C++ served model or env requires a key when it is started with
`RLMESH_MODEL_ENDPOINT_TOKEN` / `RLMESH_ENV_ENDPOINT_TOKEN`. Clients send one
with `api_key=` (`authorization: Bearer <key>`), and an `https://` address
dials over TLS, so a policy behind a TLS load balancer drives your local env
directly. Python 0.2 serves envs and models on sessions, which check no
endpoint token yet: a set `RLMESH_ENV_ENDPOINT_TOKEN` fails `EnvServer` and a
set `RLMESH_MODEL_ENDPOINT_TOKEN` fails `Model.serve`, so keep the session's
network private.

Without `api_key`, clients read `RLMESH_API_KEY`, and send it only to
`https://` addresses. `tls_ca_file=` trusts a private CA bundle instead of the
system roots. The options are `RemoteEnvOptions::api_key` in Rust,
`RLMeshRemoteEnvOptions.api_key` in C and `with_api_key` in C++.

Servers send HTTP/2 keepalive PINGs on idle connections. They detect a dead
peer and keep NAT mappings and proxies that count PINGs as traffic alive, but
many L7 load balancers (AWS ALB among them) do not reset their idle timer on a
PING: set the proxy's stream idle timeout above the longest gap between
predicts, including the env's reset between episodes.

## Rust

Grab the [`rlmesh`](crates/rlmesh/README.md) crate. You'll need `protoc` installed (`apt install protobuf-compiler` / `brew install protobuf`) since the gRPC stubs are generated at build time.

## C and C++

The [C API](crates/rlmesh-capi/README.md) (`rlmesh.h`) and the header-only [C++ API](crates/rlmesh-capi/README-cpp.md) (`rlmesh.hpp`, C++17, with RAII handles and exceptions) ship as a prebuilt package on each GitHub Release, with CMake and pkg-config support.

```cpp
rlmesh::Model model([](const rlmesh::Value&) { return rlmesh::Value(0); });
auto result = model.run("tcp://127.0.0.1:5555", {.episodes = 2, .seeds = {0, 1}});
```

RLMesh aims to feel familiar in every language: the same nouns and constructs (`Env`, `Space`, `Model`, `run`, `Binding`, `RemoteEnv`), each in that language's own conventions and with its own helpers.

## Integrations

- [Unreal Engine](integrations/unreal/RLMesh): serve actors as RLMesh environments.
- [Unity](integrations/unity): Experimental `com.rlmesh.unity` package with main-thread callbacks and reload preflight. Unity support is not yet verified on IL2CPP.

## Status

Pre-1.0. The Python package is the stable-ish surface; pin a minor version if you depend on it. See [compatibility](https://rlmesh.dev/docs/releases/compatibility/) for the details.

## Links

- Docs: https://rlmesh.dev/docs/
- Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md) · [local dev](docs/local-dev.md) · [testing](docs/testing.md)

## License

MIT or Apache-2.0, your pick. JPEG encoding is based in part on the work of the Independent JPEG Group.
