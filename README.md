<div align="center">

<img src="docs/assets/brand-mark.svg" alt="RLMesh" width="96" height="96">

# RLMesh

**Plug any model into any Gymnasium environment, locally or over the network.**

[![CI](https://github.com/ArenaX-Labs/rlmesh/actions/workflows/ci.yml/badge.svg)](https://github.com/ArenaX-Labs/rlmesh/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/rlmesh.svg)](https://pypi.org/project/rlmesh/) [![crates.io](https://img.shields.io/crates/v/rlmesh.svg)](https://crates.io/crates/rlmesh) [![Docs](https://img.shields.io/badge/docs-rlmesh.dev-blue.svg)](https://rlmesh.dev/docs/)

</div>

Serve an environment in one process, drive it from another. Same `reset` / `step` / `render` / `close` you already know, just over the wire. Python on top, Rust underneath.

> **0.2 is an alpha.** It replaces the 0.1 gRPC protocol with real-time
> sessions, and 0.1 and 0.2 do not talk to each other. This page describes
> 0.2; a plain `pip install rlmesh` still gives 0.1, which is maintained on
> the [`release/0.1`](https://github.com/ArenaX-Labs/rlmesh/tree/release/0.1) branch.

## Install

```bash
pip install --pre "rlmesh[gymnasium,numpy]"
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

## Robots

`rlmesh.realtime` (Experimental) runs robots and policies on one network:
a host on your PC, robots that declare their parts (an ESP32 with two
servos, a camera, an arm), and policies that bind to them by kind and
drive several at once, with a cockpit that shows why each robot is or
isn't moving and recordings that export to LeRobot. Start with
[real-time sessions](docs/realtime.md).

```sh
pip install --pre "rlmesh[robot]"
rlmesh host
```

## Rust

Grab the [`rlmesh`](crates/rlmesh/README.md) crate. You'll need `protoc` installed (`apt install protobuf-compiler` / `brew install protobuf`) since the session protobuf messages are generated at build time.

## C and C++

The [C API](crates/rlmesh-capi/README.md) (`rlmesh.h`) and the header-only [C++ API](crates/rlmesh-capi/README-cpp.md) (`rlmesh.hpp`, C++17, with RAII handles and exceptions) are fenced for 0.2: they stay built and tested in CI, but no prebuilt C/C++ package is published. Source builds retain CMake and pkg-config support. 0.2 publishes the Python wheels and Rust crates; C#, Unity, Unreal and npm publication is also fenced.

```cpp
rlmesh::Model model([](const rlmesh::Value&) { return rlmesh::Value(0); });
auto result = model.run("127.0.0.1:7447/counter",
                        rlmesh::RunOptions{}.with_episodes(2).with_seeds({0, 1}));
```

RLMesh aims to feel familiar in every language: the same nouns and constructs (`Env`, `Space`, `Model`, `run`, `Binding`, `RemoteEnv`), each in that language's own conventions and with its own helpers.

## Integrations

- [Unreal Engine](integrations/unreal/RLMesh): serve actors as RLMesh environments.
- [Unity](integrations/unity): Experimental `com.rlmesh.unity` package with main-thread callbacks and reload preflight. Unity support is not yet verified on IL2CPP.
- [ESP32](integrations/esp32): `RLMesh`, an ESP32 on Wi-Fi as an RLMesh robot: one Arduino library, provisioned over USB, joining your host on its own. Built from source for now.
- [Arduino](integrations/arduino): `RLMeshTether`, the board side of a USB tether for boards without a network (Nano, Uno, Mega, UNO R4). Unreleased: the computer side is not in this release.

## Status

Pre-1.0. The Python package is the stable-ish surface; pin a minor version if you depend on it. See [compatibility](https://rlmesh.dev/docs/releases/compatibility/) for the details.

## Links

- Docs: https://rlmesh.dev/docs/
- Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md) · [local dev](docs/local-dev.md) · [testing](docs/testing.md)

## License

MIT or Apache-2.0, your pick. JPEG encoding is based in part on the work of the Independent JPEG Group.
