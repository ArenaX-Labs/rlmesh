# RLMesh

RLMesh is evaluation infrastructure for RL and VLA systems. The Python package
provides APIs for connecting models to environments and running
model-environment evaluation workflows.

> Early beta: APIs and package structure may change before a stable release.

## Installation

```bash
pip install --pre rlmesh
```

Optional adapters:

```bash
pip install --pre "rlmesh[numpy]"
pip install --pre "rlmesh[gymnasium]"
pip install --pre "rlmesh[torch]"
```

## Quickstart

Terminal 1:

```bash
python examples/python/quickstart/serve.py
```

Terminal 2:

```bash
python examples/python/quickstart/eval.py
```

The same eval script can connect to any example `EnvServer` listening on
`127.0.0.1:5555`, including the SAI MuJoCo and SAI Pygame example servers when
they are launched from their own example environments.

## Links

- Homepage: https://rlmesh.dev
- Repository: https://github.com/ArenaX-Labs/rlmesh
- Default contact: research@competesai.com

## License

Licensed under either MIT or Apache-2.0.
