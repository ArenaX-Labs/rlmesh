# Python Examples

These examples are small and copyable. Most server/client examples default to `127.0.0.1:5555`.
Sandbox examples start their own Docker-backed environment process.

## Quickstart

Use this first when learning the RLMesh server/client shape. It serves Gymnasium `CartPole-v1` and
uses the NumPy client adapter.

```bash
uv run python examples/python/quickstart/serve_gymnasium.py
```

In another terminal:

```bash
uv run python examples/python/quickstart/eval.py
```

If copying these files outside the repository, install the published package:

```bash
pip install --pre "rlmesh[gymnasium,numpy]"
```

For a custom object without Gymnasium, see `quickstart/serve.py`.

## Sandbox Examples

Use these when the environment should start in an owned Docker-backed process instead of a separate
server terminal.

```bash
uv run python examples/python/sandbox/gym_sandbox.py
```

Available sandbox examples:

- [`sandbox/gym_sandbox.py`](sandbox/gym_sandbox.py): starts Gymnasium `CartPole-v1` in a sandbox.
- [`sandbox/hf_sandbox.py`](sandbox/hf_sandbox.py): starts
  `hf://lerobot/cartpole-env:cartpole_suite/0`.

## Optional Environment Examples

The SAI examples keep their own `pyproject.toml`, lockfile, and `.venv`. That is intentional: these
dependencies are optional and can be heavier than the normal RLMesh development environment.

With `mise`, opening or entering one of these folders creates and sources that example's `.venv`
automatically. Each folder also exposes local tasks:

```bash
cd examples/python/sai-pygame
mise run sync
mise run serve
```

In another terminal from the same folder:

```bash
mise run eval
```

Available optional examples:

- [`sai-pygame`](sai-pygame): serves `sai_pygame:SquidHunt-v0`.
- [`sai-mujoco`](sai-mujoco): serves `sai_mujoco:So101IkColorSortPickPlace-v0`.
