# Python Examples

These examples are meant to be runnable from the repository and easy to copy into a separate
project. Start a server in one terminal, then run a client in a second terminal. All examples use
`127.0.0.1:5555`.

## Quickstart

Use this first when learning or copying the RLMesh server/client shape. It uses a tiny in-file
environment and the NumPy client adapter.

```bash
uv run python examples/python/quickstart/serve.py
```

In another terminal:

```bash
uv run python examples/python/quickstart/eval.py
```

If copying these files outside the repository, install the published package:

```bash
pip install --pre "rlmesh[numpy]"
```

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
