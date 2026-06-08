# SAI MuJoCo Example

This example serves `sai_mujoco:So101IkColorSortPickPlace-v0` through RLMesh and reuses the shared
quickstart client for evaluation.

This folder intentionally has its own `pyproject.toml`, `uv.lock`, and mise-managed `.venv` so
MuJoCo and SAI dependencies stay out of the root development environment.

Start the server:

```bash
mise run sync
mise run serve
```

In another terminal from this same folder, run the client:

```bash
mise run eval
```

Without mise, use uv directly:

```bash
uv sync
uv run python serve.py
uv run python ../quickstart/eval.py
```
