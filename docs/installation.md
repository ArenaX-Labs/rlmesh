# Installation

Install the published Python beta:

```bash
pip install --pre "rlmesh[gymnasium,numpy]"
```

RLMesh supports Python 3.10 and newer. Start with Gymnasium for environments and the NumPy backend
for examples and notebooks.

## Optional Extras

Install only the extras you need:

```bash
pip install --pre rlmesh
pip install --pre "rlmesh[numpy]"
pip install --pre "rlmesh[gymnasium]"
pip install --pre "rlmesh[torch]"
pip install --pre "rlmesh[hf]"
```

Use `gymnasium` when serving a Gymnasium environment (or `gym` for a legacy classic-Gym stack). Use
`torch` only when you want client-side values decoded as Torch tensors. Use `hf` for host-side,
container-less resolution of `hf://` model weights and EnvHub sources; in a sandbox the container
fetches them for you.

## Repository Examples

Inside this repository, use the pinned development environment:

```bash
mise install
mise run setup
```

Then run examples with `uv run`. Sandbox examples need Docker access. Optional example folders keep
their own lockfiles and environments so heavier dependencies do not leak into the root development
environment.
