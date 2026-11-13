# Installation

Install the Python package. 0.2 is an alpha, so pip needs `--pre` to pick it; without it you get 0.1, which speaks a different protocol:

```bash
pip install --pre "rlmesh[gymnasium,numpy]"
```

To pin the alpha exactly (in a Dockerfile, say), name its version: `pip install "rlmesh[gymnasium,numpy]==0.2.0a0"`.

RLMesh supports Python 3.10 and newer. Start with Gymnasium for environments and the NumPy backend for examples and notebooks, then follow the [quickstart](https://rlmesh.dev/docs/start/quickstart/).

## Optional Extras

Install only the extras you need:

```bash
pip install --pre rlmesh
pip install --pre "rlmesh[numpy]"
pip install --pre "rlmesh[gymnasium]"
pip install --pre "rlmesh[torch]"
pip install --pre "rlmesh[jax]"
pip install --pre "rlmesh[hf]"
pip install --pre "rlmesh[robot]"
```

Pick `gymnasium` when serving a Gymnasium environment, or `gym` for a legacy classic-Gym stack. `torch` decodes client-side values as Torch tensors; `jax` decodes them as JAX arrays; `numpy` decodes them as arrays. `hf` adds host-side, container-less resolution of `hf://` model weights and EnvHub sources; in a sandbox the container fetches them for you. `robot` adds NumPy, which real-time sessions (`rlmesh.realtime`: robots, policies, recordings) use for samples and frames, on a robot's computer and a policy's alike. `rlmesh.realtime` itself and the `rlmesh host` and `rlmesh cockpit` commands come with every install (the cockpit on desktop platforms). See [backends](https://rlmesh.dev/docs/guides/value-backends/) for how the backend extras change value decoding.

## From Source

Until the alpha is on PyPI, install this branch from source. The build needs a Rust toolchain and `protoc` (`apt install protobuf-compiler` / `brew install protobuf`):

```bash
RLMESH_RELEASE_BUILD=1 pip install "rlmesh[gymnasium,numpy] @ git+https://github.com/ArenaX-Labs/rlmesh@v2/core#subdirectory=python/rlmesh"
```

`RLMESH_RELEASE_BUILD=1` stamps the alpha's workflow edition. Without it the build is a dev build, which only talks to a build of the identical commit (see [dev builds](local-dev.md#dev-builds-and-the-dev-cohort)). Install the same commit in every process and container that takes part.

## Repository Examples

Inside this repository, use the pinned development environment:

```bash
mise install
mise run setup
```

Then run examples with `uv run`. Sandbox examples need Docker access. The in-repository examples share this environment; heavier demos that need their own lockfile live in the separate `rlmesh-examples` repository.
