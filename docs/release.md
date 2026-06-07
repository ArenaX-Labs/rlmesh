# Release Process

This beta is published manually from a local machine.

## Preflight

```bash
git status --short
mise run release:check
```

Confirm the version spellings:

- Rust crates: `0.1.0-beta.1`
- Python package: `0.1.0b1`

Confirm package and project access:

- The PyPI `rlmesh` project is available to the publishing account.
- The crates.io names in the workspace are available to the publishing account.
- `rlmesh.dev` and `docs.rlmesh.dev` resolve to the intended launch pages.

If using `fnox`, configure the local keychain values referenced by
`fnox.toml` before publishing:

- `CARGO_REGISTRY_TOKEN`
- `PYPI_TOKEN`

## Rust Crates

Verify packaging without uploading:

```bash
cargo package --workspace --allow-dirty
```

Publish crates in dependency order:

```bash
cargo publish -p rlmesh-proto
cargo publish -p rlmesh-spaces
cargo publish -p rlmesh-grpc
cargo publish -p rlmesh-runtime
cargo publish -p rlmesh
cargo publish -p rlmesh-cli
cargo publish -p rlmesh-sandbox
```

## Python Wheels

RLMesh currently publishes Python wheels only. Do not build or upload a Python
source distribution; native builds are covered by the explicit wheel matrix
below.

Local smoke builds may produce plain `linux_*` platform tags. Those wheels are
useful for installed-artifact validation but cannot be uploaded to PyPI. Release
validation intentionally rejects plain `linux_*` tags; expected Linux release
wheels use uploadable tags such as `manylinux` or `musllinux`.

The wheel checker validates both ABI/platform tags and payload contents. Wheels
must contain only runtime package files, type information, the native extension,
metadata, licenses, notices, and SBOMs; tests, Rust source, caches, and build
outputs are rejected.

Build release wheels:

```bash
mise run release:python:wheels
```

Inspect the wheel matrix:

```bash
python scripts/check_python_wheels.py python/rlmesh/dist
```

Upload only after inspecting `python/rlmesh/dist`:

```bash
maturin upload python/rlmesh/dist/*
```

## Post-Publish Smoke

Install from public indexes in a clean environment:

```bash
python -m venv /tmp/rlmesh-release-smoke
/tmp/rlmesh-release-smoke/bin/python -m pip install --pre rlmesh
/tmp/rlmesh-release-smoke/bin/python -c "import rlmesh; print(rlmesh.__version__)"
```
