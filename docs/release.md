# Release Process

This beta is published manually from a local machine.

## Preflight

```bash
git status --short
mise run release:check
```

Review the compatibility policy before changing stable APIs, protocol fields, or package versions:
[`compatibility.md`](compatibility.md).

Confirm the version spellings:

- Rust crates: `0.1.0-beta.1`
- Python package: `0.1.0b1`

Confirm the release policy manifest:

```bash
python scripts/check_rlmesh_policy.py
```

Confirm protobuf compatibility against the checked-in public protocol baseline:

```bash
mise run protocol:breaking
```

This is also included in `mise run check` and therefore in `mise run release:check`.

Confirm package and project access:

- The PyPI `rlmesh` project is available to the publishing account.
- The crates.io names in the workspace are available to the publishing account.
- `rlmesh.dev` and `docs.rlmesh.dev` resolve to the intended launch pages.

If using `fnox`, configure the local keychain values referenced by `fnox.toml` before publishing:

- `CARGO_REGISTRY_TOKEN`
- `PYPI_TOKEN`

## Rust Crates

Core feature releases should move the public Rust crates together. Patch releases may be
artifact-specific when the fix is isolated, but republish any top-level crate that needs to depend
on a fixed lower-level crate.

Verify packaging without uploading:

```bash
mise run release:rust:package
```

Before the first crates.io publish, workspace crates that depend on other RLMesh crates cannot be
verified by `cargo package` as a full workspace because Cargo rewrites path dependencies to registry
dependencies during publish verification. The release task fully verifies independent crates and
then assembles all workspace tarballs with `--no-verify`; `mise run check` and `mise run test`
remain the local compilation and behavior gates.

Publish crates in dependency order:

```bash
cargo publish -p rlmesh-proto
cargo publish -p rlmesh-spaces
cargo publish -p rlmesh-cli
cargo publish -p rlmesh-runtime
cargo publish -p rlmesh-grpc
cargo publish -p rlmesh-sandbox
cargo publish -p rlmesh
```

## Python Wheels

Python is a core RLMesh artifact. Python-only fixes may produce a Python patch release without
forcing no-op publishes for unrelated bindings, but the package family in `rlmesh.toml` must stay
unchanged. Protocol generation or workflow edition changes need an explicit compatibility review.

RLMesh currently publishes Python wheels only. Do not build or upload a Python source distribution;
native builds are covered by the explicit wheel matrix below.

Local smoke builds may produce plain `linux_*` platform tags. Those wheels are useful for
installed-artifact validation but cannot be uploaded to PyPI. Release validation intentionally
rejects plain `linux_*` tags; expected Linux release wheels use uploadable tags such as `manylinux`
or `musllinux`.

The wheel checker validates both ABI/platform tags and payload contents. Wheels must contain only
runtime package files, type information, the native extension, metadata, licenses, notices, and
SBOMs; tests, Rust source, caches, and build outputs are rejected.

Build release wheels:

```bash
mise run release:python:wheels
```

Wheel builds are host-specific. Run `mise run release:python:wheels:macos` on macOS with Xcode
Command Line Tools installed to produce macOS wheels. Run `mise run release:python:wheels:linux` on
Linux to produce Linux and Windows wheels. The generic `release:python:wheels` task dispatches to
the current host's supported wheel set; it does not cross-link macOS frameworks from Linux. Release
wheel tasks remove stale `rlmesh-*.whl` files first so local smoke wheels with plain `linux_*` tags
cannot be uploaded accidentally.

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
