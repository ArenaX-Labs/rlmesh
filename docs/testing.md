---
orphan: true
---

# Testing

RLMesh uses a small set of test layers so local iteration can stay fast while release checks still
exercise packaged artifacts.

## Fast Tests

Run the default Rust, Python unit, and Python integration tests:

```bash
mise run test
```

Focused variants:

```bash
mise run test:rust
mise run test:python
mise run test:python:unit
mise run test:python:integration
```

## API Surface Tests

The Python API surface tests check exported symbols, native stub exports, and the stable API surface
snapshot:

```bash
mise run test:python:api-surface
```

Run these when changing public Python modules, generated native stubs, or package exports.

## System Harness Tests

The system harness tests validate the runner and private fixture package without building or
installing a wheel:

```bash
mise run test:system:harness
```

Fixture scenarios and deterministic trace baselines live under `tests/system`.

## Installed-Artifact System Tests

Installed-artifact system tests validate built Python wheels in clean `uv` environments. They
exercise process boundaries, optional dependencies, deterministic traces, and artifact-level
benchmark signal.

List profiles and scenarios:

```bash
mise run test:system:list
```

Run the basic profile against `python/rlmesh/dist`:

```bash
mise run test:system -- --dry-run
mise run test:system
```

Run heavier optional profiles:

```bash
mise run test:system -- --profile gymnasium
mise run test:system -- --profile torch
mise run test:system -- --profile mujoco
mise run test:system:heavy
```

Clean system-test environments, logs, and reports:

```bash
mise run test:system:clean
```

## Release Check

The local release gate combines static checks, tests, package verification, wheel builds, and
installed-artifact validation:

```bash
mise run release:check
```
