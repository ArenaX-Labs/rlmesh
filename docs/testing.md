---
orphan: true
---

# Testing

RLMesh uses a few test layers so local iteration stays fast while release checks still exercise packaged artifacts.

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

Rerun a task whenever a file changes:

```bash
mise watch test:rust
mise watch test:python:unit
```

## CI Parity

`test:ci` is the single test entrypoint CI runs after `check`. It runs `test`, the API surface tests, the example programs, the system harness tests, then builds a local wheel and runs the installed-artifact system profile against it:

```bash
mise run test:ci
```

If `check` and `test:ci` pass locally, the CI fast job passes (it additionally rebuilds the C/C++ smoke under clang and gcc and builds the docs). On push, CI also runs `release:rust:package` and `test:cxx:pkg` in a separate package job.

## API Surface Tests

The Python API surface tests check exported symbols, native stub exports, and the stable API surface snapshot:

```bash
mise run test:python:api-surface
```

Run these when changing public Python modules, generated native stubs, or package exports.

## System Harness Tests

The system harness tests validate the runner and private fixture package without building or installing a wheel:

```bash
mise run test:system:harness
```

Fixture scenarios and deterministic trace baselines live under `tests/system`.

## Installed-Artifact System Tests

Installed-artifact system tests validate built Python wheels in clean `uv` environments. They cover process boundaries, optional dependencies, deterministic traces, and artifact-level benchmark signal.

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

The local release gate combines static checks, tests, package verification, wheel builds, and installed-artifact validation:

```bash
mise run release:check
```
