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

## Cross-Version Matrix

The cross-version matrix runs this tree against the last published release, on every pull request and on pushes to `main`:

```bash
mise run test:crossver
```

The old side of every cell is the PyPI wheel pinned by exact version and sha256 in `tests/system/crossver.lock`; the new side is a wheel built from this tree with `RLMESH_RELEASE_BUILD=1`, which stamps the same workflow-edition cohort the published wheel advertises (a plain dev build stamps `2026.06-dev.<sha>` and is refused at the edition floor by design). `tests/system/crossver.py` drives the cells: old and new env servers against old and new runtimes, old and new served models against the other side's runtime, both same-version controls, and two forged refusals (a ConfigureEnv pin naming an edition no build implements, and a `rlmesh-wire-v2` handshake). Each of the six real cells asserts that the cross-version Join reproduces `tests/system/traces/counter-entrypoint.json`, and — measured on the server it runs against, by a raw wire probe rather than by the runtime's own client — that server's `HandshakeResponse.compatible` flag and its declared WANT (this tree's servers declare the current edition; the published wheel declares none). The edition column is the runtime's real `ResolveAdapter` pin on the model cells and its real `ConfigureEnv` pin on the env cells this tree's runtime drives (read off this tree's env server log, or proven by the trace when the published wheel serves the env, since a refused pin aborts before Reset); the published wheel's runtime sends no pin, so its env cells fall back to the two builds' CAN intersection acked through a forged `ConfigureEnv`. The two refusal cells assert the refusal message instead; the results table prints the same legend.

Cells 7 and 8 select _between_ two editions, so they print as pending until a second edition is sealed; they are listed in the results table rather than skipped silently.

Set `RLMESH_CROSSVER_WHEEL_DIR` to a directory holding the pinned wheel to resolve it from there instead of PyPI; the install still goes through the lock file's hashes, so a look-alike wheel (this tree's own build carries the same version) fails.

## Release Check

The local release gate combines static checks, tests, package verification, wheel builds, and installed-artifact validation:

```bash
mise run release:check
```
