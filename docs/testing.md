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
mise run test:python:conformance
```

`test:python:conformance` holds the gates other code is measured against. The
Gymnasium gate (`python/rlmesh/tests/conformance/test_gym_conformance.py`) runs
plain gymnasium and the same env behind `rlmesh.numpy.RemoteEnv` /
`RemoteVectorEnv`, over gRPC and a v2 session, and compares every
observation, reward, flag and info bit for bit; its docstring lists the
documented differences. MuJoCo cases skip when `mujoco` is not installed
(`uv sync --group mujoco`). `native_gate.py` beside it runs the same rows
over an env, a model and clients written in another language, against
Python on a session: `test:cxx` runs it for C++ and the Unreal plugin's
game-thread env, `test:csharp` for C# and the Unity host, each naming its gate
program in `RLMESH_NATIVE_GATE`; a row the language cannot express is skipped
with the reason it prints.

Run only what a change can affect (changed crates since `origin/main`, their dependents, and Python when reachable):

```bash
mise run test:affected
```

Locally, C# tests run only for direct changes under `csharp/`,
`integrations/unity/`, `tools/rlmesh_csharp_surface/` or `crates/rlmesh-capi/`
(including `rlmesh.h`), changes to `scripts/dev/affected.py`, or changes to the
`csharp` Python dependency group in `pyproject.toml` / `uv.lock`. The group
comparison includes resolved transitive dependencies and selected extras;
unrelated Python dependency edits do not require dotnet. Transitive Rust core
changes still select the affected Rust and Python tests, without installing
dotnet locally. Use `mise run test:csharp` to request the managed suite directly.

CI invokes the planner with `--ci` and deliberately selects C# more broadly:
any crate reaching `rlmesh-capi`, workspace build inputs, either Python manifest
or lockfile, `python/rlmesh/**` (the interoperability client), the C/C++ package
task, the planner and `.github/workflows/ci.yml` all select it. The existing
Linux job installs the scoped C# toolchain only when selected, then runs interop
staleness, managed tests, NativeAOT and package checks. `test:ci` itself does not
install dotnet.

Rust tests run under [cargo-nextest](https://nexte.st) (configured in `.config/nextest.toml`); doctests still run through `cargo test --doc`.

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

To run the whole CI job on Depot compute instead of locally, including uncommitted changes, use `mise run ci:remote` (see [local development](local-dev.md#worktrees-and-limited-compute)).

CI uses one required job, `Fast Checks`, on pull requests, pushes to `main`, and manual runs. It runs `check` and `test:ci`, reuses the resulting wheels for cross-version compatibility, verifies Rust and C/C++ packages, and repeats the C/C++ smoke under gcc (pull requests) or under both gcc and clang (pushes to `main` and manual runs); `test` itself already covers zig's clang with libc++. Package and compatibility failures therefore fail the same required check as the rest of CI; there are no event-specific skipped jobs.

Toolchain setup installs the committed `mise.lock` with `--locked`; CI does not regenerate it against changing upstream dependency metadata. Refresh the lock explicitly when updating tools.

CI sets `RLMESH_RELEASE_BUILD=1` so one local wheel build covers both the basic installed-artifact and compatibility profiles. To reproduce its checks locally:

```bash
RLMESH_RELEASE_BUILD=1 mise run check
RLMESH_RELEASE_BUILD=1 mise run test:ci
mise run test:crossver:installed
mise run release:rust:package
CC=clang CXX=clang++ mise run test:cxx:pkg
CC=clang CXX=clang++ mise run test:cxx
CC=gcc CXX=g++ mise run test:cxx
```

`release:rust:package` extracts and compiles the packaged crates, catching omitted source files and invalid published dependency metadata that a workspace build can miss. Keep this verification even when workspace tests pass.

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

The cross-version matrix runs this tree against the last published release as part of the required CI job, on every pull request and on pushes to `main`. The standalone local entrypoint builds its own release wheels:

```bash
mise run test:crossver
```

When release-cohort wheels have already been built, run `mise run test:crossver:installed` to validate them without rebuilding. CI and `release:check` use this entrypoint; the runner rejects a dev-cohort wheel rather than silently testing the wrong artifact.

The old side of every cell is the PyPI wheel pinned by exact version and sha256 in `tests/system/crossver.lock`; the new side is a wheel built from this tree with `RLMESH_RELEASE_BUILD=1`, which stamps the same workflow-edition cohort the published wheel advertises (a plain dev build stamps `2026.06-dev.<sha>` and is refused at the edition floor by design). `tests/system/crossver.py` drives the cells: old and new env servers against old and new runtimes, old and new served models against the other side's runtime, both same-version controls, and two forged refusals (a ConfigureEnv pin naming an edition no build implements, and a `rlmesh-wire-v2` handshake). Each of the six real cells asserts that the cross-version Join reproduces `tests/system/traces/counter-entrypoint.json`, and — measured on the server it runs against, by a raw wire probe rather than by the runtime's own client — that server's `HandshakeResponse.compatible` flag and its declared WANT (this tree's servers declare the current edition; the published wheel declares none). The edition column is the runtime's real `ResolveAdapter` pin on the model cells and its real `ConfigureEnv` pin on the env cells this tree's runtime drives (read off this tree's env server log, or proven by the trace when the published wheel serves the env, since a refused pin aborts before Reset); the published wheel's runtime sends no pin, so its env cells fall back to the two builds' CAN intersection acked through a forged `ConfigureEnv`. The two refusal cells assert the refusal message instead; the results table prints the same legend.

Cells 7 and 8 select _between_ two editions, so they print as pending until a second edition is sealed; they are listed in the results table rather than skipped silently.

Set `RLMESH_CROSSVER_WHEEL_DIR` to a directory holding the pinned wheel to resolve it from there instead of PyPI; the install still goes through the lock file's hashes, so a look-alike wheel (this tree's own build carries the same version) fails.

## Release Check

The local release gate combines static checks, tests, package verification, wheel builds, and installed-artifact validation:

```bash
mise run release:check
```

## When parity:check or parity:fixtures fails

`mise run parity:check` compares the Stable Python snapshot with the live Rust,
C and C++ APIs. `mise run parity:fixtures` runs shared behavioural scenarios in
all four languages. The reviewed source of mappings is
[`docs/specs/language-parity/manifest.toml`](specs/language-parity/manifest.toml);
checks never regenerate it. Read the cited [parity rule](specs/language-parity/rules.md)
and inspect the implementation before changing a mapping to silence a failure.

Each `["module:Symbol.member"]` table records Python `parameters`, including
`default` literals and, where needed, their snapshot `expression`. Its `.rust`,
`.c` and `.cpp` tables record `status`, native `path`, and `params` targets.
`Type#field` addresses a data field; `function#parameter` addresses an input.
Members have their own symbol tables. Preserve Python's input order (R-6/R-7)
and reset/step output order (R-15).

| Failure                                                 | Where to fix it                                                                                                                                                                                                                                                                                                                  |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| E-2.1: absent symbol or keyword                         | Add the symbol table or `parameters` entry and each language's `params`. Implement shared APIs required by R-1.                                                                                                                                                                                                                  |
| E-2.1: bridge or exemption                              | `status = "bridge"` is restricted to R-24's Python framework and typing scopes. `bridges` lists only Python bridge inputs. An `exempt` mapping needs its **own** `rule` citing a declared decision, never a generic `R-n`.                                                                                                       |
| E-2.2: missing target, name, parameter, order or return | Correct `path`, `params`, `protocol`, `returns` or `result_tuple`, or fix native code to follow R-4/R-7/R-12/R-15. C++ protocol parameters come from detected calls cross-checked against C callback slots.                                                                                                                      |
| E-2.3: stale gap                                        | Change `missing` to `present` and supply `path`/`params`. A genuinely missing implementation needs `status = "missing"` and `gap = "G-n"` from the spec's gap table.                                                                                                                                                             |
| E-2.4: public native addition                           | Map a shared field/member directly. `also` inventories individual translation helpers; structural fields (`has_*`, `_reserved*`, `*_len`, `struct_size`), mapped setters and traits are recognized. Other helpers need a per-target `also_rules` citation. Adding an arbitrary member to `also` alone does not establish parity. |
| E-2.4: native extra or plumbing                         | `[ext].rust/c/cpp` accepts only R-25 fences and its closed list of shared cancellation/serve/ABI exceptions. `[binding]` accepts binding plumbing (R-12/R-25 and vendored SPC-19 DLPack declarations). Move an inappropriate public extra into its fence or remove it.                                                           |
| E-2.5: default                                          | Fix native initializers/`Default`, or intentionally update `parameters.*.default` with Python. `translations.<parameter>` cites a declared decision for a deliberate translation (for example RUN-20's C sentinel). Regenerate assertions with `python tests/parity/generate_defaults.py`.                                       |
| E-2.6 / E-2.7: vocabulary or error                      | Fix the native enum/hierarchy to R-9/R-21; update its symbol's `path`/`also` only when the mapping itself is wrong.                                                                                                                                                                                                              |

`translations` are per-parameter decision citations, independent of the row's
`rule`; a naming rule cannot authorize a rename. Rust default assertions cover
mapped fields on every `Default` type. Types with required constructor inputs
use an explicit `default_constructor`; `default_serializers` records a typed
representation when a Python literal needs translation. A callable default
that has no data field requires `default_skip.<parameter>` with a concrete
reason. Keep behavioural coverage for those defaults; a skip is reviewable
source, not an inferred naming exception.

Fixture failures print the scenario, observed key and expected value. Locate
it under `tests/parity/fixtures/`, then inspect the corresponding operation in
`python_runner.py`, `c_runner.c`, `cpp_runner.cpp` and
`crates/rlmesh/tests/parity_fixtures.rs`. Expectations must come from library
behaviour; runners must not manufacture expected results. A restricted
`languages` list requires a `why` explanation. Check warnings counts as well as
categories/text, and preserve authored error text across local and remote calls.
Repeat a repaired gate under load with
`mise run parity:fixtures -- --repeat 5 --load`. The task takes its own heavy
lock; do not wrap it again. `run.py` honours `CARGO_TARGET_DIR` and supports
`--no-build` when reusing native consumers.

For local libclang/header-driver mismatches, run header checks with
`CC="mise exec zig@0.13.0 -- zig cc" CXX="mise exec zig@0.13.0 -- zig c++"`.
`RLMESH_PARITY_HEADER_CC` / `RLMESH_PARITY_HEADER_CXX` apply those drivers only
to parity header extraction when invoking the combined check task.
