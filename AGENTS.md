# RLMesh agent context (open-source framework)

RLMesh connects models to environments for evaluation. This repo is the OSS
framework: Rust crates in `crates/` (the `rlmesh` facade, the v2 session
`rlmesh-session` and its sans-IO core `rlmesh-session-core`, the cockpit,
adapters, spaces, CLI, C API, and the v1 bridge `rlmesh-grpc` and
`rlmesh-proto`) and the `rlmesh` Python package in
`python/rlmesh/` (a maturin build whose extension crate is
`python/rlmesh/rust`). The Python package is the supported surface. The
managed platform consumes this repo as pinned releases; keep its contracts
here.

## Workflow

`mise run setup | check | test | build`; `mise tasks` lists the rest. Prefer
repo tasks over ad hoc commands.

The host has limited CPU and memory and is shared with other worktrees:

1. Iterate with `mise run test:affected`: Rust tests (nextest) for crates
   changed since `origin/main` and their dependents, plus Python tests when
   they can be affected. `-- --dry-run` shows the selection.
2. Wrap any other heavy local command (cargo builds, full test runs, wheel
   builds) in `mise run dev:heavy -- <command>` so it queues behind other jobs
   instead of competing for memory.
3. Before handing off, run `mise run check`, then `mise run ci:remote`: the
   full CI job on Depot compute, with your uncommitted changes uploaded as a
   patch. Do not run `test:ci` or release wheel builds locally unless asked.
   For a failed remote run, `depot ci diagnose <run-id>` and
   `depot ci logs <job-id>` explain it.

Compiled crates are cached by sccache (`RUSTC_WRAPPER` in `mise.toml`), so a
fresh worktree mostly rebuilds from cache. The editable Python install loads
the in-tree extension; after Rust changes or a commit, `mise run setup:python`
refreshes it (`test` and `test:affected` do this themselves).

## Rules

- Two wire contracts, both governed in `rlmesh.toml`: the eval wire
  `crates/rlmesh-proto/proto/` (`[protocol]`, frozen per generation) and the
  real-time session protocol `crates/rlmesh-session-core/proto/`
  (`[protocols."rlmesh-realtime-v1"]`, provisional until 0.2.0 freezes it).
  buf lints both; byte goldens (`crates/rlmesh-session-core/tests/vectors/wire`)
  guard the session protocol's encodings; `protocol:baseline-verify` and
  `protocol:breaking` enforce a frozen one.
- Never hand-edit generated files: regenerate native stubs with
  `mise run stubs:generate`, and update the Python API surface snapshot
  (`python/rlmesh/tests/api_surface/snapshots/`) with the change it records.
- Conventional commit titles; squash-merged PR titles become the commit. The
  changelog is hand-written (`mise run changelog:draft` only drafts bullets).
- Unreleased work may be fenced if CI builds and tests it; nothing half-built ships.

Maintainer docs: `docs/local-dev.md`, `docs/testing.md`, `CONTRIBUTING.md`.
