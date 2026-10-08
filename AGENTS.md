# RLMesh agent context (open-source framework)

RLMesh connects models to environments for evaluation. This repo is the OSS
framework: Rust crates in `crates/` (runtime, protocol, gRPC, spaces, adapters,
sandbox, viewer, CLI, C API) and the `rlmesh` Python package in
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
the in-tree extension; after Rust changes, `mise run build:python:develop`
refreshes it (`test:affected` does this itself).

## Rules

- The wire contract is `crates/rlmesh-proto/proto/`, frozen per protocol
  generation against `rlmesh.toml`; `protocol:baseline-verify` and
  `protocol:breaking` enforce it.
- Never hand-edit generated files: regenerate native stubs with
  `mise run stubs:generate`, and update the Python API surface snapshot
  (`python/rlmesh/tests/api_surface/snapshots/`) with the change it records.
- Conventional commit titles; squash-merged PR titles become the commit. The
  changelog is hand-written (`mise run changelog:draft` only drafts bullets).
- Half-built features get removed, not left behind flags.

Maintainer docs: `docs/local-dev.md`, `docs/testing.md`, `CONTRIBUTING.md`.
