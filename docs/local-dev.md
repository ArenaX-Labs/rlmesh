# Local Development

This page covers the maintainer workflow for the RLMesh repository.

## Requirements

Install Git and [mise](https://mise.jdx.dev/getting-started.html). The remaining tools are pinned in `mise.toml`, including Python, Rust, uv, Protobuf tooling, and release build helpers.

The root `mise.toml` also creates and sources the repository `.venv` automatically when mise is active.

## Setup

Install mise-managed tools:

```bash
mise install
```

Then sync the Python development environment and install git hooks:

```bash
mise run setup
```

Useful setup subtasks:

```bash
mise run setup:python
mise run setup:hooks
```

Release wheel builds need extra Python and Rust targets:

```bash
mise run setup:python:release
mise run setup:rust:targets
```

## Daily Checks

Run static checks without modifying source files:

```bash
mise run check
```

Run the default test set:

```bash
mise run test
```

Run exactly what CI tests (adds the API surface, examples, system harness, and an installed-wheel system profile):

```bash
mise run test:ci
```

For faster iteration, run focused tasks, or `mise watch <task>` to rerun one on every file change:

```bash
mise run fmt:check
mise run lint
mise run typecheck
mise run test:rust
mise run test:python:unit
mise run test:python:integration
```

On `uv run`, uv rebuilds the native extension when Rust sources change; the git hooks installed by `setup:hooks` rebuild it on checkout and merge. See [testing](testing.md) for the full test layering.

## Worktrees and Limited Compute

The repository is set up for several worktrees (and coding agents) sharing one modest machine, with heavy work offloaded to [Depot](https://depot.dev).

- **Worktree setup.** `scripts/dev/setup-worktree.sh` prepares a fresh worktree: it links the main checkout's gitignored `*.local.toml`, trusts and installs the mise tools, and runs `setup`. T3 Code runs it automatically from `t3.json`, which also defines Check/Test/Build actions and clears `target/` and `.venv` when a thread settles.
- **Affected tests.** `mise run test:affected` runs Rust tests (under [nextest](https://nexte.st)) for crates changed since `origin/main` plus every crate that depends on them, and the Python tests when Python sources or the extension crate are affected. Pass `-- --base <ref>` to compare elsewhere, or `-- --dry-run` to print the selection.
- **Heavy-job queue.** `mise run dev:heavy -- <command>` runs a command under a machine-wide lock, so two full builds never compete for memory. A waiting job prints which command holds the lock.
- **Compile cache.** Every `rustc` call goes through [sccache](https://github.com/mozilla/sccache) (`RUSTC_WRAPPER` in `mise.toml`). By default it caches on local disk, shared by all worktrees. With `RLMESH_BUILDER=depot` in your mise environment (for example `~/.config/mise/config.toml`) and a `depot login`, it uses Depot Cache instead, the same cache CI's Depot runners use. Run `sccache --show-stats` to see hit rates, and set `RUSTC_WRAPPER = ""` in `mise.local.toml` to opt out.
- **Remote CI.** `mise run ci:remote` runs the CI `Fast Checks` job on Depot compute with your local changes uploaded as a patch, so the full `test:ci` gate runs without pushing and without loading this machine. Extra arguments go to `depot ci run` (for example `-- --ssh` to open a shell on the job's sandbox). It needs `depot login` and an organization with Depot CI enabled for this repository.

uv uses one per-user cache (`~/.cache/uv`) for every worktree and hardlinks from it, so syncing a new worktree is fast.

## Build

Build the Rust workspace and current-platform Python wheels:

```bash
mise run build
```

Build only one side:

```bash
mise run build:rust
mise run build:python
```

Local wheel builds may use plain `linux_*` tags for smoke testing. Release wheels must use uploadable platform tags such as `manylinux`, `musllinux`, `macosx`, or `win`.

Build the linux-glibc wheel pair consumed by container images (skips when the wheels for the current version and architecture already exist):

```bash
mise run build:python:docker
```

## Docs

The user docs and the Python API reference are published at [rlmesh.dev/docs](https://rlmesh.dev/docs/) and built from the managed platform repo, which snapshots this package's API surface (`rlmesh-api-surface docs-api-surface`) on each release. This `docs/` directory keeps the maintainer docs and the specs the tooling reads: the workflow editions, compatibility and versioning policy, and the describe envelope.

## Release Gate

Before publishing a release from a local machine, run:

```bash
mise run release:check
```

Publishing stays manual. See [release](release.md) for the maintainer release process.
