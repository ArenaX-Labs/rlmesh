# Local Development

This page covers the lightweight maintainer workflow for the RLMesh repository. The public user
documentation lives at https://docs.rlmesh.dev.

## Requirements

Install Git and [mise](https://mise.jdx.dev/getting-started.html). The remaining tools are pinned in
`mise.toml`, including Python, Rust, uv, Protobuf tooling, and release build helpers.

The root `mise.toml` also creates and sources the repository `.venv` automatically when mise is
active.

## Setup

Install mise-managed tools, sync the Python development environment, and install git hooks:

```bash
mise run setup
```

Useful setup subtasks:

```bash
mise run setup:tools
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

For faster iteration, run focused tasks:

```bash
mise run fmt:check
mise run lint
mise run typecheck
mise run test:rust
mise run test:python:unit
mise run test:python:integration
```

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

Local wheel builds may use plain `linux_*` tags for smoke testing. Release wheels must use
uploadable platform tags such as `manylinux`, `musllinux`, `macosx`, or `win`.

## Release Gate

Before publishing a beta release from a local machine, run:

```bash
mise run release:check
```

Publishing stays manual for this beta. See [`release.md`](release.md) for the maintainer release
process.
