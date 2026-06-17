---
orphan: true
---

# Local Development

This page covers the maintainer workflow for the RLMesh repository.

## Requirements

Install Git and [mise](https://mise.jdx.dev/getting-started.html). The remaining tools are pinned in
`mise.toml`, including Python, Rust, uv, Protobuf tooling, and release build helpers.

The root `mise.toml` also creates and sources the repository `.venv` automatically when mise is
active.

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

Build the linux-glibc wheel pair consumed by container images (skips when the wheels for the current
version and architecture already exist):

```bash
mise run build:python:docker
```

## Site

Build the site:

```bash
mise run docs:build
```

Serve it locally while editing:

```bash
mise run docs:serve
```

Both tasks write their output to `site/`.

## Release Gate

Before publishing a release from a local machine, run:

```bash
mise run release:check
```

Publishing stays manual. See {doc}`release` for the maintainer release process.
