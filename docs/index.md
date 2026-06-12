# RLMesh

**Gymnasium-compatible infrastructure for model-environment evaluation.**

[![CI](https://github.com/ArenaX-Labs/rlmesh/actions/workflows/ci.yml/badge.svg)](https://github.com/ArenaX-Labs/rlmesh/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rlmesh.svg)](https://pypi.org/project/rlmesh/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://pypi.org/project/rlmesh/)
[![crates.io](https://img.shields.io/crates/v/rlmesh.svg)](https://crates.io/crates/rlmesh)
[![Docs](https://img.shields.io/badge/docs-rlmesh.dev-blue.svg)](https://docs.rlmesh.dev)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/ArenaX-Labs/rlmesh#license)

RLMesh connects models to environments without requiring them to share one process, one dependency
set, or one machine. The Python SDK serves Gymnasium-style environments, connects local or remote
evaluators, and keeps the same workflow usable when evaluation moves behind a service boundary. Rust
crates provide the lower-level runtime, protocol, and packaging layers.

RLMesh is in beta. The published beta is for early adopters and feedback; APIs and package structure
may still change before a stable release.

## Try First

Start with the shortest local loop:

- {doc}`Install RLMesh <installation>` with Gymnasium and the NumPy backend.
- {doc}`Run the quickstart <quickstart>`: serve `CartPole-v1`, connect one evaluator.
- {doc}`Check Gymnasium compatibility <gymnasium>` for the current supported space set.
- {doc}`Try the examples <examples>`: swap environments, run sandboxed or isolated dependency
  stacks, and connect one evaluator to multiple endpoints.

## What To Notice

The important part is the iteration loop. A model or evaluator can connect to an environment with
the familiar `reset`, `step`, `render`, and `close` shape, even when that environment lives in a
separate process with separate dependencies.

That lets you:

- run a model against an environment without merging their dependency stacks;
- run multiple environment endpoints at the same time;
- reuse existing Gymnasium registrations, wrappers, and environment objects with small changes.

The managed platform builds on the same boundary for scheduling, batching, resource allocation,
dashboards, and cluster orchestration. Start with the open-source framework to see that boundary
directly.

```{toctree}
:hidden:
:caption: Get Started
:maxdepth: 1

installation
quickstart
gymnasium
examples
```

```{toctree}
:hidden:
:caption: User Guide
:maxdepth: 2

user-guide/serving-environments
user-guide/remote-clients
user-guide/backends
user-guide/sandbox
```

```{toctree}
:hidden:
:caption: Reference
:maxdepth: 2

api/index
compatibility
editions/index
```
