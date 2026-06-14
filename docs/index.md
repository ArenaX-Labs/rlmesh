# RLMesh

**Gymnasium-compatible infrastructure for model-environment evaluation.**

[![CI](https://github.com/ArenaX-Labs/rlmesh/actions/workflows/ci.yml/badge.svg)](https://github.com/ArenaX-Labs/rlmesh/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rlmesh.svg)](https://pypi.org/project/rlmesh/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://pypi.org/project/rlmesh/)
[![crates.io](https://img.shields.io/crates/v/rlmesh.svg)](https://crates.io/crates/rlmesh)
[![Docs](https://img.shields.io/badge/docs-rlmesh.dev-blue.svg)](https://docs.rlmesh.dev)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/ArenaX-Labs/rlmesh#license)

RLMesh connects models to environments without requiring them to share one process, one dependency
set, or one machine. The Python SDK serves Gymnasium-style environments and lets evaluators call
`reset`, `step`, `render`, and `close` over local or remote transports. Rust crates provide the
lower-level runtime, protocol, and packaging layers.

RLMesh is in beta. Pin versions for active projects; APIs and package structure may still change
before a stable release.

## Try First

Start with the shortest local loop:

- {doc}`Install RLMesh <installation>` with Gymnasium and the NumPy backend.
- {doc}`Run the quickstart <quickstart>`: serve `CartPole-v1`, connect one evaluator.
- {doc}`Check Gymnasium compatibility <gymnasium>` for the current supported space set.
- {doc}`Try the examples <examples>`: swap environments, run sandboxed or isolated dependency
  stacks, and connect one evaluator to multiple endpoints.

## Model-Environment Boundary

Start with the boundary between the model and environment. A model or evaluator can connect with the
familiar `reset`, `step`, `render`, and `close` calls, even when that environment lives in a
separate process with separate dependencies.

Use that boundary to:

- run a model against an environment without merging their dependency stacks;
- run multiple environment endpoints at the same time;
- reuse existing Gymnasium registrations, wrappers, and environment objects with small changes;
- name an environment's construction as a {doc}`recipe <user-guide/recipes>` and rebuild it the same
  way locally or in a sandbox.

SAI examples appear where they demonstrate optional environment packages. The RLMesh docs focus on
the open-source framework and the protocol boundary it exposes.

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
user-guide/recipes
```

```{toctree}
:hidden:
:caption: Reference
:maxdepth: 2

api/index
compatibility
editions/index
```
