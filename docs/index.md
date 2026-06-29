# RLMesh

**Gymnasium-compatible infrastructure for model-environment evaluation.**

[![CI](https://github.com/ArenaX-Labs/rlmesh/actions/workflows/ci.yml/badge.svg)](https://github.com/ArenaX-Labs/rlmesh/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/rlmesh.svg)](https://pypi.org/project/rlmesh/) [![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://pypi.org/project/rlmesh/) [![crates.io](https://img.shields.io/crates/v/rlmesh.svg)](https://crates.io/crates/rlmesh) [![Docs](https://img.shields.io/badge/docs-rlmesh.dev-blue.svg)](https://docs.rlmesh.dev) [![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/ArenaX-Labs/rlmesh#license)

RLMesh connects models to environments that need not share a process, a dependency set, or a machine. The Python SDK serves Gymnasium-style environments and lets evaluators call `reset`, `step`, `render`, and `close` over local or remote transports. Rust crates provide the lower-level runtime, protocol, and packaging layers.

RLMesh is released and pre-1.0 (`0.x`): the Python package is the supported surface, and a minor release may change a stable API with a migration note. See {doc}`compatibility` and {doc}`versioning`; pin a minor range for active projects.

```{note}
**0.1.0-rc.2 is a release candidate for 0.1.0.** It advertises the exact
`2026.06-0.1.0-rc.2` workflow cohort; the bare `2026.06` edition seals at the
final 0.1.0. The stability statements in these docs describe that upcoming
release.
```

## Try First

Start with the shortest local loop:

- {doc}`Install RLMesh <installation>` with Gymnasium and the NumPy backend.
- {doc}`Run the quickstart <quickstart>`: serve `CartPole-v1`, connect one evaluator.
- {doc}`Check Gymnasium compatibility <gymnasium>` for the current supported space set.
- {doc}`Try the examples <examples>`: swap environments, run sandboxed or isolated dependency stacks, and connect one evaluator to multiple endpoints.

## Model-Environment Boundary

Start with the boundary between the model and environment. A model or evaluator connects through the familiar `reset`, `step`, `render`, and `close` calls, even when the environment lives in a separate process with separate dependencies.

Use that boundary to:

- run a model against an environment without merging their dependency stacks;
- run multiple environment endpoints at the same time;
- reuse existing Gymnasium registrations, wrappers, and environment objects with small changes;
- package an environment or model as a container image and run it the same way locally or on the hosted platform (see the {doc}`bring-your-own-container example <examples/byo-container>`).

RLMesh has two surfaces, split by protocol. The framework is this repository. It speaks the gRPC runtime contract, so you author and run environments and models in Python or Rust, on one machine or over the wire. The hosted platform speaks a REST control plane at `api.rlmesh.dev` and is documented separately. These docs cover the framework.

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
:caption: Authoring
:maxdepth: 2

user-guide/environments
user-guide/models
user-guide/evaluation
```

```{toctree}
:hidden:
:caption: Adapters
:maxdepth: 2

user-guide/adapters
user-guide/adapters/reference
user-guide/adapters/escape-hatches
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
specs/describe.v1
```

```{toctree}
:hidden:
:caption: Releases
:maxdepth: 1

changelog
versioning
```
