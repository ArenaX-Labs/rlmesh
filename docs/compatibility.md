# Compatibility

RLMesh promises stable workflows, not frozen internals. During beta, stability labels describe the
intended support level, but APIs and package structure may still change.

## Stable

Stable workflows include documented public APIs, supported CLI flows, and supported remote
environment/model interactions.

- Imports, signatures, and documented behavior should stay usable.
- New runtimes should keep accepting older stable clients and packages.
- New features may require newer packages or capabilities, but older stable workflows should still
  fail clearly or keep working.

## Preview and Experimental

Preview APIs are intended to become stable but may still change with migration notes. Experimental
APIs may change or disappear.

Torch and JAX backends and sandbox helpers are experimental in this beta.

Beta caveat: dtype values added during the beta (`int8/16`, `uint16/32/64`, `bfloat16`) are not
negotiated — a peer from an older beta fails with a decode error naming the unknown dtype when it
meets an environment that uses them. Run both ends on the same beta release. Edition-gated dtype
negotiation is planned for the workflow-edition rollout before GA.

## Framework Version Floors

The optional framework backends declare the lowest versions their conversion paths actually require.
Each floor has a concrete reason:

| Package | Floor      | Why                                                                                      |
| ------- | ---------- | ---------------------------------------------------------------------------------------- |
| Python  | `3.10`     | Ecosystem baseline; all framework floors below ship `cp310` wheels.                      |
| numpy   | `>=1.22`   | First release with complete Python 3.10 wheel coverage.                                  |
| torch   | `>=1.11`   | First release with full `cp310` wheels and top-level `torch.from_dlpack`. [^torch-glibc] |
| jax     | `>=0.4.24` | First release with DLPack `bool` support; `jaxlib` below `0.4.18` is no longer on PyPI.  |

[^torch-glibc]:
    Torch wheels older than 1.13 fail to load on glibc 2.41+ hosts ("cannot enable executable
    stack"), so the floor harness exercises 1.13.1 there; 1.11 remains the declared install floor
    for older systems.

The floors are executed — not just declared — by `mise run test:python:floors`, which builds a
`cp310` wheel and runs the framework test suites against exactly these versions. Versions below a
floor may work but are unsupported. Within a framework, some features need newer releases:
`rlmesh.numpy` itself converts through the buffer protocol on any supported numpy, but consuming
RLMesh tensors with `np.from_dlpack` needs numpy 1.23 (`bool` needs 1.25). Torch `bool` over DLPack
needs 2.2 (older versions fall back to a copy), and Torch `uint16/32/64` need 2.3.

## Value Semantics and Caveats

`rlmesh.Tensor` is a validated transport container with DLPack and buffer-protocol edges — it is not
an ndarray. Compute, slicing, and broadcasting belong to the frameworks; RLMesh moves bytes and
metadata between them and the wire.

- **Zero-copy asymmetry:** exporting (`memoryview`, `__dlpack__`, framework views) is zero-copy;
  importing (`Tensor(...)`, `Tensor.from_dlpack`) currently always copies. Zero-copy import is
  planned.
- **Integer precision:** Box bounds checks compare through `float64`, so `int64`/`uint64` values
  beyond 2^53 lose precision there. The legacy scalar-list wire encoding stores integers in a signed
  64-bit slot, so `uint64` values above 2^63 wrap on that path (the raw byte encoding used by modern
  clients is exact).
- **Mutation:** decoded views are read-only by contract. NumPy enforces this; Torch does not (see
  the Torch backend page). JAX arrays are immutable by construction.

## Artifact Versions

Core feature releases move together. Patch releases may be artifact-specific when the fix is
isolated.

## Enforcement

`rlmesh.toml` records the current package family, artifacts, protocol generation, workflow edition,
and API surface policy:

```bash
python scripts/check_rlmesh_policy.py
```

`mise run check` includes `mise run policy:check`. Pull requests also run protobuf breaking-change
checks with `mise run protocol:breaking` against the checked-in protocol baseline at
`crates/rlmesh-proto/baselines/rlmesh.protocol.v1`.
