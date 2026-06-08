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

Torch adapters and sandbox helpers are experimental in this beta.

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
