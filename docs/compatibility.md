# Compatibility

RLMesh promises stable workflows, not frozen internals.

If a workflow is documented as Stable, newer RLMesh releases should keep that workflow working for
every workflow edition and protocol generation they still support. Package versions are not the
boundary of that promise.

A package family is only a release coordination label. During `0.x`, the package family is
`0.minor`, such as `0.1`. After `1.0`, the package family is the major version. It tells users which
artifacts are expected to move together and how patch hotfixes are scoped.

## Stable Workflows

Stable workflows include documented public APIs, supported CLI flows, and supported remote
environment/model interactions. For those workflows:

- Stable public APIs should keep their documented imports, signatures, and behavior.
- Newer runtimes should keep accepting older stable clients and packages.
- Stable protocol and API surface schemas should remain readable, or get an explicit migration path.
- New features may require newer packages or capabilities, but older stable workflows should still
  work.

This does not promise that every external dependency stack works forever. Preview and Experimental
APIs may change, and security or soundness fixes may remove unsafe behavior.

## Stability Levels

Stable APIs are covered by the compatibility policy. Preview APIs are intended to become stable but
may still change with migration notes. Experimental APIs may change or disappear.

Use the docs as the source of truth for stability labels. Unlabeled internals, generated details,
and test helpers are not stable.

## Workflow Editions

RLMesh uses workflow editions for coordinated semantic shifts.

The first edition is `2026`. An edition describes workflow behavior, not package versions. A future
edition can change defaults or semantics while keeping old stable workflows available for their
support window.

Peers exchange the requested workflow edition during the handshake. If an endpoint does not support
the requested edition, it should fail directly instead of guessing.

Dropping support for a stable workflow edition is a compatibility event. It requires release notes,
a migration path, and an explicit support decision rather than an incidental package-version bump.

## Protocol Generations

Package versions are not protocol generations.

The current protocol generation is `rlmesh.protocol.v1`. Protobuf packages such as `rlmesh.env.v1`,
`rlmesh.model.v1`, and `rlmesh.spaces.v1` are service/schema generations inside that protocol
generation.

Optional behavior should be negotiated with named capabilities. Unsupported new features should fail
directly with a message like `requires capability X`.

Dropping support for a stable protocol generation is also a compatibility event. The normal path is
to add a new protocol generation while continuing to accept the old one for its support window.

## Artifact Versions

Core feature releases move together. Patch releases may be artifact-specific.

For example, a Python-only hotfix can publish a new Python patch without forcing unrelated Rust
crates or future incubating bindings to publish no-op updates. The package family stays the same.

## Enforcement

`rlmesh.toml` records the current package family, artifacts, protocol generation, workflow edition,
and API surface policy:

```bash
python scripts/check_rlmesh_policy.py
```

`mise run check` includes `mise run policy:check`. Pull requests also run protobuf breaking-change
checks with `mise run protocol:breaking` against the checked-in protocol baseline at
`crates/rlmesh-proto/baselines/rlmesh.protocol.v1`.
