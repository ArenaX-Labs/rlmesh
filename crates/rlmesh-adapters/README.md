# rlmesh-adapters

Core implementation of RLMesh's declarative env/model IO adapters: versioned spec types (serde,
JSON-compatible with the published `rlmesh.adapters.v1.*` metadata format), spec resolution into
concrete plans, and plan descriptions. Language bindings (Python today, more later) wrap this crate;
per-pairing escape hatches (custom transforms, custom adapters) stay in the binding languages by
design.

Semantics are pinned by the conformance vectors in this crate's `conformance/v1/` directory; every
implementation and binding must pass them (`tests/conformance.rs` here, plus each binding's own
runner).
