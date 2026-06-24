# rlmesh-adapters

Core implementation of RLMesh's declarative env/model IO adapters: versioned spec types (serde, JSON-compatible with the published `rlmesh.adapters.v1.*` metadata format), spec resolution into concrete plans, and plan descriptions. Language bindings wrap this crate (Python today, more later). Per-pairing escape hatches such as custom transforms and custom adapters stay in the binding languages by design.

The conformance vectors in this crate's `conformance/v1/` directory pin the semantics. Every implementation and binding must pass them: `tests/conformance.rs` here, plus each binding's own runner.
