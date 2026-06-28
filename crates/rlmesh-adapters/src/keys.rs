//! Metadata keys under which v1 specs travel in contract metadata.
//!
//! The version segment of these key strings, not any source-module path, is the
//! adapter spec-format discriminator. A breaking spec-format change ships a v2
//! key (`rlmesh.adapters.v2.*`); a v2 reader must also keep reading v1 so a
//! newer build still resolves an older peer's specs.

/// Key carrying serialized [`EnvTags`](crate::spec::EnvTags) in env contract metadata.
pub const ENV_METADATA_KEY: &str = "rlmesh.adapters.v1.env_tags";

/// Key carrying a serialized [`ModelSpec`](crate::spec::ModelSpec) in model metadata.
pub const MODEL_METADATA_KEY: &str = "rlmesh.adapters.v1.model_spec";

/// Key carrying a serialized describe envelope (see [`build_describe_envelope`])
/// in env/model metadata or an OCI artifact. Same version-segment doctrine: a
/// breaking envelope-format change ships `rlmesh.describe.v2`.
///
/// [`build_describe_envelope`]: crate::envelope::build_describe_envelope
pub const DESCRIBE_METADATA_KEY: &str = "rlmesh.describe.v1";

/// The describe-envelope schema version stamped into every envelope. Additive
/// fields stay at `1`; a breaking restructure bumps this and the `v1` segment of
/// [`DESCRIBE_METADATA_KEY`] together. The Rust builder is the only writer, so
/// no producer (Python or a future native SDK) can disagree on it.
pub const DESCRIBE_SCHEMA_VERSION: u32 = 1;
