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
