//! Metadata keys under which v1 specs travel in contract metadata.

/// Key carrying serialized [`EnvTags`](crate::spec::EnvTags) in env contract metadata.
pub const ENV_METADATA_KEY: &str = "rlmesh.adapters.v1.env_tags";

/// Key carrying a serialized [`ModelSpec`](crate::spec::ModelSpec) in model metadata.
pub const MODEL_METADATA_KEY: &str = "rlmesh.adapters.v1.model_spec";
