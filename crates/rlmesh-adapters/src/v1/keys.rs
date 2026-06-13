//! Metadata keys under which v1 specs travel in contract metadata.

/// Key carrying serialized [`super::EnvAnnotations`] in env contract metadata.
pub const ENV_METADATA_KEY: &str = "rlmesh.adapters.v1.env_annotations";

/// Key carrying a serialized [`super::ModelIoSpec`] in model metadata.
pub const MODEL_METADATA_KEY: &str = "rlmesh.adapters.v1.model_spec";
