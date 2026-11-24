/// Key carrying serialized [`EnvTags`](crate::spec::EnvTags) in env contract metadata.
pub const ENV_METADATA_KEY: &str = "rlmesh.adapters.v1.env_tags";

/// Key carrying the contract-discriminant binding a branched `EnvFactory` chose
/// at `make()` time, in env contract metadata. The value names every declared discriminant,
/// defaults applied, so a reader never re-derives it from construction params.
pub const ENV_BRANCH_METADATA_KEY: &str = "rlmesh.adapters.v1.env_branch";

/// Key carrying a serialized [`ModelSpec`](crate::spec::ModelSpec) in model metadata.
pub const MODEL_METADATA_KEY: &str = "rlmesh.adapters.v1.model_spec";

/// Key carrying a serialized describe envelope (see [`build_describe_envelope`])
/// in env/model metadata or an OCI artifact.
///
/// [`build_describe_envelope`]: crate::envelope::build_describe_envelope
pub const DESCRIBE_METADATA_KEY: &str = "rlmesh.describe.v1";

/// The describe-envelope schema version stamped into every envelope. Additive fields keep it;
/// a breaking restructure bumps it together with the `v1` segment of
/// [`DESCRIBE_METADATA_KEY`].
pub const DESCRIBE_SCHEMA_VERSION: u32 = 1;
