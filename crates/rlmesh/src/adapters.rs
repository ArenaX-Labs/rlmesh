//! Python's `rlmesh.adapters`, as far as env and model authoring need it: the
//! [`EnvTags`] an env publishes and the [`ModelSpec`] a model declares. The
//! typed tag and spec leaves, `tag` and `resolve` arrive with the adapters
//! surface; until then Rust builds both from their wire JSON with
//! [`crate::ext::env_tags_from_json`] and [`crate::ext::model_spec_from_json`].

use std::fmt;

use rlmesh_adapters::v1::{ENV_METADATA_KEY, SpaceView, join, reject_unknowns_env};

use crate::spaces::Space;
use crate::specs::{json_to_value, value_to_json};
use crate::{Error, Metadata, Result};

/// The semantic tags an env publishes for its spaces: Python's
/// `adapters.EnvTags`.
#[derive(Clone, PartialEq)]
pub struct EnvTags(rlmesh_adapters::v1::EnvTags);

impl EnvTags {
    pub(crate) fn from_json(text: &str) -> Result<EnvTags> {
        let tags: rlmesh_adapters::v1::EnvTags = serde_json::from_str(text)
            .map_err(|err| Error::invalid_argument(format!("invalid adapter tags: {err}")))?;
        reject_unknowns_env(&tags).map_err(|message| {
            Error::invalid_argument(format!("invalid adapter tags: {message}"))
        })?;
        Ok(EnvTags(tags))
    }

    pub(crate) fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(&self.0).unwrap_or(serde_json::Value::Null)
    }

    /// The tags an env's metadata publishes under `ENV_METADATA_KEY`, if any.
    pub(crate) fn from_metadata(metadata: &Metadata) -> Result<Option<EnvTags>> {
        let Some(value) = metadata.get(ENV_METADATA_KEY) else {
            return Ok(None);
        };
        if value.is_none() {
            return Ok(None);
        }
        let json = value_to_json(value)?;
        EnvTags::from_json(&json.to_string()).map(Some)
    }

    /// Validate these tags against the spaces they describe, with Python's
    /// `AdapterResolutionError` class.
    pub(crate) fn validate(&self, observation: &Space, action: &Space) -> Result<()> {
        join(
            &self.0,
            &SpaceView::from(observation.spec().wire()),
            &SpaceView::from(action.spec().wire()),
        )
        .map(|_| ())
        .map_err(|err| {
            Error::adapter_resolution(format!("adapter tags do not fit the spaces: {err}"))
        })
    }

    /// Stamp these tags into `metadata` under `ENV_METADATA_KEY`.
    pub(crate) fn stamp(&self, metadata: &mut Metadata) {
        metadata.insert(ENV_METADATA_KEY, json_to_value(self.to_json()));
    }
}

impl fmt::Debug for EnvTags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EnvTags({})", self.to_json())
    }
}

impl EnvTags {
    pub(crate) fn wire(&self) -> &rlmesh_adapters::v1::EnvTags {
        &self.0
    }
}

/// A model's declared input tree and action layout: Python's
/// `adapters.ModelSpec`. A model with a spec has its adapter resolved per
/// route from the env's [`EnvTags`].
#[derive(Clone, PartialEq)]
pub struct ModelSpec(rlmesh_adapters::v1::ModelSpec);

impl ModelSpec {
    pub(crate) fn from_json(text: &str) -> Result<ModelSpec> {
        let spec: rlmesh_adapters::v1::ModelSpec = serde_json::from_str(text)
            .map_err(|err| Error::invalid_argument(format!("invalid model spec: {err}")))?;
        rlmesh_adapters::v1::reject_unknowns_model(&spec)
            .map_err(|message| Error::invalid_argument(format!("invalid model spec: {message}")))?;
        Ok(ModelSpec(spec))
    }

    pub(crate) fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(&self.0).unwrap_or(serde_json::Value::Null)
    }

    pub(crate) fn wire(&self) -> &rlmesh_adapters::v1::ModelSpec {
        &self.0
    }
}

impl fmt::Debug for ModelSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ModelSpec({})", self.to_json())
    }
}
