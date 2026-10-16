//! Declarative adapter specifications, resolution and per-step transforms.
mod callable;
mod codec;
mod execute;
mod leaves;
pub(crate) mod repr;
mod tag;
mod tree;
mod validation;
mod values;
mod vocabulary;
use crate::spaces::Space;
use crate::specs::value_to_json;
use crate::{Error, Metadata, Result, Value};
pub use callable::*;
pub use execute::{
    Adapter, AdapterBase, ResolveOptions, WrappedPredict, resolve, resolve_from_contract,
};
pub(crate) use execute::{contract_tags, resolve_parts, restore_shape};
pub use leaves::*;
use rlmesh_adapters::v1 as core;
use serde_json::{Value as Json, json};
use std::fmt;
pub use tag::{TagError, TagOptions, Tagged, tag, tag_vector};
pub use tree::*;
pub use values::*;
pub use vocabulary::*;

/// The shared Python spelling or ordered vocabulary.
pub const ENV_METADATA_KEY: &str = core::ENV_METADATA_KEY;
/// The shared Python spelling or ordered vocabulary.
pub const MODEL_METADATA_KEY: &str = core::MODEL_METADATA_KEY;
/// The shared Python spelling or ordered vocabulary.
pub const ENV_BRANCH_METADATA_KEY: &str = core::ENV_BRANCH_METADATA_KEY;
/// The shared Python spelling or ordered vocabulary.
pub const IMAGE_PRIMARY: &str = "image/primary";
/// The shared Python spelling or ordered vocabulary.
pub const IMAGE_SECONDARY: &str = "image/secondary";
/// The shared Python spelling or ordered vocabulary.
pub const INSTRUCTION: &str = "text/instruction";
/// The shared Python spelling or ordered vocabulary.
pub const JOINT_POS: &str = "proprio/joint_pos";
/// The shared Python spelling or ordered vocabulary.
pub const JOINT_VEL: &str = "proprio/joint_vel";
/// The shared Python spelling or ordered vocabulary.
pub const ACTION_JOINT_POS: &str = "action/joint_pos";
/// The shared Python spelling or ordered vocabulary.
pub const ACTION_JOINT_VEL: &str = "action/joint_vel";
/// The shared Python spelling or ordered vocabulary.
pub const IMAGE_WRIST: &str = "image/wrist";
/// The shared Python spelling or ordered vocabulary.
pub const EEF_POS: &str = "proprio/eef_pos";
/// The shared Python spelling or ordered vocabulary.
pub const EEF_ROT: &str = "proprio/eef_rot";
/// The shared Python spelling or ordered vocabulary.
pub const GRIPPER_POS: &str = "proprio/gripper";
/// The shared Python spelling or ordered vocabulary.
pub const EEF_WRENCH: &str = "proprio/eef_wrench";
/// The shared Python spelling or ordered vocabulary.
pub const ACTION_DELTA_POS: &str = "action/delta_eef_pos";
/// The shared Python spelling or ordered vocabulary.
pub const ACTION_DELTA_ROT: &str = "action/delta_eef_rot";
/// The shared Python spelling or ordered vocabulary.
pub const ACTION_GRIPPER: &str = "action/gripper";
/// The shared Python spelling or ordered vocabulary.
pub const ACTION_EEF_POS: &str = "action/eef_pos";
/// The shared Python spelling or ordered vocabulary.
pub const ACTION_EEF_ROT: &str = "action/eef_rot";
/// The shared Python spelling or ordered vocabulary.
pub const BASE_ANG_VEL: &str = "proprio/base_ang_vel";
/// The shared Python spelling or ordered vocabulary.
pub const BASE_ROT: &str = "proprio/base_rot";
/// The shared Python spelling or ordered vocabulary.
pub const COMMAND_BASE_VEL: &str = "command/base_vel";
/// The shared Python spelling or ordered vocabulary.
pub const LEFT_ARM: &str = "left_arm";
/// The shared Python spelling or ordered vocabulary.
pub const RIGHT_ARM: &str = "right_arm";
/// The shared Python spelling or ordered vocabulary.
pub const HEAD: &str = "head";
/// The shared Python spelling or ordered vocabulary.
pub const TORSO: &str = "torso";
/// The shared Python spelling or ordered vocabulary.
pub const BASE: &str = "base";
/// The shared Python spelling or ordered vocabulary.
pub const LEFT_LEG: &str = "left_leg";
/// The shared Python spelling or ordered vocabulary.
pub const RIGHT_LEG: &str = "right_leg";
/// The shared Python spelling or ordered vocabulary.
pub const PARTS: &[&str] = &[LEFT_ARM, RIGHT_ARM, HEAD, TORSO, BASE, LEFT_LEG, RIGHT_LEG];

/// A validated immutable model input tree and action layout.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelSpec {
    input: InputNode,
    output: Action,
}
impl ModelSpec {
    /// Construct this value with Python’s field defaults.
    pub fn new(input: impl Into<InputNode>, output: Action) -> Result<Self> {
        let mut input = input.into();
        let mut output = output;
        validation::normalize_input(&mut input);
        validation::normalize_action(&mut output);
        validation::input(&input)?;
        validation::action(&output)?;
        Ok(Self { input, output })
    }
    /// The authored immutable model input tree.
    pub fn input(&self) -> &InputNode {
        &self.input
    }
    /// The authored model action layout.
    pub fn output(&self) -> &Action {
        &self.output
    }
    pub(crate) fn wire(&self) -> Result<core::ModelSpec> {
        validation::finite_input(&self.input)?;
        validation::finite_action(&self.output)?;
        serde_json::from_value(
            json!({"input":self.input.wire_json()?,"output":codec::action_json(&self.output)?}),
        )
        .map_err(|e| Error::invalid_argument(e.to_string()))
    }
    pub(crate) fn wire_json(&self) -> Result<Json> {
        serde_json::to_value(self.wire()?).map_err(|e| Error::invalid_argument(e.to_string()))
    }
    /// Validate and serialize to canonical typed metadata.
    pub fn to_dict(&self) -> Result<Metadata> {
        if has_custom(&self.input) {
            return Err(Error::invalid_argument(
                "custom input cannot be published in v1 contract metadata",
            ));
        }
        let spec = self.wire()?;
        core::reject_unknowns_model(&spec).map_err(Error::invalid_argument)?;
        codec::metadata(
            serde_json::to_value(spec).map_err(|e| Error::invalid_argument(e.to_string()))?,
        )
    }
    /// Serialize to canonical compact, sorted UTF-8 JSON.
    pub fn to_json(&self) -> Result<String> {
        codec::format_json(&value_to_json(&Value::Dict(self.to_dict()?))?)
    }
    /// The fragment to merge into contract metadata.
    pub fn to_metadata(&self) -> Result<Metadata> {
        let mut m = Metadata::new();
        m.insert(MODEL_METADATA_KEY, self.to_dict()?);
        Ok(m)
    }
    /// Read and validate a typed metadata representation.
    pub fn from_dict(data: &Metadata) -> Result<Self> {
        Self::from_json(&codec::metadata_json(data)?)
    }
    /// Read and validate a JSON representation.
    pub fn from_json(payload: &str) -> Result<Self> {
        let wire: core::ModelSpec =
            serde_json::from_str(payload).map_err(|e| Error::invalid_argument(e.to_string()))?;
        core::reject_unknowns_model(&wire).map_err(Error::invalid_argument)?;
        Self::new(
            codec::input_from(
                serde_json::to_value(&wire.input)
                    .map_err(|e| Error::invalid_argument(e.to_string()))?,
            )?,
            codec::action_from(
                serde_json::to_value(wire.output)
                    .map_err(|e| Error::invalid_argument(e.to_string()))?,
            )?,
        )
    }
    /// Read the versioned metadata key; absent or None yields None.
    pub fn from_metadata(metadata: &Metadata) -> Result<Option<Self>> {
        metadata_payload(metadata, MODEL_METADATA_KEY)?
            .map(Self::from_dict)
            .transpose()
    }
}
fn has_custom(n: &InputNode) -> bool {
    match n {
        InputNode::Leaf(ModelLeaf::Custom(_)) => true,
        InputNode::Leaf(_) => false,
        InputNode::Dict(v) => v.iter().any(|(_, n)| has_custom(n)),
        InputNode::Tuple(v) => v.iter().any(has_custom),
    }
}
/// The semantic tags an env publishes for its observation and action spaces.
#[derive(Clone, Debug, PartialEq)]
pub struct EnvTags {
    observation: ObsNode,
    action: Action,
}
impl EnvTags {
    /// Construct this value with Python’s field defaults.
    pub fn new(observation: impl Into<ObsNode>, action: Action) -> Result<Self> {
        let mut observation = observation.into();
        let mut action = action;
        validation::normalize_obs(&mut observation);
        validation::normalize_action(&mut action);
        validation::obs(&observation)?;
        validation::action(&action)?;
        let mut keys = Vec::new();
        validation::identities(&observation, &mut keys);
        let mut seen = std::collections::HashSet::new();
        if keys.iter().any(|k| !seen.insert(k)) {
            return Err(Error::invalid_argument(
                "env tags declare a role more than once",
            ));
        }
        Ok(Self {
            observation,
            action,
        })
    }
    /// The authored immutable observation tag tree.
    pub fn observation(&self) -> &ObsNode {
        &self.observation
    }
    /// The authored environment action layout.
    pub fn action(&self) -> &Action {
        &self.action
    }
    pub(crate) fn wire(&self) -> Result<core::EnvTags> {
        validation::finite_obs(&self.observation)?;
        validation::finite_action(&self.action)?;
        serde_json::from_value(json!({"observation":self.observation.wire_json()?,"action":codec::action_json(&self.action)?})).map_err(|e|Error::invalid_argument(e.to_string()))
    }
    pub(crate) fn wire_json(&self) -> Result<Json> {
        serde_json::to_value(self.wire()?).map_err(|e| Error::invalid_argument(e.to_string()))
    }
    /// Validate and serialize to canonical typed metadata.
    pub fn to_dict(&self) -> Result<Metadata> {
        let wire = self.wire()?;
        core::reject_unknowns_env(&wire).map_err(Error::invalid_argument)?;
        codec::metadata(
            serde_json::to_value(wire).map_err(|e| Error::invalid_argument(e.to_string()))?,
        )
    }
    /// Serialize to canonical compact, sorted UTF-8 JSON.
    pub fn to_json(&self) -> Result<String> {
        codec::format_json(&value_to_json(&Value::Dict(self.to_dict()?))?)
    }
    /// The fragment to merge into contract metadata.
    pub fn to_metadata(&self) -> Result<Metadata> {
        let mut m = Metadata::new();
        m.insert(ENV_METADATA_KEY, self.to_dict()?);
        Ok(m)
    }
    /// Read and validate a typed metadata representation.
    pub fn from_dict(data: &Metadata) -> Result<Self> {
        Self::from_json(&codec::metadata_json(data)?)
    }
    /// Read and validate a JSON representation.
    pub fn from_json(payload: &str) -> Result<Self> {
        let wire: core::EnvTags =
            serde_json::from_str(payload).map_err(|e| Error::invalid_argument(e.to_string()))?;
        core::reject_unknowns_env(&wire).map_err(Error::invalid_argument)?;
        Self::new(
            codec::obs_from(
                serde_json::to_value(&wire.observation)
                    .map_err(|e| Error::invalid_argument(e.to_string()))?,
            )?,
            codec::action_from(
                serde_json::to_value(wire.action)
                    .map_err(|e| Error::invalid_argument(e.to_string()))?,
            )?,
        )
    }
    /// Read the versioned metadata key; absent or None yields None.
    pub fn from_metadata(metadata: &Metadata) -> Result<Option<Self>> {
        metadata_payload(metadata, ENV_METADATA_KEY)?
            .map(Self::from_dict)
            .transpose()
    }
    /// Declared roles grouped by modality, in authored order.
    pub fn observation_roles(&self) -> ObservationRoles {
        let mut keys = Vec::new();
        validation::identities(&self.observation, &mut keys);
        let mut r = ObservationRoles::default();
        for (kind, role, _, _) in keys {
            match kind {
                0 => r.images.push(role),
                1 => r.states.push(role),
                _ => r.texts.push(role),
            }
        }
        r
    }
    pub(crate) fn validate(&self, observation: &Space, action: &Space) -> Result<()> {
        core::join(
            &self.wire().map_err(resolution_error)?,
            &core::SpaceView::from(observation.spec().wire()),
            &core::SpaceView::from(action.spec().wire()),
        )
        .map(|_| ())
        .map_err(|e| Error::adapter_resolution(e.to_string()))
    }
    pub(crate) fn stamp(&self, metadata: &mut Metadata) -> Result<()> {
        metadata.insert(ENV_METADATA_KEY, self.to_dict().map_err(resolution_error)?);
        Ok(())
    }
}
fn metadata_payload<'a>(m: &'a Metadata, key: &str) -> Result<Option<&'a Metadata>> {
    match m.get(key) {
        None | Some(Value::None) => Ok(None),
        Some(Value::Dict(m)) => Ok(Some(m)),
        _ => Err(Error::adapter_resolution(format!(
            "metadata key {key:?} must hold a mapping"
        ))),
    }
}
pub(crate) fn resolution_error(e: impl fmt::Display) -> Error {
    Error::adapter_resolution(e.to_string())
}
impl fmt::Display for ModelSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ModelSpec(input={}, output={})", self.input, self.output)
    }
}
impl fmt::Display for EnvTags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EnvTags(observation={}, action={})",
            self.observation, self.action
        )
    }
}

#[cfg(test)]
mod tests;

pub(crate) use execute::{inject_instruction, register};

pub(crate) use codec::format_json;
