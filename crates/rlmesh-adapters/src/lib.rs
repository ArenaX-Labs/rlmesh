//! Declarative env/model IO adapter core for RLMesh.
//!
//! Environments and models describe their IO formats once as versioned,
//! pure-data specs; [`v1::resolve`] derives the concrete per-pairing
//! conversion plan by matching semantic roles. No code is ever evaluated
//! from spec data: custom transforms resolve to host-language callbacks
//! that bindings materialize themselves.
//!
//! The JSON wire format and resolution semantics are frozen per version by
//! the conformance vectors under this crate's `conformance/` directory;
//! every implementation and binding must pass them.
//!
//! The `rlmesh.adapters.v1.*` metadata key names the wire version, not the
//! source layout. As a pre-1.0 breaking wave, the v1 spec format has been
//! **redefined in place**: the two specs are now recursive trees (`ObsNode`,
//! `InputNode`) whose container type = the runtime container type, addressed by
//! structured paths (`NodePath`) rather than dotted strings, with the assembled
//! obs payload a `Value` tree. This redefinition sets the v1 *tree* contract;
//! within it the format then evolves additively only (new optional fields with
//! defaults), and a later breaking change bumps the key to v2. The [`v1`] facade
//! is only a stable import path; the implementation sits flat at the crate root.

mod apply;
mod describe;
mod error;
mod fmt;
mod join;
mod keys;
mod path;
mod plans;
mod resolver;
pub mod roles;
mod space_view;
mod spec;
mod stateful;

/// Version 1 of the adapter spec format and resolution semantics.
pub mod v1 {
    pub use crate::roles;

    pub use crate::apply::{
        ApplyError, CustomTransform, NoCustoms, SkipCustoms, Value, convert_rotation,
    };
    pub use crate::error::{AdapterResolutionError, ErrorCode};
    pub use crate::join::{JoinError, join};
    pub use crate::keys::{ENV_METADATA_KEY, MODEL_METADATA_KEY};
    pub use crate::path::{NodePath, PathSeg};
    pub use crate::plans::{
        ActionPlan, ActionSegment, CustomPlan, ImagePlan, ObsPlan, ResolvedAdapter, StatePiece,
        StatePlan, TextPlan,
    };
    pub use crate::resolver::resolve;
    pub use crate::space_view::{SpaceView, SpaceViewKind};
    pub use crate::spec::{
        Action, Actuator, ConcatPart, Custom, EnvFeature, EnvFeatures, EnvImage, EnvState, EnvTags,
        EnvText, Field, Image, ImageLayout, ImageTag, InputNode, ModelLeaf, ModelSpec, Normalize,
        ObsLeaf, ObsNode, RotationEncoding, SplitLayout, State, StateContainer, StateTag, Text,
        TextContainer, TextTag, UnknownFeature, reject_unknowns_env, reject_unknowns_model,
    };
    pub use crate::stateful::{
        EncodingTransform, FrameBuffers, NoEncodings, apply_actions, assemble_obs,
        space_value_to_obs_map, space_value_to_value, split_chunk, value_max_abs_diff,
    };
}
