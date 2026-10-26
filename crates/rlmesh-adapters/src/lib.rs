//! Declarative env/model IO specs for RLMesh: the v1 declarations, and the
//! host side of binding them.
//!
//! Environments and models describe their IO formats once as versioned,
//! pure-data specs ([`v1::EnvTags`], [`v1::ModelSpec`]). [`v1::resolve`]
//! joins an env's tags against its spaces, lowers both declarations into
//! the binder's input and binds them with the binder's Gym door
//! ([`rlmesh_session_core::spec::role`]), which decides every conversion and
//! runs it; this crate walks the host values (observation trees, Gym
//! spaces, frame windows) around it. No code is ever evaluated from spec
//! data: custom transforms resolve to host-language callbacks that bindings
//! materialize themselves.
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
mod envelope;
mod error;
mod join;
mod keys;
mod lower;
mod space_json;
mod space_view;
mod spec;
mod stateful;

pub use rlmesh_session_core::spec::role::roles;
use rlmesh_session_core::spec::role::{path, plan as plans};
use rlmesh_spaces::advisory;

/// Version 1 of the adapter spec format and resolution semantics.
pub mod v1 {
    pub use crate::roles;

    pub use crate::advisory::{Advisory, AdvisorySeverity};
    pub use crate::apply::{
        ApplyError, CustomTransform, Jpeg, NoCustoms, RESAMPLES, SkipCustoms, Value,
        transform_action, transform_obs,
    };
    pub use crate::envelope::{EnvelopeError, Kind, build_describe_envelope};
    pub use crate::error::{AdapterResolutionError, ErrorCode};
    pub use crate::join::{JoinError, join};
    pub use crate::keys::{
        DESCRIBE_METADATA_KEY, DESCRIBE_SCHEMA_VERSION, ENV_BRANCH_METADATA_KEY, ENV_METADATA_KEY,
        MODEL_METADATA_KEY,
    };
    pub use crate::lower::{advisories, resolve};
    pub use crate::path::{NodePath, PathSeg};
    pub use crate::plans::{
        ActionPlan, ActionSegment, Binding as ResolvedAdapter, CropPlan, CustomPlan, HistoryWindow,
        ImagePlan, ObsPlan, PreviousAction, StatePiece, StatePlan, TextPlan,
    };
    pub use crate::space_json::{
        SpaceJsonError, space_dtype_name, space_kind_name, space_spec_from_json, space_spec_to_json,
    };
    pub use crate::space_view::{SpaceView, SpaceViewKind};
    pub use crate::spec::{
        Action, ActionEncoding, Actuator, Attr, CHANNEL_ORDERS, CROP_MODES, ConcatPart, Custom,
        CustomEncoding, EnvFeature, EnvFeatures, EnvImage, EnvState, EnvTags, EnvText, FRAMES,
        Field, FrameLaw, FramePolicy, FrameRef, Image, ImageLayout, ImageTag, InputNode, ModelLeaf,
        ModelSpec, Normalize, ObsLeaf, ObsNode, PROVENANCES, PartSource, Provenance, REFERENCES,
        ReferenceLaw, RolePolicy, RotationEncoding, RotationLiteral, SplitLayout, StackPad, State,
        StateContainer, StateEncoding, StateTag, Text, TextContainer, TextTag, Timing,
        UnknownFeature, reject_bare_fields_env, reject_bare_fields_model,
        reject_unframed_roles_env, reject_unframed_roles_model, reject_unknowns_env,
        reject_unknowns_model, reject_unsanctioned_roles_env, reject_unsanctioned_roles_model,
        render_requests,
    };
    pub use crate::stateful::{
        EncodingTransform, FrameBuffers, MAX_EXECUTION_HORIZON, NoEncodings, apply_actions,
        assemble_obs, observe_obs, record_action, space_value_to_obs_map, space_value_to_value,
        split_chunk, value_max_abs_diff,
    };
}
