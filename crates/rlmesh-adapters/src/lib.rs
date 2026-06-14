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
//! Within v1 the JSON format evolves additively only (new optional fields with
//! defaults). The wire version is the `rlmesh.adapters.v1.*` metadata key, not
//! the source layout. A breaking spec-format change bumps that key to v2 while
//! still dual-reading v1, independent of how the Rust modules are organized. The
//! [`v1`] facade is only a stable import path; the implementation sits flat at
//! the crate root.

mod apply;
mod describe;
mod error;
mod fmt;
mod join;
mod keys;
mod plans;
mod resolver;
pub mod roles;
mod space_view;
mod spec;

/// Version 1 of the adapter spec format and resolution semantics.
pub mod v1 {
    pub use crate::roles;

    pub use crate::apply::{
        ApplyError, CustomTransform, NoCustoms, SkipCustoms, Value, convert_rotation,
    };
    pub use crate::error::{AdapterResolutionError, ErrorCode};
    pub use crate::join::{JoinError, join};
    pub use crate::keys::{ENV_METADATA_KEY, MODEL_METADATA_KEY};
    pub use crate::plans::{
        ActionPlan, ActionSegment, CustomPlan, ImagePlan, ObsPlan, ResolvedAdapter, StatePiece,
        StatePlan, TextPlan,
    };
    pub use crate::resolver::resolve;
    pub use crate::space_view::{SpaceView, SpaceViewKind};
    pub use crate::spec::{
        ActionComponent, ActionLayout, CustomInput, EnvFeature, EnvFeatures, EnvImage, EnvState,
        EnvTags, EnvText, ImageInput, ImageLayout, ImageTag, ModelInput, ModelSpec, ObsTag,
        RotationEncoding, StateComponent, StateContainer, StateField, StateInput, StateLayout,
        StateTag, TextContainer, TextInput, TextTag,
    };
}
