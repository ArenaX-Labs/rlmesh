//! Version 1 of the adapter spec format and resolution semantics.
//!
//! Within v1 the JSON format evolves additively only (new optional fields
//! with defaults); breaking changes ship as a `v2` module and metadata key.

mod apply;
mod describe;
mod error;
mod keys;
mod plans;
mod pyfmt;
mod resolver;
pub mod roles;
mod space_view;
mod spec;

pub use apply::{
    ApplyError, Array, ArrayData, CustomTransform, Dtype, NoCustoms, SkipCustoms, Value,
    convert_rotation,
};
pub use error::AdapterResolutionError;
pub use keys::{ENV_METADATA_KEY, MODEL_METADATA_KEY};
pub use plans::{
    ActionPlan, ActionSegment, CustomPlan, ImagePlan, ObsPlan, ResolvedAdapter, StatePiece,
    StatePlan, TextPlan,
};
pub use resolver::resolve;
pub use space_view::{SpaceView, SpaceViewKind};
pub use spec::{
    ActionComponent, ActionLayout, CustomInput, EnvFeature, EnvFeatures, EnvImage, EnvState,
    EnvText, ImageInput, ImageLayout, ModelInput, ModelIoSpec, RotationEncoding, StateComponent,
    StateContainer, StateInput, TextContainer, TextInput,
};
