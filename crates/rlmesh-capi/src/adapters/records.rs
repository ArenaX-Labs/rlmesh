//! C adapter records and vocabulary conversion.
#![allow(unsafe_code)]
use super::*;
pub(crate) fn enum_image_layout(v: i32) -> Result<a::ImageLayout, CapiError> {
    match v {
        1 => Ok(a::ImageLayout::Hwc),
        2 => Ok(a::ImageLayout::Chw),
        _ => Err(CapiError::invalid_arg("invalid ImageLayout")),
    }
}
pub(crate) fn number_image_layout(v: a::ImageLayout) -> i32 {
    match v {
        a::ImageLayout::Hwc => 1,
        a::ImageLayout::Chw => 2,
        _ => 0,
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_image_layout_str(v: i32) -> *const c_char {
    enum_image_layout(v).map_or(std::ptr::null(), |v| {
        crate::value::dtype::static_str(v.as_str())
    })
}
pub(crate) fn enum_fit_mode(v: i32) -> Result<a::FitMode, CapiError> {
    match v {
        1 => Ok(a::FitMode::Stretch),
        2 => Ok(a::FitMode::Crop),
        3 => Ok(a::FitMode::Pad),
        _ => Err(CapiError::invalid_arg("invalid FitMode")),
    }
}
pub(crate) fn number_fit_mode(v: a::FitMode) -> i32 {
    match v {
        a::FitMode::Stretch => 1,
        a::FitMode::Crop => 2,
        a::FitMode::Pad => 3,
        _ => 0,
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_fit_mode_str(v: i32) -> *const c_char {
    enum_fit_mode(v).map_or(std::ptr::null(), |v| {
        crate::value::dtype::static_str(v.as_str())
    })
}
pub(crate) fn enum_frame(v: i32) -> Result<a::Frame, CapiError> {
    match v {
        1 => Ok(a::Frame::World),
        2 => Ok(a::Frame::RobotBase),
        3 => Ok(a::Frame::Tool),
        _ => Err(CapiError::invalid_arg("invalid Frame")),
    }
}
pub(crate) fn number_frame(v: a::Frame) -> i32 {
    match v {
        a::Frame::World => 1,
        a::Frame::RobotBase => 2,
        a::Frame::Tool => 3,
        _ => 0,
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_frame_str(v: i32) -> *const c_char {
    enum_frame(v).map_or(std::ptr::null(), |v| {
        crate::value::dtype::static_str(v.as_str())
    })
}
pub(crate) fn enum_reference(v: i32) -> Result<a::Reference, CapiError> {
    match v {
        1 => Ok(a::Reference::Current),
        2 => Ok(a::Reference::Target),
        _ => Err(CapiError::invalid_arg("invalid Reference")),
    }
}
pub(crate) fn number_reference(v: a::Reference) -> i32 {
    match v {
        a::Reference::Current => 1,
        a::Reference::Target => 2,
        _ => 0,
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_reference_str(v: i32) -> *const c_char {
    enum_reference(v).map_or(std::ptr::null(), |v| {
        crate::value::dtype::static_str(v.as_str())
    })
}
pub(crate) fn enum_provenance(v: i32) -> Result<a::Provenance, CapiError> {
    match v {
        1 => Ok(a::Provenance::Sensed),
        2 => Ok(a::Provenance::Estimated),
        3 => Ok(a::Provenance::Privileged),
        _ => Err(CapiError::invalid_arg("invalid Provenance")),
    }
}
pub(crate) fn number_provenance(v: a::Provenance) -> i32 {
    match v {
        a::Provenance::Sensed => 1,
        a::Provenance::Estimated => 2,
        a::Provenance::Privileged => 3,
        _ => 0,
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_provenance_str(v: i32) -> *const c_char {
    enum_provenance(v).map_or(std::ptr::null(), |v| {
        crate::value::dtype::static_str(v.as_str())
    })
}
pub(crate) fn enum_resample(v: i32) -> Result<a::Resample, CapiError> {
    match v {
        1 => Ok(a::Resample::Bilinear),
        2 => Ok(a::Resample::BilinearAa),
        3 => Ok(a::Resample::BicubicAa),
        4 => Ok(a::Resample::Lanczos3Aa),
        5 => Ok(a::Resample::Area),
        _ => Err(CapiError::invalid_arg("invalid Resample")),
    }
}
pub(crate) fn number_resample(v: a::Resample) -> i32 {
    match v {
        a::Resample::Bilinear => 1,
        a::Resample::BilinearAa => 2,
        a::Resample::BicubicAa => 3,
        a::Resample::Lanczos3Aa => 4,
        a::Resample::Area => 5,
        _ => 0,
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_resample_str(v: i32) -> *const c_char {
    enum_resample(v).map_or(std::ptr::null(), |v| {
        crate::value::dtype::static_str(v.as_str())
    })
}
pub(crate) fn enum_rotation_encoding(v: i32) -> Result<a::RotationEncoding, CapiError> {
    match v {
        1 => Ok(a::RotationEncoding::QuatXyzw),
        2 => Ok(a::RotationEncoding::QuatWxyz),
        3 => Ok(a::RotationEncoding::AxisAngle),
        4 => Ok(a::RotationEncoding::Rot6d),
        5 => Ok(a::RotationEncoding::Rot6dRowmajor),
        6 => Ok(a::RotationEncoding::EulerXyz),
        7 => Ok(a::RotationEncoding::GravityXyz),
        _ => Err(CapiError::invalid_arg("invalid RotationEncoding")),
    }
}
pub(crate) fn number_rotation_encoding(v: a::RotationEncoding) -> i32 {
    match v {
        a::RotationEncoding::QuatXyzw => 1,
        a::RotationEncoding::QuatWxyz => 2,
        a::RotationEncoding::AxisAngle => 3,
        a::RotationEncoding::Rot6d => 4,
        a::RotationEncoding::Rot6dRowmajor => 5,
        a::RotationEncoding::EulerXyz => 6,
        a::RotationEncoding::GravityXyz => 7,
        _ => 0,
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_rotation_encoding_str(v: i32) -> *const c_char {
    enum_rotation_encoding(v).map_or(std::ptr::null(), |v| {
        crate::value::dtype::static_str(v.as_str())
    })
}
pub(crate) fn enum_stack_pad(v: i32) -> Result<a::StackPad, CapiError> {
    match v {
        1 => Ok(a::StackPad::First),
        2 => Ok(a::StackPad::Black),
        _ => Err(CapiError::invalid_arg("invalid StackPad")),
    }
}
pub(crate) fn number_stack_pad(v: a::StackPad) -> i32 {
    match v {
        a::StackPad::First => 1,
        a::StackPad::Black => 2,
        _ => 0,
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_stack_pad_str(v: i32) -> *const c_char {
    enum_stack_pad(v).map_or(std::ptr::null(), |v| {
        crate::value::dtype::static_str(v.as_str())
    })
}
pub(crate) fn enum_crop_mode(v: i32) -> Result<a::CropMode, CapiError> {
    match v {
        1 => Ok(a::CropMode::Zoom),
        2 => Ok(a::CropMode::Slice),
        _ => Err(CapiError::invalid_arg("invalid CropMode")),
    }
}
pub(crate) fn number_crop_mode(v: a::CropMode) -> i32 {
    match v {
        a::CropMode::Zoom => 1,
        a::CropMode::Slice => 2,
        _ => 0,
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_crop_mode_str(v: i32) -> *const c_char {
    enum_crop_mode(v).map_or(std::ptr::null(), |v| {
        crate::value::dtype::static_str(v.as_str())
    })
}
pub(crate) fn enum_channel_order(v: i32) -> Result<a::ChannelOrder, CapiError> {
    match v {
        1 => Ok(a::ChannelOrder::Rgb),
        2 => Ok(a::ChannelOrder::Bgr),
        _ => Err(CapiError::invalid_arg("invalid ChannelOrder")),
    }
}
pub(crate) fn number_channel_order(v: a::ChannelOrder) -> i32 {
    match v {
        a::ChannelOrder::Rgb => 1,
        a::ChannelOrder::Bgr => 2,
        _ => 0,
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_channel_order_str(v: i32) -> *const c_char {
    enum_channel_order(v).map_or(std::ptr::null(), |v| {
        crate::value::dtype::static_str(v.as_str())
    })
}
pub(crate) fn enum_state_container(v: i32) -> Result<a::StateContainer, CapiError> {
    match v {
        1 => Ok(a::StateContainer::Array),
        2 => Ok(a::StateContainer::List),
        _ => Err(CapiError::invalid_arg("invalid StateContainer")),
    }
}
pub(crate) fn number_state_container(v: a::StateContainer) -> i32 {
    match v {
        a::StateContainer::Array => 1,
        a::StateContainer::List => 2,
        _ => 0,
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_state_container_str(v: i32) -> *const c_char {
    enum_state_container(v).map_or(std::ptr::null(), |v| {
        crate::value::dtype::static_str(v.as_str())
    })
}
pub(crate) fn enum_text_container(v: i32) -> Result<a::TextContainer, CapiError> {
    match v {
        1 => Ok(a::TextContainer::Str),
        2 => Ok(a::TextContainer::List),
        _ => Err(CapiError::invalid_arg("invalid TextContainer")),
    }
}
pub(crate) fn number_text_container(v: a::TextContainer) -> i32 {
    match v {
        a::TextContainer::Str => 1,
        a::TextContainer::List => 2,
        _ => 0,
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_text_container_str(v: i32) -> *const c_char {
    enum_text_container(v).map_or(std::ptr::null(), |v| {
        crate::value::dtype::static_str(v.as_str())
    })
}
pub(crate) fn enum_state_source(v: i32) -> Result<a::StateSource, CapiError> {
    match v {
        1 => Ok(a::StateSource::Observation),
        2 => Ok(a::StateSource::Action),
        _ => Err(CapiError::invalid_arg("invalid StateSource")),
    }
}
pub(crate) fn number_state_source(v: a::StateSource) -> i32 {
    match v {
        a::StateSource::Observation => 1,
        a::StateSource::Action => 2,
        _ => 0,
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_state_source_str(v: i32) -> *const c_char {
    enum_state_source(v).map_or(std::ptr::null(), |v| {
        crate::value::dtype::static_str(v.as_str())
    })
}
pub(crate) fn enum_advisory_severity(v: i32) -> Result<a::AdvisorySeverity, CapiError> {
    match v {
        1 => Ok(a::AdvisorySeverity::Info),
        2 => Ok(a::AdvisorySeverity::Caution),
        _ => Err(CapiError::invalid_arg("invalid AdvisorySeverity")),
    }
}
pub(crate) fn number_advisory_severity(v: a::AdvisorySeverity) -> i32 {
    match v {
        a::AdvisorySeverity::Info => 1,
        a::AdvisorySeverity::Caution => 2,
        _ => 0,
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_advisory_severity_str(v: i32) -> *const c_char {
    enum_advisory_severity(v).map_or(std::ptr::null(), |v| {
        crate::value::dtype::static_str(v.as_str())
    })
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshInputNode {
    pub kind: i32,
    pub leaf: *const c_void,
    pub keys: *const RLMeshStr,
    pub children: *const RLMeshInputNode,
    pub children_len: usize,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshObsNode {
    pub kind: i32,
    pub leaf: *const c_void,
    pub keys: *const RLMeshStr,
    pub children: *const RLMeshObsNode,
    pub children_len: usize,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshConcatPart {
    pub kind: i32,
    pub role: *const c_char,
    pub leaf: *const c_void,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshEncodingArm {
    pub transform: *const RLMeshRotationTransform,
    pub entrypoint: *const c_char,
}

#[cfg(target_pointer_width = "64")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshCustomEncoding {
    pub struct_size: usize,
    pub base: i32,
    pub _reserved0: [u8; 4],
    pub from_base: RLMeshEncodingArm,
    pub to_base: RLMeshEncodingArm,
    pub name: *const c_char,
}
#[cfg(target_pointer_width = "32")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshCustomEncoding {
    pub struct_size: usize,
    pub base: i32,
    pub from_base: RLMeshEncodingArm,
    pub to_base: RLMeshEncodingArm,
    pub name: *const c_char,
}
#[cfg(target_pointer_width = "64")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshImage {
    pub struct_size: usize,
    pub role: *const c_char,
    pub has_height: bool,
    pub _reserved0: [u8; 3],
    pub height: u32,
    pub has_width: bool,
    pub _reserved1: [u8; 3],
    pub width: u32,
    pub layout: i32,
    pub has_channels: bool,
    pub _reserved2: [u8; 3],
    pub channels: u32,
    pub dtype: i32,
    pub dtype_other: *const c_char,
    pub normalize: i32,
    pub _reserved3: [u8; 4],
    pub normalize_range: [f64; 2],
    pub lead_dims: u32,
    pub upside_down: bool,
    pub _reserved4: [u8; 3],
    pub resample: i32,
    pub _reserved5: [u8; 4],
    pub resample_other: *const c_char,
    pub allow_upscale: bool,
    pub _reserved6: [u8; 7],
    pub fit: *const i32,
    pub fit_len: usize,
    pub fit_other: *const *const c_char,
    pub optional: bool,
    pub has_fill: bool,
    pub fill: u8,
    pub _reserved7: [u8; 1],
    pub stack: u32,
    pub has_crop: bool,
    pub _reserved8: [u8; 7],
    pub crop: f64,
    pub has_crop_area: bool,
    pub _reserved9: [u8; 7],
    pub crop_area: f64,
    pub crop_mode: i32,
    pub _reserved10: [u8; 4],
    pub crop_mode_other: *const c_char,
    pub has_jpeg_quality: bool,
    pub jpeg_quality: u8,
    pub _reserved11: [u8; 2],
    pub channel_order: i32,
    pub channel_order_other: *const c_char,
    pub has_render: bool,
    pub _reserved12: [u8; 3],
    pub render: [u32; 2],
    pub _reserved13: [u8; 4],
    pub offsets: *const i32,
    pub offsets_len: usize,
    pub stack_pad: i32,
    pub _reserved14: [u8; 4],
    pub part: *const c_char,
    pub has_stride: bool,
    pub _reserved15: [u8; 3],
    pub stride: u32,
}
#[cfg(target_pointer_width = "32")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshImage {
    pub struct_size: usize,
    pub role: *const c_char,
    pub has_height: bool,
    pub _reserved0: [u8; 3],
    pub height: u32,
    pub has_width: bool,
    pub _reserved1: [u8; 3],
    pub width: u32,
    pub layout: i32,
    pub has_channels: bool,
    pub _reserved2: [u8; 3],
    pub channels: u32,
    pub dtype: i32,
    pub dtype_other: *const c_char,
    pub normalize: i32,
    pub normalize_range: [f64; 2],
    pub lead_dims: u32,
    pub upside_down: bool,
    pub _reserved3: [u8; 3],
    pub resample: i32,
    pub resample_other: *const c_char,
    pub allow_upscale: bool,
    pub _reserved4: [u8; 3],
    pub fit: *const i32,
    pub fit_len: usize,
    pub fit_other: *const *const c_char,
    pub optional: bool,
    pub has_fill: bool,
    pub fill: u8,
    pub _reserved5: [u8; 1],
    pub stack: u32,
    pub has_crop: bool,
    pub _reserved6: [u8; 7],
    pub crop: f64,
    pub has_crop_area: bool,
    pub _reserved7: [u8; 7],
    pub crop_area: f64,
    pub crop_mode: i32,
    pub crop_mode_other: *const c_char,
    pub has_jpeg_quality: bool,
    pub jpeg_quality: u8,
    pub _reserved8: [u8; 2],
    pub channel_order: i32,
    pub channel_order_other: *const c_char,
    pub has_render: bool,
    pub _reserved9: [u8; 3],
    pub render: [u32; 2],
    pub offsets: *const i32,
    pub offsets_len: usize,
    pub stack_pad: i32,
    pub part: *const c_char,
    pub has_stride: bool,
    pub _reserved10: [u8; 3],
    pub stride: u32,
}
#[cfg(target_pointer_width = "64")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshState {
    pub struct_size: usize,
    pub role: *const c_char,
    pub encoding: *const i32,
    pub encoding_len: usize,
    pub encoding_other: *const *const c_char,
    pub custom_encoding: *const RLMeshCustomEncoding,
    pub has_dim: bool,
    pub _reserved0: [u8; 3],
    pub dim: u32,
    pub has_index: bool,
    pub _reserved1: [u8; 3],
    pub index: u32,
    pub optional: bool,
    pub has_range: bool,
    pub _reserved2: [u8; 6],
    pub range: [f64; 2],
    pub has_pad_to: bool,
    pub _reserved3: [u8; 3],
    pub pad_to: u32,
    pub dtype: i32,
    pub _reserved4: [u8; 4],
    pub dtype_other: *const c_char,
    pub reshape: *const i64,
    pub reshape_len: usize,
    pub container: i32,
    pub _reserved5: [u8; 4],
    pub fill: f64,
    pub post_rotate: *const RLMeshRotation,
    pub has_scale: bool,
    pub _reserved6: [u8; 7],
    pub scale: f64,
    pub axis_scale: *const f64,
    pub axis_scale_len: usize,
    pub has_offset: bool,
    pub _reserved7: [u8; 7],
    pub offset: f64,
    pub axis_offset: *const f64,
    pub axis_offset_len: usize,
    pub has_frame: bool,
    pub _reserved8: [u8; 3],
    pub frame: i32,
    pub frame_other: *const c_char,
    pub provenance: *const i32,
    pub provenance_len: usize,
    pub provenance_other: *const *const c_char,
    pub part: *const c_char,
    pub labels: *const *const c_char,
    pub labels_len: usize,
    pub source: i32,
    pub has_clip: bool,
    pub _reserved9: [u8; 3],
    pub clip: [f64; 2],
}
#[cfg(target_pointer_width = "32")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshState {
    pub struct_size: usize,
    pub role: *const c_char,
    pub encoding: *const i32,
    pub encoding_len: usize,
    pub encoding_other: *const *const c_char,
    pub custom_encoding: *const RLMeshCustomEncoding,
    pub has_dim: bool,
    pub _reserved0: [u8; 3],
    pub dim: u32,
    pub has_index: bool,
    pub _reserved1: [u8; 3],
    pub index: u32,
    pub optional: bool,
    pub has_range: bool,
    pub _reserved2: [u8; 6],
    pub range: [f64; 2],
    pub has_pad_to: bool,
    pub _reserved3: [u8; 3],
    pub pad_to: u32,
    pub dtype: i32,
    pub dtype_other: *const c_char,
    pub reshape: *const i64,
    pub reshape_len: usize,
    pub container: i32,
    pub _reserved4: [u8; 4],
    pub fill: f64,
    pub post_rotate: *const RLMeshRotation,
    pub has_scale: bool,
    pub _reserved5: [u8; 3],
    pub scale: f64,
    pub axis_scale: *const f64,
    pub axis_scale_len: usize,
    pub has_offset: bool,
    pub _reserved6: [u8; 7],
    pub offset: f64,
    pub axis_offset: *const f64,
    pub axis_offset_len: usize,
    pub has_frame: bool,
    pub _reserved7: [u8; 3],
    pub frame: i32,
    pub frame_other: *const c_char,
    pub provenance: *const i32,
    pub provenance_len: usize,
    pub provenance_other: *const *const c_char,
    pub part: *const c_char,
    pub labels: *const *const c_char,
    pub labels_len: usize,
    pub source: i32,
    pub has_clip: bool,
    pub _reserved8: [u8; 7],
    pub clip: [f64; 2],
}
#[cfg(target_pointer_width = "64")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshConcat {
    pub struct_size: usize,
    pub parts: *const RLMeshConcatPart,
    pub parts_len: usize,
    pub has_pad_to: bool,
    pub _reserved0: [u8; 3],
    pub pad_to: u32,
    pub dtype: i32,
    pub _reserved1: [u8; 4],
    pub dtype_other: *const c_char,
    pub reshape: *const i64,
    pub reshape_len: usize,
    pub container: i32,
    pub has_clip: bool,
    pub _reserved2: [u8; 3],
    pub clip: [f64; 2],
}
#[cfg(target_pointer_width = "32")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshConcat {
    pub struct_size: usize,
    pub parts: *const RLMeshConcatPart,
    pub parts_len: usize,
    pub has_pad_to: bool,
    pub _reserved0: [u8; 3],
    pub pad_to: u32,
    pub dtype: i32,
    pub dtype_other: *const c_char,
    pub reshape: *const i64,
    pub reshape_len: usize,
    pub container: i32,
    pub has_clip: bool,
    pub _reserved1: [u8; 7],
    pub clip: [f64; 2],
}
#[cfg(target_pointer_width = "64")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshConstant {
    pub struct_size: usize,
    pub dim: u32,
    pub _reserved0: [u8; 4],
    pub fill: f64,
}
#[cfg(target_pointer_width = "32")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshConstant {
    pub struct_size: usize,
    pub dim: u32,
    pub fill: f64,
}
#[cfg(target_pointer_width = "64")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshAdaptersText {
    pub struct_size: usize,
    pub role: *const c_char,
    pub container: i32,
    pub has_fill: bool,
    pub _reserved0: [u8; 3],
    pub fill: RLMeshStr,
}
#[cfg(target_pointer_width = "32")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshAdaptersText {
    pub struct_size: usize,
    pub role: *const c_char,
    pub container: i32,
    pub has_fill: bool,
    pub _reserved0: [u8; 3],
    pub fill: RLMeshStr,
}
#[cfg(target_pointer_width = "64")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshCustom {
    pub struct_size: usize,
    pub transform: *const RLMeshObsTransform,
    pub entrypoint: *const c_char,
}
#[cfg(target_pointer_width = "32")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshCustom {
    pub struct_size: usize,
    pub transform: *const RLMeshObsTransform,
    pub entrypoint: *const c_char,
}
#[cfg(target_pointer_width = "64")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshActuator {
    pub struct_size: usize,
    pub role: *const c_char,
    pub dim: u32,
    pub has_encoding: bool,
    pub _reserved0: [u8; 3],
    pub encoding: i32,
    pub _reserved1: [u8; 4],
    pub custom_encoding: *const RLMeshCustomEncoding,
    pub has_range: bool,
    pub _reserved2: [u8; 7],
    pub range: [f64; 2],
    pub binary: bool,
    pub has_scale: bool,
    pub _reserved3: [u8; 6],
    pub scale: f64,
    pub axis_scale: *const f64,
    pub axis_scale_len: usize,
    pub invert: bool,
    pub has_threshold: bool,
    pub _reserved4: [u8; 6],
    pub threshold: f64,
    pub clip: bool,
    pub _reserved5: [u8; 7],
    pub fill: f64,
    pub optional: bool,
    pub has_frame: bool,
    pub _reserved6: [u8; 2],
    pub frame: i32,
    pub frame_other: *const c_char,
    pub has_reference: bool,
    pub _reserved7: [u8; 3],
    pub reference: i32,
    pub reference_other: *const c_char,
    pub part: *const c_char,
    pub has_offset: bool,
    pub _reserved8: [u8; 7],
    pub offset: f64,
    pub axis_offset: *const f64,
    pub axis_offset_len: usize,
    pub labels: *const *const c_char,
    pub labels_len: usize,
}
#[cfg(target_pointer_width = "32")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshActuator {
    pub struct_size: usize,
    pub role: *const c_char,
    pub dim: u32,
    pub has_encoding: bool,
    pub _reserved0: [u8; 3],
    pub encoding: i32,
    pub custom_encoding: *const RLMeshCustomEncoding,
    pub has_range: bool,
    pub _reserved1: [u8; 7],
    pub range: [f64; 2],
    pub binary: bool,
    pub has_scale: bool,
    pub _reserved2: [u8; 6],
    pub scale: f64,
    pub axis_scale: *const f64,
    pub axis_scale_len: usize,
    pub invert: bool,
    pub has_threshold: bool,
    pub _reserved3: [u8; 6],
    pub threshold: f64,
    pub clip: bool,
    pub _reserved4: [u8; 7],
    pub fill: f64,
    pub optional: bool,
    pub has_frame: bool,
    pub _reserved5: [u8; 2],
    pub frame: i32,
    pub frame_other: *const c_char,
    pub has_reference: bool,
    pub _reserved6: [u8; 3],
    pub reference: i32,
    pub reference_other: *const c_char,
    pub part: *const c_char,
    pub has_offset: bool,
    pub _reserved7: [u8; 3],
    pub offset: f64,
    pub axis_offset: *const f64,
    pub axis_offset_len: usize,
    pub labels: *const *const c_char,
    pub labels_len: usize,
}
#[cfg(target_pointer_width = "64")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshAction {
    pub struct_size: usize,
    pub components: *const *const RLMeshActuator,
    pub components_len: usize,
    pub has_clip: bool,
    pub _reserved0: [u8; 7],
    pub clip: [f64; 2],
}
#[cfg(target_pointer_width = "32")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshAction {
    pub struct_size: usize,
    pub components: *const *const RLMeshActuator,
    pub components_len: usize,
    pub has_clip: bool,
    pub _reserved0: [u8; 3],
    pub clip: [f64; 2],
}
#[cfg(target_pointer_width = "64")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshImageTag {
    pub struct_size: usize,
    pub role: *const c_char,
    pub layout: i32,
    pub upside_down: bool,
    pub _reserved0: [u8; 3],
    pub part: *const c_char,
}
#[cfg(target_pointer_width = "32")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshImageTag {
    pub struct_size: usize,
    pub role: *const c_char,
    pub layout: i32,
    pub upside_down: bool,
    pub _reserved0: [u8; 3],
    pub part: *const c_char,
}
#[cfg(target_pointer_width = "64")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshStateTag {
    pub struct_size: usize,
    pub role: *const c_char,
    pub encoding: *const i32,
    pub encoding_len: usize,
    pub encoding_other: *const *const c_char,
    pub has_range: bool,
    pub _reserved0: [u8; 7],
    pub range: [f64; 2],
    pub has_frame: bool,
    pub _reserved1: [u8; 3],
    pub frame: i32,
    pub frame_other: *const c_char,
    pub has_provenance: bool,
    pub _reserved2: [u8; 3],
    pub provenance: i32,
    pub provenance_other: *const c_char,
    pub part: *const c_char,
    pub labels: *const *const c_char,
    pub labels_len: usize,
}
#[cfg(target_pointer_width = "32")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshStateTag {
    pub struct_size: usize,
    pub role: *const c_char,
    pub encoding: *const i32,
    pub encoding_len: usize,
    pub encoding_other: *const *const c_char,
    pub has_range: bool,
    pub _reserved0: [u8; 3],
    pub range: [f64; 2],
    pub has_frame: bool,
    pub _reserved1: [u8; 3],
    pub frame: i32,
    pub frame_other: *const c_char,
    pub has_provenance: bool,
    pub _reserved2: [u8; 3],
    pub provenance: i32,
    pub provenance_other: *const c_char,
    pub part: *const c_char,
    pub labels: *const *const c_char,
    pub labels_len: usize,
    pub _reserved3: [u8; 4],
}
#[cfg(target_pointer_width = "64")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshTextTag {
    pub struct_size: usize,
    pub role: *const c_char,
}
#[cfg(target_pointer_width = "32")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshTextTag {
    pub struct_size: usize,
    pub role: *const c_char,
}
#[cfg(target_pointer_width = "64")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshField {
    pub struct_size: usize,
    pub role: *const c_char,
    pub dim: u32,
    pub _reserved0: [u8; 4],
    pub encoding: *const i32,
    pub encoding_len: usize,
    pub encoding_other: *const *const c_char,
    pub has_range: bool,
    pub _reserved1: [u8; 7],
    pub range: [f64; 2],
    pub has_frame: bool,
    pub _reserved2: [u8; 3],
    pub frame: i32,
    pub frame_other: *const c_char,
    pub has_provenance: bool,
    pub _reserved3: [u8; 3],
    pub provenance: i32,
    pub provenance_other: *const c_char,
    pub part: *const c_char,
    pub labels: *const *const c_char,
    pub labels_len: usize,
}
#[cfg(target_pointer_width = "32")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshField {
    pub struct_size: usize,
    pub role: *const c_char,
    pub dim: u32,
    pub encoding: *const i32,
    pub encoding_len: usize,
    pub encoding_other: *const *const c_char,
    pub has_range: bool,
    pub _reserved0: [u8; 7],
    pub range: [f64; 2],
    pub has_frame: bool,
    pub _reserved1: [u8; 3],
    pub frame: i32,
    pub frame_other: *const c_char,
    pub has_provenance: bool,
    pub _reserved2: [u8; 3],
    pub provenance: i32,
    pub provenance_other: *const c_char,
    pub part: *const c_char,
    pub labels: *const *const c_char,
    pub labels_len: usize,
    pub _reserved3: [u8; 4],
}
#[cfg(target_pointer_width = "64")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshSplit {
    pub struct_size: usize,
    pub fields: *const *const RLMeshField,
    pub fields_len: usize,
}
#[cfg(target_pointer_width = "32")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshSplit {
    pub struct_size: usize,
    pub fields: *const *const RLMeshField,
    pub fields_len: usize,
}
#[cfg(target_pointer_width = "64")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshPreviousOptions {
    pub struct_size: usize,
    pub part: *const c_char,
    pub fill: f64,
}
#[cfg(target_pointer_width = "32")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshPreviousOptions {
    pub struct_size: usize,
    pub part: *const c_char,
    pub fill: f64,
}
#[cfg(target_pointer_width = "64")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshObservationRoles {
    pub struct_size: usize,
    pub images: *const *const c_char,
    pub images_len: usize,
    pub states: *const *const c_char,
    pub states_len: usize,
    pub texts: *const *const c_char,
    pub texts_len: usize,
}
#[cfg(target_pointer_width = "32")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshObservationRoles {
    pub struct_size: usize,
    pub images: *const *const c_char,
    pub images_len: usize,
    pub states: *const *const c_char,
    pub states_len: usize,
    pub texts: *const *const c_char,
    pub texts_len: usize,
}

pub(crate) fn number_d_type(v: rlmesh::DType) -> i32 {
    crate::value::dtype::dtype_to_c(v)
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshRotation {
    pub struct_size: usize,
    pub encoding: i32,
    pub _reserved: u32,
    pub value: *const f64,
    pub value_len: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshAdvisory {
    pub struct_size: usize,
    pub severity: i32,
    pub _reserved: u32,
    pub message: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshTagOptions {
    pub struct_size: usize,
    pub validate: bool,
    pub _reserved0: u8,
    pub _reserved1: u16,
    pub _reserved2: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RLMeshResolveOptions {
    pub struct_size: usize,
    pub trust_entrypoints: bool,
    pub check_inverse: bool,
    pub _reserved0: u16,
    pub _reserved1: u32,
}
