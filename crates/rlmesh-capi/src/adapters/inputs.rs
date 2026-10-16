//! Typed C leaf inputs converted directly to facade values.
#![allow(unsafe_code)]
use super::*;
pub(crate) unsafe fn parse_rotation(p: *const RLMeshRotation) -> Result<a::Rotation, CapiError> {
    let v = unsafe { sized(p, "RLMeshRotation") }?;
    if v._reserved != 0 {
        return Err(CapiError::type_error(
            "RLMeshRotation reserved bytes must be zero",
        ));
    }
    let value = a::Rotation {
        encoding: enum_rotation_encoding(v.encoding)?,
        value: unsafe { slice_arg(v.value, v.value_len, "value") }?.to_vec(),
    };
    Ok(value)
}
pub(crate) unsafe fn parse_custom_encoding(
    p: *const RLMeshCustomEncoding,
) -> Result<a::CustomEncoding, CapiError> {
    let v = unsafe { sized(p, "RLMeshCustomEncoding") }?;
    #[cfg(target_pointer_width = "64")]
    if v._reserved0.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshCustomEncoding reserved bytes must be zero",
        ));
    }
    let value = a::CustomEncoding {
        base: enum_rotation_encoding(v.base)?,
        from_base: unsafe { encoding_arm(v.from_base) }?,
        to_base: unsafe { encoding_arm(v.to_base) }?,
        name: unsafe { required(v.name, "name") }?,
    };
    Ok(value)
}
pub(crate) unsafe fn parse_image(p: *const RLMeshImage) -> Result<a::Image, CapiError> {
    let v = unsafe { sized(p, "RLMeshImage") }?;
    #[cfg(target_pointer_width = "64")]
    if v._reserved0.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImage reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved1.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImage reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved2.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImage reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved3.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImage reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved4.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImage reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved5.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImage reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved6.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImage reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved7.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImage reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved8.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImage reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved9.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImage reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved10.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImage reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved11.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImage reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved12.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImage reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved13.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImage reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved14.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImage reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved15.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImage reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved0.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImage reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved1.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImage reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved2.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImage reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved3.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImage reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved4.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImage reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved5.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImage reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved6.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImage reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved7.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImage reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved8.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImage reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved9.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImage reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved10.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImage reserved bytes must be zero",
        ));
    }
    let value = a::Image {
        role: unsafe { required(v.role, "role") }?,
        height: v.has_height.then_some(v.height),
        width: v.has_width.then_some(v.width),
        layout: enum_image_layout(v.layout)?,
        channels: v.has_channels.then_some(v.channels),
        dtype: unsafe { dtype_value(v.dtype, v.dtype_other) }?,
        normalize: match v.normalize {
            1 => a::Normalize::Off,
            2 => a::Normalize::Unit,
            3 => a::Normalize::Range(v.normalize_range[0], v.normalize_range[1]),
            _ => return Err(CapiError::invalid_arg("invalid normalize")),
        },
        lead_dims: v.lead_dims,
        upside_down: v.upside_down,
        resample: unsafe { tolerant_resample(v.resample, v.resample_other) }?,
        allow_upscale: v.allow_upscale,
        fit: unsafe { accept(v.fit, v.fit_len, v.fit_other, tolerant_fit_mode) }?,
        optional: v.optional,
        fill: v.has_fill.then_some(v.fill),
        stack: v.stack,
        crop: v.has_crop.then_some(v.crop),
        crop_area: v.has_crop_area.then_some(v.crop_area),
        crop_mode: unsafe { tolerant_crop_mode(v.crop_mode, v.crop_mode_other) }?,
        jpeg_quality: v.has_jpeg_quality.then_some(v.jpeg_quality),
        channel_order: unsafe { tolerant_channel_order(v.channel_order, v.channel_order_other) }?,
        render: v.has_render.then_some((v.render[0], v.render[1])),
        offsets: unsafe { optional_slice(v.offsets, v.offsets_len, "offsets") }?,
        stack_pad: enum_stack_pad(v.stack_pad)?,
        part: unsafe { ffi::text(v.part) }?,
    };
    let value = if v.has_stride {
        value.with_stride(v.stride)?
    } else {
        value
    };
    Ok(value)
}
pub(crate) unsafe fn parse_state(p: *const RLMeshState) -> Result<a::State, CapiError> {
    let v = unsafe { sized(p, "RLMeshState") }?;
    #[cfg(target_pointer_width = "64")]
    if v._reserved0.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshState reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved1.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshState reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved2.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshState reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved3.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshState reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved4.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshState reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved5.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshState reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved6.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshState reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved7.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshState reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved8.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshState reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved9.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshState reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved0.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshState reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved1.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshState reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved2.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshState reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved3.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshState reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved4.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshState reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved5.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshState reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved6.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshState reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved7.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshState reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved8.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshState reserved bytes must be zero",
        ));
    }
    let value = a::State {
        role: unsafe { required(v.role, "role") }?,
        encoding: unsafe {
            state_encoding(
                v.encoding,
                v.encoding_len,
                v.encoding_other,
                v.custom_encoding,
            )
        }?,
        dim: v.has_dim.then_some(v.dim),
        index: v.has_index.then_some(v.index),
        optional: v.optional,
        range: v.has_range.then_some((v.range[0], v.range[1])),
        pad_to: v.has_pad_to.then_some(v.pad_to),
        dtype: unsafe { dtype_value(v.dtype, v.dtype_other) }?,
        reshape: unsafe { optional_slice(v.reshape, v.reshape_len, "reshape") }?,
        container: enum_state_container(v.container)?,
        fill: v.fill,
        post_rotate: if v.post_rotate.is_null() {
            None
        } else {
            Some(unsafe { parse_rotation(v.post_rotate) }?)
        },
        scale: unsafe {
            axis(
                v.has_scale,
                v.scale,
                v.axis_scale,
                v.axis_scale_len,
                "scale",
            )
        }?,
        offset: unsafe {
            axis(
                v.has_offset,
                v.offset,
                v.axis_offset,
                v.axis_offset_len,
                "offset",
            )
        }?,
        frame: if v.has_frame {
            Some(unsafe { tolerant_frame(v.frame, v.frame_other) }?)
        } else {
            None
        },
        provenance: unsafe {
            accept(
                v.provenance,
                v.provenance_len,
                v.provenance_other,
                tolerant_provenance,
            )
        }?,
        part: unsafe { ffi::text(v.part) }?,
        labels: unsafe { strings(v.labels, v.labels_len) }?,
        source: enum_state_source(v.source)?,
        clip: v.has_clip.then_some((v.clip[0], v.clip[1])),
    };
    Ok(value)
}
pub(crate) unsafe fn parse_concat(p: *const RLMeshConcat) -> Result<a::Concat, CapiError> {
    let v = unsafe { sized(p, "RLMeshConcat") }?;
    #[cfg(target_pointer_width = "64")]
    if v._reserved0.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshConcat reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved1.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshConcat reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved2.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshConcat reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved0.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshConcat reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved1.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshConcat reserved bytes must be zero",
        ));
    }
    let value = a::Concat {
        parts: unsafe { slice_arg(v.parts, v.parts_len, "parts") }?
            .iter()
            .map(|p| unsafe { parse_part(p) })
            .collect::<Result<_, _>>()?,
        pad_to: v.has_pad_to.then_some(v.pad_to),
        dtype: unsafe { dtype_value(v.dtype, v.dtype_other) }?,
        reshape: unsafe { optional_slice(v.reshape, v.reshape_len, "reshape") }?,
        container: enum_state_container(v.container)?,
        clip: v.has_clip.then_some((v.clip[0], v.clip[1])),
    };
    Ok(value)
}
pub(crate) unsafe fn parse_constant(p: *const RLMeshConstant) -> Result<a::Constant, CapiError> {
    let v = unsafe { sized(p, "RLMeshConstant") }?;
    #[cfg(target_pointer_width = "64")]
    if v._reserved0.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshConstant reserved bytes must be zero",
        ));
    }
    let value = a::Constant {
        dim: v.dim,
        fill: v.fill,
    };
    Ok(value)
}
pub(crate) unsafe fn parse_text(p: *const RLMeshAdaptersText) -> Result<a::Text, CapiError> {
    let v = unsafe { sized(p, "RLMeshAdaptersText") }?;
    #[cfg(target_pointer_width = "64")]
    if v._reserved0.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshAdaptersText reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved0.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshAdaptersText reserved bytes must be zero",
        ));
    }
    let value = a::Text {
        role: unsafe { required(v.role, "role") }?,
        container: enum_text_container(v.container)?,
        fill: if v.has_fill {
            Some(unsafe { v.fill.read("fill") }?.to_owned())
        } else {
            None
        },
    };
    Ok(value)
}
pub(crate) unsafe fn parse_custom(p: *const RLMeshCustom) -> Result<a::Custom, CapiError> {
    let v = unsafe { sized(p, "RLMeshCustom") }?;
    let value = a::Custom {
        transform: unsafe { v.transform.as_ref() }.map(|t| t.0.clone()),
        entrypoint: unsafe { ffi::text(v.entrypoint) }?,
    };
    Ok(value)
}
pub(crate) unsafe fn parse_actuator(p: *const RLMeshActuator) -> Result<a::Actuator, CapiError> {
    let v = unsafe { sized(p, "RLMeshActuator") }?;
    #[cfg(target_pointer_width = "64")]
    if v._reserved0.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshActuator reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved1.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshActuator reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved2.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshActuator reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved3.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshActuator reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved4.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshActuator reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved5.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshActuator reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved6.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshActuator reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved7.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshActuator reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved8.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshActuator reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved0.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshActuator reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved1.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshActuator reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved2.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshActuator reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved3.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshActuator reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved4.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshActuator reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved5.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshActuator reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved6.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshActuator reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved7.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshActuator reserved bytes must be zero",
        ));
    }
    let value = a::Actuator {
        role: unsafe { ffi::text(v.role) }?,
        dim: v.dim,
        encoding: unsafe { action_encoding(v.has_encoding, v.encoding, v.custom_encoding) }?,
        range: v.has_range.then_some((v.range[0], v.range[1])),
        binary: v.binary,
        scale: unsafe {
            axis(
                v.has_scale,
                v.scale,
                v.axis_scale,
                v.axis_scale_len,
                "scale",
            )
        }?,
        invert: v.invert,
        threshold: v.has_threshold.then_some(v.threshold),
        clip: v.clip,
        fill: v.fill,
        optional: v.optional,
        frame: if v.has_frame {
            Some(unsafe { tolerant_frame(v.frame, v.frame_other) }?)
        } else {
            None
        },
        reference: if v.has_reference {
            Some(unsafe { tolerant_reference(v.reference, v.reference_other) }?)
        } else {
            None
        },
        part: unsafe { ffi::text(v.part) }?,
        offset: unsafe {
            axis(
                v.has_offset,
                v.offset,
                v.axis_offset,
                v.axis_offset_len,
                "offset",
            )
        }?,
        labels: unsafe { strings(v.labels, v.labels_len) }?,
    };
    Ok(value)
}
pub(crate) unsafe fn parse_action(p: *const RLMeshAction) -> Result<a::Action, CapiError> {
    let v = unsafe { sized(p, "RLMeshAction") }?;
    #[cfg(target_pointer_width = "64")]
    if v._reserved0.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshAction reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved0.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshAction reserved bytes must be zero",
        ));
    }
    let value = a::Action {
        components: unsafe { slice_arg(v.components, v.components_len, "components") }?
            .iter()
            .map(|&p| unsafe { parse_actuator(p) })
            .collect::<Result<_, _>>()?,
        clip: v.has_clip.then_some((v.clip[0], v.clip[1])),
    };
    Ok(value)
}
pub(crate) unsafe fn parse_image_tag(p: *const RLMeshImageTag) -> Result<a::ImageTag, CapiError> {
    let v = unsafe { sized(p, "RLMeshImageTag") }?;
    #[cfg(target_pointer_width = "64")]
    if v._reserved0.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImageTag reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved0.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshImageTag reserved bytes must be zero",
        ));
    }
    let value = a::ImageTag {
        role: unsafe { required(v.role, "role") }?,
        layout: enum_image_layout(v.layout)?,
        upside_down: v.upside_down,
        part: unsafe { ffi::text(v.part) }?,
    };
    Ok(value)
}
pub(crate) unsafe fn parse_state_tag(p: *const RLMeshStateTag) -> Result<a::StateTag, CapiError> {
    let v = unsafe { sized(p, "RLMeshStateTag") }?;
    #[cfg(target_pointer_width = "64")]
    if v._reserved0.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshStateTag reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved1.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshStateTag reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved2.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshStateTag reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved0.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshStateTag reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved1.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshStateTag reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved2.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshStateTag reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved3.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshStateTag reserved bytes must be zero",
        ));
    }
    let value = a::StateTag {
        role: unsafe { required(v.role, "role") }?,
        encoding: unsafe {
            accept(
                v.encoding,
                v.encoding_len,
                v.encoding_other,
                tolerant_rotation_encoding,
            )
        }?,
        range: v.has_range.then_some((v.range[0], v.range[1])),
        frame: if v.has_frame {
            Some(unsafe { tolerant_frame(v.frame, v.frame_other) }?)
        } else {
            None
        },
        provenance: if v.has_provenance {
            Some(unsafe { tolerant_provenance(v.provenance, v.provenance_other) }?)
        } else {
            None
        },
        part: unsafe { ffi::text(v.part) }?,
        labels: unsafe { strings(v.labels, v.labels_len) }?,
    };
    Ok(value)
}
pub(crate) unsafe fn parse_text_tag(p: *const RLMeshTextTag) -> Result<a::TextTag, CapiError> {
    let v = unsafe { sized(p, "RLMeshTextTag") }?;
    let value = a::TextTag {
        role: unsafe { required(v.role, "role") }?,
    };
    Ok(value)
}
pub(crate) unsafe fn parse_field(p: *const RLMeshField) -> Result<a::Field, CapiError> {
    let v = unsafe { sized(p, "RLMeshField") }?;
    #[cfg(target_pointer_width = "64")]
    if v._reserved0.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshField reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved1.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshField reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved2.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshField reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "64")]
    if v._reserved3.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshField reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved0.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshField reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved1.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshField reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved2.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshField reserved bytes must be zero",
        ));
    }
    #[cfg(target_pointer_width = "32")]
    if v._reserved3.iter().any(|&b| b != 0) {
        return Err(CapiError::type_error(
            "RLMeshField reserved bytes must be zero",
        ));
    }
    let value = a::Field {
        role: unsafe { ffi::text(v.role) }?,
        dim: v.dim,
        encoding: unsafe {
            accept(
                v.encoding,
                v.encoding_len,
                v.encoding_other,
                tolerant_rotation_encoding,
            )
        }?,
        range: v.has_range.then_some((v.range[0], v.range[1])),
        frame: if v.has_frame {
            Some(unsafe { tolerant_frame(v.frame, v.frame_other) }?)
        } else {
            None
        },
        provenance: if v.has_provenance {
            Some(unsafe { tolerant_provenance(v.provenance, v.provenance_other) }?)
        } else {
            None
        },
        part: unsafe { ffi::text(v.part) }?,
        labels: unsafe { strings(v.labels, v.labels_len) }?,
    };
    Ok(value)
}
pub(crate) unsafe fn parse_split(p: *const RLMeshSplit) -> Result<a::Split, CapiError> {
    let v = unsafe { sized(p, "RLMeshSplit") }?;
    let value = a::Split {
        fields: unsafe { slice_arg(v.fields, v.fields_len, "fields") }?
            .iter()
            .map(|&p| unsafe { parse_field(p) })
            .collect::<Result<_, _>>()?,
    };
    Ok(value)
}
pub(crate) unsafe fn parse_previous_options(
    p: *const RLMeshPreviousOptions,
) -> Result<a::PreviousOptions, CapiError> {
    let v = unsafe { sized(p, "RLMeshPreviousOptions") }?;
    let value = a::PreviousOptions {
        part: unsafe { ffi::text(v.part) }?,
        fill: v.fill,
    };
    Ok(value)
}
pub(crate) unsafe fn parse_observation_roles(
    p: *const RLMeshObservationRoles,
) -> Result<a::ObservationRoles, CapiError> {
    let v = unsafe { sized(p, "RLMeshObservationRoles") }?;
    let value = a::ObservationRoles {
        images: unsafe { strings(v.images, v.images_len) }?.unwrap_or_default(),
        states: unsafe { strings(v.states, v.states_len) }?.unwrap_or_default(),
        texts: unsafe { strings(v.texts, v.texts_len) }?.unwrap_or_default(),
    };
    Ok(value)
}
pub(crate) unsafe fn parse_advisory(p: *const RLMeshAdvisory) -> Result<a::Advisory, CapiError> {
    let v = unsafe { sized(p, "RLMeshAdvisory") }?;
    if v._reserved != 0 {
        return Err(CapiError::type_error(
            "RLMeshAdvisory reserved bytes must be zero",
        ));
    }
    let value = a::Advisory {
        severity: enum_advisory_severity(v.severity)?,
        message: unsafe { required(v.message, "message") }?,
    };
    Ok(value)
}
pub(crate) unsafe fn parse_tag_options(
    p: *const RLMeshTagOptions,
) -> Result<a::TagOptions, CapiError> {
    let v = unsafe { sized(p, "RLMeshTagOptions") }?;
    if v._reserved0 != 0 {
        return Err(CapiError::type_error(
            "RLMeshTagOptions reserved bytes must be zero",
        ));
    }
    if v._reserved1 != 0 {
        return Err(CapiError::type_error(
            "RLMeshTagOptions reserved bytes must be zero",
        ));
    }
    if v._reserved2 != 0 {
        return Err(CapiError::type_error(
            "RLMeshTagOptions reserved bytes must be zero",
        ));
    }
    let value = a::TagOptions {
        validate: v.validate,
    };
    Ok(value)
}
pub(crate) unsafe fn parse_resolve_options(
    p: *const RLMeshResolveOptions,
) -> Result<a::ResolveOptions, CapiError> {
    let v = unsafe { sized(p, "RLMeshResolveOptions") }?;
    if v._reserved0 != 0 {
        return Err(CapiError::type_error(
            "RLMeshResolveOptions reserved bytes must be zero",
        ));
    }
    if v._reserved1 != 0 {
        return Err(CapiError::type_error(
            "RLMeshResolveOptions reserved bytes must be zero",
        ));
    }
    let value = a::ResolveOptions {
        trust_entrypoints: v.trust_entrypoints,
        check_inverse: v.check_inverse,
    };
    Ok(value)
}
pub(crate) unsafe fn dtype_value(
    v: i32,
    other: *const c_char,
) -> Result<a::Tolerant<rlmesh::DType>, CapiError> {
    if v == 1000 {
        Ok(a::Tolerant::parse(&unsafe {
            required(other, "DType_other")
        }?))
    } else {
        Ok((crate::value::dtype::dtype_from_c(v)
            .ok_or_else(|| CapiError::invalid_arg("invalid dtype"))?)
        .into())
    }
}
pub(crate) unsafe fn tolerant_fit_mode(
    v: i32,
    other: *const c_char,
) -> Result<a::Tolerant<a::FitMode>, CapiError> {
    if v == 1000 {
        Ok(a::Tolerant::parse(&unsafe {
            required(other, "FitMode_other")
        }?))
    } else {
        Ok((enum_fit_mode(v)?).into())
    }
}
pub(crate) unsafe fn tolerant_frame(
    v: i32,
    other: *const c_char,
) -> Result<a::Tolerant<a::Frame>, CapiError> {
    if v == 1000 {
        Ok(a::Tolerant::parse(&unsafe {
            required(other, "Frame_other")
        }?))
    } else {
        Ok((enum_frame(v)?).into())
    }
}
pub(crate) unsafe fn tolerant_reference(
    v: i32,
    other: *const c_char,
) -> Result<a::Tolerant<a::Reference>, CapiError> {
    if v == 1000 {
        Ok(a::Tolerant::parse(&unsafe {
            required(other, "Reference_other")
        }?))
    } else {
        Ok((enum_reference(v)?).into())
    }
}
pub(crate) unsafe fn tolerant_provenance(
    v: i32,
    other: *const c_char,
) -> Result<a::Tolerant<a::Provenance>, CapiError> {
    if v == 1000 {
        Ok(a::Tolerant::parse(&unsafe {
            required(other, "Provenance_other")
        }?))
    } else {
        Ok((enum_provenance(v)?).into())
    }
}
pub(crate) unsafe fn tolerant_resample(
    v: i32,
    other: *const c_char,
) -> Result<a::Tolerant<a::Resample>, CapiError> {
    if v == 1000 {
        Ok(a::Tolerant::parse(&unsafe {
            required(other, "Resample_other")
        }?))
    } else {
        Ok((enum_resample(v)?).into())
    }
}
pub(crate) unsafe fn tolerant_rotation_encoding(
    v: i32,
    other: *const c_char,
) -> Result<a::Tolerant<a::RotationEncoding>, CapiError> {
    if v == 1000 {
        Ok(a::Tolerant::parse(&unsafe {
            required(other, "RotationEncoding_other")
        }?))
    } else {
        Ok((enum_rotation_encoding(v)?).into())
    }
}
pub(crate) unsafe fn tolerant_crop_mode(
    v: i32,
    other: *const c_char,
) -> Result<a::Tolerant<a::CropMode>, CapiError> {
    if v == 1000 {
        Ok(a::Tolerant::parse(&unsafe {
            required(other, "CropMode_other")
        }?))
    } else {
        Ok((enum_crop_mode(v)?).into())
    }
}
pub(crate) unsafe fn tolerant_channel_order(
    v: i32,
    other: *const c_char,
) -> Result<a::Tolerant<a::ChannelOrder>, CapiError> {
    if v == 1000 {
        Ok(a::Tolerant::parse(&unsafe {
            required(other, "ChannelOrder_other")
        }?))
    } else {
        Ok((enum_channel_order(v)?).into())
    }
}

pub(crate) unsafe fn parse_part(p: *const RLMeshConcatPart) -> Result<a::ConcatPart, CapiError> {
    let p = unsafe { p.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null ConcatPart"))?;
    Ok(match p.kind {
        1 => a::ConcatPart::Role(unsafe { required(p.role, "role") }?),
        2 => a::ConcatPart::State(unsafe { parse_state(p.leaf.cast()) }?),
        3 => a::ConcatPart::Constant(unsafe { parse_constant(p.leaf.cast()) }?),
        _ => return Err(CapiError::invalid_arg("invalid ConcatPart kind")),
    })
}
pub(crate) unsafe fn parse_input(
    p: *const RLMeshInputNode,
    active: &mut Vec<usize>,
) -> Result<a::InputNode, CapiError> {
    if active.contains(&(p as usize)) {
        return Err(CapiError::invalid_arg("cyclic InputNode tree"));
    }
    active.push(p as usize);
    let p = unsafe { p.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null InputNode"))?;
    let node = match p.kind {
        1 => unsafe { parse_image(p.leaf.cast()) }?.into(),
        2 => unsafe { parse_state(p.leaf.cast()) }?.into(),
        3 => unsafe { parse_concat(p.leaf.cast()) }?.into(),
        4 => unsafe { parse_text(p.leaf.cast()) }?.into(),
        5 => unsafe { parse_custom(p.leaf.cast()) }?.into(),
        6 | 7 => {
            let nodes = unsafe { slice_arg(p.children, p.children_len, "children") }?;
            let children = nodes
                .iter()
                .map(|n| unsafe { parse_input(n, active) })
                .collect::<Result<Vec<_>, _>>()?;
            if p.kind == 7 {
                a::InputNode::tuple(children)
            } else {
                let keys = unsafe { slice_arg(p.keys, p.children_len, "keys") }?;
                let keys = keys
                    .iter()
                    .map(|k| unsafe { k.read("key") }.map(str::to_owned))
                    .collect::<Result<Vec<_>, _>>()?;
                a::InputNode::dict(keys.into_iter().zip(children))
            }
        }
        _ => return Err(CapiError::invalid_arg("invalid InputNode kind")),
    };
    active.pop();
    Ok(node)
}
pub(crate) unsafe fn parse_obs(
    p: *const RLMeshObsNode,
    active: &mut Vec<usize>,
) -> Result<a::ObsNode, CapiError> {
    if active.contains(&(p as usize)) {
        return Err(CapiError::invalid_arg("cyclic ObsNode tree"));
    }
    active.push(p as usize);
    let p = unsafe { p.as_ref() }.ok_or_else(|| CapiError::invalid_arg("null ObsNode"))?;
    let node = match p.kind {
        1 => unsafe { parse_image_tag(p.leaf.cast()) }?.into(),
        2 => unsafe { parse_state_tag(p.leaf.cast()) }?.into(),
        3 => unsafe { parse_text_tag(p.leaf.cast()) }?.into(),
        4 => unsafe { parse_split(p.leaf.cast()) }?.into(),
        5 | 6 => {
            let nodes = unsafe { slice_arg(p.children, p.children_len, "children") }?;
            let children = nodes
                .iter()
                .map(|n| unsafe { parse_obs(n, active) })
                .collect::<Result<Vec<_>, _>>()?;
            if p.kind == 6 {
                a::ObsNode::tuple(children)
            } else {
                let keys = unsafe { slice_arg(p.keys, p.children_len, "keys") }?;
                let keys = keys
                    .iter()
                    .map(|k| unsafe { k.read("key") }.map(str::to_owned))
                    .collect::<Result<Vec<_>, _>>()?;
                a::ObsNode::dict(keys.into_iter().zip(children))
            }
        }
        _ => return Err(CapiError::invalid_arg("invalid ObsNode kind")),
    };
    active.pop();
    Ok(node)
}
