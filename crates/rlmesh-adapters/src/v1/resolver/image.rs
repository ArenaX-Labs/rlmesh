//! Pair one model image input with an env camera and derive its plan.

use std::collections::BTreeMap;

use super::super::fmt::{quoted, quoted_keys};
use super::super::plans::ImagePlan;
use super::super::spec::{EnvImage, ImageInput};
use super::{Result, err};

pub(super) fn plan_image(
    model_input: &ImageInput,
    images_by_role: &BTreeMap<String, &EnvImage>,
) -> Result<ImagePlan> {
    let mut env_image = images_by_role.get(&model_input.role).copied();
    if env_image.is_none() && images_by_role.len() == 1 {
        env_image = images_by_role.values().next().copied();
    }
    let Some(env_image) = env_image else {
        return Err(err(format!(
            "model input {} wants an image with role {} but the env offers {}",
            quoted(&model_input.key),
            quoted(&model_input.role),
            quoted_keys(images_by_role)
        )));
    };
    if model_input.resample != "bilinear" && model_input.resample != "bilinear_aa" {
        return Err(err(format!(
            "model input {}: unsupported resample {}; expected 'bilinear' or \
             'bilinear_aa'",
            quoted(&model_input.key),
            quoted(&model_input.resample)
        )));
    }
    // When the model declares only one target axis, fill the other from the
    // env's native resolution (derived into the env image by `join`) rather
    // than silently skipping the resize.
    let size = match (model_input.height, model_input.width) {
        (Some(height), Some(width)) => Some((height, width)),
        (Some(height), None) => Some((height, env_image.width)),
        (None, Some(width)) => Some((env_image.height, width)),
        (None, None) => None,
    };
    Ok(ImagePlan {
        model_key: model_input.key.clone(),
        env_key: env_image.key.clone(),
        src_layout: env_image.layout,
        dst_layout: model_input.layout,
        flip: env_image.upside_down != model_input.upside_down,
        size,
        resample: model_input.resample.clone(),
        dtype: model_input.dtype.clone(),
        normalize: model_input.normalize,
        lead_dims: model_input.lead_dims,
    })
}
