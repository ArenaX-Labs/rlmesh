#[allow(unused_imports)]
use crate::prelude::*;

use alloc::collections::BTreeMap;

use crate::spec::image::crop_cut;
use crate::spec::plan::Fit;
use crate::spec::policy::ImageLayout;
use crate::spec::projection::Projection;
use crate::spec::role::bind::{LeafKey, Projections, Result, err, seek};
use crate::spec::role::error::{Advisory, ErrorCode};
use crate::spec::role::fmt::{quoted, quoted_leaf_keys};
use crate::spec::role::input::{Crop, EnvImage, Image, StackPad};
use crate::spec::role::path::NodePath;
use crate::spec::role::plan::{CropPlan, ImagePlan};
use crate::spec::role::roles::registry;
use crate::spec::role::state::part_suffix;
use crate::wire::part::DType;

const MAX_STACK_SPAN: i32 = 128;

pub(super) fn plan_image(
    model_input: &Image,
    placement: NodePath,
    images_by_role: &BTreeMap<LeafKey, &EnvImage>,
    unknown_roles: &BTreeMap<String, String>,
    advisories: &mut Vec<Advisory>,
    projections: &mut Projections,
) -> Result<ImagePlan> {
    let at = quoted(&placement.to_string());
    if model_input.fill.is_some() && !model_input.optional {
        return Err(err(
            ErrorCode::Unsupported,
            format!(
                "model input {at}: fill only applies to an optional camera; \
                 set optional or drop fill"
            ),
        ));
    }
    let bound = seek(
        images_by_role,
        &model_input.role,
        model_input.part.as_deref(),
        None,
        &format!("model input {at}"),
        "image role",
        "env",
        advisories,
    )?;
    let (mut env_image, mut part) = match bound {
        Some(bound) => (Some(*bound.feature), bound.part),
        None => (None, model_input.part.clone()),
    };
    if env_image.is_none() {
        crate::spec::role::bind::reject_referenced_unknown(
            &model_input.role,
            &placement,
            unknown_roles,
        )?;
    }
    let mut role_rebound = None;
    if env_image.is_none()
        && !model_input.optional
        && images_by_role.len() == 1
        && model_input.part.is_none()
        && registry::is_sanctioned_role(&model_input.role)
    {
        let ((_, only_part, _), only) = images_by_role.iter().next().expect("one camera");
        projections.apply(
            Projection::RebindCamera,
            format!(
                "model input {at} (role {}) reads the env's only camera (role {})",
                quoted(&model_input.role),
                quoted(&only.role)
            ),
            ErrorCode::MissingRole,
            format!(
                "model input {at} needs camera role {} but the env's only camera has role {}",
                quoted(&model_input.role),
                quoted(&only.role)
            ),
        )?;
        env_image = Some(*only);
        part = only_part.clone();
        role_rebound = Some((model_input.role.clone(), only.role.clone()));
    }
    let Some(env_image) = env_image else {
        if model_input.optional {
            return zero_fill_image_plan(model_input, placement);
        }
        return Err(err(
            ErrorCode::MissingRole,
            format!(
                "model input {at} wants an image with role {}{} but the env offers {}",
                quoted(&model_input.role),
                part_suffix(model_input.part.as_deref()),
                quoted_leaf_keys(images_by_role)
            ),
        ));
    };
    if let Some(expected) = model_input.channels
        && env_image.channels != 0
        && env_image.channels != expected
    {
        return Err(err(
            ErrorCode::Unsupported,
            format!(
                "model input {at}: expects {expected} channel(s) but the env image has {}; the \
             adapter does not convert between channel counts (e.g. RGB vs grayscale)",
                env_image.channels
            ),
        ));
    }
    let render = model_input.render.filter(|_| known_size(env_image));
    if let Some((height, width)) = render
        && (env_image.height, env_image.width) != (height, width)
    {
        return Err(err(
            ErrorCode::RenderMismatch,
            format!(
                "model input {at}: declares render {height}x{width} but the bound camera {} \
                 renders {}x{}",
                quoted(&env_image.role),
                env_image.height,
                env_image.width
            ),
        ));
    }
    let size = match (model_input.height, model_input.width) {
        (Some(height), Some(width)) => Some((height, width)),
        (Some(height), None) => Some((height, env_image.width)),
        (None, Some(width)) => Some((env_image.height, width)),
        (None, None) => None,
    };
    let swap_rb = model_input.bgr;
    if let Some(channels) = model_input
        .channels
        .or(Some(env_image.channels))
        .filter(|&c| c != 0)
        && channels != 3
    {
        for (declared, what) in [
            (swap_rb, "channel_order \"bgr\""),
            (model_input.jpeg_quality.is_some(), "jpeg_quality"),
        ] {
            if declared {
                return Err(err(
                    ErrorCode::Unsupported,
                    format!(
                        "model input {at}: {what} needs a 3-channel image, \
                         got {channels} channel(s)"
                    ),
                ));
            }
        }
    }
    let crop = crop_plan(model_input, env_image);
    let fit = resolve_fit(model_input, &at, env_image, size, crop.as_ref())?;
    let offsets = stack_window(model_input, &at)?;
    let processed = match size {
        Some(size) => Some(size),
        None => match crop.as_ref().and_then(|crop| crop.cut) {
            Some(cut) => Some(cut),
            None if known_size(env_image) => Some((env_image.height, env_image.width)),
            None => None,
        },
    };
    let channels = model_input
        .channels
        .or(Some(env_image.channels))
        .filter(|&channels| channels != 0);
    Ok(ImagePlan {
        placement,
        source: env_image.source.clone(),
        src_layout: env_image.layout,
        dst_layout: model_input.layout,
        flip: env_image.upside_down != model_input.upside_down,
        size,
        fit,
        resample: model_input.resample,
        dtype: model_input.dtype,
        normalize: model_input.normalize,
        lead_dims: model_input.lead_dims,
        src_range: env_image.value_range,
        stack: model_input.stack,
        zero_fill: None,
        fill: model_input.fill.unwrap_or(0),
        crop,
        jpeg_quality: model_input.jpeg_quality,
        swap_rb,
        render,
        offsets,
        stack_pad: model_input.stack_pad,
        frame_bytes: frame_bytes(model_input.dtype, processed, channels),
        role_rebound,
        part,
    })
}

fn known_size(env_image: &EnvImage) -> bool {
    env_image.height != 0 && env_image.width != 0
}

fn crop_plan(model_input: &Image, env_image: &EnvImage) -> Option<CropPlan> {
    let Crop {
        fraction,
        area,
        slice,
    } = model_input.crop?;
    let cut = (slice && env_image.height != 0 && env_image.width != 0).then(|| {
        (
            crop_cut(env_image.height as usize, fraction) as u32,
            crop_cut(env_image.width as usize, fraction) as u32,
        )
    });
    Some(CropPlan {
        fraction,
        area,
        slice,
        cut,
    })
}

fn zero_fill_image_plan(model_input: &Image, placement: NodePath) -> Result<ImagePlan> {
    let offsets = stack_window(model_input, &quoted(&placement.to_string()))?;
    let (Some(height), Some(width), Some(channels)) =
        (model_input.height, model_input.width, model_input.channels)
    else {
        return Err(err(
            ErrorCode::MissingWidth,
            format!(
                "model input {}: an optional image the env does not provide needs height, width, \
             and channels to size the zero-filled frame",
                quoted(&placement.to_string())
            ),
        ));
    };
    Ok(ImagePlan {
        placement,
        source: NodePath::root(),
        src_layout: ImageLayout::Hwc,
        dst_layout: model_input.layout,
        flip: false,
        size: None,
        fit: Fit::Stretch,
        resample: model_input.resample,
        dtype: model_input.dtype,
        normalize: model_input.normalize,
        lead_dims: model_input.lead_dims,
        src_range: None,
        stack: model_input.stack,
        zero_fill: Some((height, width, channels)),
        fill: model_input.fill.unwrap_or(0),
        crop: None,
        jpeg_quality: None,
        swap_rb: false,
        render: None,
        offsets,
        stack_pad: model_input.stack_pad,
        frame_bytes: frame_bytes(model_input.dtype, Some((height, width)), Some(channels)),
        role_rebound: None,
        part: model_input.part.clone(),
    })
}

fn frame_bytes(dtype: DType, size: Option<(u32, u32)>, channels: Option<u32>) -> u64 {
    let (Some((height, width)), Some(channels)) = (size, channels) else {
        return 0;
    };
    u64::from(height) * u64::from(width) * u64::from(channels) * dtype.size()
}

fn stack_window(model_input: &Image, at: &str) -> Result<Option<Vec<i32>>> {
    if model_input.stack_pad != StackPad::First && model_input.stack <= 1 {
        return Err(err(
            ErrorCode::Unsupported,
            format!(
                "model input {at}: stack_pad only applies to a stacked image; set stack > 1 or \
                 drop stack_pad"
            ),
        ));
    }
    let Some(offsets) = &model_input.offsets else {
        return Ok(None);
    };
    let ordered = offsets.windows(2).all(|pair| pair[0] < pair[1]);
    if offsets.is_empty() || !ordered || offsets.last() != Some(&0) || offsets[0] > 0 {
        return Err(err(
            ErrorCode::Unsupported,
            format!(
                "model input {at}: offsets must be non-positive and strictly increasing, ending \
                 at 0 (the current frame), got {offsets:?}"
            ),
        ));
    }
    if offsets.len() != model_input.stack as usize {
        return Err(err(
            ErrorCode::Unsupported,
            format!(
                "model input {at}: stack={} disagrees with {} offset(s) {offsets:?}; a frame \
                 window declares both and they are never inferred from each other",
                model_input.stack,
                offsets.len()
            ),
        ));
    }
    let span = 1 - offsets[0];
    if span > MAX_STACK_SPAN {
        return Err(err(
            ErrorCode::Unsupported,
            format!(
                "model input {at}: stack={} over offsets {offsets:?} spans {span} consecutive \
                 frames, more than the {MAX_STACK_SPAN} a window may hold",
                model_input.stack
            ),
        ));
    }
    Ok(Some(offsets.clone()))
}

fn resolve_fit(
    model_input: &Image,
    at: &str,
    env_image: &EnvImage,
    size: Option<(u32, u32)>,
    crop: Option<&CropPlan>,
) -> Result<Fit> {
    let Some((target_height, target_width)) = size else {
        return Ok(Fit::Stretch);
    };
    let (env_height, env_width) = match crop {
        Some(crop) => (
            crop_cut(env_image.height as usize, crop.fraction) as u32,
            crop_cut(env_image.width as usize, crop.fraction) as u32,
        ),
        None => (env_image.height, env_image.width),
    };
    let known = env_height != 0 && env_width != 0;

    let upscales = |mode: Fit| -> bool {
        if !known {
            return false;
        }
        match mode {
            Fit::Stretch | Fit::Crop => target_height > env_height || target_width > env_width,
            Fit::Pad => target_height > env_height && target_width > env_width,
        }
    };

    let aspect_differs = known
        && u64::from(env_height) * u64::from(target_width)
            != u64::from(env_width) * u64::from(target_height);

    let permitted: Vec<Fit> = model_input
        .fit
        .as_ref()
        .map(|set| set.known.clone())
        .unwrap_or_default();

    if aspect_differs {
        let first_usable = permitted
            .iter()
            .copied()
            .find(|&mode| model_input.allow_upscale || !upscales(mode));
        if let Some(mode) = first_usable {
            return Ok(mode);
        }
        let message = match &model_input.fit {
            None => format!(
                "model input {at}: target {target_height}x{target_width} changes the env's \
             {env_height}x{env_width} aspect ratio; set fit to 'stretch', 'crop', or 'pad'"
            ),
            Some(set) if permitted.is_empty() => format!(
                "model input {at}: target {target_height}x{target_width} changes the env's \
             {env_height}x{env_width} aspect ratio and the declared fit {:?} names no mode this \
             build recognizes; expected 'stretch', 'crop', or 'pad'",
                set.names
            ),
            Some(_) => format!(
                "model input {at}: every declared fit would upscale {} to \
                 {target_height}x{target_width}; declare a fit that downscales (e.g. 'pad'), or {}",
                upscale_source(env_image, crop, env_height, env_width),
                upscale_remedy(crop),
            ),
        };
        return Err(err(ErrorCode::Unsupported, message));
    }

    if let Some(set) = &model_input.fit
        && permitted.is_empty()
    {
        return Err(err(
            ErrorCode::Unsupported,
            format!(
                "model input {at}: the declared fit {:?} names no mode this build \
             recognizes; expected 'stretch', 'crop', or 'pad'",
                set.names
            ),
        ));
    }
    let mode = permitted.first().copied().unwrap_or(Fit::Stretch);
    if !model_input.allow_upscale && upscales(mode) {
        return Err(err(
            ErrorCode::Unsupported,
            format!(
                "model input {at}: target {target_height}x{target_width} upscales {}; {}",
                upscale_source(env_image, crop, env_height, env_width),
                upscale_remedy(crop),
            ),
        ));
    }
    Ok(if known { Fit::Stretch } else { mode })
}

fn upscale_source(
    env_image: &EnvImage,
    crop: Option<&CropPlan>,
    source_height: u32,
    source_width: u32,
) -> String {
    let camera = format!("the env's {}x{} camera", env_image.height, env_image.width);
    match crop {
        None => camera,
        Some(crop) => {
            let declared = match crop.area {
                Some(area) => format!("crop_area={area}"),
                None => format!("crop={}", crop.fraction),
            };
            format!(
                "the {source_height}x{source_width} crop box the model's {declared} cuts from \
                 {camera}"
            )
        }
    }
}

fn upscale_remedy(crop: Option<&CropPlan>) -> String {
    let crop_less = if crop.is_some() { ", or crop less" } else { "" };
    format!(
        "set allow_upscale on the model's Image input (`adapt.Image(..., allow_upscale=True)`, \
         JSON `\"allow_upscale\": true`) to interpolate detail that is not there, or have the \
         env publish a larger camera{crop_less}"
    )
}

#[cfg(test)]
mod image_resolve_tests {
    #[allow(unused_imports)]
    use crate::prelude::*;

    use alloc::collections::BTreeMap;

    use super::plan_image;
    use crate::spec::plan::Fit;
    use crate::spec::policy::{ImageLayout, Resample};
    use crate::spec::projection::Projection;
    use crate::spec::role::bind::{LeafKey, Projections};
    use crate::spec::role::input::{Accept, Crop, EnvImage, Image, StackPad};
    use crate::spec::role::path::NodePath;
    use crate::spec::role::{Error, ErrorCode};
    use crate::wire::part::DType;

    fn plan(
        model: &Image,
        images: &BTreeMap<LeafKey, &EnvImage>,
    ) -> Result<crate::spec::role::plan::ImagePlan, Error> {
        plan_image(
            model,
            NodePath::root(),
            images,
            &BTreeMap::new(),
            &mut Vec::new(),
            &mut Projections::allowing(&[Projection::RebindCamera]),
        )
    }

    fn fits(names: &[&str]) -> Accept<Fit> {
        let known = |name: &str| match name {
            "stretch" => Some(Fit::Stretch),
            "crop" => Some(Fit::Crop),
            "pad" => Some(Fit::Pad),
            _ => None,
        };
        Accept {
            names: names.iter().map(|name| (*name).to_owned()).collect(),
            known: names.iter().filter_map(|name| known(name)).collect(),
        }
    }

    fn env_image(height: u32, width: u32) -> EnvImage {
        EnvImage {
            source: NodePath::root().push_key("cam"),
            role: "image/primary".to_owned(),
            part: None,
            layout: ImageLayout::Hwc,
            upside_down: false,
            height,
            width,
            channels: 3,
            value_range: None,
        }
    }

    fn model_image(height: u32, width: u32, allow_upscale: bool) -> Image {
        Image {
            offsets: None,
            stack_pad: StackPad::First,
            role: "image/primary".to_owned(),
            part: None,
            height: Some(height),
            width: Some(width),
            layout: ImageLayout::Hwc,
            channels: None,
            dtype: DType::U8,
            normalize: None,
            lead_dims: 0,
            upside_down: false,
            resample: Resample::Bilinear,
            allow_upscale,
            fit: None,
            optional: false,
            fill: None,
            stack: 1,
            crop: None,
            jpeg_quality: None,
            bgr: false,
            render: None,
        }
    }

    fn images(env: &EnvImage) -> BTreeMap<LeafKey, &EnvImage> {
        BTreeMap::from([((env.role.clone(), env.part.clone(), None), env)])
    }

    #[test]
    fn upscale_without_opt_in_is_a_resolve_error() {
        let env = env_image(128, 128);
        let error = plan(&model_image(256, 256, false), &images(&env)).expect_err("err");
        assert_eq!(error.code, ErrorCode::Unsupported);
        let message = &error.message;
        assert!(message.contains("the env's 128x128 camera"), "{message}");
        assert!(message.contains("allow_upscale=True"), "{message}");
        assert!(message.contains(r#""allow_upscale": true"#), "{message}");
        assert!(message.contains("larger camera"), "{message}");
        assert!(!message.contains("crop"), "{message}");
        assert!(!message.contains("  "), "{message:?}");
    }

    #[test]
    fn frame_window_errors_read_as_single_spaced_sentences() {
        let env = env_image(8, 8);
        let mut inert_pad = model_image(8, 8, false);
        inert_pad.stack_pad = StackPad::Black;
        let mut unordered = model_image(8, 8, false);
        unordered.stack = 2;
        unordered.offsets = Some(vec![0, -1]);
        let mut disagreeing = model_image(8, 8, false);
        disagreeing.stack = 3;
        disagreeing.offsets = Some(vec![-1, 0]);
        let mut too_wide = model_image(8, 8, false);
        too_wide.stack = 2;
        too_wide.offsets = Some(vec![-200, 0]);
        for (model, expect) in [
            (inert_pad, "set stack > 1 or drop stack_pad"),
            (unordered, "strictly increasing, ending at 0"),
            (disagreeing, "a frame window declares both"),
            (too_wide, "consecutive frames, more than"),
        ] {
            let error = plan(&model, &images(&env)).expect_err("err");
            assert!(error.message.contains(expect), "{}", error.message);
            assert!(!error.message.contains("  "), "{:?}", error.message);
        }
    }

    #[test]
    fn a_frame_window_budgets_its_processed_frame_and_takes_its_limits() {
        let env = env_image(8, 8);
        let mut model = model_image(4, 4, false);
        model.dtype = DType::F32;
        assert_eq!(
            plan(&model, &images(&env)).expect("ok").frame_bytes,
            4 * 4 * 3 * 4
        );
        (model.height, model.width) = (None, None);
        assert_eq!(
            plan(&model, &images(&env)).expect("ok").frame_bytes,
            8 * 8 * 3 * 4
        );
        let unknown = env_image(0, 0);
        assert_eq!(plan(&model, &images(&unknown)).expect("ok").frame_bytes, 0);
        let window = |offsets: &[i32]| {
            let mut model = model_image(8, 8, false);
            model.stack = u32::try_from(offsets.len()).expect("small");
            model.offsets = Some(offsets.to_vec());
            plan(&model, &images(&env))
        };
        for ok in [&[0][..], &[-127, 0], &[-2, -1, 0]] {
            window(ok).unwrap_or_else(|e| panic!("{ok:?}: {}", e.message));
        }
        for bad in [&[-1, -1, 0][..], &[-2, -1], &[-128, 0]] {
            assert!(window(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn a_crop_at_the_cameras_own_height_or_width_is_not_an_upscale() {
        let env = env_image(128, 128);
        for (height, width) in [(128, 64), (64, 128)] {
            let mut model = model_image(height, width, false);
            model.fit = Some(fits(&["crop"]));
            plan(&model, &images(&env))
                .unwrap_or_else(|e| panic!("{height}x{width}: {}", e.message));
        }
        for (height, width) in [(128, 256), (64, 256), (256, 128)] {
            let mut model = model_image(height, width, false);
            model.fit = Some(fits(&["pad"]));
            plan(&model, &images(&env))
                .unwrap_or_else(|e| panic!("pad {height}x{width}: {}", e.message));
        }
        let half = env_image(0, 128);
        plan(&model_image(256, 256, false), &images(&half)).expect("nothing to compare");
        let mut model = model_image(224, 224, false);
        model.render = Some((448, 448));
        assert_eq!(plan(&model, &images(&half)).expect("ok").render, None);
    }

    #[test]
    fn upscale_with_opt_in_resolves() {
        let env = env_image(128, 128);
        assert!(plan(&model_image(256, 256, true), &images(&env)).is_ok());
    }

    #[test]
    fn downscale_needs_no_opt_in() {
        let env = env_image(256, 256);
        assert!(plan(&model_image(128, 128, false), &images(&env)).is_ok());
    }

    #[test]
    fn a_matching_aspect_records_a_plain_scale_whatever_fit_says() {
        let env = env_image(256, 256);
        let mut model = model_image(256, 256, false);
        model.fit = Some(fits(&["pad"]));
        assert_eq!(plan(&model, &images(&env)).expect("ok").fit, Fit::Stretch);
        let unknown = env_image(0, 0);
        assert_eq!(plan(&model, &images(&unknown)).expect("ok").fit, Fit::Pad);
    }

    #[test]
    fn aspect_mismatch_without_fit_is_a_resolve_error() {
        let env = env_image(8, 8);
        let error = plan(&model_image(3, 4, false), &images(&env)).expect_err("err");
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert!(
            error.message.contains("aspect ratio"),
            "got: {}",
            error.message
        );
    }

    #[test]
    fn aspect_mismatch_with_fit_resolves() {
        let env = env_image(8, 8);
        let mut model = model_image(3, 4, false);
        model.fit = Some(fits(&["crop"]));
        let plan = plan(&model, &images(&env)).expect("ok");
        assert_eq!(plan.fit, Fit::Crop);
    }

    #[test]
    fn matching_aspect_needs_no_fit() {
        let env = env_image(8, 8);
        let plan = plan(&model_image(4, 4, false), &images(&env)).expect("ok");
        assert_eq!(plan.fit, Fit::Stretch);
    }

    #[test]
    fn fit_list_picks_first_usable_per_env() {
        let mut model = model_image(100, 100, false);
        model.fit = Some(fits(&["crop", "pad"]));
        let big = env_image(200, 150);
        assert_eq!(plan(&model, &images(&big)).expect("ok").fit, Fit::Crop);
        let small = env_image(50, 150);
        assert_eq!(plan(&model, &images(&small)).expect("ok").fit, Fit::Pad);
    }

    #[test]
    fn every_fit_upscaling_names_the_model_knob_and_the_env_camera() {
        let mut model = model_image(100, 100, false);
        model.fit = Some(fits(&["crop"]));
        let error = plan(&model, &images(&env_image(50, 150))).expect_err("err");
        assert_eq!(error.code, ErrorCode::Unsupported);
        let message = &error.message;
        assert!(
            message.contains("every declared fit would upscale the env's 50x150 camera"),
            "{message}"
        );
        assert!(
            message.contains("or set allow_upscale on the model's Image input"),
            "{message}"
        );
        assert!(!message.contains("  "), "{message:?}");
    }

    #[test]
    fn aspect_mismatch_with_only_unrecognized_fit_is_a_resolve_error() {
        let env = env_image(8, 8);
        let mut model = model_image(3, 4, false);
        model.fit = Some(fits(&["squish"]));
        let error = plan(&model, &images(&env)).expect_err("err");
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert!(
            error.message.contains("recognize"),
            "got: {}",
            error.message
        );
    }

    #[test]
    fn matching_aspect_with_only_unrecognized_fit_is_a_resolve_error() {
        let env = env_image(8, 8);
        let mut model = model_image(4, 4, false);
        model.fit = Some(fits(&["squish"]));
        let error = plan(&model, &images(&env)).expect_err("err");
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert!(
            error.message.contains("recognize"),
            "got: {}",
            error.message
        );
    }

    #[test]
    fn normalize_range_overload_maps_into_the_declared_range() {
        let env = env_image(8, 8);
        let mut model = model_image(8, 8, false);
        model.normalize = Some((-1.0, 1.0));
        let plan = plan(&model, &images(&env)).expect("ok");
        assert_eq!(plan.normalize, Some((-1.0, 1.0)));
    }

    #[test]
    fn normalize_true_uses_the_unit_interval() {
        let env = env_image(8, 8);
        let mut model = model_image(8, 8, false);
        model.normalize = Some((0.0, 1.0));
        let plan = plan(&model, &images(&env)).expect("ok");
        assert_eq!(plan.normalize, Some((0.0, 1.0)));
    }

    #[test]
    fn normalize_off_skips_normalization() {
        let env = env_image(8, 8);
        let plan = plan(&model_image(8, 8, false), &images(&env)).expect("ok");
        assert_eq!(plan.normalize, None);
    }

    #[test]
    fn channel_mismatch_is_a_resolve_error() {
        let mut env = env_image(8, 8);
        env.channels = 1;
        let mut model = model_image(8, 8, false);
        model.channels = Some(3);
        let error = plan(&model, &images(&env)).expect_err("err");
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert!(error.message.contains("channel"), "got: {}", error.message);
    }

    #[test]
    fn matching_channels_resolve() {
        let env = env_image(8, 8);
        let mut model = model_image(8, 8, false);
        model.channels = Some(3);
        assert!(plan(&model, &images(&env)).is_ok());
    }

    #[test]
    fn undeclared_channels_skip_the_check() {
        let mut env = env_image(8, 8);
        env.channels = 1;
        assert!(plan(&model_image(8, 8, false), &images(&env)).is_ok());
    }

    #[test]
    fn optional_image_absent_zero_fills() {
        let mut model = model_image(8, 8, false);
        model.optional = true;
        model.channels = Some(3);
        let empty: BTreeMap<LeafKey, &EnvImage> = BTreeMap::new();
        let plan = plan(&model, &empty).expect("ok");
        assert_eq!(plan.zero_fill, Some((8, 8, 3)));
        assert!(plan.source.is_root());
    }

    #[test]
    fn optional_image_with_absent_role_zero_fills_even_with_one_camera() {
        let env = env_image(8, 8);
        let mut model = model_image(8, 8, false);
        model.role = "image/overhead".to_owned();
        model.optional = true;
        model.channels = Some(3);
        let plan = plan(&model, &images(&env)).expect("ok");
        assert_eq!(plan.zero_fill, Some((8, 8, 3)));
        assert!(plan.source.is_root());
    }

    #[test]
    fn optional_image_without_channels_is_an_error() {
        let mut model = model_image(8, 8, false);
        model.optional = true;
        let empty: BTreeMap<LeafKey, &EnvImage> = BTreeMap::new();
        let error = plan(&model, &empty).expect_err("err");
        assert_eq!(error.code, ErrorCode::MissingWidth);
    }

    #[test]
    fn non_optional_absent_image_is_missing_role() {
        let empty: BTreeMap<LeafKey, &EnvImage> = BTreeMap::new();
        let error = plan(&model_image(8, 8, false), &empty).expect_err("err");
        assert_eq!(error.code, ErrorCode::MissingRole);
    }

    #[test]
    fn fill_without_optional_is_a_resolve_error() {
        let env = env_image(8, 8);
        let mut model = model_image(8, 8, false);
        model.fill = Some(128);
        let error = plan(&model, &images(&env)).expect_err("err");
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert!(
            error.message.contains("fill only applies"),
            "got: {}",
            error.message
        );
    }

    #[test]
    fn typo_role_with_one_camera_is_a_resolve_error() {
        let env = env_image(8, 8);
        let mut model = model_image(8, 8, false);
        model.role = "image/priamry".to_owned();
        let error = plan(&model, &images(&env)).expect_err("err");
        assert_eq!(error.code, ErrorCode::MissingRole);
    }

    #[test]
    fn sanctioned_role_with_one_camera_binds_and_records_the_rebind() {
        let env = env_image(8, 8);
        let mut model = model_image(8, 8, false);
        model.role = "image/wrist".to_owned();
        let plan = plan(&model, &images(&env)).expect("ok");
        assert_eq!(
            plan.role_rebound,
            Some(("image/wrist".to_owned(), "image/primary".to_owned()))
        );
    }

    #[test]
    fn role_mismatch_with_two_cameras_is_a_resolve_error() {
        let primary = env_image(8, 8);
        let mut wrist = env_image(8, 8);
        wrist.role = "image/wrist".to_owned();
        let mut images = images(&primary);
        images.extend(self::images(&wrist));
        let mut model = model_image(8, 8, false);
        model.role = "image/overhead".to_owned();
        let error = plan(&model, &images).expect_err("err");
        assert_eq!(error.code, ErrorCode::MissingRole);
    }

    #[test]
    fn render_matching_the_bound_camera_resolves() {
        let env = env_image(448, 448);
        let mut model = model_image(224, 224, false);
        model.render = Some((448, 448));
        let plan = plan(&model, &images(&env)).expect("ok");
        assert_eq!(plan.render, Some((448, 448)));
    }

    #[test]
    fn render_disagreeing_with_the_bound_camera_is_a_render_mismatch() {
        let env = env_image(256, 256);
        let mut model = model_image(224, 224, false);
        model.render = Some((448, 448));
        let error = plan(&model, &images(&env)).expect_err("err");
        assert_eq!(error.code, ErrorCode::RenderMismatch);
        assert!(
            error.message.contains("declares render 448x448")
                && error.message.contains("renders 256x256"),
            "got: {}",
            error.message
        );
    }

    #[test]
    fn render_asserts_nothing_when_the_camera_size_is_underivable() {
        let env = env_image(0, 0);
        let mut model = model_image(224, 224, false);
        model.render = Some((448, 448));
        let plan = plan(&model, &images(&env)).expect("ok");
        assert_eq!(plan.render, None);
    }

    #[test]
    fn render_is_checked_against_the_rebound_camera() {
        let env = env_image(256, 256);
        let mut model = model_image(224, 224, false);
        model.role = "image/wrist".to_owned();
        model.render = Some((448, 448));
        let error = plan(&model, &images(&env)).expect_err("err");
        assert_eq!(error.code, ErrorCode::RenderMismatch);
        assert!(
            error.message.contains("\"image/primary\""),
            "got: {}",
            error.message
        );
        let env = env_image(448, 448);
        let plan = plan(&model, &images(&env)).expect("ok");
        assert_eq!(plan.render, Some((448, 448)));
        assert!(plan.role_rebound.is_some());
    }

    #[test]
    fn render_on_an_absent_optional_camera_asserts_nothing() {
        let mut model = model_image(224, 224, false);
        model.optional = true;
        model.channels = Some(3);
        model.render = Some((448, 448));
        let empty: BTreeMap<LeafKey, &EnvImage> = BTreeMap::new();
        let plan = plan(&model, &empty).expect("ok");
        assert_eq!(plan.zero_fill, Some((224, 224, 3)));
        assert_eq!(plan.render, None);
    }

    #[test]
    fn crop_area_resolves_to_its_side_fraction() {
        let env = env_image(8, 8);
        let mut model = model_image(8, 8, false);
        model.crop = Some(Crop {
            fraction: 0.9f64.sqrt(),
            area: Some(0.9),
            slice: false,
        });
        let crop = plan(&model, &images(&env)).expect("ok").crop.expect("crop");
        assert!((crop.fraction - 0.9f64.sqrt()).abs() < 1e-12);
        assert_eq!(crop.area, Some(0.9));
        assert!(!crop.slice);
        assert_eq!(crop.cut, None);
    }

    #[test]
    fn slice_crop_precomputes_the_integer_cut() {
        let env = env_image(480, 480);
        let mut model = model_image(448, 448, true);
        model.crop = Some(Crop {
            fraction: 2.0 / 3.0,
            area: None,
            slice: true,
        });
        let crop = plan(&model, &images(&env)).expect("ok").crop.expect("crop");
        assert!(crop.slice);
        assert_eq!(crop.cut, Some((320, 320)));
    }

    #[test]
    fn a_crop_that_shrinks_the_source_below_the_target_needs_allow_upscale() {
        let env = env_image(12, 12);
        let mut model = model_image(8, 8, false);
        model.crop = Some(Crop {
            fraction: 0.5,
            area: None,
            slice: false,
        });
        let error = plan(&model, &images(&env)).expect_err("err");
        assert_eq!(error.code, ErrorCode::Unsupported);
        let message = &error.message;
        assert!(
            message
                .contains("the 6x6 crop box the model's crop=0.5 cuts from the env's 12x12 camera"),
            "{message}"
        );
        assert!(
            message.contains("allow_upscale=True") && message.contains("crop less"),
            "{message}"
        );
        assert!(!message.contains("  "), "{message:?}");
        assert!(plan(&model_image(8, 8, false), &images(&env)).is_ok());
    }

    #[test]
    fn bgr_needs_three_channels() {
        let mut env = env_image(8, 8);
        env.channels = 1;
        let mut model = model_image(8, 8, false);
        model.bgr = true;
        let error = plan(&model, &images(&env)).expect_err("err");
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert!(
            error.message.contains("3-channel"),
            "got: {}",
            error.message
        );

        let rgb = env_image(8, 8);
        assert!(plan(&model, &images(&rgb)).expect("ok").swap_rb);
    }
}
