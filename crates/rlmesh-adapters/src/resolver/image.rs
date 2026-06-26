//! Pair one model image input with an env camera and derive its plan.

use std::collections::BTreeMap;

use super::{Result, err};
use crate::error::ErrorCode;
use crate::fmt::{quoted, quoted_keys};
use crate::plans::ImagePlan;
use crate::spec::{EnvImage, FitMode, ImageInput, ImageLayout};

pub(super) fn plan_image(
    model_input: &ImageInput,
    images_by_role: &BTreeMap<String, &EnvImage>,
) -> Result<ImagePlan> {
    // `absent_fill` only colors the zero-filled frame an `optional` camera
    // produces when the env lacks it; without `optional` it can never take
    // effect, so a set-but-inert fill is a spec error, not a silent no-op.
    if model_input.absent_fill.is_some() && !model_input.optional {
        return Err(err(
            ErrorCode::Unsupported,
            format!(
                "model input {}: absent_fill only applies to an optional camera; \
                 set optional or drop absent_fill",
                quoted(&model_input.key)
            ),
        ));
    }
    let mut env_image = images_by_role.get(&model_input.role).copied();
    // The lone-camera fallback papers over role-name mismatches when the env
    // offers exactly one camera. An `optional` input, though, has explicitly
    // opted into zero-filling when its role is absent -- that contract wins, so
    // don't let the fallback bind it to an unrelated camera.
    if env_image.is_none() && !model_input.optional && images_by_role.len() == 1 {
        env_image = images_by_role.values().next().copied();
    }
    let Some(env_image) = env_image else {
        // An optional camera the env does not provide is zero-filled (a black
        // frame), not a hard error -- the image-side analogue of an optional
        // state component.
        if model_input.optional {
            return zero_fill_image_plan(model_input);
        }
        return Err(err(
            ErrorCode::MissingRole,
            format!(
                "model input {} wants an image with role {} but the env offers {}",
                quoted(&model_input.key),
                quoted(&model_input.role),
                quoted_keys(images_by_role)
            ),
        ));
    };
    if model_input.resample != "bilinear" && model_input.resample != "bilinear_aa" {
        return Err(err(
            ErrorCode::Unsupported,
            format!(
                "model input {}: unsupported resample {}; expected 'bilinear' or \
             'bilinear_aa'",
                quoted(&model_input.key),
                quoted(&model_input.resample)
            ),
        ));
    }
    // Validate dtype at resolve (like resample above) so a typo'd name fails
    // resolution once, not per-step in apply (finalize_dtype) at serve time.
    if rlmesh_spaces::DType::from_name(&model_input.dtype).is_none() {
        return Err(err(
            ErrorCode::Unsupported,
            format!(
                "model input {}: unknown dtype {}",
                quoted(&model_input.key),
                quoted(&model_input.dtype)
            ),
        ));
    }
    // A channel-count mismatch (e.g. RGB vs grayscale) is not converted; left
    // unchecked it silently feeds the model a wrong-shaped tensor, so reject it.
    // Only checked when the model declares its expected channels and the env's
    // channel count was derivable.
    if let Some(expected) = model_input.channels
        && env_image.channels != 0
        && env_image.channels != expected
    {
        return Err(err(
            ErrorCode::Unsupported,
            format!(
                "model input {}: expects {expected} channel(s) but the env image has {}; the \
             adapter does not convert between channel counts (e.g. RGB vs grayscale)",
                quoted(&model_input.key),
                env_image.channels
            ),
        ));
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
    let fit = resolve_fit(model_input, env_image, size)?;
    Ok(ImagePlan {
        model_key: model_input.key.clone(),
        env_key: env_image.key.clone(),
        src_layout: env_image.layout,
        dst_layout: model_input.layout,
        flip: env_image.upside_down != model_input.upside_down,
        size,
        fit,
        resample: model_input.resample.clone(),
        dtype: model_input.dtype.clone(),
        normalize: resolve_normalize(model_input),
        lead_dims: model_input.lead_dims,
        src_range: env_image.value_range,
        stack: model_input.stack,
        zero_fill: None,
        absent_fill: model_input.absent_fill.unwrap_or(0),
    })
}

/// Build a zero-fill (black-frame) plan for an optional image the env lacks.
///
/// The blank is sized from the model's declared `height`/`width`/`channels`
/// (there is no env image to derive them from), then run through the normal
/// normalize/dtype/layout/lead steps so it matches a real black frame.
fn zero_fill_image_plan(model_input: &ImageInput) -> Result<ImagePlan> {
    let (Some(height), Some(width), Some(channels)) =
        (model_input.height, model_input.width, model_input.channels)
    else {
        return Err(err(
            ErrorCode::MissingWidth,
            format!(
                "model input {}: an optional image the env does not provide needs height, width, \
             and channels to size the zero-filled frame",
                quoted(&model_input.key)
            ),
        ));
    };
    Ok(ImagePlan {
        model_key: model_input.key.clone(),
        env_key: String::new(),
        src_layout: ImageLayout::Hwc,
        dst_layout: model_input.layout,
        flip: false,
        size: None,
        fit: FitMode::Stretch,
        resample: model_input.resample.clone(),
        dtype: model_input.dtype.clone(),
        normalize: resolve_normalize(model_input),
        lead_dims: model_input.lead_dims,
        src_range: None,
        stack: model_input.stack,
        zero_fill: Some((height, width, channels)),
        absent_fill: model_input.absent_fill.unwrap_or(0),
    })
}

/// The normalize range to apply to this image, or `None` for no normalization.
///
/// A declared `normalize_range` implies normalization even when the `normalize`
/// flag is unset, so a range is never silently dropped (setting a range but
/// forgetting the flag used to feed the model raw, unnormalized pixels). When
/// normalizing without an explicit range, default to the conventional `[0, 1]`.
fn resolve_normalize(model_input: &ImageInput) -> Option<(f64, f64)> {
    (model_input.normalize || model_input.normalize_range.is_some())
        .then(|| model_input.normalize_range.unwrap_or((0.0, 1.0)))
}

/// Choose the fit mode for this env from the model's permitted modes.
///
/// The model declares an ordered preference list (a single mode is a one-entry
/// list); per env, the first mode that does not need a *disallowed* upscale wins,
/// so one spec can crop a large camera and letterbox a small one. Both the
/// aspect guard (an aspect-changing resize with no fit declared) and the upscale
/// guard (a resize that scales up without `allow_upscale`) live here.
fn resolve_fit(
    model_input: &ImageInput,
    env_image: &EnvImage,
    size: Option<(u32, u32)>,
) -> Result<FitMode> {
    let Some((target_height, target_width)) = size else {
        return Ok(FitMode::Stretch); // no resize
    };
    let (env_height, env_width) = (env_image.height, env_image.width);
    // The env's native resolution is derived by `join`; if it could not be
    // determined (0), do not block on size — there is nothing to compare against.
    let known = env_height != 0 && env_width != 0;

    // Whether `mode` must scale the env image *up* to reach the target. Crop
    // covers, so any axis short of the target upscales; pad only contains, so it
    // upscales only when both axes are short; stretch scales each axis directly.
    let upscales = |mode: FitMode| -> bool {
        if !known {
            return false;
        }
        match mode {
            FitMode::Stretch | FitMode::Crop => {
                target_height > env_height || target_width > env_width
            }
            FitMode::Pad => target_height > env_height && target_width > env_width,
        }
    };

    let aspect_differs = known
        && u64::from(env_height) * u64::from(target_width)
            != u64::from(env_width) * u64::from(target_height);

    // Permitted modes in preference order; unrecognized (future) modes are
    // skipped so an old core degrades gracefully. `None` = no fit declared.
    let permitted: Vec<FitMode> = model_input
        .fit
        .as_ref()
        .map(|set| set.known().collect())
        .unwrap_or_default();

    if aspect_differs {
        let first_usable = permitted
            .iter()
            .copied()
            .find(|&mode| model_input.allow_upscale || !upscales(mode));
        if let Some(mode) = first_usable {
            return Ok(mode);
        }
        // Nothing usable: distinguish "no fit", "fit unrecognized", and "every
        // declared fit would upscale" so the author knows what to change.
        let message = match &model_input.fit {
            None => format!(
                "model input {}: target {target_height}x{target_width} changes the env's \
             {env_height}x{env_width} aspect ratio; set fit to 'stretch', 'crop', or 'pad'",
                quoted(&model_input.key)
            ),
            Some(set) if permitted.is_empty() => format!(
                "model input {}: target {target_height}x{target_width} changes the env's \
             {env_height}x{env_width} aspect ratio and the declared fit {:?} names no mode this \
             build recognizes; expected 'stretch', 'crop', or 'pad'",
                quoted(&model_input.key),
                set.wire_names()
            ),
            Some(_) => format!(
                "model input {}: every declared fit would upscale the env's \
             {env_height}x{env_width} image to {target_height}x{target_width}; set allow_upscale \
             or declare a fit that downscales (e.g. 'pad')",
                quoted(&model_input.key)
            ),
        };
        return Err(err(ErrorCode::Unsupported, message));
    }

    // Aspect matches (or is unknown): every mode is the same uniform scale. A
    // declared fit that names only modes this build doesn't recognize is still
    // rejected here (consistent with the aspect-differs branch) -- it signals a
    // version/typo mismatch the author should fix, not silently ignore. A missing
    // fit (`None`) still defaults to a plain scale.
    if let Some(set) = &model_input.fit
        && permitted.is_empty()
    {
        return Err(err(
            ErrorCode::Unsupported,
            format!(
                "model input {}: the declared fit {:?} names no mode this build \
             recognizes; expected 'stretch', 'crop', or 'pad'",
                quoted(&model_input.key),
                set.wire_names()
            ),
        ));
    }
    let mode = permitted.first().copied().unwrap_or(FitMode::Stretch);
    if !model_input.allow_upscale && upscales(mode) {
        return Err(err(
            ErrorCode::Unsupported,
            format!(
                "model input {}: target {target_height}x{target_width} upscales the env's \
             {env_height}x{env_width} image; set allow_upscale to interpolate detail that is not there",
                quoted(&model_input.key)
            ),
        ));
    }
    Ok(mode)
}

#[cfg(test)]
mod image_resolve_tests {
    use std::collections::BTreeMap;

    use super::plan_image;
    use crate::error::ErrorCode;
    use crate::spec::{AcceptSet, EnvImage, FitMode, ImageInput, ImageLayout};

    fn env_image(height: u32, width: u32) -> EnvImage {
        EnvImage {
            key: "cam".to_owned(),
            role: "image/primary".to_owned(),
            layout: ImageLayout::Hwc,
            upside_down: false,
            height,
            width,
            channels: 3,
            value_range: None,
        }
    }

    fn model_image(height: u32, width: u32, allow_upscale: bool) -> ImageInput {
        ImageInput {
            key: "image".to_owned(),
            role: "image/primary".to_owned(),
            height: Some(height),
            width: Some(width),
            layout: ImageLayout::Hwc,
            channels: None,
            dtype: "uint8".to_owned(),
            normalize: false,
            normalize_range: None,
            lead_dims: 0,
            upside_down: false,
            resample: "bilinear_aa".to_owned(),
            allow_upscale,
            fit: None,
            optional: false,
            absent_fill: None,
            stack: 1,
        }
    }

    fn images(env: &EnvImage) -> BTreeMap<String, &EnvImage> {
        BTreeMap::from([(env.role.clone(), env)])
    }

    #[test]
    fn upscale_without_opt_in_is_a_resolve_error() {
        let env = env_image(128, 128);
        let error = plan_image(&model_image(256, 256, false), &images(&env)).expect_err("err");
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert!(error.message.contains("upscale"), "got: {}", error.message);
    }

    #[test]
    fn upscale_with_opt_in_resolves() {
        let env = env_image(128, 128);
        assert!(plan_image(&model_image(256, 256, true), &images(&env)).is_ok());
    }

    #[test]
    fn downscale_needs_no_opt_in() {
        let env = env_image(256, 256);
        assert!(plan_image(&model_image(128, 128, false), &images(&env)).is_ok());
    }

    #[test]
    fn aspect_mismatch_without_fit_is_a_resolve_error() {
        let env = env_image(8, 8);
        // 3x4 changes the 1:1 aspect; with no fit declared this is rejected.
        let error = plan_image(&model_image(3, 4, false), &images(&env)).expect_err("err");
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
        model.fit = Some(AcceptSet::single(FitMode::Crop));
        let plan = plan_image(&model, &images(&env)).expect("ok");
        assert_eq!(plan.fit, FitMode::Crop);
    }

    #[test]
    fn matching_aspect_needs_no_fit() {
        let env = env_image(8, 8);
        // 4x4 preserves the 1:1 aspect -> no fit required, defaults to stretch.
        let plan = plan_image(&model_image(4, 4, false), &images(&env)).expect("ok");
        assert_eq!(plan.fit, FitMode::Stretch);
    }

    #[test]
    fn fit_list_picks_first_usable_per_env() {
        // fit=[crop, pad]: crop must cover (upscales a too-small env), pad only
        // contains. The same spec therefore crops a large camera and letterboxes
        // a small one -- chosen per env, not pre-committed.
        let mut model = model_image(100, 100, false);
        model.fit = Some(serde_json::from_str(r#"["crop", "pad"]"#).expect("parse"));
        // Large enough to crop-cover by downscaling -> crop (first preference).
        let big = env_image(200, 150);
        assert_eq!(
            plan_image(&model, &images(&big)).expect("ok").fit,
            FitMode::Crop
        );
        // Too short to crop-cover without upscaling -> falls back to pad.
        let small = env_image(50, 150);
        assert_eq!(
            plan_image(&model, &images(&small)).expect("ok").fit,
            FitMode::Pad
        );
    }

    #[test]
    fn aspect_mismatch_with_only_unrecognized_fit_is_a_resolve_error() {
        let env = env_image(8, 8);
        let mut model = model_image(3, 4, false);
        // A future/typo'd mode parses (tolerated) but resolves to no usable fit.
        model.fit = Some(serde_json::from_str(r#""squish""#).expect("parse"));
        let error = plan_image(&model, &images(&env)).expect_err("err");
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert!(
            error.message.contains("recognize"),
            "got: {}",
            error.message
        );
    }

    #[test]
    fn matching_aspect_with_only_unrecognized_fit_is_a_resolve_error() {
        // Same unrecognized-fit spec as above but with a matching (1:1) aspect:
        // this used to silently degrade to stretch; now it is rejected too.
        let env = env_image(8, 8);
        let mut model = model_image(4, 4, false);
        model.fit = Some(serde_json::from_str(r#""squish""#).expect("parse"));
        let error = plan_image(&model, &images(&env)).expect_err("err");
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert!(
            error.message.contains("recognize"),
            "got: {}",
            error.message
        );
    }

    #[test]
    fn normalize_range_implies_normalization() {
        // A range set without the normalize flag must still normalize (not feed
        // the model raw pixels).
        let env = env_image(8, 8);
        let mut model = model_image(8, 8, false);
        model.normalize = false;
        model.normalize_range = Some((-1.0, 1.0));
        let plan = plan_image(&model, &images(&env)).expect("ok");
        assert_eq!(plan.normalize, Some((-1.0, 1.0)));
    }

    #[test]
    fn no_normalize_and_no_range_skips_normalization() {
        let env = env_image(8, 8);
        let plan = plan_image(&model_image(8, 8, false), &images(&env)).expect("ok");
        assert_eq!(plan.normalize, None);
    }

    #[test]
    fn channel_mismatch_is_a_resolve_error() {
        let mut env = env_image(8, 8);
        env.channels = 1; // grayscale env
        let mut model = model_image(8, 8, false);
        model.channels = Some(3); // model wants RGB
        let error = plan_image(&model, &images(&env)).expect_err("err");
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert!(error.message.contains("channel"), "got: {}", error.message);
    }

    #[test]
    fn matching_channels_resolve() {
        let env = env_image(8, 8); // 3 channels
        let mut model = model_image(8, 8, false);
        model.channels = Some(3);
        assert!(plan_image(&model, &images(&env)).is_ok());
    }

    #[test]
    fn undeclared_channels_skip_the_check() {
        let mut env = env_image(8, 8);
        env.channels = 1;
        // model declares no channel count -> the check is skipped (back-compat).
        assert!(plan_image(&model_image(8, 8, false), &images(&env)).is_ok());
    }

    #[test]
    fn optional_image_absent_zero_fills() {
        let mut model = model_image(8, 8, false);
        model.optional = true;
        model.channels = Some(3);
        let empty: BTreeMap<String, &EnvImage> = BTreeMap::new();
        let plan = plan_image(&model, &empty).expect("ok");
        assert_eq!(plan.zero_fill, Some((8, 8, 3)));
        assert!(plan.env_key.is_empty());
    }

    #[test]
    fn optional_image_with_absent_role_zero_fills_even_with_one_camera() {
        // Regression: the lone-camera fallback must not bind an optional input
        // whose role is absent to an unrelated single camera -- it must zero-fill.
        let env = env_image(8, 8); // role "image/primary"
        let mut model = model_image(8, 8, false);
        model.role = "image/overhead".to_owned(); // absent from the single-camera env
        model.optional = true;
        model.channels = Some(3);
        let plan = plan_image(&model, &images(&env)).expect("ok");
        assert_eq!(plan.zero_fill, Some((8, 8, 3)));
        assert!(plan.env_key.is_empty());
    }

    #[test]
    fn optional_image_without_channels_is_an_error() {
        let mut model = model_image(8, 8, false);
        model.optional = true; // height+width set, channels None -> cannot size
        let empty: BTreeMap<String, &EnvImage> = BTreeMap::new();
        let error = plan_image(&model, &empty).expect_err("err");
        assert_eq!(error.code, ErrorCode::MissingWidth);
    }

    #[test]
    fn non_optional_absent_image_is_missing_role() {
        let empty: BTreeMap<String, &EnvImage> = BTreeMap::new();
        let error = plan_image(&model_image(8, 8, false), &empty).expect_err("err");
        assert_eq!(error.code, ErrorCode::MissingRole);
    }

    #[test]
    fn absent_fill_without_optional_is_a_resolve_error() {
        // A fill level set without `optional` can never take effect; reject it
        // rather than silently ignoring the configured value.
        let env = env_image(8, 8);
        let mut model = model_image(8, 8, false);
        model.absent_fill = Some(128);
        let error = plan_image(&model, &images(&env)).expect_err("err");
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert!(
            error.message.contains("absent_fill only applies"),
            "got: {}",
            error.message
        );
    }

    #[test]
    fn unknown_dtype_is_a_resolve_error() {
        // dtype is checked at resolve (like resample), not deferred to apply.
        let env = env_image(8, 8);
        let mut model = model_image(8, 8, false);
        model.dtype = "flat32".to_owned(); // typo of float32
        let error = plan_image(&model, &images(&env)).expect_err("err");
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert!(
            error.message.contains("unknown dtype"),
            "got: {}",
            error.message
        );
    }
}
