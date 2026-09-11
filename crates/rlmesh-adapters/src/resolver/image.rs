//! Pair one model image input with an env camera and derive its plan.

use std::collections::BTreeMap;

use super::{Result, err};
use crate::error::ErrorCode;
use crate::fmt::{quoted, quoted_keys};
use crate::path::NodePath;
use crate::plans::{CropPlan, ImagePlan};
use crate::spec::{CHANNEL_ORDERS, CROP_MODES, EnvImage, FitMode, Image, ImageLayout};

pub(super) fn plan_image(
    model_input: &Image,
    placement: NodePath,
    images_by_role: &BTreeMap<String, &EnvImage>,
    unknown_roles: &BTreeMap<String, String>,
) -> Result<ImagePlan> {
    let at = quoted(&placement.to_string());
    // `fill` only colors the zero-filled frame an `optional` camera
    // produces when the env lacks it; without `optional` it can never take
    // effect, so a set-but-inert fill is a spec error, not a silent no-op.
    if model_input.fill.is_some() && !model_input.optional {
        return Err(err(
            ErrorCode::Unsupported,
            format!(
                "model input {at}: fill only applies to an optional camera; \
                 set optional or drop fill"
            ),
        ));
    }
    let mut env_image = images_by_role.get(&model_input.role).copied();
    if env_image.is_none() {
        // The role's data is present but under a kind this core can't read: fail
        // loud before any fallback (lone-camera bind, optional zero-fill) papers
        // over it. A role the env genuinely lacks passes through to the fallbacks.
        super::reject_referenced_unknown(&model_input.role, &placement, unknown_roles)?;
    }
    // The lone-camera fallback binds a role-mismatched input to the env's
    // single camera -- but only for a sanctioned (registered or `x/`) role, so
    // a typo'd role name fails loudly instead of silently feeding the camera.
    // An `optional` input has opted into zero-filling instead; that wins. A
    // `_2` role names the *second* of a pair, and an env with one camera has no
    // second of anything: rebinding it would quietly feed the first arm's view
    // to an input asking for the other arm's.
    let mut role_rebound = None;
    if env_image.is_none()
        && !model_input.optional
        && images_by_role.len() == 1
        && !model_input.role.ends_with("_2")
        && crate::roles::registry::is_sanctioned_role(&model_input.role)
    {
        env_image = images_by_role.values().next().copied();
        role_rebound = env_image.map(|env| (model_input.role.clone(), env.role.clone()));
    }
    let Some(env_image) = env_image else {
        // An optional camera the env does not provide is zero-filled (a black
        // frame), not a hard error -- the image-side analogue of an optional
        // state component.
        if model_input.optional {
            return zero_fill_image_plan(model_input, placement);
        }
        return Err(err(
            ErrorCode::MissingRole,
            format!(
                "model input {at} wants an image with role {} but the env offers {}",
                quoted(&model_input.role),
                quoted_keys(images_by_role)
            ),
        ));
    };
    if !crate::apply::RESAMPLES.contains(&model_input.resample.as_str()) {
        return Err(err(
            ErrorCode::Unsupported,
            format!(
                "model input {at}: unsupported resample {}; expected one of {:?}",
                quoted(&model_input.resample),
                crate::apply::RESAMPLES
            ),
        ));
    }
    // The two other constrained-string vocabularies, checked here for the same
    // reason as `resample`: they are strings on the wire so a future additive
    // value degrades to this typed error instead of a parse failure, which only
    // pays off if a typo is caught at resolve rather than ignored as inert.
    for (field, value, vocabulary) in [
        ("crop_mode", &model_input.crop_mode, CROP_MODES),
        ("channel_order", &model_input.channel_order, CHANNEL_ORDERS),
    ] {
        if !vocabulary.contains(&value.as_str()) {
            return Err(err(
                ErrorCode::Unsupported,
                format!(
                    "model input {at}: unsupported {field} {}; expected one of {vocabulary:?}",
                    quoted(value)
                ),
            ));
        }
    }
    // Validate dtype at resolve (like resample above) so a typo'd name fails
    // resolution once, not per-step in apply (finalize_dtype) at serve time.
    if rlmesh_spaces::DType::from_name(&model_input.dtype).is_none() {
        return Err(err(
            ErrorCode::Unsupported,
            format!(
                "model input {at}: unknown dtype {}",
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
                "model input {at}: expects {expected} channel(s) but the env image has {}; the \
             adapter does not convert between channel counts (e.g. RGB vs grayscale)",
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
    let swap_rb = model_input.channel_order == "bgr";
    // The swap is a 3-channel op; a grayscale or RGBA feed would silently get
    // its bytes reordered (or fail per-step in apply), so reject it here.
    if swap_rb
        && let Some(channels) = model_input
            .channels
            .or(Some(env_image.channels))
            .filter(|&c| c != 0)
        && channels != 3
    {
        return Err(err(
            ErrorCode::Unsupported,
            format!(
                "model input {at}: channel_order \"bgr\" needs a 3-channel image, \
                 got {channels} channel(s)"
            ),
        ));
    }
    let crop = crop_plan(model_input, &at, env_image)?;
    let fit = resolve_fit(model_input, &at, env_image, size)?;
    Ok(ImagePlan {
        placement,
        source: env_image.source.clone(),
        src_layout: env_image.layout,
        dst_layout: model_input.layout,
        flip: env_image.upside_down != model_input.upside_down,
        size,
        fit,
        resample: model_input.resample.clone(),
        dtype: model_input.dtype.clone(),
        normalize: model_input.normalize.range(),
        lead_dims: model_input.lead_dims,
        src_range: env_image.value_range,
        stack: model_input.stack,
        zero_fill: None,
        fill: model_input.fill.unwrap_or(0),
        crop,
        swap_rb,
        role_rebound,
    })
}

/// Resolve `crop` / `crop_area` / `crop_mode` into the center box to keep.
///
/// The two fractions say the same thing two ways — a side fraction and an area
/// fraction — so declaring both is a spec error rather than a silent
/// precedence rule. The plan carries the side fraction; `area` rides along only
/// so describe can echo the form the author wrote.
fn crop_plan(model_input: &Image, at: &str, env_image: &EnvImage) -> Result<Option<CropPlan>> {
    if model_input.crop.is_some() && model_input.crop_area.is_some() {
        return Err(err(
            ErrorCode::Unsupported,
            format!(
                "model input {at}: set crop (a side fraction) or crop_area (an area \
                 fraction), not both"
            ),
        ));
    }
    let Some(fraction) = model_input
        .crop
        .or_else(|| model_input.crop_area.map(f64::sqrt))
    else {
        return Ok(None);
    };
    let slice = model_input.crop_mode == "slice";
    // The integer cut at the env's declared resolution, for the describe text;
    // apply recomputes it from the frame it is handed, through the same helper.
    let cut = (slice && env_image.height != 0 && env_image.width != 0).then(|| {
        (
            crate::apply::crop_cut(env_image.height as usize, fraction) as u32,
            crate::apply::crop_cut(env_image.width as usize, fraction) as u32,
        )
    });
    Ok(Some(CropPlan {
        fraction,
        area: model_input.crop_area,
        slice,
        cut,
    }))
}

/// Build a zero-fill (black-frame) plan for an optional image the env lacks.
///
/// The blank is sized from the model's declared `height`/`width`/`channels`
/// (there is no env image to derive them from), then run through the normal
/// normalize/dtype/layout/lead steps so it matches a real black frame.
fn zero_fill_image_plan(model_input: &Image, placement: NodePath) -> Result<ImagePlan> {
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
        fit: FitMode::Stretch,
        resample: model_input.resample.clone(),
        dtype: model_input.dtype.clone(),
        normalize: model_input.normalize.range(),
        lead_dims: model_input.lead_dims,
        src_range: None,
        stack: model_input.stack,
        zero_fill: Some((height, width, channels)),
        fill: model_input.fill.unwrap_or(0),
        // A synthesized frame is one flat level: cropping or swapping its
        // channels cannot change a pixel, so neither step is planned.
        crop: None,
        swap_rb: false,
        role_rebound: None,
    })
}

/// Choose the fit mode for this env from the model's permitted modes.
///
/// The model declares an ordered preference list (a single mode is a one-entry
/// list); per env, the first mode that does not need a *disallowed* upscale wins,
/// so one spec can crop a large camera and letterbox a small one. Both the
/// aspect guard (an aspect-changing resize with no fit declared) and the upscale
/// guard (a resize that scales up without `allow_upscale`) live here.
fn resolve_fit(
    model_input: &Image,
    at: &str,
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
                "model input {at}: target {target_height}x{target_width} changes the env's \
             {env_height}x{env_width} aspect ratio; set fit to 'stretch', 'crop', or 'pad'"
            ),
            Some(set) if permitted.is_empty() => format!(
                "model input {at}: target {target_height}x{target_width} changes the env's \
             {env_height}x{env_width} aspect ratio and the declared fit {:?} names no mode this \
             build recognizes; expected 'stretch', 'crop', or 'pad'",
                set.wire_names()
            ),
            Some(_) => format!(
                "model input {at}: every declared fit would upscale the env's \
             {env_height}x{env_width} image to {target_height}x{target_width}; set allow_upscale \
             or declare a fit that downscales (e.g. 'pad')"
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
                "model input {at}: the declared fit {:?} names no mode this build \
             recognizes; expected 'stretch', 'crop', or 'pad'",
                set.wire_names()
            ),
        ));
    }
    let mode = permitted.first().copied().unwrap_or(FitMode::Stretch);
    if !model_input.allow_upscale && upscales(mode) {
        return Err(err(
            ErrorCode::Unsupported,
            format!(
                "model input {at}: target {target_height}x{target_width} upscales the env's \
             {env_height}x{env_width} image; set allow_upscale to interpolate detail that is not there"
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
    use crate::path::NodePath;
    use crate::spec::{AcceptSet, EnvImage, FitMode, Image, ImageLayout, Normalize};

    /// Resolve at the root placement (the common single-leaf-payload case).
    fn plan(
        model: &Image,
        images: &BTreeMap<String, &EnvImage>,
    ) -> Result<crate::plans::ImagePlan, crate::error::AdapterResolutionError> {
        plan_image(model, NodePath::root(), images, &BTreeMap::new())
    }

    fn env_image(height: u32, width: u32) -> EnvImage {
        EnvImage {
            source: NodePath::root().push_key("cam"),
            role: "image/primary".to_owned(),
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
            role: "image/primary".to_owned(),
            height: Some(height),
            width: Some(width),
            layout: ImageLayout::Hwc,
            channels: None,
            dtype: "uint8".to_owned(),
            normalize: Normalize::Off,
            lead_dims: 0,
            upside_down: false,
            resample: "bilinear".to_owned(),
            allow_upscale,
            fit: None,
            optional: false,
            fill: None,
            stack: 1,
            crop: None,
            crop_area: None,
            crop_mode: "zoom".to_owned(),
            channel_order: "rgb".to_owned(),
            unknown: Default::default(),
        }
    }

    fn images(env: &EnvImage) -> BTreeMap<String, &EnvImage> {
        BTreeMap::from([(env.role.clone(), env)])
    }

    #[test]
    fn upscale_without_opt_in_is_a_resolve_error() {
        let env = env_image(128, 128);
        let error = plan(&model_image(256, 256, false), &images(&env)).expect_err("err");
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert!(error.message.contains("upscale"), "got: {}", error.message);
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
    fn aspect_mismatch_without_fit_is_a_resolve_error() {
        let env = env_image(8, 8);
        // 3x4 changes the 1:1 aspect; with no fit declared this is rejected.
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
        model.fit = Some(AcceptSet::single(FitMode::Crop));
        let plan = plan(&model, &images(&env)).expect("ok");
        assert_eq!(plan.fit, FitMode::Crop);
    }

    #[test]
    fn matching_aspect_needs_no_fit() {
        let env = env_image(8, 8);
        // 4x4 preserves the 1:1 aspect -> no fit required, defaults to stretch.
        let plan = plan(&model_image(4, 4, false), &images(&env)).expect("ok");
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
        assert_eq!(plan(&model, &images(&big)).expect("ok").fit, FitMode::Crop);
        // Too short to crop-cover without upscaling -> falls back to pad.
        let small = env_image(50, 150);
        assert_eq!(plan(&model, &images(&small)).expect("ok").fit, FitMode::Pad);
    }

    #[test]
    fn aspect_mismatch_with_only_unrecognized_fit_is_a_resolve_error() {
        let env = env_image(8, 8);
        let mut model = model_image(3, 4, false);
        // A future/typo'd mode parses (tolerated) but resolves to no usable fit.
        model.fit = Some(serde_json::from_str(r#""squish""#).expect("parse"));
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
        // Same unrecognized-fit spec as above but with a matching (1:1) aspect:
        // this used to silently degrade to stretch; now it is rejected too.
        let env = env_image(8, 8);
        let mut model = model_image(4, 4, false);
        model.fit = Some(serde_json::from_str(r#""squish""#).expect("parse"));
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
        // A `[min, max]` normalize maps pixels into that range.
        let env = env_image(8, 8);
        let mut model = model_image(8, 8, false);
        model.normalize = Normalize::Range(-1.0, 1.0);
        let plan = plan(&model, &images(&env)).expect("ok");
        assert_eq!(plan.normalize, Some((-1.0, 1.0)));
    }

    #[test]
    fn normalize_true_uses_the_unit_interval() {
        let env = env_image(8, 8);
        let mut model = model_image(8, 8, false);
        model.normalize = Normalize::Unit;
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
        env.channels = 1; // grayscale env
        let mut model = model_image(8, 8, false);
        model.channels = Some(3); // model wants RGB
        let error = plan(&model, &images(&env)).expect_err("err");
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert!(error.message.contains("channel"), "got: {}", error.message);
    }

    #[test]
    fn matching_channels_resolve() {
        let env = env_image(8, 8); // 3 channels
        let mut model = model_image(8, 8, false);
        model.channels = Some(3);
        assert!(plan(&model, &images(&env)).is_ok());
    }

    #[test]
    fn undeclared_channels_skip_the_check() {
        let mut env = env_image(8, 8);
        env.channels = 1;
        // model declares no channel count -> the check is skipped (back-compat).
        assert!(plan(&model_image(8, 8, false), &images(&env)).is_ok());
    }

    #[test]
    fn optional_image_absent_zero_fills() {
        let mut model = model_image(8, 8, false);
        model.optional = true;
        model.channels = Some(3);
        let empty: BTreeMap<String, &EnvImage> = BTreeMap::new();
        let plan = plan(&model, &empty).expect("ok");
        assert_eq!(plan.zero_fill, Some((8, 8, 3)));
        assert!(plan.source.is_root());
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
        let plan = plan(&model, &images(&env)).expect("ok");
        assert_eq!(plan.zero_fill, Some((8, 8, 3)));
        assert!(plan.source.is_root());
    }

    #[test]
    fn optional_image_without_channels_is_an_error() {
        let mut model = model_image(8, 8, false);
        model.optional = true; // height+width set, channels None -> cannot size
        let empty: BTreeMap<String, &EnvImage> = BTreeMap::new();
        let error = plan(&model, &empty).expect_err("err");
        assert_eq!(error.code, ErrorCode::MissingWidth);
    }

    #[test]
    fn non_optional_absent_image_is_missing_role() {
        let empty: BTreeMap<String, &EnvImage> = BTreeMap::new();
        let error = plan(&model_image(8, 8, false), &empty).expect_err("err");
        assert_eq!(error.code, ErrorCode::MissingRole);
    }

    #[test]
    fn fill_without_optional_is_a_resolve_error() {
        // A fill level set without `optional` can never take effect; reject it
        // rather than silently ignoring the configured value.
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
        // A role outside the registry (and not an `x/` escape) never rides the
        // lone-camera fallback: a typo fails loudly instead of feeding the camera.
        let env = env_image(8, 8); // role "image/primary"
        let mut model = model_image(8, 8, false);
        model.role = "image/priamry".to_owned();
        let error = plan(&model, &images(&env)).expect_err("err");
        assert_eq!(error.code, ErrorCode::MissingRole);
    }

    #[test]
    fn sanctioned_role_with_one_camera_binds_and_records_the_rebind() {
        let env = env_image(8, 8); // role "image/primary"
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
        let images = BTreeMap::from([
            (primary.role.clone(), &primary),
            (wrist.role.clone(), &wrist),
        ]);
        let mut model = model_image(8, 8, false);
        model.role = "image/overhead".to_owned();
        let error = plan(&model, &images).expect_err("err");
        assert_eq!(error.code, ErrorCode::MissingRole);
    }

    #[test]
    fn crop_area_resolves_to_its_side_fraction() {
        // The two fractions are one box said two ways: an area fraction is the
        // square of the side fraction the plan carries.
        let env = env_image(8, 8);
        let mut model = model_image(8, 8, false);
        model.crop_area = Some(0.9);
        let crop = plan(&model, &images(&env)).expect("ok").crop.expect("crop");
        assert!((crop.fraction - 0.9f64.sqrt()).abs() < 1e-12);
        assert_eq!(crop.area, Some(0.9));
        assert!(!crop.slice);
        assert_eq!(crop.cut, None); // a zoom takes no integer cut
    }

    #[test]
    fn slice_crop_precomputes_the_integer_cut() {
        let env = env_image(480, 480);
        let mut model = model_image(448, 448, true);
        model.crop = Some(2.0 / 3.0);
        model.crop_mode = "slice".to_owned();
        let crop = plan(&model, &images(&env)).expect("ok").crop.expect("crop");
        assert!(crop.slice);
        assert_eq!(crop.cut, Some((320, 320)));
    }

    #[test]
    fn crop_and_crop_area_together_are_a_resolve_error() {
        let env = env_image(8, 8);
        let mut model = model_image(8, 8, false);
        model.crop = Some(0.5);
        model.crop_area = Some(0.25);
        let error = plan(&model, &images(&env)).expect_err("err");
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert!(error.message.contains("not both"), "got: {}", error.message);
    }

    #[test]
    fn typo_crop_mode_and_channel_order_are_resolve_errors() {
        // Constrained strings, like `resample`: a typo fails resolution instead
        // of silently defaulting to zoom/rgb.
        let env = env_image(8, 8);
        for (field, value) in [("crop_mode", "zooom"), ("channel_order", "rbg")] {
            let mut model = model_image(8, 8, false);
            if field == "crop_mode" {
                model.crop_mode = value.to_owned();
            } else {
                model.channel_order = value.to_owned();
            }
            let error = plan(&model, &images(&env)).expect_err("err");
            assert_eq!(error.code, ErrorCode::Unsupported);
            assert!(error.message.contains(field), "got: {}", error.message);
        }
    }

    #[test]
    fn bgr_needs_three_channels() {
        let mut env = env_image(8, 8);
        env.channels = 1;
        let mut model = model_image(8, 8, false);
        model.channel_order = "bgr".to_owned();
        let error = plan(&model, &images(&env)).expect_err("err");
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert!(
            error.message.contains("3-channel"),
            "got: {}",
            error.message
        );

        let rgb = env_image(8, 8); // 3 channels
        assert!(plan(&model, &images(&rgb)).expect("ok").swap_rb);
    }

    #[test]
    fn unknown_dtype_is_a_resolve_error() {
        // dtype is checked at resolve (like resample), not deferred to apply.
        let env = env_image(8, 8);
        let mut model = model_image(8, 8, false);
        model.dtype = "flat32".to_owned(); // typo of float32
        let error = plan(&model, &images(&env)).expect_err("err");
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert!(
            error.message.contains("unknown dtype"),
            "got: {}",
            error.message
        );
    }
}
