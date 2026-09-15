//! Pair each model state component with an env feature and derive the plan.

use std::collections::BTreeMap;

use super::action::ModelOutputs;
use super::{Indexed, LeafKey, Result, bind, check_geometry, check_provenance, err};
use crate::advisory::Advisory;
use crate::error::ErrorCode;
use crate::fmt::{quoted, quoted_accept_set, quoted_encoding, quoted_leaf_keys};
use crate::path::NodePath;
use crate::plans::{PreviousAction, StatePiece, StatePlan};
use crate::spec::{AcceptSet, Attr, ConcatPart, EnvState, PartSource, RotationEncoding, State};

/// Width of an optional component's fill when the env lacks it.
fn fill_width(component: &ConcatPart, role: &str, at: &str) -> Result<u32> {
    if component.index.is_some() {
        return Ok(1);
    }
    if let Some(labels) = &component.labels {
        return Ok(labels.len() as u32);
    }
    if let Some(dim) = component.dim {
        return Ok(dim);
    }
    if let Some(native) = component
        .encoding
        .as_ref()
        .and_then(|encoding| encoding.base())
    {
        return Ok(native.dims());
    }
    Err(err(
        ErrorCode::MissingWidth,
        format!(
            "model input {at}: optional state role {} needs dim, index, or encoding \
         to size its zero fill",
            quoted(role)
        ),
    ))
}

/// ` (part "p")` for a message about a leaf that named one, else nothing.
pub(super) fn part_suffix(part: Option<&str>) -> String {
    match part {
        Some(part) => format!(" (part {})", quoted(part)),
        None => String::new(),
    }
}

/// The constant an absent (or role-less) part contributes, with the model-side
/// affine folded in — so `apply_state` never needs the spec again.
fn folded_fill(component: &ConcatPart) -> f64 {
    component.fill * component.scale.unwrap_or(1.0) + component.offset.unwrap_or(0.0)
}

/// A piece with no env source: `dim` copies of `fill`. The scalar affine is
/// folded into `fill` by the caller; a per-axis one rides along unfolded and
/// applies per element.
fn fill_piece(
    width: u32,
    fill: f64,
    absent_role: bool,
    part: Option<String>,
    component: &ConcatPart,
) -> StatePiece {
    StatePiece {
        source: NodePath::root(),
        src_offset: None,
        src_dim: None,
        src_encoding: None,
        dst_encoding: None,
        post_rotate: None,
        dim: Some(width),
        index: None,
        src_range: None,
        dst_range: None,
        scale: None,
        offset: None,
        axis_scale: component.axis_scale.clone(),
        axis_offset: component.axis_offset.clone(),
        gather: None,
        labels: component.labels.clone(),
        src_labels: None,
        fill: Some(fill),
        absent_role,
        previous: None,
        frame: None,
        provenance: None,
        part,
        width: Some(width),
    }
}

/// An action-source part reads the spec's own output actuator of the same
/// `(role, part)`: the raw slice of the action the model executed at the
/// previous step, in model order, or `fill` before the episode's first one.
/// The width is the actuator's; `labels` select from the actuator's labels.
fn previous_action_piece(
    component: &ConcatPart,
    role: &str,
    at: &str,
    outputs: &ModelOutputs<'_>,
    advisories: &mut Vec<Advisory>,
) -> Result<StatePiece> {
    let bound = bind(
        &outputs.by_key,
        role,
        component.part.as_deref(),
        None,
        &format!("model input {at}"),
        "previous action role",
        "model",
        advisories,
    )?;
    let Some(bound) = bound else {
        return Err(err(
            ErrorCode::MissingRole,
            format!(
                "model input {at} reads the previous action for role {}{} but no actuator \
                 emits it; the model outputs {}",
                quoted(role),
                part_suffix(component.part.as_deref()),
                quoted_leaf_keys(&outputs.by_key)
            ),
        ));
    };
    let (start, actuator) = *bound.feature;
    let gather = label_gather(
        role,
        actuator.labels.as_deref(),
        component.labels.as_deref(),
        at,
        "actuator",
    )?;
    let width = component
        .labels
        .as_ref()
        .map_or(actuator.dim, |labels| labels.len() as u32);
    if let Some(dim) = component.dim
        && dim != width
    {
        return Err(err(
            ErrorCode::DimMismatch,
            format!(
                "model input {at}: previous action role {} declares dim {dim} but the actuator \
                 emits {width}",
                quoted(role)
            ),
        ));
    }
    Ok(StatePiece {
        source: NodePath::root(),
        src_offset: None,
        src_dim: None,
        src_encoding: None,
        dst_encoding: None,
        post_rotate: None,
        dim: Some(width),
        index: None,
        src_range: None,
        dst_range: None,
        scale: None,
        offset: None,
        axis_scale: None,
        axis_offset: None,
        gather,
        labels: component.labels.clone().or_else(|| actuator.labels.clone()),
        src_labels: actuator.labels.clone(),
        fill: Some(component.fill),
        absent_role: false,
        previous: Some(PreviousAction {
            role: role.to_owned(),
            start,
            stop: start + actuator.dim,
        }),
        frame: None,
        provenance: None,
        part: bound.part,
        width: Some(width),
    })
}

/// A per-axis vector must name exactly one value per resolved axis.
pub(super) fn check_axis_width(
    values: Option<&[f64]>,
    name: &str,
    width: Option<u32>,
    locus: &str,
) -> Result<()> {
    if let (Some(values), Some(width)) = (values, width)
        && values.len() != width as usize
    {
        return Err(err(
            ErrorCode::DimMismatch,
            format!(
                "{locus}: {name} has {} values but the resolved width is {width}",
                values.len()
            ),
        ));
    }
    Ok(())
}

/// Where each of `wanted` sits in `available`, or the labels `available`
/// lacks. Shared by the observation gather and the action scatter.
pub(super) fn positions(
    wanted: &[String],
    available: &[String],
) -> std::result::Result<Vec<u32>, Vec<String>> {
    let mut missing = Vec::new();
    let mut found = Vec::with_capacity(wanted.len());
    for label in wanted {
        match available.iter().position(|have| have == label) {
            Some(index) => found.push(index as u32),
            None => missing.push(label.clone()),
        }
    }
    if missing.is_empty() {
        Ok(found)
    } else {
        Err(missing)
    }
}

/// The gather a model label tuple implies against the leaf it reads (an env
/// leaf, or the model's own actuator for an action-source part; `offerer`
/// names which): `None` when the model names none, the offerer's own order
/// (identity), or a [`LabelMismatch`](ErrorCode::LabelMismatch) when the
/// offerer declares no labels or lacks one the model names. A labeled model
/// against an unlabeled leaf is an error, not a caution: identity-on-hope is
/// the bug.
fn label_gather(
    role: &str,
    env: Option<&[String]>,
    model: Option<&[String]>,
    at: &str,
    offerer: &str,
) -> Result<Option<Vec<u32>>> {
    let Some(model) = model else {
        return Ok(None);
    };
    let Some(env) = env else {
        return Err(err(
            ErrorCode::LabelMismatch,
            format!(
                "model input {at}: state role {} names labels {:?} but the {offerer} declares \
                 none; label the {offerer} (labels=) so the axes can be aligned",
                quoted(role),
                model
            ),
        ));
    };
    match positions(model, env) {
        Ok(gather) => {
            let identity =
                gather.len() == env.len() && gather.iter().enumerate().all(|(i, &j)| i as u32 == j);
            Ok((!identity).then_some(gather))
        }
        Err(missing) => Err(err(
            ErrorCode::LabelMismatch,
            format!(
                "model input {at}: state role {} names labels {:?} that the {offerer} lacks; \
                 the {offerer} declares {:?}",
                quoted(role),
                missing,
                env
            ),
        )),
    }
}

/// Choose the (source, destination) rotation encodings for a state piece.
///
/// `env` is the producer: its first recognized entry is the native (raw)
/// encoding the runtime value is in. `model` is the consumer: its entries are
/// an accept-set in preference order. Prefer no conversion — when the model
/// accepts the env's native, use it on both sides; otherwise convert the native
/// into the model's most-preferred recognized encoding. `(None, None)` means
/// neither side declares a rotation. A side that declares only *unrecognized*
/// encodings is a typed resolve error (graceful degradation's loud edge), not a
/// silent pass.
fn select_state_encoding(
    role: &str,
    env: Option<&AcceptSet<RotationEncoding>>,
    model: Option<&AcceptSet<RotationEncoding>>,
) -> Result<(Option<RotationEncoding>, Option<RotationEncoding>)> {
    let (env_set, model_set) = match (env, model) {
        (None, None) => return Ok((None, None)),
        (Some(_), None) | (None, Some(_)) => {
            return Err(err(
                ErrorCode::EncodingMismatch,
                format!(
                    "state role {}: cannot convert encoding {} to {}; both sides \
                 must declare a rotation encoding",
                    quoted(role),
                    quoted_accept_set(env),
                    quoted_accept_set(model)
                ),
            ));
        }
        (Some(env_set), Some(model_set)) => (env_set, model_set),
    };
    let Some(native) = env_set.first_known() else {
        return Err(err(
            ErrorCode::EncodingMismatch,
            format!(
                "state role {}: env declares only unrecognized rotation encoding(s) {:?}; \
             upgrade the runtime to one that recognizes them",
                quoted(role),
                env_set.wire_names()
            ),
        ));
    };
    if model_set.first_known().is_none() {
        return Err(err(
            ErrorCode::EncodingMismatch,
            format!(
                "state role {}: model declares only unrecognized rotation encoding(s) {:?}; \
             upgrade the runtime to one that recognizes them",
                quoted(role),
                model_set.wire_names()
            ),
        ));
    }
    let dst = if model_set.accepts(native) {
        native
    } else {
        model_set
            .first_known()
            .expect("model has a recognized encoding")
    };
    // The direction law: a sink carries a direction, not a rotation, so an
    // env that publishes one binds only a model that reads that same sink.
    if native.is_sink() && dst != native {
        return Err(err(
            ErrorCode::EncodingMismatch,
            format!(
                "state role {}: the env declares encoding {} which is a direction, not a \
                 rotation, so it cannot be converted to {}; only a rotation encoding \
                 converts into it",
                quoted(role),
                quoted_encoding(Some(native)),
                quoted_encoding(Some(dst))
            ),
        ));
    }
    Ok((Some(native), Some(dst)))
}

pub(super) fn plan_state(
    model_input: &State,
    placement: NodePath,
    states_by_role: &BTreeMap<LeafKey, Indexed<&EnvState>>,
    outputs: &ModelOutputs<'_>,
    unknown_roles: &BTreeMap<String, String>,
    advisories: &mut Vec<Advisory>,
) -> Result<StatePlan> {
    let at = quoted(&placement.to_string());
    let mut pieces: Vec<StatePiece> = Vec::with_capacity(model_input.components.len());
    for component in &model_input.components {
        // A constant part reads nothing: it contributes its declared width of
        // the fill value wherever it sits in the concat.
        let Some(role) = component.role.as_deref() else {
            let width = component
                .dim
                .expect("a constant part is codec-checked for dim");
            pieces.push(fill_piece(width, component.fill, false, None, component));
            continue;
        };
        // An action-source part reads the model's own output, never an env
        // leaf: it binds the spec's actuator and consults nothing here.
        if component.source == PartSource::Action {
            pieces.push(previous_action_piece(
                component, role, &at, outputs, advisories,
            )?);
            continue;
        }
        // A custom encoding resolves structurally to its `base` here and the
        // host-side repack runs on its own slice, addressed by the resolved
        // piece widths this plan records — so it may sit at any offset of a
        // multi-part concat. What it can never be is absent: an optional part
        // the env does not declare has no zero form of the custom packing.
        let is_custom = component
            .encoding
            .as_ref()
            .is_some_and(|encoding| encoding.custom().is_some());
        if is_custom && component.optional {
            return Err(err(
                ErrorCode::Unsupported,
                format!(
                    "state role {}: a custom encoding cannot be optional (its host-side repack \
                     has no zero form)",
                    quoted(role)
                ),
            ));
        }
        let locus = format!("model input {at}: state role {}", quoted(role));
        let bound = bind(
            states_by_role,
            role,
            component.part.as_deref(),
            component.provenance.as_ref(),
            &format!("model input {at}"),
            "state role",
            "env",
            advisories,
        )?;
        let Some(bound) = bound else {
            // The role's data is present but under a kind this core can't read:
            // fail loud before the optional zero-fill silently degrades it. A
            // role the env genuinely lacks falls through to the optional branch.
            super::reject_referenced_unknown(role, &placement, unknown_roles)?;
            if component.optional {
                let width = fill_width(component, role, &at)?;
                for (name, axis) in [
                    ("axis_scale", component.axis_scale.as_deref()),
                    ("axis_offset", component.axis_offset.as_deref()),
                ] {
                    check_axis_width(axis, name, Some(width), &locus)?;
                }
                pieces.push(fill_piece(
                    width,
                    folded_fill(component),
                    true,
                    component.part.clone(),
                    component,
                ));
                continue;
            }
            return Err(err(
                ErrorCode::MissingRole,
                format!(
                    "model input {at} needs state role {}{} but the env offers {}",
                    quoted(role),
                    part_suffix(component.part.as_deref()),
                    quoted_leaf_keys(states_by_role)
                ),
            ));
        };
        let env_state: &EnvState = bound.feature;
        // Align the axes by name before anything else reads them. A model
        // that names a label the env lacks fills when optional (a named
        // axis, like a named part, never rebinds), else it is a mismatch.
        let gather = match label_gather(
            role,
            env_state.labels.as_deref(),
            component.labels.as_deref(),
            &at,
            "env leaf",
        ) {
            Ok(gather) => gather,
            Err(_) if component.optional && env_state.labels.is_some() => {
                let width = fill_width(component, role, &at)?;
                pieces.push(fill_piece(
                    width,
                    folded_fill(component),
                    true,
                    bound.part.clone(),
                    component,
                ));
                continue;
            }
            Err(error) => return Err(error),
        };
        // A custom encoding shadows to its `base` for the structural
        // negotiation; the host-side arm is never imported or run here (only a
        // trusted in-process resolve does). Validate the obs-side invariants the
        // platform can check without running the arm.
        let model_set = match component.encoding.as_ref() {
            Some(encoding) => {
                if let Some(custom) = encoding.custom() {
                    if custom.from_base.is_none() {
                        return Err(err(
                            ErrorCode::Unsupported,
                            format!(
                                "state role {}: an observation custom encoding needs from_base",
                                quoted(role)
                            ),
                        ));
                    }
                    // The repack preserves the base width, so an index (one
                    // element) never fits; a dim is legal exactly when it
                    // restates that width, which is how a concat part declares
                    // its own slice.
                    let base_dims = custom.base().dims();
                    if component.index.is_some()
                        || component.dim.is_some_and(|dim| dim != base_dims)
                    {
                        return Err(err(
                            ErrorCode::Unsupported,
                            format!(
                                "state role {}: a custom encoding keeps its base width; drop \
                                 index and set dim to {base_dims} or omit it",
                                quoted(role)
                            ),
                        ));
                    }
                }
                Some(encoding.accept_set())
            }
            None => None,
        };
        let (src_encoding, dst_encoding) =
            select_state_encoding(role, env_state.encoding.as_ref(), model_set.as_ref())?;
        // A post-rotation decodes the source into a matrix, which a sink has
        // none of (the codec already refuses a sink as the literal itself).
        if let Some(src) = src_encoding
            && src.is_sink()
            && component.post_rotate.is_some()
        {
            return Err(err(
                ErrorCode::EncodingMismatch,
                format!(
                    "state role {}: post_rotate needs a rotation to decode but the env \
                     declares encoding {}, a direction",
                    quoted(role),
                    quoted_encoding(Some(src))
                ),
            ));
        }
        let frame = check_geometry(
            Attr::Frame,
            role,
            env_state.frame.as_ref(),
            component.frame.as_ref(),
            advisories,
        )?;
        let provenance = check_provenance(
            role,
            env_state.provenance.as_ref(),
            component.provenance.as_ref(),
            advisories,
        )?;
        // When converting, the env feature's declared width must match the
        // native (source) encoding the raw value is in.
        if let (Some(src), Some(dst)) = (src_encoding, dst_encoding)
            && src != dst
            && let Some(env_dim) = env_state.dim
            && env_dim != src.dims()
        {
            return Err(err(
                ErrorCode::DimMismatch,
                format!(
                    "state role {}: env feature {} declares {env_dim} dims but \
                 encoding {} has {}",
                    quoted(role),
                    quoted(&env_state.source.to_string()),
                    quoted_encoding(Some(src)),
                    src.dims()
                ),
            ));
        }
        // Bounds-check the requested slice against the source width. The
        // width is the env feature's, unless a rotation conversion reshapes it
        // first (in which case the converted width applies). Without this an
        // out-of-range index or dim silently yields fewer values. A
        // post-rotation always goes through the matrix, so it reshapes to the
        // destination encoding even when both sides name the same one.
        let converts = matches!(
            (src_encoding, dst_encoding),
            (Some(src), Some(dst)) if src != dst || component.post_rotate.is_some()
        );
        let source_width = if converts {
            dst_encoding.map(|encoding| encoding.dims())
        } else if let Some(labels) = &component.labels {
            // Labels fix the width: the gather (or the identical order) yields
            // exactly one element per named axis.
            Some(labels.len() as u32)
        } else {
            env_state.dim
        };
        if let Some(width) = source_width {
            if let Some(index) = component.index {
                if index >= width {
                    return Err(err(
                        ErrorCode::SliceOutOfRange,
                        format!(
                            "state role {}: index {index} is out of range for the \
                         width-{width} source feature {}",
                            quoted(role),
                            quoted(&env_state.source.to_string())
                        ),
                    ));
                }
            } else if let Some(dim) = component.dim
                && dim > width
            {
                return Err(err(
                    ErrorCode::SliceOutOfRange,
                    format!(
                        "state role {}: requested {dim} dims but the source feature \
                     {} has width {width}",
                        quoted(role),
                        quoted(&env_state.source.to_string())
                    ),
                ));
            }
        }
        // The piece's resolved output width, when statically known: an index
        // keeps one element, a dim truncates to it, otherwise the source width
        // (the converted width when a rotation conversion reshapes it) stands.
        let width = match (component.index, component.dim) {
            (Some(_), _) => Some(1),
            (_, Some(dim)) => Some(dim),
            (None, None) => source_width,
        };
        for (name, axis) in [
            ("axis_scale", component.axis_scale.as_deref()),
            ("axis_offset", component.axis_offset.as_deref()),
        ] {
            check_axis_width(axis, name, width, &locus)?;
        }
        // The output axis names: the model's when it declared them (the
        // gather put the values in that order), else the env's, which the
        // identity read preserves.
        let labels = component
            .labels
            .clone()
            .or_else(|| env_state.labels.clone())
            .filter(|labels| width.is_none_or(|width| labels.len() == width as usize));
        pieces.push(StatePiece {
            source: env_state.source.clone(),
            src_offset: env_state.slice_offset,
            // src_dim is the slice width, meaningful only for a layout field
            // (where slice_offset is set); a whole-leaf state leaves it None so
            // the documented "used only when src_offset is set" invariant holds
            // (env_state.dim there is the advisory space width, not a slice).
            src_dim: env_state.slice_offset.and(env_state.dim),
            src_encoding,
            dst_encoding,
            post_rotate: component.post_rotate.clone(),
            // Labels fix the width the way a declared `dim` does.
            dim: component
                .dim
                .or_else(|| component.labels.as_ref().map(|labels| labels.len() as u32)),
            index: component.index,
            src_range: env_state.range,
            dst_range: component.range,
            scale: component.scale,
            offset: component.offset,
            axis_scale: component.axis_scale.clone(),
            axis_offset: component.axis_offset.clone(),
            gather,
            labels,
            src_labels: env_state.labels.clone(),
            fill: None,
            absent_role: false,
            previous: None,
            frame,
            provenance,
            part: bound.part,
            width,
        });
    }
    // Known only when every piece's width is (and no pathological spec overflows
    // the sum): a host-side shim can address a slice only in that case.
    let native_width = pieces.iter().try_fold(0u32, |total, piece| {
        piece.width.and_then(|width| total.checked_add(width))
    });
    Ok(StatePlan {
        placement,
        pieces,
        pad_to: model_input.pad_to,
        clip: model_input.clip,
        native_width,
        dtype: model_input.dtype.clone(),
        reshape: model_input.reshape.clone(),
        container: model_input.container,
    })
}

#[cfg(test)]
mod encoding_selection_tests {
    use super::*;

    fn set(json: &str) -> AcceptSet<RotationEncoding> {
        serde_json::from_str(json).expect("parse accept-set")
    }

    #[test]
    fn matching_single_encoding_does_not_convert() {
        let (env, model) = (set(r#""quat_xyzw""#), set(r#""quat_xyzw""#));
        let (src, dst) =
            select_state_encoding("proprio/rot", Some(&env), Some(&model)).expect("ok");
        assert_eq!(src, Some(RotationEncoding::QuatXyzw));
        assert_eq!(dst, Some(RotationEncoding::QuatXyzw));
    }

    #[test]
    fn differing_single_encodings_convert_native_to_target() {
        let (env, model) = (set(r#""quat_xyzw""#), set(r#""rot6d""#));
        let (src, dst) =
            select_state_encoding("proprio/rot", Some(&env), Some(&model)).expect("ok");
        assert_eq!(src, Some(RotationEncoding::QuatXyzw)); // env native = source
        assert_eq!(dst, Some(RotationEncoding::Rot6d)); // model target
    }

    #[test]
    fn prefers_no_conversion_when_model_accepts_the_native() {
        // The model lists rot6d first but also accepts the env's native quat:
        // take the native (no conversion), even though rot6d is preferred.
        let (env, model) = (set(r#""quat_xyzw""#), set(r#"["rot6d", "quat_xyzw"]"#));
        let (src, dst) =
            select_state_encoding("proprio/rot", Some(&env), Some(&model)).expect("ok");
        assert_eq!(src, Some(RotationEncoding::QuatXyzw));
        assert_eq!(dst, Some(RotationEncoding::QuatXyzw));
    }

    #[test]
    fn falls_back_past_an_unrecognized_preference() {
        // Scenario 1: a model trained on a future encoding lists it first but
        // accepts the frozen env's native as a fallback — the unknown is
        // skipped at resolve, the native chosen, no runtime error.
        let (env, model) = (set(r#""quat_xyzw""#), set(r#"["rot10d", "quat_xyzw"]"#));
        let (src, dst) =
            select_state_encoding("proprio/rot", Some(&env), Some(&model)).expect("ok");
        assert_eq!(src, Some(RotationEncoding::QuatXyzw));
        assert_eq!(dst, Some(RotationEncoding::QuatXyzw));
    }

    #[test]
    fn wholly_unrecognized_declaration_errors_at_resolve() {
        // Graceful degradation's loud edge: a side that names only an unknown
        // encoding parses, but resolves to a typed error (not a silent pass).
        let (env, model) = (set(r#""quat_xyzw""#), set(r#""rot10d""#));
        let error =
            select_state_encoding("proprio/rot", Some(&env), Some(&model)).expect_err("err");
        assert_eq!(error.code, ErrorCode::EncodingMismatch);
        assert!(
            error.message.contains("unrecognized"),
            "got: {}",
            error.message
        );
    }

    #[test]
    fn the_gravity_sink_converts_in_but_never_out() {
        // Any rotation projects to gravity...
        let (env, model) = (set(r#""quat_wxyz""#), set(r#""gravity_xyz""#));
        let (src, dst) =
            select_state_encoding("proprio/base_rot", Some(&env), Some(&model)).expect("ok");
        assert_eq!(src, Some(RotationEncoding::QuatWxyz));
        assert_eq!(dst, Some(RotationEncoding::GravityXyz));
        // ...gravity reads as itself...
        let (env, model) = (
            set(r#""gravity_xyz""#),
            set(r#"["quat_wxyz", "gravity_xyz"]"#),
        );
        let (src, dst) =
            select_state_encoding("proprio/base_rot", Some(&env), Some(&model)).expect("ok");
        assert_eq!(
            (src, dst),
            (
                Some(RotationEncoding::GravityXyz),
                Some(RotationEncoding::GravityXyz)
            )
        );
        // ...and never becomes a rotation again.
        let (env, model) = (set(r#""gravity_xyz""#), set(r#""quat_wxyz""#));
        let error =
            select_state_encoding("proprio/base_rot", Some(&env), Some(&model)).expect_err("err");
        assert_eq!(error.code, ErrorCode::EncodingMismatch);
        assert!(
            error.message.contains("a direction, not a rotation"),
            "{}",
            error.message
        );
    }

    #[test]
    fn one_sided_declaration_is_a_mismatch() {
        let model = set(r#""quat_xyzw""#);
        let error = select_state_encoding("proprio/rot", None, Some(&model)).expect_err("err");
        assert_eq!(error.code, ErrorCode::EncodingMismatch);
        assert!(
            error.message.contains("both sides"),
            "got: {}",
            error.message
        );
    }

    #[test]
    fn neither_side_declares_means_no_rotation() {
        let (src, dst) = select_state_encoding("proprio/rot", None, None).expect("ok");
        assert_eq!(src, None);
        assert_eq!(dst, None);
    }
}
