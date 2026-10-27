//! Map the model's action layout onto the env's, role by role.

#[allow(unused_imports)]
use crate::prelude::*;

use alloc::collections::{BTreeMap, BTreeSet};

use crate::spec::projection::Projection;
use crate::spec::role::bind::{
    LeafKey, Projections, Result, check_geometry, err, index_by_key, seek,
};
use crate::spec::role::error::{Advisory, ErrorCode};
use crate::spec::role::fmt::{missing_role_hint, quoted, quoted_encoding, quoted_leaf_keys};
use crate::spec::role::frames::Attr;
use crate::spec::role::input::Actuator;
use crate::spec::role::plan::{ActionPlan, ActionSegment};
use crate::spec::role::roles::registry;
use crate::spec::role::state::{check_axis_width, label_permutation, part_suffix};

/// Enforce a registered role's `Fixed` dim law on a model actuator (the env side
/// is checked at `join`). An ad-hoc/`Variable`/`ByEncoding` role is left alone.
fn check_role_dim_law(role: &str, dim: u32) -> Result<()> {
    if let Some(def) = registry::role_def(role)
        && let registry::DimLaw::Fixed(expected) = def.dim
        && dim != expected
    {
        return Err(err(
            ErrorCode::DimMismatch,
            format!(
                "action role {}: model declares {dim} dims but the role is {expected}-D by convention",
                quoted(role)
            ),
        ));
    }
    Ok(())
}

/// The scatter a model label tuple implies against an env actuator: per env
/// axis, the model index that drives it. `None` when the model names no
/// labels or the orders agree. The two tuples must name the same set.
fn label_scatter(role: &str, env: &Actuator, model: &Actuator) -> Result<Option<Vec<u32>>> {
    let Some(model_labels) = &model.labels else {
        return Ok(None);
    };
    let Some(env_labels) = &env.labels else {
        return Err(err(
            ErrorCode::LabelMismatch,
            format!(
                "action role {}: the model names labels {:?} but the env actuator declares \
                 none; label the env actuator (labels=) so the axes can be aligned",
                quoted(role),
                model_labels
            ),
        ));
    };
    label_permutation(env_labels, model_labels).map_err(|diff| {
        err(
            ErrorCode::LabelMismatch,
            format!(
                "action role {}: the model names labels {:?} but {}; the two must name the \
                 same set",
                quoted(role),
                model_labels,
                diff.describe("env actuator", "model"),
            ),
        )
    })
}

/// Validate that a model/env action component pairing is convertible. `role` is
/// the matched (always present) role both sides share.
fn check_action_dims(model: &Actuator, env: &Actuator, role: &str) -> Result<()> {
    check_role_dim_law(role, model.dim)?;
    // A custom action encoding shadows to its `base` for the structural checks;
    // the host-side arm (`to_base`) is never imported or run here. Validate that
    // a model-side custom carries the action arm it needs.
    if let (Some(custom), Some(base)) = (model.custom, model.encoding) {
        if !custom.to_base {
            return Err(err(
                ErrorCode::Unsupported,
                format!(
                    "action role {}: an action custom encoding needs to_base",
                    quoted(role)
                ),
            ));
        }
        if model.dim != base.width() {
            return Err(err(
                ErrorCode::DimMismatch,
                format!(
                    "action role {}: a custom encoding preserves its base width {} but the actuator declares dim {}",
                    quoted(role),
                    base.width(),
                    model.dim
                ),
            ));
        }
    }
    let model_encoding = model.encoding;
    let env_encoding = env.encoding;
    let converting = match (model_encoding, env_encoding) {
        (Some(model_encoding), Some(env_encoding)) => model_encoding != env_encoding,
        _ => false,
    };
    if converting {
        let model_encoding = model_encoding.expect("checked above");
        let env_encoding = env_encoding.expect("checked above");
        if model.dim != model_encoding.width() || env.dim != env_encoding.width() {
            return Err(err(
                ErrorCode::DimMismatch,
                format!(
                    "action role {}: dims {}->{} do not match encodings {}->{}",
                    quoted(role),
                    model.dim,
                    env.dim,
                    quoted_encoding(Some(model_encoding)),
                    quoted_encoding(Some(env_encoding))
                ),
            ));
        }
        // Model per-axis corrections name the model's own axes, which no longer
        // exist once the rotation is re-encoded for the env.
        if model.axis_scale.is_some() || model.axis_offset.is_some() {
            return Err(err(
                ErrorCode::Unsupported,
                format!(
                    "action role {}: the model's axis_scale/axis_offset cannot apply across a \
                 rotation conversion {}->{}; use a scalar scale/offset or declare them on the env",
                    quoted(role),
                    quoted_encoding(Some(model_encoding)),
                    quoted_encoding(Some(env_encoding))
                ),
            ));
        }
        return Ok(());
    }
    if model_encoding != env_encoding && (model_encoding.is_none() || env_encoding.is_none()) {
        return Err(err(
            ErrorCode::EncodingMismatch,
            format!(
                "action role {}: cannot convert encoding {} to {}; both sides must \
             declare a rotation encoding",
                quoted(role),
                quoted_encoding(model_encoding),
                quoted_encoding(env_encoding)
            ),
        ));
    }
    if model.dim != env.dim {
        return Err(err(
            ErrorCode::DimMismatch,
            format!(
                "action role {}: model outputs {} dims but the env expects {}",
                quoted(role),
                model.dim,
                env.dim
            ),
        ));
    }
    Ok(())
}

/// A model output keyed for [`index_by_key`]: `(role, part, no provenance,
/// (start offset, actuator))`.
type PlacedOutput<'a> = (
    &'a str,
    Option<&'a str>,
    Option<&'a str>,
    (u32, &'a Actuator),
);

/// The model's roled outputs keyed by `(role, part)`, each with its start
/// offset in the action vector, plus the vector's width. Shared by the action
/// planner (the env actuator seeks the output that drives it) and the state
/// planner (an action-source part seeks the output it reads back).
pub(super) struct ModelOutputs<'a> {
    pub by_key: BTreeMap<LeafKey, (u32, &'a Actuator)>,
    pub in_dim: u32,
}

pub(super) fn index_outputs(model: &[Actuator]) -> Result<ModelOutputs<'_>> {
    // A role-less (opaque) model actuator emits dims the env ignores: it
    // advances the cursor but is matched by nothing, so only roled components
    // are keyed.
    let mut placed: Vec<PlacedOutput<'_>> = Vec::new();
    let mut cursor: u32 = 0;
    for component in model {
        if let Some(role) = &component.role {
            placed.push((role, component.part.as_deref(), None, (cursor, component)));
        }
        cursor = cursor.checked_add(component.dim).ok_or_else(|| {
            err(
                ErrorCode::DimMismatch,
                "model action component dims overflow u32".to_owned(),
            )
        })?;
    }
    Ok(ModelOutputs {
        by_key: index_by_key(placed.into_iter(), "model action")?,
        in_dim: cursor,
    })
}

/// The roled model outputs no segment of `plan` reads, as `(role, part,
/// start, dim)` in model output order: what binding them would discard.
pub(super) fn unconsumed<'a>(
    outputs: &ModelOutputs<'a>,
    plan: &ActionPlan,
) -> Vec<(&'a str, Option<&'a str>, u32, u32)> {
    let read: BTreeSet<u32> = plan
        .segments
        .iter()
        .filter(|segment| segment.fill.is_none())
        .map(|segment| segment.start)
        .collect();
    let mut out: Vec<_> = outputs
        .by_key
        .values()
        .filter(|(start, _)| !read.contains(start))
        .map(|(start, actuator)| {
            (
                actuator.role.as_deref().unwrap_or_default(),
                actuator.part.as_deref(),
                *start,
                actuator.dim,
            )
        })
        .collect();
    out.sort_by_key(|(_, _, start, _)| *start);
    out
}

pub(super) fn plan_action(
    env: &[Actuator],
    clip: Option<(f64, f64)>,
    outputs: &ModelOutputs<'_>,
    advisories: &mut Vec<Advisory>,
    projections: &mut Projections,
) -> Result<ActionPlan> {
    let in_dim = outputs.in_dim;
    let offsets = &outputs.by_key;

    let mut segments: Vec<ActionSegment> = Vec::with_capacity(env.len());
    let mut seen_env: BTreeMap<LeafKey, ()> = BTreeMap::new();
    for env_component in env {
        // A role-less (opaque) env actuator occupies its dims with a constant
        // fill, matched by no model output (the action-side mirror of a role-less
        // Field). It reads nothing from the model.
        let Some(role) = &env_component.role else {
            segments.push(ActionSegment {
                role: None,
                start: 0,
                stop: 0,
                src_encoding: None,
                dst_encoding: None,
                src_range: None,
                dst_range: None,
                model_scale: None,
                model_offset: None,
                model_axis_scale: None,
                model_axis_offset: None,
                model_invert: false,
                model_threshold: None,
                scale: None,
                offset: None,
                axis_scale: None,
                axis_offset: None,
                invert: false,
                threshold: None,
                scatter: None,
                labels: None,
                model_labels: None,
                binarize: false,
                clip: None,
                frame: None,
                reference: None,
                part: None,
                fill: Some((env_component.dim, env_component.fill)),
                flip: None,
                unit_factor: None,
            });
            continue;
        };
        let env_key: LeafKey = (role.clone(), env_component.part.clone(), None);
        if seen_env.insert(env_key.clone(), ()).is_some() {
            // Mirror the model-side dedup above (and the env-side StateLayout
            // role check): a role repeated in the env layout would resolve
            // every copy against the same model slice, building the env action
            // by repetition instead of a real mapping.
            return Err(err(
                ErrorCode::Duplicate,
                match &env_key.1 {
                    Some(part) => format!(
                        "duplicate env action role {} under part {}",
                        quoted(&env_key.0),
                        quoted(part)
                    ),
                    None => format!("duplicate env action role {}", quoted(&env_key.0)),
                },
            ));
        }
        // The env actuator is the seeker here: it looks for the model output
        // that drives it, under the same part rules a model input follows.
        let bound = seek(
            offsets,
            role,
            env_component.part.as_deref(),
            None,
            "env action",
            "role",
            "model",
            advisories,
        )?;
        let Some(bound) = bound else {
            // An optional roled actuator the model does not output falls back to
            // its constant `fill` -- the action-side mirror of a model input's
            // `optional` zero-fill, reusing the opaque branch's fill segment. The
            // role is kept on the segment so `describe` can flag the fabrication;
            // apply ignores it once `fill` is set.
            if env_component.optional {
                segments.push(ActionSegment {
                    role: Some(role.clone()),
                    start: 0,
                    stop: 0,
                    src_encoding: None,
                    dst_encoding: None,
                    src_range: None,
                    dst_range: None,
                    model_scale: None,
                    model_offset: None,
                    model_axis_scale: None,
                    model_axis_offset: None,
                    model_invert: false,
                    model_threshold: None,
                    scale: None,
                    offset: None,
                    axis_scale: None,
                    axis_offset: None,
                    invert: false,
                    threshold: None,
                    scatter: None,
                    labels: env_component.labels.clone(),
                    model_labels: None,
                    binarize: false,
                    clip: None,
                    frame: None,
                    reference: None,
                    part: env_component.part.clone(),
                    fill: Some((env_component.dim, env_component.fill)),
                    flip: None,
                    unit_factor: None,
                });
                continue;
            }
            return Err(err(
                ErrorCode::MissingRole,
                format!(
                    "env action needs role {}{} but the model only outputs {}{}",
                    quoted(role),
                    part_suffix(env_component.part.as_deref()),
                    quoted_leaf_keys(offsets),
                    missing_role_hint(role, offsets.keys().map(|(r, _, _)| r.as_str()))
                ),
            ));
        };
        let (start, model_component) = *bound.feature;
        let scatter = label_scatter(role, env_component, model_component)?;
        check_action_dims(model_component, env_component, role)?;
        // Each side's per-axis vector runs in that side's own label order, so
        // it must be exactly that side's width.
        let locus = format!("action role {}", quoted(role));
        for (name, axis, width) in [
            (
                "the model's axis_scale",
                model_component.axis_scale.as_deref(),
                model_component.dim,
            ),
            (
                "the model's axis_offset",
                model_component.axis_offset.as_deref(),
                model_component.dim,
            ),
            (
                "the env's axis_scale",
                env_component.axis_scale.as_deref(),
                env_component.dim,
            ),
            (
                "the env's axis_offset",
                env_component.axis_offset.as_deref(),
                env_component.dim,
            ),
        ] {
            check_axis_width(axis, name, Some(width), &locus)?;
        }
        // clip is an env-side clamp to the env actuator's range; it has no meaning
        // on the model output (whose range is a mapping source, not a final bound).
        if model_component.clip {
            return Err(err(
                ErrorCode::Unsupported,
                format!(
                    "action role {}: clip is an env-side clamp; declare it on the env actuator",
                    quoted(role)
                ),
            ));
        }
        let binarize = env_component.binary || model_component.binary;
        // threshold is a binary decision boundary on either side; without a binary
        // snap it would silently become a constant offset on a continuous action.
        if (env_component.threshold.is_some() || model_component.threshold.is_some()) && !binarize {
            return Err(err(
                ErrorCode::Unsupported,
                format!(
                    "action role {}: threshold requires a binary component (set binary=true)",
                    quoted(role)
                ),
            ));
        }
        // clip-to-range needs a range to clamp to; a clip without one is an
        // authoring contradiction (Python __post_init__ catches it too).
        let clip = if env_component.clip {
            let Some(range) = env_component.range else {
                return Err(err(
                    ErrorCode::Unsupported,
                    format!(
                        "action role {}: clip=true clamps to range, but no range is declared",
                        quoted(role)
                    ),
                ));
            };
            Some(range)
        } else {
            None
        };
        // An absolute pose command carries a frame; a delta carries the
        // reference pose it is integrated against. Same rule, same errors.
        let frame = check_geometry(
            Attr::Frame,
            role,
            env_component.frame.as_ref(),
            model_component.frame.as_ref(),
            advisories,
        )?;
        let reference = check_geometry(
            Attr::Reference,
            role,
            env_component.reference.as_ref(),
            model_component.reference.as_ref(),
            advisories,
        )?;
        let same_range = model_component.range == env_component.range;
        let flip = check_polarity(
            role,
            env_component,
            model_component,
            advisories,
            projections,
        )?;
        let unit_factor = check_unit(
            role,
            env_component,
            model_component,
            same_range,
            advisories,
            projections,
        )?;
        segments.push(ActionSegment {
            role: Some(role.clone()),
            start,
            stop: start + model_component.dim,
            src_encoding: model_component.encoding,
            dst_encoding: env_component.encoding,
            src_range: if same_range {
                None
            } else {
                model_component.range
            },
            dst_range: if same_range {
                None
            } else {
                env_component.range
            },
            // Corrections compose as literal transforms after the format bridge:
            // model-side first (bridge the model's output convention toward the
            // env), then env-side. A model that emits the env's convention leaves
            // its side unset; a shared env declares its quirk once on its side.
            model_scale: model_component.scale,
            model_offset: model_component.offset,
            model_axis_scale: model_component.axis_scale.clone(),
            model_axis_offset: model_component.axis_offset.clone(),
            model_invert: model_component.invert,
            model_threshold: model_component.threshold,
            unit_factor,
            flip,
            scale: env_component.scale,
            offset: env_component.offset,
            axis_scale: env_component.axis_scale.clone(),
            axis_offset: env_component.axis_offset.clone(),
            invert: env_component.invert,
            threshold: env_component.threshold,
            scatter,
            labels: env_component.labels.clone(),
            // With no scatter the model writes the env's axes in the env's
            // order, so the env's names apply to the model's values too.
            model_labels: model_component
                .labels
                .clone()
                .or_else(|| env_component.labels.clone()),
            binarize,
            clip,
            frame,
            reference,
            part: bound.part,
            fill: None,
        });
    }
    Ok(ActionPlan {
        segments,
        clip,
        in_dim,
    })
}

/// The gripper polarity check: the model's command, through both sides' own
/// sign corrections, must reach the env in the env's declared polarity. A
/// difference is the named projection `flip_gripper`, which reflects the
/// value in the env's range; returned as that reflection's sum.
fn check_polarity(
    role: &str,
    env: &Actuator,
    model: &Actuator,
    advisories: &mut Vec<Advisory>,
    projections: &mut Projections,
) -> Result<Option<f64>> {
    let (model_polarity, env_polarity) = match (model.polarity, env.polarity) {
        (Some(model_polarity), Some(env_polarity)) => (model_polarity, env_polarity),
        (None, None) => return Ok(None),
        (model_polarity, _) => {
            advisories.push(Advisory::info(format!(
                "action role {}: only the {} declares a gripper polarity, so it is not checked",
                quoted(role),
                if model_polarity.is_some() {
                    "model"
                } else {
                    "env"
                }
            )));
            return Ok(None);
        }
    };
    let delivered = model_polarity
        .through(model.invert, model.scale)
        .through(env.invert, env.scale);
    if delivered == env_polarity {
        return Ok(None);
    }
    projections.apply(
        Projection::FlipGripper,
        format!(
            "action role {}: the model's {} command is reversed into the env's {}",
            quoted(role),
            quoted(delivered.as_str()),
            quoted(env_polarity.as_str())
        ),
        ErrorCode::ConventionMismatch,
        format!(
            "action role {}: the model's gripper command reaches the env as {} (polarity {} \
             through both sides' invert/scale) but the env's gripper is {}",
            quoted(role),
            quoted(delivered.as_str()),
            quoted(model_polarity.as_str()),
            quoted(env_polarity.as_str())
        ),
    )?;
    Ok(Some(env.range.map_or(0.0, |(low, high)| low + high)))
}

/// The delta unit check: one model unit, through the range map and both
/// sides' scales, must be the same physical step as the env's. A difference
/// is the named projection `convert_units`; returned as its factor.
fn check_unit(
    role: &str,
    env: &Actuator,
    model: &Actuator,
    same_range: bool,
    advisories: &mut Vec<Advisory>,
    projections: &mut Projections,
) -> Result<Option<f64>> {
    let (model_unit, env_unit) = match (model.unit, env.unit) {
        (Some(model_unit), Some(env_unit)) => (model_unit, env_unit),
        (None, None) => return Ok(None),
        (model_unit, _) => {
            advisories.push(Advisory::info(format!(
                "action role {}: only the {} declares a unit, so it is not checked",
                quoted(role),
                if model_unit.is_some() { "model" } else { "env" }
            )));
            return Ok(None);
        }
    };
    let span = |(low, high): (f64, f64)| high - low;
    let range_gain = match (model.range, env.range) {
        (Some(src), Some(dst)) if !same_range && span(src) != 0.0 => span(dst) / span(src),
        _ => 1.0,
    };
    let gain = range_gain * model.scale.unwrap_or(1.0).abs() * env.scale.unwrap_or(1.0).abs();
    let delivered = env_unit * gain;
    if ((model_unit - delivered) / model_unit).abs() <= 1e-9 {
        return Ok(None);
    }
    let factor = model_unit / delivered;
    projections.apply(
        Projection::ConvertUnits,
        format!(
            "action role {}: model unit {model_unit} rescaled by {factor} to the env's {env_unit}",
            quoted(role)
        ),
        ErrorCode::ConventionMismatch,
        format!(
            "action role {}: one model unit is {model_unit} but reaches the env as {delivered} \
             (env unit {env_unit} through the range map and scales)",
            quoted(role)
        ),
    )?;
    Ok(Some(factor))
}

#[cfg(test)]
mod tests {
    use super::{index_outputs, plan_action};
    use crate::spec::role::bind::{Projections, Result};
    use crate::spec::role::input::{Actuator, CustomArms};
    use crate::spec::role::plan::ActionPlan;
    use crate::spec::role::{Advisory, ErrorCode};
    use crate::spec::rotation::Rotation;

    fn plan(
        model: &[Actuator],
        env: &[Actuator],
        advisories: &mut Vec<Advisory>,
    ) -> Result<ActionPlan> {
        plan_action(
            env,
            None,
            &index_outputs(model)?,
            advisories,
            &mut Projections::allowing(&[]),
        )
    }

    fn component(role: &str) -> Actuator {
        Actuator {
            role: Some(role.to_owned()),
            dim: 1,
            encoding: None,
            custom: None,
            range: None,
            binary: false,
            scale: None,
            offset: None,
            axis_scale: None,
            axis_offset: None,
            invert: false,
            threshold: None,
            clip: false,
            fill: 0.0,
            optional: false,
            frame: None,
            part: None,
            labels: None,
            reference: None,
            polarity: None,
            unit: None,
        }
    }

    fn opaque(dim: u32, fill: f64) -> Actuator {
        Actuator {
            role: None,
            dim,
            encoding: None,
            custom: None,
            range: None,
            binary: false,
            scale: None,
            offset: None,
            axis_scale: None,
            axis_offset: None,
            invert: false,
            threshold: None,
            clip: false,
            fill,
            optional: false,
            frame: None,
            part: None,
            labels: None,
            reference: None,
            polarity: None,
            unit: None,
        }
    }

    fn layout(components: Vec<Actuator>) -> Vec<Actuator> {
        components
    }

    #[test]
    fn rejects_threshold_without_a_binary_component() {
        let model = layout(vec![component("action/gripper")]);
        let mut gripper = component("action/gripper");
        gripper.threshold = Some(0.5);
        let env = layout(vec![gripper]);
        let error = plan(&model, &env, &mut Vec::new()).unwrap_err();
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert!(error.message.contains("threshold requires a binary"));
    }

    #[test]
    fn rejects_custom_action_encoding_with_a_non_base_width() {
        let mut model_rot = component("custom/rot");
        model_rot.dim = 3;
        model_rot.encoding = Some(Rotation::Rot6d);
        model_rot.custom = Some(CustomArms {
            from_base: false,
            to_base: true,
        });
        let mut env_rot = component("custom/rot");
        env_rot.dim = 6;
        env_rot.encoding = Some(Rotation::Rot6d);
        let error = plan(
            &layout(vec![model_rot]),
            &layout(vec![env_rot]),
            &mut Vec::new(),
        )
        .unwrap_err();
        assert_eq!(error.code, ErrorCode::DimMismatch);
        assert!(
            error.message.contains("preserves its base width"),
            "{}",
            error.message
        );
    }

    #[test]
    fn rejects_model_axis_corrections_across_a_rotation_conversion() {
        let rotation = |dim, encoding| {
            let mut rot = component("custom/rot");
            rot.dim = dim;
            rot.encoding = Some(encoding);
            rot
        };
        let mut model_rot = rotation(4, Rotation::QuatXyzw);
        model_rot.axis_scale = Some(vec![1.0, 1.0, 1.0, 10.0]);
        let env_rot = rotation(3, Rotation::EulerXyz);
        let error = plan(
            &layout(vec![model_rot.clone()]),
            &layout(vec![env_rot]),
            &mut Vec::new(),
        )
        .unwrap_err();
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert!(
            error
                .message
                .contains("cannot apply across a rotation conversion"),
            "{}",
            error.message
        );
        // The same encoding on both sides keeps the model's axes meaningful.
        let same = rotation(4, Rotation::QuatXyzw);
        assert!(
            plan(
                &layout(vec![model_rot]),
                &layout(vec![same]),
                &mut Vec::new()
            )
            .is_ok()
        );
    }

    #[test]
    fn threshold_with_a_binary_component_is_accepted() {
        let model = layout(vec![component("action/gripper")]);
        let mut gripper = component("action/gripper");
        gripper.threshold = Some(0.5);
        gripper.binary = true;
        let env = layout(vec![gripper]);
        assert!(plan(&model, &env, &mut Vec::new()).is_ok());
    }

    #[test]
    fn model_side_corrections_resolve_onto_the_segment() {
        // A model declares its own output convention (e.g. a gripper polarity
        // flip): it resolves onto the segment's model-side fields rather than
        // being rejected, so no imperative bridge is needed in predict().
        let mut model_gripper = component("action/gripper");
        model_gripper.scale = Some(2.0);
        model_gripper.invert = true;
        let model = layout(vec![model_gripper]);
        let env = layout(vec![component("action/gripper")]);
        let plan = plan(&model, &env, &mut Vec::new()).unwrap();
        assert_eq!(plan.segments[0].model_scale, Some(2.0));
        assert!(plan.segments[0].model_invert);
        // The env declared no corrections, so its side stays unset.
        assert_eq!(plan.segments[0].scale, None);
        assert!(!plan.segments[0].invert);
    }

    #[test]
    fn clip_resolves_to_the_env_actuator_range() {
        let model = layout(vec![component("action/x")]);
        let mut env_x = component("action/x");
        env_x.range = Some((-1.5, 1.5));
        env_x.clip = true;
        let env = layout(vec![env_x]);
        let plan = plan(&model, &env, &mut Vec::new()).unwrap();
        assert_eq!(plan.segments[0].clip, Some((-1.5, 1.5)));
    }

    #[test]
    fn rejects_clip_without_a_range() {
        let model = layout(vec![component("action/x")]);
        let mut env_x = component("action/x");
        env_x.clip = true; // no range declared
        let env = layout(vec![env_x]);
        let error = plan(&model, &env, &mut Vec::new()).unwrap_err();
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert!(
            error.message.contains("clamps to range"),
            "{}",
            error.message
        );
    }

    #[test]
    fn rejects_clip_declared_on_the_model_component() {
        // clip stays env-side only (it clamps to the env actuator's range).
        let mut model_x = component("action/x");
        model_x.clip = true;
        let model = layout(vec![model_x]);
        let env = layout(vec![component("action/x")]);
        let error = plan(&model, &env, &mut Vec::new()).unwrap_err();
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert!(
            error.message.contains("env-side clamp"),
            "{}",
            error.message
        );
    }

    #[test]
    fn opaque_env_actuator_resolves_to_a_fill_segment() {
        // The env requires a dim no model produces (e.g. a control-mode selector);
        // a role-less actuator resolves to an opaque fill, not a MissingRole error.
        let model = layout(vec![component("action/gripper")]);
        let env = layout(vec![component("action/gripper"), opaque(2, 0.25)]);
        let plan = plan(&model, &env, &mut Vec::new()).unwrap();
        assert_eq!(plan.in_dim, 1); // the model only outputs the gripper dim
        assert_eq!(plan.segments.len(), 2);
        assert!(plan.segments[0].role.is_some());
        assert_eq!(plan.segments[1].role, None);
        assert_eq!(plan.segments[1].fill, Some((2, 0.25)));
    }

    #[test]
    fn optional_roled_actuator_with_no_model_output_resolves_to_a_fill_segment() {
        // An optional role the model omits falls back to its fill instead of a
        // MissingRole error -- the action-side mirror of an optional model input.
        // The role is kept on the segment (for `describe`); apply ignores it.
        let model = layout(vec![component("action/gripper")]);
        let mut base = component("x/base_motion");
        base.dim = 4;
        base.optional = true;
        base.fill = 0.0;
        let env = layout(vec![component("action/gripper"), base]);
        let plan = plan(&model, &env, &mut Vec::new()).unwrap();
        assert_eq!(plan.in_dim, 1); // the model still only outputs the gripper dim
        assert_eq!(plan.segments.len(), 2);
        assert_eq!(plan.segments[1].role.as_deref(), Some("x/base_motion"));
        assert_eq!(plan.segments[1].fill, Some((4, 0.0)));
    }

    #[test]
    fn optional_role_the_model_does_output_is_driven_normally() {
        // optional only changes the missing case: a model that outputs the role
        // maps it like any other actuator (no fill).
        let model = layout(vec![component("action/gripper")]);
        let mut env_gripper = component("action/gripper");
        env_gripper.optional = true;
        let env = layout(vec![env_gripper]);
        let plan = plan(&model, &env, &mut Vec::new()).unwrap();
        assert_eq!(plan.segments.len(), 1);
        assert_eq!(plan.segments[0].fill, None);
        assert_eq!(plan.in_dim, 1);
    }

    #[test]
    fn non_optional_unmatched_role_still_errors() {
        let model = layout(vec![component("action/gripper")]);
        let env = layout(vec![
            component("action/gripper"),
            component("x/base_motion"),
        ]);
        let error = plan(&model, &env, &mut Vec::new()).unwrap_err();
        assert_eq!(error.code, ErrorCode::MissingRole);
        assert!(error.message.contains("env action needs role"));
    }
}
