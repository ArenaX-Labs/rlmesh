//! Map the model's action layout onto the env's, role by role.

use std::collections::BTreeMap;

use super::{Result, err};
use crate::error::ErrorCode;
use crate::fmt::{quoted, quoted_encoding, quoted_keys};
use crate::plans::{ActionPlan, ActionSegment};
use crate::spec::{Action, Actuator};

/// Enforce a registered role's `Fixed` dim law on a model actuator (the env side
/// is checked at `join`). An ad-hoc/`Variable`/`ByEncoding` role is left alone.
fn check_role_dim_law(role: &str, dim: u32) -> Result<()> {
    if let Some(def) = crate::roles::registry::role_def(role)
        && let crate::roles::registry::DimLaw::Fixed(expected) = def.dim
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

/// Validate that a model/env action component pairing is convertible. `role` is
/// the matched (always present) role both sides share.
fn check_action_dims(model: &Actuator, env: &Actuator, role: &str) -> Result<()> {
    check_role_dim_law(role, model.dim)?;
    let converting = match (model.encoding, env.encoding) {
        (Some(model_encoding), Some(env_encoding)) => model_encoding != env_encoding,
        _ => false,
    };
    if converting {
        let model_encoding = model.encoding.expect("checked above");
        let env_encoding = env.encoding.expect("checked above");
        if model.dim != model_encoding.dims() || env.dim != env_encoding.dims() {
            return Err(err(
                ErrorCode::DimMismatch,
                format!(
                    "action role {}: dims {}->{} do not match encodings {}->{}",
                    quoted(role),
                    model.dim,
                    env.dim,
                    quoted_encoding(model.encoding),
                    quoted_encoding(env.encoding)
                ),
            ));
        }
        return Ok(());
    }
    if model.encoding != env.encoding && (model.encoding.is_none() || env.encoding.is_none()) {
        return Err(err(
            ErrorCode::EncodingMismatch,
            format!(
                "action role {}: cannot convert encoding {} to {}; both sides must \
             declare a rotation encoding",
                quoted(role),
                quoted_encoding(model.encoding),
                quoted_encoding(env.encoding)
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

pub(super) fn plan_action(model: &Action, env: &Action) -> Result<ActionPlan> {
    // `clip` is an env-side actuator clamp on the assembled action vector (read
    // as `env.clip` below); a clip declared on the *model* action layout is
    // silently dropped. Reject it, mirroring the per-component scale/invert/
    // threshold guard that likewise rejects an env-side knob on the model.
    if model.clip.is_some() {
        return Err(err(
            ErrorCode::Unsupported,
            "clip is an env-side actuator clamp; the model action declaration must leave it unset"
                .to_owned(),
        ));
    }
    let mut offsets: BTreeMap<String, (u32, &Actuator)> = BTreeMap::new();
    let mut cursor: u32 = 0;
    for component in &model.components {
        // A role-less (opaque) model actuator emits dims the env ignores: it
        // advances the cursor but is matched by nothing, so only roled components
        // join the offset map.
        if let Some(role) = &component.role {
            if offsets.contains_key(role) {
                return Err(err(
                    ErrorCode::Duplicate,
                    format!("duplicate model action role {}", quoted(role)),
                ));
            }
            offsets.insert(role.clone(), (cursor, component));
        }
        cursor = cursor.checked_add(component.dim).ok_or_else(|| {
            err(
                ErrorCode::DimMismatch,
                "model action component dims overflow u32".to_owned(),
            )
        })?;
    }
    let in_dim = cursor;

    let mut segments: Vec<ActionSegment> = Vec::with_capacity(env.components.len());
    let mut seen_env_roles: BTreeMap<&str, ()> = BTreeMap::new();
    for env_component in &env.components {
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
                model_invert: false,
                model_threshold: None,
                scale: None,
                invert: false,
                threshold: None,
                binarize: false,
                clip: None,
                fill: Some((env_component.dim, env_component.fill)),
            });
            continue;
        };
        if seen_env_roles.insert(role.as_str(), ()).is_some() {
            // Mirror the model-side dedup above (and the env-side StateLayout
            // role check): a role repeated in the env layout would resolve
            // every copy against the same model slice, building the env action
            // by repetition instead of a real mapping.
            return Err(err(
                ErrorCode::Duplicate,
                format!("duplicate env action role {}", quoted(role)),
            ));
        }
        let Some(&(start, model_component)) = offsets.get(role) else {
            return Err(err(
                ErrorCode::MissingRole,
                format!(
                    "env action needs role {} but the model only outputs {}",
                    quoted(role),
                    quoted_keys(&offsets)
                ),
            ));
        };
        check_action_dims(model_component, env_component, role)?;
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
        let same_range = model_component.range == env_component.range;
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
            model_invert: model_component.invert,
            model_threshold: model_component.threshold,
            scale: env_component.scale,
            invert: env_component.invert,
            threshold: env_component.threshold,
            binarize,
            clip,
            fill: None,
        });
    }
    Ok(ActionPlan {
        segments,
        clip: env.clip,
        in_dim,
    })
}

#[cfg(test)]
mod tests {
    use super::plan_action;
    use crate::error::ErrorCode;
    use crate::spec::{Action, Actuator};

    fn component(role: &str) -> Actuator {
        Actuator {
            role: Some(role.to_owned()),
            dim: 1,
            encoding: None,
            range: None,
            binary: false,
            scale: None,
            invert: false,
            threshold: None,
            clip: false,
            fill: 0.0,
            unknown: Default::default(),
        }
    }

    fn opaque(dim: u32, fill: f64) -> Actuator {
        Actuator {
            role: None,
            dim,
            encoding: None,
            range: None,
            binary: false,
            scale: None,
            invert: false,
            threshold: None,
            clip: false,
            fill,
            unknown: Default::default(),
        }
    }

    fn layout(components: Vec<Actuator>) -> Action {
        Action {
            components,
            clip: None,
        }
    }

    #[test]
    fn rejects_threshold_without_a_binary_component() {
        let model = layout(vec![component("action/gripper")]);
        let mut gripper = component("action/gripper");
        gripper.threshold = Some(0.5);
        let env = layout(vec![gripper]);
        let error = plan_action(&model, &env).unwrap_err();
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert!(error.message.contains("threshold requires a binary"));
    }

    #[test]
    fn threshold_with_a_binary_component_is_accepted() {
        let model = layout(vec![component("action/gripper")]);
        let mut gripper = component("action/gripper");
        gripper.threshold = Some(0.5);
        gripper.binary = true;
        let env = layout(vec![gripper]);
        assert!(plan_action(&model, &env).is_ok());
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
        let plan = plan_action(&model, &env).unwrap();
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
        let plan = plan_action(&model, &env).unwrap();
        assert_eq!(plan.segments[0].clip, Some((-1.5, 1.5)));
    }

    #[test]
    fn rejects_clip_without_a_range() {
        let model = layout(vec![component("action/x")]);
        let mut env_x = component("action/x");
        env_x.clip = true; // no range declared
        let env = layout(vec![env_x]);
        let error = plan_action(&model, &env).unwrap_err();
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
        let error = plan_action(&model, &env).unwrap_err();
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert!(
            error.message.contains("env-side clamp"),
            "{}",
            error.message
        );
    }

    #[test]
    fn rejects_clip_declared_on_the_model_action() {
        // clip is an env-side clamp applied to the assembled vector; a clip on
        // the model layout is read from the env side only, so reject it loudly
        // rather than silently dropping it.
        let mut model = layout(vec![component("action/gripper")]);
        model.clip = Some((-1.0, 1.0));
        let env = layout(vec![component("action/gripper")]);
        let error = plan_action(&model, &env).unwrap_err();
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert!(
            error.message.contains("env-side actuator clamp"),
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
        let plan = plan_action(&model, &env).unwrap();
        assert_eq!(plan.in_dim, 1); // the model only outputs the gripper dim
        assert_eq!(plan.segments.len(), 2);
        assert!(plan.segments[0].role.is_some());
        assert_eq!(plan.segments[1].role, None);
        assert_eq!(plan.segments[1].fill, Some((2, 0.25)));
    }
}
