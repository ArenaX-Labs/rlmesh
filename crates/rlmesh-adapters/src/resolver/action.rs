//! Map the model's action layout onto the env's, role by role.

use std::collections::BTreeMap;

use super::{Result, err};
use crate::error::ErrorCode;
use crate::fmt::{quoted, quoted_encoding, quoted_keys};
use crate::plans::{ActionPlan, ActionSegment};
use crate::spec::{Action, Actuator};

/// Validate that a model/env action component pairing is convertible.
fn check_action_dims(model: &Actuator, env: &Actuator) -> Result<()> {
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
                    quoted(&model.role),
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
                quoted(&model.role),
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
                quoted(&model.role),
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
        if offsets.contains_key(&component.role) {
            return Err(err(
                ErrorCode::Duplicate,
                format!("duplicate model action role {}", quoted(&component.role)),
            ));
        }
        offsets.insert(component.role.clone(), (cursor, component));
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
        if seen_env_roles
            .insert(env_component.role.as_str(), ())
            .is_some()
        {
            // Mirror the model-side dedup above (and the env-side StateLayout
            // role check): a role repeated in the env layout would resolve
            // every copy against the same model slice, building the env action
            // by repetition instead of a real mapping.
            return Err(err(
                ErrorCode::Duplicate,
                format!("duplicate env action role {}", quoted(&env_component.role)),
            ));
        }
        let Some(&(start, model_component)) = offsets.get(&env_component.role) else {
            return Err(err(
                ErrorCode::MissingRole,
                format!(
                    "env action needs role {} but the model only outputs {}",
                    quoted(&env_component.role),
                    quoted_keys(&offsets)
                ),
            ));
        };
        check_action_dims(model_component, env_component)?;
        // Corrections (scale/invert/threshold) describe the env's actuator
        // convention; the model declares none. They are read only from the env
        // component below, so a model-side correction would be silently dropped --
        // reject it instead.
        if model_component.scale.is_some()
            || model_component.invert
            || model_component.threshold.is_some()
        {
            return Err(err(
                ErrorCode::Unsupported,
                format!(
                    "action role {}: scale/invert/threshold belong on the env component, not the model",
                    quoted(&env_component.role)
                ),
            ));
        }
        let binarize = env_component.binary || model_component.binary;
        // threshold is a binary decision boundary; without a binary snap it would
        // silently become a constant offset on a continuous action.
        if env_component.threshold.is_some() && !binarize {
            return Err(err(
                ErrorCode::Unsupported,
                format!(
                    "action role {}: threshold requires a binary component (set binary=true)",
                    quoted(&env_component.role)
                ),
            ));
        }
        let same_range = model_component.range == env_component.range;
        segments.push(ActionSegment {
            role: env_component.role.clone(),
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
            // The env owns its actuator convention, so its corrections drive the
            // segment; a model that ships against many envs declares none.
            scale: env_component.scale,
            invert: env_component.invert,
            threshold: env_component.threshold,
            binarize,
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
            role: role.to_owned(),
            dim: 1,
            encoding: None,
            range: None,
            binary: false,
            scale: None,
            invert: false,
            threshold: None,
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
    fn rejects_corrections_declared_on_the_model_component() {
        let mut model_gripper = component("action/gripper");
        model_gripper.scale = Some(2.0);
        let model = layout(vec![model_gripper]);
        let env = layout(vec![component("action/gripper")]);
        let error = plan_action(&model, &env).unwrap_err();
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert!(error.message.contains("belong on the env component"));
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
}
