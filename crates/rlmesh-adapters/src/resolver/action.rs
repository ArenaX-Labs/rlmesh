//! Map the model's action layout onto the env's, role by role.

use std::collections::BTreeMap;

use super::{Result, err};
use crate::error::ErrorCode;
use crate::fmt::{quoted, quoted_encoding, quoted_keys};
use crate::plans::{ActionPlan, ActionSegment};
use crate::spec::{ActionComponent, ActionLayout};

/// Validate that a model/env action component pairing is convertible.
fn check_action_dims(model: &ActionComponent, env: &ActionComponent) -> Result<()> {
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

pub(super) fn plan_action(model: &ActionLayout, env: &ActionLayout) -> Result<ActionPlan> {
    let mut offsets: BTreeMap<String, (u32, &ActionComponent)> = BTreeMap::new();
    let mut cursor: u32 = 0;
    for component in &model.components {
        if offsets.contains_key(&component.role) {
            return Err(err(
                ErrorCode::Duplicate,
                format!("duplicate model action role {}", quoted(&component.role)),
            ));
        }
        offsets.insert(component.role.clone(), (cursor, component));
        cursor += component.dim;
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
            binarize: env_component.binary || model_component.binary,
            out_dim: env_component.dim,
        });
    }
    Ok(ActionPlan {
        segments,
        clip: env.clip,
        in_dim,
    })
}
