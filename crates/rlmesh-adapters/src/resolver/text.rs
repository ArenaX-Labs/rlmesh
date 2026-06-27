//! Pair one model text input with an env text feature or its default.

use std::collections::BTreeMap;

use super::{Result, err};
use crate::error::ErrorCode;
use crate::fmt::{quoted, quoted_keys};
use crate::path::NodePath;
use crate::plans::TextPlan;
use crate::spec::{EnvText, Text};

pub(super) fn plan_text(
    model_input: &Text,
    placement: NodePath,
    texts_by_role: &BTreeMap<String, &EnvText>,
    unknown_roles: &BTreeMap<String, String>,
) -> Result<TextPlan> {
    let env_text = texts_by_role.get(&model_input.role).copied();
    if env_text.is_none() {
        // The role's data is present but under a kind this core can't read: fail
        // loud before a default silently masks it. A role the env genuinely lacks
        // falls through to the default (or the missing-role error below).
        super::reject_referenced_unknown(&model_input.role, &placement, unknown_roles)?;
        if model_input.default.is_none() {
            return Err(err(
                ErrorCode::MissingRole,
                format!(
                    "model input {} needs text role {} but the env offers {} and no \
                 default is set",
                    quoted(&placement.to_string()),
                    quoted(&model_input.role),
                    quoted_keys(texts_by_role)
                ),
            ));
        }
    }
    Ok(TextPlan {
        placement,
        // A default-only text input (no matching env role) has no source.
        source: env_text.map(|text| text.source.clone()),
        container: model_input.container,
        default: model_input.default.clone(),
    })
}
