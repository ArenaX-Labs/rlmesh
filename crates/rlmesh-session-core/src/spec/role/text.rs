//! Pair one model text input with an env text feature or its fill.

#[allow(unused_imports)]
use crate::prelude::*;

use alloc::collections::BTreeMap;

use crate::spec::role::bind::{Result, err, reject_referenced_unknown};
use crate::spec::role::error::ErrorCode;
use crate::spec::role::fmt::{quoted, quoted_keys};
use crate::spec::role::input::{EnvText, Text};
use crate::spec::role::path::NodePath;
use crate::spec::role::plan::TextPlan;

pub(super) fn plan_text(
    model_input: &Text,
    placement: NodePath,
    texts_by_role: &BTreeMap<String, &EnvText>,
    unknown_roles: &BTreeMap<String, String>,
) -> Result<TextPlan> {
    let env_text = texts_by_role.get(&model_input.role).copied();
    if env_text.is_none() {
        // The role's data is present but under a kind this core can't read: fail
        // loud before a fill silently masks it. A role the env genuinely lacks
        // falls through to the fill (or the missing-role error below).
        reject_referenced_unknown(&model_input.role, &placement, unknown_roles)?;
        if model_input.fill.is_none() {
            return Err(err(
                ErrorCode::MissingRole,
                format!(
                    "model input {} needs text role {} but the env offers {} and no \
                 fill is set",
                    quoted(&placement.to_string()),
                    quoted(&model_input.role),
                    quoted_keys(texts_by_role)
                ),
            ));
        }
    }
    Ok(TextPlan {
        placement,
        // A fill-only text input (no matching env role) has no source.
        source: env_text.map(|text| text.source.clone()),
        list: model_input.list,
        fill: model_input.fill.clone(),
    })
}
