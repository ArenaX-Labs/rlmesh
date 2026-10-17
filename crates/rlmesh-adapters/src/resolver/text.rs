//! Pair one model text input with an env text feature or its fill.

use std::collections::BTreeMap;

use super::{Fills, Result, err};
use crate::advisory::Advisory;
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
    fills: &mut Fills,
) -> Result<TextPlan> {
    let env_text = texts_by_role.get(&model_input.role).copied();
    let mut fill = model_input.fill.clone();
    if env_text.is_none() {
        // The role's data is present but under a kind this core can't read: fail
        // loud before a fill silently masks it. A role the env genuinely lacks
        // falls through to the fill (or the missing-role error below).
        super::reject_referenced_unknown(&model_input.role, &placement, unknown_roles)?;
        if fill.is_none() && fills.missing {
            fills.made.push(Advisory::caution(format!(
                "transfer: model input {} needs text role {} the env does not offer; it \
                 gets an empty string",
                quoted(&placement.to_string()),
                quoted(&model_input.role),
            )));
            fill = Some(String::new());
        }
        if fill.is_none() {
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
        container: model_input.container,
        fill,
    })
}
