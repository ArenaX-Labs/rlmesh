//! Pair one model text input with an env text feature or its default.

use std::collections::BTreeMap;

use super::super::plans::TextPlan;
use super::super::pyfmt::{py_repr, py_repr_sorted_keys};
use super::super::spec::{EnvText, TextInput};
use super::{Result, err};

pub(super) fn plan_text(
    model_input: &TextInput,
    texts_by_role: &BTreeMap<String, &EnvText>,
) -> Result<TextPlan> {
    let env_text = texts_by_role.get(&model_input.role).copied();
    if env_text.is_none() && model_input.default.is_none() {
        return Err(err(format!(
            "model input {} needs text role {} but the env offers {} and no \
             default is set",
            py_repr(&model_input.key),
            py_repr(&model_input.role),
            py_repr_sorted_keys(texts_by_role)
        )));
    }
    Ok(TextPlan {
        model_key: model_input.key.clone(),
        env_key: env_text.map(|text| text.key.clone()).unwrap_or_default(),
        container: model_input.container,
        default: model_input.default.clone(),
    })
}
