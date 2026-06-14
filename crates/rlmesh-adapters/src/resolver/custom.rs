//! Admit a custom input as a host-language hole, gating entrypoint trust.

use super::{Result, err};
use crate::error::ErrorCode;
use crate::fmt::quoted;
use crate::plans::CustomPlan;
use crate::spec::CustomInput;

pub(super) fn plan_custom(
    model_input: &CustomInput,
    trust_entrypoints: bool,
) -> Result<CustomPlan> {
    if !trust_entrypoints {
        return Err(err(
            ErrorCode::UntrustedEntrypoint,
            format!(
                "custom input {} references entrypoint {}; pass \
             resolve(..., trust_entrypoints=True) to allow importing it",
                quoted(&model_input.key),
                quoted(&model_input.transform)
            ),
        ));
    }
    Ok(CustomPlan {
        model_key: model_input.key.clone(),
        transform: model_input.transform.clone(),
    })
}
