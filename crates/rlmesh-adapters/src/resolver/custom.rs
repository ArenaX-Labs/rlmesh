//! Admit a custom input as a host-language hole, gating entrypoint trust.

use super::{Result, err};
use crate::error::ErrorCode;
use crate::fmt::quoted;
use crate::path::NodePath;
use crate::plans::CustomPlan;
use crate::spec::Custom;

pub(super) fn plan_custom(
    model_input: &Custom,
    placement: NodePath,
    trust_entrypoints: bool,
) -> Result<CustomPlan> {
    if !trust_entrypoints {
        return Err(err(
            ErrorCode::UntrustedEntrypoint,
            format!(
                "custom input {} references entrypoint {}; pass \
             resolve(..., trust_entrypoints=True) to allow importing it",
                quoted(&placement.to_string()),
                quoted(&model_input.transform)
            ),
        ));
    }
    let placement_key = placement.to_string();
    Ok(CustomPlan {
        placement,
        placement_key,
        transform: model_input.transform.clone(),
    })
}
