//! Admit a custom input as a host-language hole, gating entrypoint trust.

use crate::spec::role::bind::{Result, err};
use crate::spec::role::error::ErrorCode;
use crate::spec::role::fmt::quoted;
use crate::spec::role::path::NodePath;
use crate::spec::role::plan::CustomPlan;

pub(super) fn plan_custom(
    transform: &str,
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
                quoted(transform)
            ),
        ));
    }
    let placement_key = placement.to_string();
    Ok(CustomPlan {
        placement,
        placement_key,
        transform: transform.to_owned(),
    })
}
