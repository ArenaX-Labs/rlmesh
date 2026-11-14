//! Admit a custom input as a host-language hole, gating entrypoint trust.

#[allow(unused_imports)]
use crate::prelude::*;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_custom_entrypoint_is_refused_unless_trusted() {
        let at = || NodePath::root().push_key("feat");
        let error = plan_custom("pkg.mod:fn", at(), false).expect_err("untrusted");
        assert_eq!(error.code, ErrorCode::UntrustedEntrypoint);
        let plan = plan_custom("pkg.mod:fn", at(), true).expect("trusted");
        assert_eq!(plan.transform, "pkg.mod:fn");
    }
}
