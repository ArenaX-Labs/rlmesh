//! The role registry: the framework-owned mechanism over the domain-owned vocabulary.
//!
//! Each domain module ships a [`ROLES`](core::ROLES) table; the registry is their union. It only
//! validates the dim an author declares, never supplies one.
//!
//! - **Raw quantities.** A role names a raw sensed or commanded quantity. A function of one (a
//!   projected gravity, a previous action) is an encoding or transform, not a role.
//! - **Closed kinds.** A role is `<kind>/<name>` with the kind one of [`KINDS`]; any other prefix
//!   is refused. `x/` is the escape. A role with no `/` stays ad-hoc.
//! - **Parts, not suffixes.** A repeated role carries a `part` on the leaf
//!   ([`parts`](super::parts)); the name never repeats.

#[allow(unused_imports)]
use crate::prelude::*;

use super::{body, core, manipulation};
use crate::spec::role::frames::{FrameLaw, ReferenceLaw};

/// The closed set of feature kinds a role's prefix may name.
pub const KINDS: [&str; 5] = ["image/", "proprio/", "text/", "action/", "command/"];

/// Whether `role` carries a kind this core defines, or is an `x/` escape, or
/// names no kind at all (no `/`).
pub fn is_known_kind(role: &str) -> bool {
    match role.split_once('/') {
        Some((kind, _)) => kind == "x" || KINDS.contains(&format!("{kind}/").as_str()),
        None => true,
    }
}

/// Validate the role a leaf declares: the kind prefix must be one this core
/// defines (or the `x/` escape). The message reads as the reason alone;
/// callers prefix the locus.
pub fn check_role(role: &str) -> Result<(), String> {
    if !is_known_kind(role) {
        return Err(format!(
            "role {role:?} carries a kind this core does not define; the kinds are {KINDS:?}. \
             Use one of them, or prefix a deliberately non-standard role with the x/ escape \
             (e.g. x/target_pos)"
        ));
    }
    Ok(())
}

/// Whether `role` names no kind at all (no `/`). Such a role still passes
/// [`check_role`] -- older specs declare them -- but it sidesteps the kind
/// check, so it draws [`kindless_hint`] wherever an ad-hoc role is nudged.
pub fn is_kindless(role: &str) -> bool {
    !role.contains('/')
}

/// The advisory clause for a kind-less role, or `None` when `role` names a
/// kind. Reads as a sentence on its own; callers append it to their message.
pub fn kindless_hint(role: &str) -> Option<String> {
    is_kindless(role).then(|| {
        format!(
            "{role:?} names no kind, so the closed-kind check never sees it: prefix it with \
             one of {KINDS:?}, or with the x/ escape if it is deliberately non-standard \
             (e.g. x/{role})"
        )
    })
}

/// How a registered role constrains the dim of the leaf that declares it.
///
/// Validation only: the author always writes `dim=` explicitly; these variants
/// say how (or whether) that declared dim is checked. Nothing here fills a dim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimLaw {
    /// Fixed by the role's semantics; a declared dim that differs is a hard error.
    Fixed(u32),
    /// Fixed by the rotation encoding, validated by the encoding check.
    ByEncoding,
    /// Varies by embodiment or carries no numeric dim; no check beyond env and model agreement.
    Variable,
}

/// A blessed role: a stable published name plus its validation metadata.
pub struct RoleDef {
    /// The wire role string, e.g. `"action/delta_eef_pos"`.
    pub name: &'static str,
    /// How the declared dim is validated.
    pub dim: DimLaw,
    /// One-line human description (for docs / `describe`).
    pub doc: &'static str,
    /// Whether the role's values are expressed in a reference frame. Mutually
    /// exclusive with [`reference`](Self::reference) -- a pose has a frame, a
    /// delta has a reference (pinned by `no_role_is_both_framed_and_referenced`).
    pub frame: FrameLaw,
    /// Whether the role is a delta integrated against a reference pose.
    pub reference: ReferenceLaw,
}

const DOMAINS: &[&[RoleDef]] = &[core::ROLES, manipulation::ROLES, body::ROLES];

/// The registry entry for `name`, or `None` for an ad-hoc (unregistered) role.
pub fn role_def(name: &str) -> Option<&'static RoleDef> {
    DOMAINS
        .iter()
        .copied()
        .flatten()
        .find(|role| role.name == name)
}

/// Whether `name` is a registered (blessed, published) role.
pub fn is_known_role(name: &str) -> bool {
    role_def(name).is_some()
}

/// Whether `name` is in the reserved `x/` escape namespace. A role under `x/` is
/// *intentionally* outside the registry -- the pressure valve so the closed
/// vocabulary is never a hard gate for a not-yet-blessed domain or local
/// experimentation. Escape roles are never nudged (no advisory) and pass the
/// `Strict` publish gate; only `Forbid`, which admits registered roles alone,
/// rejects them. They declare "I know this isn't standard": an `x/` state or
/// action role binds only to the identical string, so it is a contract between
/// an env and the models written against it, not a shared one. (A required,
/// unparted camera is the one exception: it may bind an env's lone camera under
/// another role, with a caution.)
pub fn is_escape_role(name: &str) -> bool {
    name.starts_with("x/")
}

/// Whether `name` needs no blessing: either a registered role or an `x/` escape.
/// An unsanctioned role is ad-hoc -- it resolves only on exact-string agreement,
/// so it earns the authoring nudge and the publish-gate `Forbid` rejection.
pub fn is_sanctioned_role(name: &str) -> bool {
    is_escape_role(name) || is_known_role(name)
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use crate::prelude::*;

    use super::{DimLaw, KINDS, check_role, is_known_role, kindless_hint, role_def};

    const RAW_QUANTITIES: &[&str] = &[
        "image/primary",
        "image/secondary",
        "image/wrist",
        "text/instruction",
        "proprio/joint_pos",
        "proprio/joint_vel",
        "proprio/eef_pos",
        "proprio/eef_rot",
        "proprio/gripper",
        "proprio/eef_wrench",
        "action/joint_pos",
        "action/joint_vel",
        "action/delta_eef_pos",
        "action/delta_eef_rot",
        "action/gripper",
        "action/eef_pos",
        "action/eef_rot",
        "proprio/base_ang_vel",
        "proprio/base_rot",
        "command/base_vel",
    ];

    const DERIVED_EXCEPTIONS: &[(&str, &str)] = &[];

    #[test]
    fn every_registered_role_is_a_raw_quantity() {
        for role in super::DOMAINS.iter().copied().flatten() {
            let allowed = RAW_QUANTITIES.contains(&role.name)
                || DERIVED_EXCEPTIONS
                    .iter()
                    .any(|(name, _)| *name == role.name);
            assert!(
                allowed,
                "role {:?} is registered but not listed as a raw sensed or commanded \
                 quantity; a derived quantity is an encoding or a transform, not a role \
                 (see the registry docs)",
                role.name
            );
        }
        for (name, why) in DERIVED_EXCEPTIONS {
            assert!(is_known_role(name), "exception {name:?} is not registered");
            assert!(!why.is_empty(), "exception {name:?} gives no reason");
        }
    }

    #[test]
    fn check_role_enforces_the_closed_kinds() {
        assert_eq!(KINDS.len(), 5);
        for role in [
            "image/primary",
            "proprio/eef_pos",
            "text/instruction",
            "action/gripper",
            "command/base_vel",
            "x/anything",
            "bare",
        ] {
            assert!(check_role(role).is_ok(), "{role}");
        }
        let err = check_role("audio/mic").unwrap_err();
        assert!(err.contains("kind this core does not define"), "{err}");
        assert!(err.contains("e.g. x/target_pos"), "{err}");
        assert!(!err.contains("  "), "{err:?}");
    }

    #[test]
    fn kindless_roles_draw_a_hint_and_kinded_ones_do_not() {
        let hint = kindless_hint("bare").expect("a kind-less role is hinted");
        assert!(
            hint.contains("names no kind") && hint.contains("e.g. x/bare"),
            "{hint}"
        );
        assert!(!hint.contains("  "), "{hint:?}");
        for role in ["proprio/eef_pos", "x/anything", "audio/mic"] {
            assert!(kindless_hint(role).is_none(), "{role}");
        }
    }

    #[test]
    fn fixed_variable_and_unknown_roles_are_distinguished() {
        assert_eq!(
            role_def("proprio/eef_pos").map(|r| r.dim),
            Some(DimLaw::Fixed(3))
        );
        assert_eq!(
            role_def("proprio/gripper").map(|r| r.dim),
            Some(DimLaw::Variable)
        );
        assert_eq!(
            role_def("proprio/eef_rot").map(|r| r.dim),
            Some(DimLaw::ByEncoding)
        );
        assert!(is_known_role("action/joint_pos"));
        assert!(!is_known_role("action/joint_pos_2"));
        assert!(!is_known_role("image/wrist_2"));
        assert!(!is_known_role("image/front"));
        assert!(!is_known_role("action/base_motion"));
        assert!(!is_known_role("action/something_ad_hoc"));
    }

    #[test]
    fn no_role_is_both_framed_and_referenced() {
        use super::{FrameLaw, ReferenceLaw};
        for role in super::DOMAINS.iter().copied().flatten() {
            assert!(
                !(role.frame == FrameLaw::Framed && role.reference == ReferenceLaw::Referenced),
                "role {:?} claims both a frame and a reference",
                role.name
            );
        }
    }

    #[test]
    fn the_end_effector_roles_are_framed_and_the_deltas_referenced() {
        use super::{FrameLaw, ReferenceLaw, role_def};
        for name in [
            "proprio/eef_pos",
            "proprio/eef_rot",
            "action/eef_pos",
            "action/eef_rot",
        ] {
            assert_eq!(
                role_def(name).map(|role| role.frame),
                Some(FrameLaw::Framed),
                "{name} should be framed"
            );
        }
        for name in ["action/delta_eef_pos", "action/delta_eef_rot"] {
            let role = role_def(name).expect("registered");
            assert_eq!(
                role.frame,
                FrameLaw::Frameless,
                "{name} should be frameless"
            );
            assert_eq!(
                role.reference,
                ReferenceLaw::Referenced,
                "{name} should be referenced"
            );
        }
        let gripper = role_def("proprio/gripper").expect("registered");
        assert_eq!(gripper.frame, FrameLaw::Frameless);
        assert_eq!(gripper.reference, ReferenceLaw::Unreferenced);
    }

    #[test]
    fn the_wrench_is_six_wide_and_framed() {
        use super::FrameLaw;
        let def = role_def("proprio/eef_wrench").expect("registered");
        assert_eq!(def.dim, DimLaw::Fixed(6));
        assert_eq!(def.frame, FrameLaw::Framed);
        assert!(!is_known_role("action/eef_wrench"));
    }

    #[test]
    fn the_body_roles_are_three_framed_raw_quantities() {
        use super::FrameLaw;
        for (name, dim) in [
            ("proprio/base_ang_vel", DimLaw::Fixed(3)),
            ("proprio/base_rot", DimLaw::ByEncoding),
            ("command/base_vel", DimLaw::Fixed(3)),
        ] {
            let def = role_def(name).expect("registered");
            assert_eq!(def.dim, dim, "{name}");
            assert_eq!(def.frame, FrameLaw::Framed, "{name}");
        }
        assert!(!is_known_role("proprio/projected_gravity"));
        assert!(!is_known_role("proprio/base_lin_vel"));
    }

    #[test]
    fn every_role_carries_a_doc() {
        for role in super::DOMAINS.iter().copied().flatten() {
            assert!(!role.doc.is_empty(), "role {:?} has no doc", role.name);
        }
    }

    #[test]
    fn no_duplicate_role_names_across_domains() {
        let mut names: Vec<&str> = super::DOMAINS
            .iter()
            .copied()
            .flatten()
            .map(|r| r.name)
            .collect();
        let total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), total, "a role name is registered twice");
    }
}
