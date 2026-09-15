//! The role registry: the framework-owned mechanism over the domain-owned
//! vocabulary.
//!
//! Each domain module (`core`, `manipulation`, `locomotion`, ...) ships a
//! [`ROLES`](core::ROLES) table; the registry is their union, looked up by name.
//! The registry only ever *validates* the dim an author declares -- it never
//! supplies one. The author always writes `dim=`; the dim law just checks it.
//!
//! **The raw-quantity law.** A role names a raw sensed or commanded quantity:
//! a joint angle the encoder reads, a Cartesian target the controller takes.
//! A function of one is never a role -- it is an *encoding* (a quaternion
//! projected to a gravity vector), a *transform* (an affine, a previous
//! action), or a model-side concern. So `proprio/projected_gravity` and
//! `x/last_action` do not belong here, however common they are in training
//! code. The test `every_registered_role_is_a_raw_quantity` pins the list;
//! a derived role cannot land without editing it and saying why.
//!
//! **Kinds are closed.** A role is `<kind>/<name>` and the kind is one of
//! [`KINDS`]; any other prefix is refused at authoring, join and resolve (a
//! newer peer's spec still parses and relays). `x/` stays the escape for a
//! whole role. A role with no `/` names no kind and stays ad-hoc.
//!
//! **Parts, not suffixes.** Where a role repeats on a body the leaf carries a
//! `part` ([`parts`](super::parts)). The ten `_2` roles predate parts and stay
//! as data; [`ALIASES`] folds each onto its base role under `part="arm_2"`
//! before any lookup, so the two spellings bind each other. No further `_N`
//! role is ever added (`no_further_suffixed_roles`).

use super::{core, manipulation, parts};
use crate::spec::{FrameLaw, ReferenceLaw};

/// The closed set of feature kinds a role's prefix may name.
pub const KINDS: [&str; 5] = ["image/", "proprio/", "text/", "action/", "command/"];

/// The legacy second-arm roles, each folded onto `(base role, part)`. Data,
/// not logic: the resolver canonicalizes both sides through [`canonical`] so
/// a v1 `proprio/eef_pos_2` model binds an env `proprio/eef_pos` leaf under
/// `part="arm_2"` and vice versa. The list is closed (see
/// `no_further_suffixed_roles`).
pub const ALIASES: &[(&str, &str, &str)] = &[
    (
        manipulation::IMAGE_WRIST_2,
        manipulation::IMAGE_WRIST,
        parts::ARM_2,
    ),
    (manipulation::EEF_POS_2, manipulation::EEF_POS, parts::ARM_2),
    (manipulation::EEF_ROT_2, manipulation::EEF_ROT, parts::ARM_2),
    (
        manipulation::GRIPPER_POS_2,
        manipulation::GRIPPER_POS,
        parts::ARM_2,
    ),
    (
        manipulation::ACTION_DELTA_POS_2,
        manipulation::ACTION_DELTA_POS,
        parts::ARM_2,
    ),
    (
        manipulation::ACTION_DELTA_ROT_2,
        manipulation::ACTION_DELTA_ROT,
        parts::ARM_2,
    ),
    (
        manipulation::ACTION_GRIPPER_2,
        manipulation::ACTION_GRIPPER,
        parts::ARM_2,
    ),
    (
        manipulation::ACTION_EEF_POS_2,
        manipulation::ACTION_EEF_POS,
        parts::ARM_2,
    ),
    (
        manipulation::ACTION_EEF_ROT_2,
        manipulation::ACTION_EEF_ROT,
        parts::ARM_2,
    ),
    (
        core::ACTION_JOINT_POS_2,
        core::ACTION_JOINT_POS,
        parts::ARM_2,
    ),
];

/// The `(base role, part)` a legacy alias role spells, or `None` for any
/// other role.
pub fn alias_of(role: &str) -> Option<(&'static str, &'static str)> {
    ALIASES
        .iter()
        .find(|(alias, _, _)| *alias == role)
        .map(|(_, base, part)| (*base, *part))
}

/// The identity a leaf binds by: its role and part with the legacy `_2`
/// spelling folded onto `(base, "arm_2")`. A declared part always stands
/// (an alias role carrying one is refused by [`check_role`] before this
/// runs, so the two never meet here).
pub fn canonical<'a>(role: &'a str, part: Option<&'a str>) -> (&'a str, Option<&'a str>) {
    match (alias_of(role), part) {
        (Some((base, alias_part)), None) => (base, Some(alias_part)),
        _ => (role, part),
    }
}

/// Whether `role` carries a kind this core defines, or is an `x/` escape, or
/// names no kind at all (no `/`).
pub fn is_known_kind(role: &str) -> bool {
    match role.split_once('/') {
        Some((kind, _)) => kind == "x" || KINDS.contains(&format!("{kind}/").as_str()),
        None => true,
    }
}

/// Validate the role and part identity a leaf declares: the kind prefix must
/// be one this core defines (or the `x/` escape), and a legacy `_2` role,
/// which already spells `part="arm_2"`, may not also carry a part. The
/// message reads as the reason alone; callers prefix the locus.
pub fn check_role(role: &str, part: Option<&str>) -> Result<(), String> {
    if !is_known_kind(role) {
        return Err(format!(
            "role {role:?} carries a kind this core does not define; the kinds are {KINDS:?}              (or the x/ escape for the whole role)"
        ));
    }
    if let (Some((base, alias_part)), Some(part)) = (alias_of(role), part) {
        return Err(format!(
            "role {role:?} is the legacy spelling of role {base:?} under part {alias_part:?}              and cannot also carry part {part:?}"
        ));
    }
    Ok(())
}

/// How a registered role constrains the dim of the leaf that declares it.
///
/// Validation only: the author always writes `dim=` explicitly; these variants
/// say how (or whether) that declared dim is checked. Nothing here fills a dim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimLaw {
    /// Fixed by the role's semantics -- `proprio/eef_pos` is always 3-D Cartesian.
    /// A declared dim that differs is a hard error.
    Fixed(u32),
    /// Fixed by the rotation encoding instead, validated where encodings already
    /// are (`join`/`check_action_dims`). This variant documents intent; the dim
    /// law defers to the encoding check rather than re-deriving it.
    ByEncoding,
    /// Varies by embodiment (gripper finger count, joint DoF) or carries no
    /// numeric dim (image/text). No check beyond the existing env<->model
    /// agreement.
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

/// Every domain's role table. A new domain is a new module plus one entry here --
/// the registry is their union. Data, not logic. (A domain earns a slot only when
/// a real env/model *pair* defines its contract: a producer alone -- e.g. a
/// mobile base with no policy that consumes it -- is not enough, and a
/// heterogeneous bundle like a base motion command must be decomposed into Fixed
/// primitives or stay ad-hoc/opaque until that contract exists.)
const DOMAINS: &[&[RoleDef]] = &[core::ROLES, manipulation::ROLES];

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
/// experimentation. Escape roles are never nudged (no advisory) and never
/// rejected at the publish gate; they just declare "I know this isn't standard."
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
    use super::{ALIASES, DimLaw, KINDS, canonical, check_role, is_known_role, role_def};

    /// The raw-quantity law as data: every registered role, each a quantity a
    /// sensor reads or a controller takes verbatim. Adding a role means adding
    /// it here; a *derived* quantity (a projection, a difference, a model's own
    /// past output) belongs in `DERIVED_EXCEPTIONS` with its justification --
    /// and today there are none.
    const RAW_QUANTITIES: &[&str] = &[
        // Cameras: the sensor's frame, verbatim.
        "image/primary",
        "image/secondary",
        "image/wrist",
        "image/wrist_2",
        // The task string the runner hands the policy.
        "text/instruction",
        // Encoders and the pose they imply through the robot's own kinematics.
        "proprio/joint_pos",
        "proprio/joint_vel",
        "proprio/eef_pos",
        "proprio/eef_rot",
        "proprio/gripper",
        "proprio/eef_pos_2",
        "proprio/eef_rot_2",
        "proprio/gripper_2",
        // Commands the controller takes as written.
        "action/joint_pos",
        "action/joint_vel",
        "action/joint_pos_2",
        "action/delta_eef_pos",
        "action/delta_eef_rot",
        "action/gripper",
        "action/delta_eef_pos_2",
        "action/delta_eef_rot_2",
        "action/gripper_2",
        "action/eef_pos",
        "action/eef_rot",
        "action/eef_pos_2",
        "action/eef_rot_2",
    ];

    /// Registered roles that are *not* raw quantities, each with the reason it
    /// was allowed anyway. Empty by design; adding an entry is a reviewed
    /// decision, not a convenience.
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
    fn no_further_suffixed_roles() {
        // The ten `_2` roles are the closed legacy list: each is an alias for
        // its base role under part "arm_2", and no `_N` role is ever added
        // again -- a repeated role gets a `part`.
        let suffixed: Vec<&str> = super::DOMAINS
            .iter()
            .copied()
            .flatten()
            .map(|role| role.name)
            .filter(|name| {
                name.rsplit_once('_')
                    .is_some_and(|(_, tail)| tail.chars().all(|c| c.is_ascii_digit()))
            })
            .collect();
        let mut expected = vec![
            "image/wrist_2",
            "proprio/eef_pos_2",
            "proprio/eef_rot_2",
            "proprio/gripper_2",
            "action/joint_pos_2",
            "action/delta_eef_pos_2",
            "action/delta_eef_rot_2",
            "action/gripper_2",
            "action/eef_pos_2",
            "action/eef_rot_2",
        ];
        let mut actual = suffixed.clone();
        actual.sort_unstable();
        expected.sort_unstable();
        assert_eq!(actual, expected, "the `_2` list is closed");
        let mut aliased: Vec<&str> = ALIASES.iter().map(|(alias, _, _)| *alias).collect();
        aliased.sort_unstable();
        assert_eq!(aliased, expected, "every `_2` role has an alias row");
        for (alias, base, part) in ALIASES {
            assert!(
                is_known_role(base),
                "alias {alias:?} base {base:?} unregistered"
            );
            assert_eq!(*part, "arm_2");
            assert_eq!(
                role_def(alias).map(|def| def.dim),
                role_def(base).map(|def| def.dim),
                "alias {alias:?} and its base disagree on the dim law"
            );
        }
    }

    #[test]
    fn canonical_folds_the_alias_onto_its_base_and_part() {
        assert_eq!(
            canonical("proprio/eef_pos_2", None),
            ("proprio/eef_pos", Some("arm_2"))
        );
        assert_eq!(
            canonical("proprio/eef_pos", Some("arm_2")),
            ("proprio/eef_pos", Some("arm_2"))
        );
        assert_eq!(
            canonical("proprio/eef_pos", None),
            ("proprio/eef_pos", None)
        );
        assert_eq!(
            canonical("x/thing", Some("head")),
            ("x/thing", Some("head"))
        );
    }

    #[test]
    fn check_role_enforces_the_closed_kinds_and_the_alias_rule() {
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
            assert!(check_role(role, Some("head")).is_ok(), "{role}");
        }
        let err = check_role("audio/mic", None).unwrap_err();
        assert!(err.contains("kind this core does not define"), "{err}");
        let err = check_role("proprio/eef_pos_2", Some("left_arm")).unwrap_err();
        assert!(err.contains("legacy spelling"), "{err}");
        assert!(check_role("proprio/eef_pos_2", None).is_ok());
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
        assert!(is_known_role("action/joint_pos")); // newly blessed (clear contract)
        // A bimanual joint command and the second wrist camera: each has a
        // producer and a reader, and each is embodiment-widthed.
        assert_eq!(
            role_def("action/joint_pos_2").map(|r| r.dim),
            Some(DimLaw::Variable)
        );
        assert_eq!(
            role_def("image/wrist_2").map(|r| r.dim),
            Some(DimLaw::Variable)
        );
        assert!(!is_known_role("image/front")); // unearned: no consuming model
        assert!(!is_known_role("action/base_motion")); // unearned: no consuming model
        assert!(!is_known_role("action/something_ad_hoc"));
    }

    #[test]
    fn every_second_arm_role_mirrors_a_registered_first_arm_role() {
        // The `_2` mirror is one-directional: a second-arm role is meaningless
        // without its first-arm original, but a first-arm role stands alone (a
        // single-arm env never declares `_2`), so the converse must NOT hold.
        let names: Vec<&str> = super::DOMAINS
            .iter()
            .copied()
            .flatten()
            .map(|role| role.name)
            .collect();
        for name in &names {
            if let Some(base) = name.strip_suffix("_2") {
                assert!(
                    names.contains(&base),
                    "second-arm role {name:?} has no registered first-arm {base:?}"
                );
            }
        }
        assert!(is_known_role("proprio/joint_pos") && !is_known_role("proprio/joint_pos_2"));
    }

    #[test]
    fn no_role_is_both_framed_and_referenced() {
        // A pose is qualified by the frame it is expressed in; a delta by the
        // pose it is integrated against. Nothing is both, so a role that claimed
        // both laws would be asking for two answers to the same question.
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
    fn the_eight_end_effector_roles_are_framed_and_the_deltas_referenced() {
        use super::{FrameLaw, ReferenceLaw, role_def};
        for name in [
            "proprio/eef_pos",
            "proprio/eef_rot",
            "proprio/eef_pos_2",
            "proprio/eef_rot_2",
            "action/eef_pos",
            "action/eef_rot",
            "action/eef_pos_2",
            "action/eef_rot_2",
        ] {
            assert_eq!(
                role_def(name).map(|role| role.frame),
                Some(FrameLaw::Framed),
                "{name} should be framed"
            );
        }
        for name in [
            "action/delta_eef_pos",
            "action/delta_eef_rot",
            "action/delta_eef_pos_2",
            "action/delta_eef_rot_2",
        ] {
            let role = role_def(name).expect("registered");
            // A delta carries no frame -- it lives in the controller's own.
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
        // A gripper is a scalar: neither law applies.
        let gripper = role_def("proprio/gripper").expect("registered");
        assert_eq!(gripper.frame, FrameLaw::Frameless);
        assert_eq!(gripper.reference, ReferenceLaw::Unreferenced);
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
