//! The role registry: the framework-owned mechanism over the domain-owned
//! vocabulary.
//!
//! Each domain module (`core`, `manipulation`, `body`, ...) ships a
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
//! **Parts, not suffixes.** A repeated role carries a `part` on the leaf
//! ([`parts`](super::parts)); the role name itself never repeats.

use super::{body, core, manipulation};
use crate::spec::{FrameLaw, ReferenceLaw};

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
            "role {role:?} carries a kind this core does not define; the kinds are {KINDS:?}              (or the x/ escape for the whole role)"
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
    use super::{DimLaw, KINDS, check_role, is_known_role, role_def};

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
        // The task string the runner hands the policy.
        "text/instruction",
        // Encoders and the pose they imply through the robot's own kinematics.
        "proprio/joint_pos",
        "proprio/joint_vel",
        "proprio/eef_pos",
        "proprio/eef_rot",
        "proprio/gripper",
        "proprio/eef_wrench", // the wrist force/torque sensor, read verbatim
        // Commands the controller takes as written.
        "action/joint_pos",
        "action/joint_vel",
        "action/delta_eef_pos",
        "action/delta_eef_rot",
        "action/gripper",
        "action/eef_pos",
        "action/eef_rot",
        // The floating base: the gyro reads it, the IMU reports it, the runner
        // commands it. Projected gravity is the `gravity_xyz` encoding of
        // `base_rot`, never a role of its own.
        "proprio/base_ang_vel", // the IMU gyro, read verbatim
        "proprio/base_rot",     // the IMU orientation, read verbatim
        "command/base_vel",     // the velocity setpoint the runner hands the policy
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
        // A second arm is a part, never a suffixed role.
        assert!(!is_known_role("action/joint_pos_2"));
        assert!(!is_known_role("image/wrist_2"));
        assert!(!is_known_role("image/front")); // unearned: no consuming model
        assert!(!is_known_role("action/base_motion")); // unearned: no consuming model
        assert!(!is_known_role("action/something_ad_hoc"));
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
    fn the_wrench_is_six_wide_and_framed() {
        use super::FrameLaw;
        let def = role_def("proprio/eef_wrench").expect("registered");
        assert_eq!(def.dim, DimLaw::Fixed(6));
        assert_eq!(def.frame, FrameLaw::Framed);
        // A sensed wrench is the only registered force quantity: a contact
        // model's output or a commanded force is not a role.
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
        // Derived of the orientation, and an estimate on hardware: neither is
        // a role (see the module docs and the research on the Go2 pair).
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
