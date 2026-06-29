//! The role registry: the framework-owned mechanism over the domain-owned
//! vocabulary.
//!
//! Each domain module (`core`, `manipulation`, `locomotion`, ...) ships a
//! [`ROLES`](core::ROLES) table; the registry is their union, looked up by name.
//! The registry only ever *validates* the dim an author declares -- it never
//! supplies one. The author always writes `dim=`; the dim law just checks it.

use super::{core, manipulation};

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
    use super::{DimLaw, is_known_role, role_def};

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
        assert!(!is_known_role("action/base_motion")); // unearned: no consuming model
        assert!(!is_known_role("action/something_ad_hoc"));
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
