//! The strict-v1 gate, decoupled from the serde layer.
//!
//! The serde codec is *unconditionally tolerant*: every growable leaf carries a
//! `#[serde(flatten)]` capture map (see [`ImageTag`](super::ImageTag) et al.) and
//! unknown kinds parse into `Unknown` arms, so any structurally-valid spec round-
//! trips without loss. Strictness is this separate post-parse pass.
//!
//! Two recognition events are tolerated; everything else stays a hard parse
//! error. They are gated at two altitudes:
//!
//! - **Unknown FIELD on a recognized kind.** A *bare* (unprefixed) unknown field
//!   is "must-understand": fail-closed, because an old core silently applying its
//!   own default for a modifier it never parsed is worse than failing (§8). A
//!   field in the reserved [`x-`/`ext.`](is_reserved_ext) namespace is the
//!   producer's opt-in "safe to ignore" and passes. This rule runs at **both**
//!   doors: PUBLISH (a typo dies at the trust boundary) and READ (a newer peer's
//!   bare additive field taints — [`reject_bare_fields_env`]/`_model`).
//! - **Unknown KIND.** Rejected at PUBLISH only (an author cannot publish a kind
//!   their own core cannot build). The READ door *retains* it for relay and lets
//!   the resolver decide — ignored with an advisory unless a model input
//!   references it, then a localized `UnsupportedKind`.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::path::NodePath;

use super::action::Action;
use super::env_tags::{EnvTags, ObsLeaf, ObsNode};
use super::model::{InputNode, ModelLeaf, ModelSpec};

/// Reject any bare unknown field **or** unknown leaf kind in an env spec (the
/// PUBLISH gate: an author's own core must understand every kind and bare field).
pub fn reject_unknowns_env(tags: &EnvTags) -> Result<(), String> {
    walk_obs(&tags.observation, &NodePath::root(), true)?;
    reject_action(&tags.action)
}

/// Reject any bare unknown field **or** unknown leaf kind in a model spec.
pub fn reject_unknowns_model(spec: &ModelSpec) -> Result<(), String> {
    walk_input(&spec.input, &NodePath::root(), true)?;
    reject_action(&spec.output)
}

/// Reject only *bare* unknown fields in an env spec, tolerating unknown kinds
/// (the READ taint: a peer's bare additive field is fail-closed, but an unknown
/// *kind* is the resolver's to ignore-or-fail, not this pass's).
pub fn reject_bare_fields_env(tags: &EnvTags) -> Result<(), String> {
    walk_obs(&tags.observation, &NodePath::root(), false)?;
    reject_action(&tags.action)
}

/// Reject only *bare* unknown fields in a model spec (the READ taint).
pub fn reject_bare_fields_model(spec: &ModelSpec) -> Result<(), String> {
    walk_input(&spec.input, &NodePath::root(), false)?;
    reject_action(&spec.output)
}

/// The publish-gate policy for ad-hoc (unregistered) roles, from most to least
/// permissive.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RolePolicy {
    /// Every role passes -- the open vocabulary (an authoring nudge only). The
    /// local/OSS default.
    Passthrough,
    /// Ad-hoc roles are rejected, but the `x/` escape is allowed: registered roles
    /// or explicit escapes, no *accidental* ad-hoc. The curated managed default.
    Strict,
    /// Every unregistered role is rejected, even the `x/` escape: a registered-only
    /// lockdown, the absolute tier.
    Forbid,
}

impl RolePolicy {
    /// Whether `role` is allowed under this policy.
    fn allows(self, role: &str) -> bool {
        match self {
            RolePolicy::Passthrough => true,
            RolePolicy::Strict => crate::roles::registry::is_sanctioned_role(role),
            RolePolicy::Forbid => crate::roles::registry::is_known_role(role),
        }
    }
}

/// Reject one role that `policy` disallows.
fn reject_role(role: &str, locus: &str, policy: RolePolicy) -> Result<(), String> {
    if policy.allows(role) {
        return Ok(());
    }
    let hint = if policy == RolePolicy::Forbid {
        "use a blessed role (this gate forbids every unregistered role, including the `x/` escape)"
    } else {
        "use a blessed role, or the `x/` escape namespace for an intentionally non-standard one"
    };
    Err(format!(
        "{locus} declares unregistered role {role:?}; {hint}"
    ))
}

/// Reject any role an env spec declares that `policy` disallows. A separate pass
/// from the field/kind gate, run only when the policy is not `Passthrough`.
pub fn reject_unsanctioned_roles_env(tags: &EnvTags, policy: RolePolicy) -> Result<(), String> {
    walk_obs_roles(&tags.observation, &NodePath::root(), policy)?;
    reject_action_roles(&tags.action, policy)
}

/// Reject any role a model spec declares that `policy` disallows.
pub fn reject_unsanctioned_roles_model(spec: &ModelSpec, policy: RolePolicy) -> Result<(), String> {
    walk_input_roles(&spec.input, &NodePath::root(), policy)?;
    reject_action_roles(&spec.output, policy)
}

fn walk_obs_roles(node: &ObsNode, path: &NodePath, policy: RolePolicy) -> Result<(), String> {
    match node {
        ObsNode::Leaf(leaf) => obs_leaf_roles(leaf, path, policy),
        ObsNode::Dict(map) => map.iter().try_for_each(|(key, child)| {
            walk_obs_roles(child, &path.push_key(key.clone()), policy)
        }),
        ObsNode::Tuple(items) => items
            .iter()
            .enumerate()
            .try_for_each(|(index, child)| walk_obs_roles(child, &path.push_index(index), policy)),
    }
}

fn obs_leaf_roles(leaf: &ObsLeaf, path: &NodePath, policy: RolePolicy) -> Result<(), String> {
    let locus = format!("observation {:?}", path.to_string());
    match leaf {
        ObsLeaf::Image(tag) => reject_role(&tag.role, &locus, policy),
        ObsLeaf::State(tag) => reject_role(&tag.role, &locus, policy),
        ObsLeaf::Text(tag) => reject_role(&tag.role, &locus, policy),
        ObsLeaf::Split(layout) => layout
            .fields
            .iter()
            .try_for_each(|field| match &field.role {
                Some(role) => reject_role(role, &locus, policy),
                None => Ok(()),
            }),
        ObsLeaf::Unknown { .. } => Ok(()),
    }
}

fn walk_input_roles(node: &InputNode, path: &NodePath, policy: RolePolicy) -> Result<(), String> {
    match node {
        InputNode::Leaf(leaf) => model_leaf_roles(leaf, path, policy),
        InputNode::Dict(map) => map.iter().try_for_each(|(key, child)| {
            walk_input_roles(child, &path.push_key(key.clone()), policy)
        }),
        InputNode::Tuple(items) => items.iter().enumerate().try_for_each(|(index, child)| {
            walk_input_roles(child, &path.push_index(index), policy)
        }),
    }
}

fn model_leaf_roles(leaf: &ModelLeaf, path: &NodePath, policy: RolePolicy) -> Result<(), String> {
    let locus = format!("model input {:?}", path.to_string());
    match leaf {
        ModelLeaf::Image(input) => reject_role(&input.role, &locus, policy),
        ModelLeaf::State(input) => input
            .components
            .iter()
            .try_for_each(|part| reject_role(&part.role, &locus, policy)),
        ModelLeaf::Text(input) => reject_role(&input.role, &locus, policy),
        ModelLeaf::Custom(_) | ModelLeaf::Unknown { .. } => Ok(()),
    }
}

fn reject_action_roles(action: &Action, policy: RolePolicy) -> Result<(), String> {
    for (index, actuator) in action.components.iter().enumerate() {
        if let Some(role) = &actuator.role {
            reject_role(role, &format!("action component[{index}]"), policy)?;
        }
    }
    Ok(())
}

/// A field name in the reserved experimental/vendor namespace is "safe to
/// ignore" — the producer, who knows its semantics, marked it cosmetic. Bare
/// (unprefixed) fields are must-understand and fail closed.
fn is_reserved_ext(key: &str) -> bool {
    key.starts_with("x-") || key.starts_with("ext.")
}

/// The first non-`x-` capture key, if any — the field that fails the gate.
fn first_bare_field(unknown: &BTreeMap<String, Value>) -> Option<&String> {
    unknown.keys().find(|key| !is_reserved_ext(key))
}

fn walk_obs(node: &ObsNode, path: &NodePath, reject_kinds: bool) -> Result<(), String> {
    match node {
        ObsNode::Leaf(leaf) => obs_leaf(leaf, path, reject_kinds),
        ObsNode::Dict(map) => map.iter().try_for_each(|(key, child)| {
            walk_obs(child, &path.push_key(key.clone()), reject_kinds)
        }),
        ObsNode::Tuple(items) => items
            .iter()
            .enumerate()
            .try_for_each(|(index, child)| walk_obs(child, &path.push_index(index), reject_kinds)),
    }
}

fn obs_leaf(leaf: &ObsLeaf, path: &NodePath, reject_kinds: bool) -> Result<(), String> {
    match leaf {
        ObsLeaf::Image(tag) => bare_field(&tag.unknown, path),
        ObsLeaf::State(tag) => bare_field(&tag.unknown, path),
        ObsLeaf::Text(tag) => bare_field(&tag.unknown, path),
        // SplitLayout / Field stay strict (their wire structs keep deny).
        ObsLeaf::Split(_) => Ok(()),
        ObsLeaf::Unknown { kind, .. } if reject_kinds => {
            Err(unknown_kind_msg("observation", kind, path))
        }
        ObsLeaf::Unknown { .. } => Ok(()),
    }
}

fn walk_input(node: &InputNode, path: &NodePath, reject_kinds: bool) -> Result<(), String> {
    match node {
        InputNode::Leaf(leaf) => model_leaf(leaf, path, reject_kinds),
        InputNode::Dict(map) => map.iter().try_for_each(|(key, child)| {
            walk_input(child, &path.push_key(key.clone()), reject_kinds)
        }),
        InputNode::Tuple(items) => items.iter().enumerate().try_for_each(|(index, child)| {
            walk_input(child, &path.push_index(index), reject_kinds)
        }),
    }
}

fn model_leaf(leaf: &ModelLeaf, path: &NodePath, reject_kinds: bool) -> Result<(), String> {
    match leaf {
        ModelLeaf::Image(input) => bare_field(&input.unknown, path),
        ModelLeaf::State(input) => bare_field(&input.unknown, path),
        ModelLeaf::Text(input) => bare_field(&input.unknown, path),
        ModelLeaf::Custom(input) => bare_field(&input.unknown, path),
        ModelLeaf::Unknown { kind, .. } if reject_kinds => {
            Err(unknown_kind_msg("model input", kind, path))
        }
        ModelLeaf::Unknown { .. } => Ok(()),
    }
}

/// The gate message for an unrecognized leaf kind (PUBLISH only).
fn unknown_kind_msg(domain: &str, kind: &str, path: &NodePath) -> String {
    format!(
        "{domain} {:?} declares unrecognized kind {kind:?}; this core cannot build it \
         (upgrade the runtime, or this spec cannot be published here)",
        path.to_string()
    )
}

/// Action components (`Actuator`) are the only growable action leaves; the
/// `Action`/`ActionWire` envelope itself stays strict (`deny_unknown_fields`).
fn reject_action(action: &Action) -> Result<(), String> {
    for (index, actuator) in action.components.iter().enumerate() {
        if let Some(field) = first_bare_field(&actuator.unknown) {
            return Err(format!(
                "action component[{index}] (role {:?}) declares unrecognized field {field:?}; \
                 upgrade the runtime or drop the field (or prefix it `x-` to mark it ignorable)",
                actuator.role.as_deref().unwrap_or("opaque")
            ));
        }
    }
    Ok(())
}

fn bare_field(unknown: &BTreeMap<String, Value>, path: &NodePath) -> Result<(), String> {
    match first_bare_field(unknown) {
        None => Ok(()),
        Some(field) => Err(format!(
            "feature {:?} declares unrecognized field {field:?}; upgrade the runtime \
             or drop the field (or prefix it `x-` to mark it ignorable)",
            path.to_string()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{reject_bare_fields_env, reject_unknowns_env, reject_unknowns_model};
    use crate::spec::{EnvTags, ModelSpec};

    #[test]
    fn env_clean_spec_passes_unknown_field_fails() {
        let clean: EnvTags = serde_json::from_str(
            r#"{"observation": {"cam": {"type": "image", "role": "image/primary"}},
                "action": {"components": [{"role": "a", "dim": 1}]}}"#,
        )
        .unwrap();
        assert!(reject_unknowns_env(&clean).is_ok());

        let dirty: EnvTags = serde_json::from_str(
            r#"{"observation": {"cam": {"type": "image", "role": "image/primary", "wat": 1}},
                "action": {"components": [{"role": "a", "dim": 1}]}}"#,
        )
        .unwrap();
        let err = reject_unknowns_env(&dirty).unwrap_err();
        assert!(err.contains("cam") && err.contains("wat"), "got: {err}");
    }

    #[test]
    fn x_prefixed_field_is_tolerated_at_both_doors() {
        // A producer-marked `x-`/`ext.` field is safe-to-ignore: it passes the
        // publish gate and the read taint, and never taints the leaf.
        let spec: EnvTags = serde_json::from_str(
            r#"{"observation": {"cam": {"type": "image", "role": "image/primary",
                "x-vendor-note": "hi", "ext.team": 7}},
                "action": {"components": [{"role": "a", "dim": 1, "x-tune": 1}]}}"#,
        )
        .unwrap();
        assert!(reject_unknowns_env(&spec).is_ok(), "publish gate");
        assert!(reject_bare_fields_env(&spec).is_ok(), "read taint");
    }

    #[test]
    fn read_taint_flags_bare_field_but_tolerates_unknown_kind() {
        // The READ door taints a bare additive field (fail-closed) but lets an
        // unknown *kind* through (the resolver decides its fate).
        let bare: EnvTags = serde_json::from_str(
            r#"{"observation": {"cam": {"type": "image", "role": "image/primary", "normalize": false}},
                "action": {"components": []}}"#,
        )
        .unwrap();
        let err = reject_bare_fields_env(&bare).unwrap_err();
        assert!(err.contains("normalize"), "got: {err}");

        let unknown_kind: EnvTags = serde_json::from_str(
            r#"{"observation": {"mic": {"type": "audio", "role": "audio/mic"}},
                "action": {"components": []}}"#,
        )
        .unwrap();
        assert!(
            reject_bare_fields_env(&unknown_kind).is_ok(),
            "read door tolerates unknown kinds"
        );
    }

    #[test]
    fn model_unknown_field_on_state_leaf_fails() {
        // Fixes the old StateWire silent drop: a stray field on a state input is
        // retained and rejected at the publish gate, not dropped.
        let dirty: ModelSpec = serde_json::from_str(
            r#"{"input": {"type": "state", "components": ["r"], "huh": true},
                "output": {"components": []}}"#,
        )
        .unwrap();
        let err = reject_unknowns_model(&dirty).unwrap_err();
        assert!(err.contains("huh"), "got: {err}");
    }

    #[test]
    fn action_component_unknown_field_fails() {
        let dirty: ModelSpec = serde_json::from_str(
            r#"{"input": {"type": "text", "role": "instruction"},
                "output": {"components": [{"role": "g", "dim": 1, "wobble": 3}]}}"#,
        )
        .unwrap();
        let err = reject_unknowns_model(&dirty).unwrap_err();
        assert!(err.contains("wobble"), "got: {err}");
    }

    #[test]
    fn strict_allows_escape_and_opaque_but_forbid_rejects_escape() {
        use super::{RolePolicy, reject_unsanctioned_roles_env, reject_unsanctioned_roles_model};

        let with_escape: EnvTags = serde_json::from_str(
            r#"{"observation": {"cam": {"type": "image", "role": "image/primary"},
                "ext": {"type": "state", "role": "x/custom"}},
                "action": {"components": [{"role": "action/gripper", "dim": 1}, {"dim": 2}]}}"#,
        )
        .unwrap();
        assert!(reject_unsanctioned_roles_env(&with_escape, RolePolicy::Strict).is_ok());

        let err = reject_unsanctioned_roles_env(&with_escape, RolePolicy::Forbid).unwrap_err();
        assert!(
            err.contains("x/custom") && err.contains("including the `x/`"),
            "{err}"
        );

        let bad: EnvTags = serde_json::from_str(
            r#"{"observation": {"cam": {"type": "image", "role": "image/primary"}},
                "action": {"components": [{"role": "action/wiggle", "dim": 1}]}}"#,
        )
        .unwrap();
        let err = reject_unsanctioned_roles_env(&bad, RolePolicy::Strict).unwrap_err();
        assert!(
            err.contains("action/wiggle") && err.contains("unregistered"),
            "{err}"
        );

        let bad_model: ModelSpec = serde_json::from_str(
            r#"{"input": {"type": "state", "components": ["proprio/eef_pos", "proprio/made_up"]},
                "output": {"components": []}}"#,
        )
        .unwrap();
        let err = reject_unsanctioned_roles_model(&bad_model, RolePolicy::Strict).unwrap_err();
        assert!(err.contains("proprio/made_up"), "{err}");
    }
}
