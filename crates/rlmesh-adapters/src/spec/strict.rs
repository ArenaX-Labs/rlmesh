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
use super::frames::{Attr, FrameLaw, FrameRef, ReferenceLaw};
use super::model::{InputNode, ModelLeaf, ModelSpec};

/// Reject any bare unknown field **or** unknown leaf kind in an env spec (the
/// PUBLISH gate: an author's own core must understand every kind and bare field).
/// The role identity rules (a closed kind prefix, no part on a legacy `_2`
/// role) are part of the same door: they hold under every role policy.
pub fn reject_unknowns_env(tags: &EnvTags) -> Result<(), String> {
    walk_obs(&tags.observation, &NodePath::root(), true)?;
    reject_action(&tags.action)?;
    reject_unsanctioned_roles_env(tags, RolePolicy::Passthrough)
}

/// Reject any bare unknown field **or** unknown leaf kind in a model spec.
pub fn reject_unknowns_model(spec: &ModelSpec) -> Result<(), String> {
    walk_input(&spec.input, &NodePath::root(), true)?;
    reject_action(&spec.output)?;
    reject_unsanctioned_roles_model(spec, RolePolicy::Passthrough)
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

/// The publish-gate policy for ad-hoc (unregistered) roles and parts, from
/// most to least permissive. A part is governed by the same tier as a role:
/// it is the same kind of vocabulary, grown the same way.
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

    /// Whether `part` is allowed under this policy (the same tiers as a role).
    fn allows_part(self, part: &str) -> bool {
        match self {
            RolePolicy::Passthrough => true,
            RolePolicy::Strict => crate::roles::parts::is_sanctioned_part(part),
            RolePolicy::Forbid => crate::roles::parts::is_known_part(part),
        }
    }
}

/// Reject one `(role, part)` that `policy` disallows, or whose identity is
/// malformed under any policy (an unknown kind prefix, a part on a legacy
/// `_2` role).
fn reject_role(
    role: &str,
    part: Option<&str>,
    locus: &str,
    policy: RolePolicy,
) -> Result<(), String> {
    crate::roles::registry::check_role(role, part)
        .map_err(|reason| format!("{locus}: {reason}"))?;
    if !policy.allows(role) {
        let hint = if policy == RolePolicy::Forbid {
            "use a blessed role (this gate forbids every unregistered role, including the `x/` escape)"
        } else {
            "use a blessed role, or the `x/` escape namespace for an intentionally non-standard one"
        };
        return Err(format!(
            "{locus} declares unregistered role {role:?}; {hint}"
        ));
    }
    if let Some(part) = part
        && !policy.allows_part(part)
    {
        let hint = if policy == RolePolicy::Forbid {
            "use a registered part (this gate forbids every unregistered part, including the `x/` escape)"
        } else {
            "use a registered part, or the `x/` escape namespace for an intentionally non-standard one"
        };
        return Err(format!(
            "{locus} declares unregistered part {part:?} on role {role:?}; {hint}"
        ));
    }
    Ok(())
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
        ObsLeaf::Image(tag) => reject_role(&tag.role, tag.part.as_deref(), &locus, policy),
        ObsLeaf::State(tag) => reject_role(&tag.role, tag.part.as_deref(), &locus, policy),
        ObsLeaf::Text(tag) => reject_role(&tag.role, None, &locus, policy),
        ObsLeaf::Split(layout) => layout
            .fields
            .iter()
            .try_for_each(|field| match &field.role {
                Some(role) => reject_role(role, field.part.as_deref(), &locus, policy),
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
        ModelLeaf::Image(input) => reject_role(&input.role, input.part.as_deref(), &locus, policy),
        // A role-less part is a declared constant, not a role claim: there is
        // nothing for the role tier to sanction.
        ModelLeaf::State(input) => input
            .components
            .iter()
            .filter_map(|part| {
                part.role
                    .as_deref()
                    .map(|role| (role, part.part.as_deref()))
            })
            .try_for_each(|(role, part)| reject_role(role, part, &locus, policy)),
        ModelLeaf::Text(input) => reject_role(&input.role, None, &locus, policy),
        ModelLeaf::Custom(_) | ModelLeaf::Unknown { .. } => Ok(()),
    }
}

fn reject_action_roles(action: &Action, policy: RolePolicy) -> Result<(), String> {
    for (index, actuator) in action.components.iter().enumerate() {
        if let Some(role) = &actuator.role {
            reject_role(
                role,
                actuator.part.as_deref(),
                &format!("action component[{index}]"),
                policy,
            )?;
        }
    }
    Ok(())
}

/// The publish-gate policy for the geometry attributes (`frame`, `reference`).
///
/// The registry says which roles they apply to ([`FrameLaw::Framed`],
/// [`ReferenceLaw::Referenced`]); this says whether a spec must actually
/// declare them. Opt-in, because an absent frame is legal v1 and every
/// pre-geometry spec would fail the strict tier.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FramePolicy {
    /// A declared frame is checked for agreement at resolve, but an absent one
    /// is fine -- the v1 default.
    Allow,
    /// Every role whose registry entry says a geometry attribute applies must
    /// declare it. The managed opt-in (`spec-normalize --require-frames`).
    Require,
}

/// Reject any role in an env spec that owes a geometry attribute and omits one.
pub fn reject_unframed_roles_env(tags: &EnvTags, policy: FramePolicy) -> Result<(), String> {
    if policy == FramePolicy::Allow {
        return Ok(());
    }
    walk_obs_frames(&tags.observation, &NodePath::root())?;
    reject_unframed_action(&tags.action)
}

/// Reject any role in a model spec that owes a geometry attribute and omits one.
pub fn reject_unframed_roles_model(spec: &ModelSpec, policy: FramePolicy) -> Result<(), String> {
    if policy == FramePolicy::Allow {
        return Ok(());
    }
    walk_input_frames(&spec.input, &NodePath::root())?;
    reject_unframed_action(&spec.output)
}

/// Reject one role that owes `attr` and declares nothing.
fn require_attr(
    role: &str,
    attr: Attr,
    declared: Option<&FrameRef>,
    locus: &str,
) -> Result<(), String> {
    let owed = match (attr, crate::roles::registry::role_def(role)) {
        (Attr::Frame, Some(def)) => def.frame == FrameLaw::Framed,
        (Attr::Reference, Some(def)) => def.reference == ReferenceLaw::Referenced,
        (_, None) => false,
    };
    if !owed || declared.is_some() {
        return Ok(());
    }
    Err(format!(
        "{locus} declares role {role:?} without a {}; this gate requires one of {:?}",
        attr.name(),
        attr.vocabulary()
    ))
}

fn walk_obs_frames(node: &ObsNode, path: &NodePath) -> Result<(), String> {
    match node {
        ObsNode::Leaf(leaf) => {
            let locus = format!("observation {:?}", path.to_string());
            match leaf {
                ObsLeaf::State(tag) => {
                    require_attr(&tag.role, Attr::Frame, tag.frame.as_ref(), &locus)
                }
                ObsLeaf::Split(layout) => {
                    layout
                        .fields
                        .iter()
                        .try_for_each(|field| match &field.role {
                            Some(role) => {
                                require_attr(role, Attr::Frame, field.frame.as_ref(), &locus)
                            }
                            None => Ok(()),
                        })
                }
                _ => Ok(()),
            }
        }
        ObsNode::Dict(map) => map
            .iter()
            .try_for_each(|(key, child)| walk_obs_frames(child, &path.push_key(key.clone()))),
        ObsNode::Tuple(items) => items
            .iter()
            .enumerate()
            .try_for_each(|(index, child)| walk_obs_frames(child, &path.push_index(index))),
    }
}

fn walk_input_frames(node: &InputNode, path: &NodePath) -> Result<(), String> {
    match node {
        InputNode::Leaf(ModelLeaf::State(input)) => {
            let locus = format!("model input {:?}", path.to_string());
            input
                .components
                .iter()
                .try_for_each(|part| match &part.role {
                    Some(role) => require_attr(role, Attr::Frame, part.frame.as_ref(), &locus),
                    None => Ok(()),
                })
        }
        InputNode::Leaf(_) => Ok(()),
        InputNode::Dict(map) => map
            .iter()
            .try_for_each(|(key, child)| walk_input_frames(child, &path.push_key(key.clone()))),
        InputNode::Tuple(items) => items
            .iter()
            .enumerate()
            .try_for_each(|(index, child)| walk_input_frames(child, &path.push_index(index))),
    }
}

fn reject_unframed_action(action: &Action) -> Result<(), String> {
    for (index, actuator) in action.components.iter().enumerate() {
        let Some(role) = &actuator.role else { continue };
        let locus = format!("action component[{index}]");
        require_attr(role, Attr::Frame, actuator.frame.as_ref(), &locus)?;
        require_attr(role, Attr::Reference, actuator.reference.as_ref(), &locus)?;
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
        // The SplitLayout envelope stays strict (its wire struct keeps deny);
        // its `Field` leaves are growable and carry a capture map.
        ObsLeaf::Split(layout) => {
            layout
                .fields
                .iter()
                .enumerate()
                .try_for_each(|(index, field)| {
                    bare_field_at(
                        &field.unknown,
                        &format!("feature {:?} field[{index}]", path.to_string()),
                    )
                })
        }
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
        ModelLeaf::State(input) => {
            bare_field(&input.unknown, path)?;
            input
                .components
                .iter()
                .enumerate()
                .try_for_each(|(index, part)| {
                    bare_field_at(
                        &part.unknown,
                        &format!("feature {:?} part[{index}]", path.to_string()),
                    )
                })
        }
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
        bare_field_at(
            &actuator.unknown,
            &format!(
                "action component[{index}] (role {:?})",
                actuator.role.as_deref().unwrap_or("opaque")
            ),
        )?;
    }
    Ok(())
}

fn bare_field(unknown: &BTreeMap<String, Value>, path: &NodePath) -> Result<(), String> {
    bare_field_at(unknown, &format!("feature {:?}", path.to_string()))
}

fn bare_field_at(unknown: &BTreeMap<String, Value>, locus: &str) -> Result<(), String> {
    match first_bare_field(unknown) {
        None => Ok(()),
        Some(field) => Err(format!(
            "{locus} declares unrecognized field {field:?}; upgrade the runtime \
             or drop the field (or prefix it `x-` to mark it ignorable)"
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
    fn inner_leaf_unknown_field_fails_the_gate() {
        // The growable *inner* leaves: a split layout's `Field` and a state
        // input's `ConcatPart`. Both parse leniently (this is where `frame`
        // landed before PR-11 typed it) and both are named by the publish gate.
        let dirty: EnvTags = serde_json::from_str(
            r#"{"observation": {"s": {"type": "split",
                    "fields": [{"role": "r", "dim": 1, "wobble": 1}]}},
                "action": {"components": []}}"#,
        )
        .unwrap();
        let err = reject_unknowns_env(&dirty).unwrap_err();
        assert!(
            err.contains("field[0]")
                && err.contains("\"wobble\"")
                && err.contains("prefix it `x-`"),
            "got: {err}"
        );
        assert!(reject_bare_fields_env(&dirty).is_err(), "read taint");

        let escaped: EnvTags = serde_json::from_str(
            r#"{"observation": {"s": {"type": "split",
                    "fields": [{"role": "r", "dim": 1, "x-note": 1}]}},
                "action": {"components": []}}"#,
        )
        .unwrap();
        assert!(reject_unknowns_env(&escaped).is_ok(), "`x-` is ignorable");

        let dirty: ModelSpec = serde_json::from_str(
            r#"{"input": {"type": "state", "components": [{"role": "r", "wobble": 1}]},
                "output": {"components": []}}"#,
        )
        .unwrap();
        let err = reject_unknowns_model(&dirty).unwrap_err();
        assert!(
            err.contains("part[0]") && err.contains("\"wobble\"") && err.contains("prefix it `x-`"),
            "got: {err}"
        );
    }

    #[test]
    fn require_frames_rejects_only_the_roles_that_owe_one() {
        use super::{FramePolicy, reject_unframed_roles_env, reject_unframed_roles_model};

        // The env declares a framed pose, a delta, and a gripper, all bare.
        // `Allow` (the v1 default) is silent; `Require` names the first owed one.
        let bare: EnvTags = serde_json::from_str(
            r#"{"observation": {"p": {"type": "state", "role": "proprio/eef_pos"}},
                "action": {"components": [
                    {"role": "action/delta_eef_pos", "dim": 3},
                    {"role": "action/gripper", "dim": 1}]}}"#,
        )
        .unwrap();
        assert!(reject_unframed_roles_env(&bare, FramePolicy::Allow).is_ok());
        let err = reject_unframed_roles_env(&bare, FramePolicy::Require).unwrap_err();
        assert!(
            err.contains("proprio/eef_pos") && err.contains("without a frame"),
            "got: {err}"
        );

        // Declared on every owed role, `Require` passes -- and the gripper, which
        // owes neither attribute, never had to say anything.
        let declared: EnvTags = serde_json::from_str(
            r#"{"observation": {"p": {"type": "state", "role": "proprio/eef_pos",
                    "frame": "robot_base"}},
                "action": {"components": [
                    {"role": "action/delta_eef_pos", "dim": 3, "reference": "current"},
                    {"role": "action/gripper", "dim": 1}]}}"#,
        )
        .unwrap();
        assert!(reject_unframed_roles_env(&declared, FramePolicy::Require).is_ok());

        // A delta owes a `reference`, not a `frame`.
        let no_reference: EnvTags = serde_json::from_str(
            r#"{"observation": {"c": {"type": "image", "role": "image/primary"}},
                "action": {"components": [{"role": "action/delta_eef_pos", "dim": 3}]}}"#,
        )
        .unwrap();
        let err = reject_unframed_roles_env(&no_reference, FramePolicy::Require).unwrap_err();
        assert!(
            err.contains("without a reference") && err.contains("current"),
            "got: {err}"
        );

        // The model side walks state parts and its own action layout.
        let model: ModelSpec = serde_json::from_str(
            r#"{"input": {"type": "state", "components": [{"role": "proprio/eef_pos", "dim": 3}]},
                "output": {"components": [{"role": "action/gripper", "dim": 1}]}}"#,
        )
        .unwrap();
        assert!(reject_unframed_roles_model(&model, FramePolicy::Allow).is_ok());
        let err = reject_unframed_roles_model(&model, FramePolicy::Require).unwrap_err();
        assert!(err.contains("proprio/eef_pos"), "got: {err}");
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

    #[test]
    fn parts_follow_the_role_tiers_and_kinds_hold_at_the_publish_door() {
        use super::{RolePolicy, reject_unsanctioned_roles_env, reject_unsanctioned_roles_model};

        // A registered part passes every tier; an ad-hoc one is nudged only,
        // Strict refuses it, and `x/` is the escape Strict allows and Forbid does not.
        let registered: EnvTags = serde_json::from_str(
            r#"{"observation": {"l": {"type": "state", "role": "proprio/eef_pos", "part": "left_arm"}},
                "action": {"components": [{"role": "action/gripper", "dim": 1, "part": "left_arm"}]}}"#,
        )
        .unwrap();
        assert!(reject_unsanctioned_roles_env(&registered, RolePolicy::Forbid).is_ok());

        let ad_hoc: EnvTags = serde_json::from_str(
            r#"{"observation": {"l": {"type": "state", "role": "proprio/eef_pos", "part": "franka"}},
                "action": {"components": []}}"#,
        )
        .unwrap();
        assert!(reject_unsanctioned_roles_env(&ad_hoc, RolePolicy::Passthrough).is_ok());
        let err = reject_unsanctioned_roles_env(&ad_hoc, RolePolicy::Strict).unwrap_err();
        assert!(
            err.contains(r#"unregistered part "franka" on role "proprio/eef_pos""#),
            "{err}"
        );

        let escaped: ModelSpec = serde_json::from_str(
            r#"{"input": {"type": "state", "components": [{"role": "proprio/eef_pos", "part": "x/tail"}]},
                "output": {"components": [{"role": "action/gripper", "dim": 1, "part": "x/tail"}]}}"#,
        )
        .unwrap();
        assert!(reject_unsanctioned_roles_model(&escaped, RolePolicy::Strict).is_ok());
        let err = reject_unsanctioned_roles_model(&escaped, RolePolicy::Forbid).unwrap_err();
        assert!(err.contains("including the `x/` escape"), "{err}");

        // The identity rules are not a tier: an unknown kind prefix and a part
        // on a legacy `_2` role fail the publish door under every policy.
        let bad_kind: EnvTags = serde_json::from_str(
            r#"{"observation": {"l": {"type": "state", "role": "audio/mic"}},
                "action": {"components": []}}"#,
        )
        .unwrap();
        let err = reject_unknowns_env(&bad_kind).unwrap_err();
        assert!(err.contains("kind this core does not define"), "{err}");
        let aliased: ModelSpec = serde_json::from_str(
            r#"{"input": {"type": "text", "role": "text/instruction"},
                "output": {"components": [{"role": "action/gripper_2", "dim": 1, "part": "left_arm"}]}}"#,
        )
        .unwrap();
        let err = reject_unknowns_model(&aliased).unwrap_err();
        assert!(err.contains("legacy spelling"), "{err}");
        // The READ taint stays tolerant of both (resolve is where they fail).
        assert!(reject_bare_fields_env(&bad_kind).is_ok());
    }
}
