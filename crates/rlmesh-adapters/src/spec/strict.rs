//! The strict-v1 gate, a post-parse pass separate from the always-tolerant serde codec.
//!
//! - An unknown *bare* field on a recognized kind is must-understand and fails, at both
//!   publish and read. A field in the reserved `x-`/`ext.` namespace is safe to ignore and
//!   passes.
//! - An unknown *kind* fails at publish only. Read keeps it for relay and the resolver
//!   decides: ignored with an advisory unless a model input references it, then
//!   `UnsupportedKind`.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::path::NodePath;

use super::action::Action;
use super::env_tags::{EnvTags, ObsLeaf, ObsNode};
use super::model::{InputNode, ModelLeaf, ModelSpec};
use super::{Attr, FrameLaw, FrameRef, ReferenceLaw};

/// Reject any bare unknown field **or** unknown leaf kind in an env spec (the
/// PUBLISH gate: an author's own core must understand every kind and bare field).
pub fn reject_unknowns_env(tags: &EnvTags) -> Result<(), String> {
    walk_obs(&tags.observation, &NodePath::root(), true)?;
    reject_action(&tags.action)?;
    reject_unsanctioned_roles_env(tags, RolePolicy::Passthrough)
}

/// Reject any bare unknown field **or** unknown leaf kind in a model spec.
pub fn reject_unknowns_model(spec: &ModelSpec) -> Result<(), String> {
    spec.timing.check()?;
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

/// The publish-gate policy for ad-hoc (unregistered) roles, from most to
/// least permissive.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RolePolicy {
    /// Every role passes -- the open vocabulary (an authoring nudge only).
    Passthrough,
    /// Ad-hoc roles are rejected, but the `x/` escape is allowed: registered roles
    /// or explicit escapes, no *accidental* ad-hoc.
    Strict,
    /// Every unregistered role is rejected, even the `x/` escape: a registered-only
    /// lockdown, the absolute tier.
    Forbid,
}

impl RolePolicy {
    fn allows(self, role: &str) -> bool {
        match self {
            RolePolicy::Passthrough => true,
            RolePolicy::Strict => crate::roles::registry::is_sanctioned_role(role),
            RolePolicy::Forbid => crate::roles::registry::is_known_role(role),
        }
    }
}

fn reject_role(role: &str, locus: &str, policy: RolePolicy) -> Result<(), String> {
    crate::roles::registry::check_role(role).map_err(|reason| format!("{locus}: {reason}"))?;
    if !policy.allows(role) {
        let hint = if policy == RolePolicy::Forbid {
            "use a blessed role (this gate forbids every unregistered role, including the `x/` escape)"
        } else {
            "use a blessed role, or the `x/` escape namespace for an intentionally non-standard one"
        };
        let kindless = crate::roles::registry::kindless_hint(role)
            .map(|hint| format!(". {hint}"))
            .unwrap_or_default();
        return Err(format!(
            "{locus} declares unregistered role {role:?}; {hint}{kindless}"
        ));
    }
    Ok(())
}

/// Reject any role an env spec declares that `policy` disallows.
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
            .filter_map(|part| part.role.as_deref())
            .try_for_each(|role| reject_role(role, &locus, policy)),
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

/// The publish-gate policy for the geometry attributes (`frame`, `reference`).
///
/// The registry says which roles they apply to ([`FrameLaw::Framed`],
/// [`ReferenceLaw::Referenced`]); this says whether a spec must declare them. Opt-in: an
/// absent frame is legal v1.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FramePolicy {
    /// A declared frame is checked for agreement at resolve, but an absent one
    /// is fine -- the v1 default.
    Allow,
    /// Every role whose registry entry says a geometry attribute applies must
    /// declare it.
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

fn require_attr(
    role: &str,
    attr: Attr,
    declared: Option<&FrameRef>,
    locus: &str,
) -> Result<(), String> {
    let owed = match (attr, crate::roles::registry::role_def(role)) {
        (Attr::Frame, Some(def)) => def.frame == FrameLaw::Framed,
        (Attr::Reference, Some(def)) => def.reference == ReferenceLaw::Referenced,
        (Attr::Provenance, Some(_)) | (_, None) => false,
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

fn is_reserved_ext(key: &str) -> bool {
    key.starts_with("x-") || key.starts_with("ext.")
}

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

fn unknown_kind_msg(domain: &str, kind: &str, path: &NodePath) -> String {
    format!(
        "{domain} {:?} declares unrecognized kind {kind:?}; this core cannot build it \
         (upgrade the runtime, or this spec cannot be published here)",
        path.to_string()
    )
}

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

        let declared: EnvTags = serde_json::from_str(
            r#"{"observation": {"p": {"type": "state", "role": "proprio/eef_pos",
                    "frame": "robot_base"}},
                "action": {"components": [
                    {"role": "action/delta_eef_pos", "dim": 3, "reference": "current"},
                    {"role": "action/gripper", "dim": 1}]}}"#,
        )
        .unwrap();
        assert!(reject_unframed_roles_env(&declared, FramePolicy::Require).is_ok());

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
        assert!(!err.contains("names no kind"), "{err}");

        let kindless: EnvTags = serde_json::from_str(
            r#"{"observation": {"cam": {"type": "image", "role": "image/primary"}},
                "action": {"components": [{"role": "wiggle", "dim": 1}]}}"#,
        )
        .unwrap();
        assert!(reject_unknowns_env(&kindless).is_ok());
        let err = reject_unsanctioned_roles_env(&kindless, RolePolicy::Strict).unwrap_err();
        assert!(
            err.contains("names no kind") && err.contains("e.g. x/wiggle"),
            "{err}"
        );
        assert!(!err.contains("  "), "{err:?}");
    }

    #[test]
    fn parts_are_never_tiered_and_kinds_hold_at_the_publish_door() {
        use super::{RolePolicy, reject_unsanctioned_roles_env, reject_unsanctioned_roles_model};

        let custom: EnvTags = serde_json::from_str(
            r#"{"observation": {"l": {"type": "state", "role": "proprio/eef_pos", "part": "franka"}},
                "action": {"components": [{"role": "action/gripper", "dim": 1, "part": "tail"}]}}"#,
        )
        .unwrap();
        assert!(reject_unsanctioned_roles_env(&custom, RolePolicy::Forbid).is_ok());
        let custom: ModelSpec = serde_json::from_str(
            r#"{"input": {"type": "state", "components": [{"role": "proprio/eef_pos", "part": "x/tail"}]},
                "output": {"components": [{"role": "action/gripper", "dim": 1, "part": "tail"}]}}"#,
        )
        .unwrap();
        assert!(reject_unsanctioned_roles_model(&custom, RolePolicy::Forbid).is_ok());

        let bad_kind: EnvTags = serde_json::from_str(
            r#"{"observation": {"l": {"type": "state", "role": "audio/mic"}},
                "action": {"components": []}}"#,
        )
        .unwrap();
        let err = reject_unknowns_env(&bad_kind).unwrap_err();
        assert!(err.contains("kind this core does not define"), "{err}");
        assert!(reject_bare_fields_env(&bad_kind).is_ok());
    }
}
