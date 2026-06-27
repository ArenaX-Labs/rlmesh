//! Resolve env and model IO specs into a concrete adapter plan.

mod action;
mod custom;
mod image;
mod state;
mod text;

use std::collections::BTreeMap;

use super::error::{AdapterResolutionError, ErrorCode};
use super::fmt::quoted;
use super::join::join;
use super::path::NodePath;
use super::plans::{ObsPlan, ResolvedAdapter};
use super::space_view::SpaceView;
use super::spec::{
    EnvFeature, EnvImage, EnvState, EnvTags, EnvText, InputNode, ModelLeaf, ModelSpec,
};

type Result<T> = std::result::Result<T, AdapterResolutionError>;

fn err(code: ErrorCode, message: String) -> AdapterResolutionError {
    AdapterResolutionError::new(code, message)
}

/// Upgrade a would-be `MissingRole` to [`ErrorCode::UnsupportedKind`] when the
/// env declares the referenced `role` only as an *unrecognized observation
/// kind*. Called at each planner's role-lookup miss *before* any fallback
/// (optional zero-fill, text default, lone-camera bind): a role whose data the
/// env actually provides but under a kind this core cannot read is present-but-
/// unreadable, so we fail loud ("upgrade the runtime") rather than silently
/// degrading — and rather than misdirecting the operator to "add the role". A
/// `role` the env understands (not in `unknown_roles`) passes through.
pub(super) fn reject_referenced_unknown(
    role: &str,
    placement: &NodePath,
    unknown_roles: &BTreeMap<String, String>,
) -> Result<()> {
    if let Some(kind) = unknown_roles.get(role) {
        return Err(err(
            ErrorCode::UnsupportedKind,
            format!(
                "model input {} needs role {} but the env declares it as unrecognized \
                 observation kind {}; upgrade the runtime",
                quoted(&placement.to_string()),
                quoted(role),
                quoted(kind)
            ),
        ));
    }
    Ok(())
}

fn index_by_role<'spec, T>(
    features: impl Iterator<Item = (&'spec String, T)>,
    label: &str,
) -> Result<BTreeMap<String, T>> {
    let mut by_role: BTreeMap<String, T> = BTreeMap::new();
    for (role, feature) in features {
        if by_role.contains_key(role) {
            return Err(err(
                ErrorCode::Duplicate,
                format!("duplicate {label} role {}", quoted(role)),
            ));
        }
        by_role.insert(role.clone(), feature);
    }
    Ok(by_role)
}

/// One model input leaf paired with its placement (tree position) in the
/// payload, produced by walking the [`InputNode`] tree.
struct PlacedLeaf<'spec> {
    leaf: &'spec ModelLeaf,
    placement: NodePath,
}

/// Flatten the model input tree into a list of leaves, each carrying the
/// [`NodePath`] placement of where its produced tensor lands in the payload. A
/// `Dict` node recurses each key, a `Tuple` each index, a `Leaf` emits itself.
fn collect_leaves<'spec>(
    node: &'spec InputNode,
    placement: NodePath,
    out: &mut Vec<PlacedLeaf<'spec>>,
) {
    match node {
        InputNode::Leaf(leaf) => out.push(PlacedLeaf { leaf, placement }),
        InputNode::Dict(map) => {
            for (key, child) in map {
                collect_leaves(child, placement.push_key(key.clone()), out);
            }
        }
        InputNode::Tuple(items) => {
            for (index, item) in items.iter().enumerate() {
                collect_leaves(item, placement.push_index(index), out);
            }
        }
    }
}

/// Derive a [`ResolvedAdapter`] for an env/model pair.
///
/// The env side is given as the [`EnvTags`] observation/action tree over its
/// observation and action spaces; [`join`] derives the keyed env features (with
/// widths and ranges) those plus the spaces imply, then each model input leaf is
/// matched to an env feature by role and placed at its tree position. A model
/// role may be reused across leaves (one camera → several input slots).
pub fn resolve(
    env_tags: &EnvTags,
    observation_space: &SpaceView,
    action_space: &SpaceView,
    model_spec: &ModelSpec,
    trust_entrypoints: bool,
) -> Result<ResolvedAdapter> {
    // READ-door taint (§8): a *bare* (non-`x-`) unknown field on a recognized
    // kind is must-understand — an old core silently applying its own default
    // for a modifier it never parsed is worse than failing. Fail closed, on both
    // sides, before any plan is built. Unknown *kinds* are tolerated here (the
    // loop below ignores or fails them precisely); only fields taint.
    crate::spec::reject_bare_fields_env(env_tags)
        .map_err(|message| err(ErrorCode::UnsupportedKind, message))?;
    crate::spec::reject_bare_fields_model(model_spec)
        .map_err(|message| err(ErrorCode::UnsupportedKind, message))?;

    let env_spec = join(env_tags, observation_space, action_space)
        .map_err(|error| err(ErrorCode::InvalidTag, error.to_string()))?;
    let images = env_spec
        .observation
        .iter()
        .filter_map(|feature| match feature {
            EnvFeature::Image(image) => Some((&image.role, image)),
            _ => None,
        });
    let states = env_spec
        .observation
        .iter()
        .filter_map(|feature| match feature {
            EnvFeature::State(state) => Some((&state.role, state)),
            _ => None,
        });
    let texts = env_spec
        .observation
        .iter()
        .filter_map(|feature| match feature {
            EnvFeature::Text(text) => Some((&text.role, text)),
            _ => None,
        });
    let images_by_role: BTreeMap<String, &EnvImage> = index_by_role(images, "env image")?;
    let states_by_role: BTreeMap<String, &EnvState> = index_by_role(states, "env state")?;
    let texts_by_role: BTreeMap<String, &EnvText> = index_by_role(texts, "env text")?;

    // Side table for referenced-unknown detection: a role the env declares only
    // as an unrecognized observation kind. A roleless unknown leaf is
    // unreferenceable, so it never enters here (drop-only).
    let unknown_roles: BTreeMap<String, String> = env_spec
        .unknown
        .iter()
        .filter_map(|unknown| {
            unknown
                .role
                .clone()
                .map(|role| (role, unknown.kind.clone()))
        })
        .collect();

    let mut leaves: Vec<PlacedLeaf> = Vec::new();
    collect_leaves(&model_spec.input, NodePath::root(), &mut leaves);

    let mut obs_plans: Vec<ObsPlan> = Vec::with_capacity(leaves.len());
    for PlacedLeaf { leaf, placement } in leaves {
        obs_plans.push(match leaf {
            ModelLeaf::Image(input) => ObsPlan::Image(image::plan_image(
                input,
                placement,
                &images_by_role,
                &unknown_roles,
            )?),
            ModelLeaf::State(input) => ObsPlan::State(state::plan_state(
                input,
                placement,
                &states_by_role,
                &unknown_roles,
            )?),
            ModelLeaf::Text(input) => ObsPlan::Text(text::plan_text(
                input,
                placement,
                &texts_by_role,
                &unknown_roles,
            )?),
            ModelLeaf::Custom(input) => {
                ObsPlan::Custom(custom::plan_custom(input, placement, trust_entrypoints)?)
            }
            // A model input of an unrecognized kind has no apply path on an old
            // core, even if the env offers a matching unknown feature — agreement
            // is irrelevant. Localized, named by placement.
            ModelLeaf::Unknown { kind, .. } => {
                return Err(err(
                    ErrorCode::UnsupportedKind,
                    format!(
                        "model input {} is of unrecognized kind {}; upgrade the runtime",
                        quoted(&placement.to_string()),
                        quoted(kind)
                    ),
                ));
            }
        });
    }

    // Reaching here means no model input referenced an unknown kind (a reference
    // hard-errors above), so every recorded unknown leaf was unreferenced: emit
    // one deterministic advisory per leaf, sorted by tree path. Run proceeds.
    let mut advisories: Vec<String> = env_spec
        .unknown
        .iter()
        .map(|unknown| {
            format!(
                "env feature {} (role {}): unrecognized kind {}; ignored (no model input requires it)",
                quoted(&unknown.source.to_string()),
                quoted(unknown.role.as_deref().unwrap_or("<none>")),
                quoted(&unknown.kind)
            )
        })
        .collect();
    advisories.sort();

    let action_plan = action::plan_action(&model_spec.output, &env_spec.action)?;
    let resolved = ResolvedAdapter::new(obs_plans, action_plan, advisories);
    // The frame-stacking × action-chunk-replay guard used to live here, but the
    // replay horizon is no longer part of the spec — it is a runtime decision
    // (`action_horizon` on ConfigureRoute). The guard moved to the engine's
    // configure_route, where the resolved stacks and the runtime horizon are both
    // known; see `AdaptedRouteSetup::configure_route`.
    Ok(resolved)
}

#[cfg(test)]
mod unknown_kind_tests {
    use super::resolve;
    use crate::error::ErrorCode;
    use crate::space_view::SpaceView;
    use crate::spec::{EnvTags, ModelSpec};

    fn space(json: &str) -> SpaceView {
        serde_json::from_str(json).expect("parse space")
    }

    fn do_resolve(
        env_tags: &str,
        obs_space: &str,
        action_space: &str,
        model_spec: &str,
    ) -> Result<crate::plans::ResolvedAdapter, crate::error::AdapterResolutionError> {
        let tags: EnvTags = serde_json::from_str(env_tags).expect("parse env tags");
        let spec: ModelSpec = serde_json::from_str(model_spec).expect("parse model spec");
        resolve(&tags, &space(obs_space), &space(action_space), &spec, false)
    }

    // A one-actuator action shared by the cases below.
    const ACTION_SPACE: &str = r#"{"kind":"box","shape":[1],"dtype":"float32"}"#;
    const ACTION_OUT: &str = r#"{"components":[{"role":"a","dim":1}]}"#;
    const ACTION_TAGS: &str = r#"{"components":[{"role":"a","dim":1}]}"#;

    #[test]
    fn unreferenced_unknown_obs_kind_resolves_with_advisory() {
        // The env declares an `audio` observation an old core can't build. The
        // model references only the camera, so resolution succeeds and the
        // unknown leaf is ignored with one deterministic advisory.
        let env_tags = format!(
            r#"{{"observation":{{"cam":{{"type":"image","role":"image/primary"}},
                "mic":{{"type":"audio","role":"audio/mic","sample_rate":16000}}}},
                "action":{ACTION_TAGS}}}"#
        );
        let obs_space = r#"{"kind":"dict","dtype":"unspecified","keys":["cam","mic"],"children":[
            {"kind":"box","shape":[4,4,3],"dtype":"uint8"},
            {"kind":"box","shape":[16],"dtype":"float32"}]}"#;
        let model = format!(
            r#"{{"input":{{"pixels":{{"type":"image","role":"image/primary"}}}},"output":{ACTION_OUT}}}"#
        );
        let adapter = do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect("resolves");
        let advisories = adapter.advisories();
        assert!(
            advisories
                .iter()
                .any(|a| a.contains("mic") && a.contains("audio")),
            "expected an unknown-kind advisory, got: {advisories:?}"
        );
        // The dropped modality is also surfaced in the human summary (it produces
        // no obs plan, so without this it would be invisible to describe()).
        let described = adapter.describe();
        assert!(
            described.contains("dropped:") && described.contains("mic"),
            "expected a dropped-modality note in describe(), got:\n{described}"
        );
    }

    #[test]
    fn referenced_unknown_obs_kind_is_unsupported_kind() {
        // The model needs role "weird", which the env offers only as an
        // unrecognized kind: a localized UnsupportedKind ("upgrade the runtime"),
        // not a misdirecting MissingRole.
        let env_tags = format!(
            r#"{{"observation":{{"sensor":{{"type":"lidar","role":"weird"}}}},"action":{ACTION_TAGS}}}"#
        );
        let obs_space = r#"{"kind":"dict","dtype":"unspecified","keys":["sensor"],"children":[
            {"kind":"box","shape":[4],"dtype":"float32"}]}"#;
        let model = format!(
            r#"{{"input":{{"s":{{"type":"state","components":["weird"]}}}},"output":{ACTION_OUT}}}"#
        );
        let err = do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect_err("unsupported");
        assert_eq!(err.code, ErrorCode::UnsupportedKind);
        assert!(
            err.message.contains("upgrade the runtime"),
            "got: {}",
            err.message
        );
    }

    #[test]
    fn optional_state_referencing_unknown_kind_is_unsupported_kind() {
        // An *optional* component still fails loud when the env provides its role
        // under an unrecognized kind: the data exists but is unreadable here, so
        // the operator is told to upgrade rather than being silently fed zeros.
        let env_tags = format!(
            r#"{{"observation":{{"sensor":{{"type":"lidar","role":"weird"}}}},"action":{ACTION_TAGS}}}"#
        );
        let obs_space = r#"{"kind":"dict","dtype":"unspecified","keys":["sensor"],"children":[
            {"kind":"box","shape":[4],"dtype":"float32"}]}"#;
        let model = format!(
            r#"{{"input":{{"s":{{"type":"state","components":[{{"role":"weird","dim":4,"optional":true}}]}}}},"output":{ACTION_OUT}}}"#
        );
        let err = do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect_err("unsupported");
        assert_eq!(err.code, ErrorCode::UnsupportedKind);
        assert!(
            err.message.contains("upgrade the runtime"),
            "got: {}",
            err.message
        );
    }

    #[test]
    fn defaulted_text_referencing_unknown_kind_is_unsupported_kind() {
        // A text input with a default still fails loud when the env provides its
        // role under an unrecognized kind -- the default must not mask present-
        // but-unreadable data.
        let env_tags = format!(
            r#"{{"observation":{{"note":{{"type":"richtext","role":"instruction"}}}},"action":{ACTION_TAGS}}}"#
        );
        let obs_space = r#"{"kind":"dict","dtype":"unspecified","keys":["note"],"children":[
            {"kind":"box","shape":[4],"dtype":"float32"}]}"#;
        let model = format!(
            r#"{{"input":{{"t":{{"type":"text","role":"instruction","default":"hi"}}}},"output":{ACTION_OUT}}}"#
        );
        let err = do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect_err("unsupported");
        assert_eq!(err.code, ErrorCode::UnsupportedKind);
        assert!(
            err.message.contains("upgrade the runtime"),
            "got: {}",
            err.message
        );
    }

    #[test]
    fn lone_camera_fallback_does_not_bind_an_unknown_kind_role() {
        // With exactly one real camera, the lone-camera fallback papers over
        // role-name mismatches -- but not when the requested role is the env's
        // *unknown-kind* leaf: that fails loud (upgrade), it does not silently
        // bind the wrong camera.
        let env_tags = format!(
            r#"{{"observation":{{"cam":{{"type":"image","role":"image/primary"}},
                "extra":{{"type":"lidar","role":"image/overhead"}}}},"action":{ACTION_TAGS}}}"#
        );
        let obs_space = r#"{"kind":"dict","dtype":"unspecified","keys":["cam","extra"],"children":[
            {"kind":"box","shape":[4,4,3],"dtype":"uint8"},
            {"kind":"box","shape":[4],"dtype":"float32"}]}"#;
        let model = format!(
            r#"{{"input":{{"pixels":{{"type":"image","role":"image/overhead"}}}},"output":{ACTION_OUT}}}"#
        );
        let err = do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect_err("unsupported");
        assert_eq!(err.code, ErrorCode::UnsupportedKind);
        assert!(
            err.message.contains("upgrade the runtime"),
            "got: {}",
            err.message
        );
    }

    #[test]
    fn model_input_of_unknown_kind_is_unsupported_kind() {
        // A model input of an unrecognized kind has no apply path on an old core;
        // it fails at resolve regardless of what the env offers.
        let env_tags = format!(
            r#"{{"observation":{{"cam":{{"type":"image","role":"image/primary"}}}},"action":{ACTION_TAGS}}}"#
        );
        let obs_space = r#"{"kind":"dict","dtype":"unspecified","keys":["cam"],"children":[
            {"kind":"box","shape":[4,4,3],"dtype":"uint8"}]}"#;
        let model = format!(
            r#"{{"input":{{"x":{{"type":"audio","role":"image/primary"}}}},"output":{ACTION_OUT}}}"#
        );
        let err = do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect_err("unsupported");
        assert_eq!(err.code, ErrorCode::UnsupportedKind);
        assert!(
            err.message.contains("unrecognized kind"),
            "got: {}",
            err.message
        );
    }

    #[test]
    fn bare_unknown_field_on_known_kind_taints_at_resolve() {
        // §8 central asymmetry: a bare additive field on a recognized kind is
        // must-understand — fail closed at resolve even though the leaf is
        // referenced and otherwise valid.
        let env_tags = format!(
            r#"{{"observation":{{"cam":{{"type":"image","role":"image/primary","normalize":false}}}},"action":{ACTION_TAGS}}}"#
        );
        let obs_space = r#"{"kind":"dict","dtype":"unspecified","keys":["cam"],"children":[
            {"kind":"box","shape":[4,4,3],"dtype":"uint8"}]}"#;
        let model = format!(
            r#"{{"input":{{"pixels":{{"type":"image","role":"image/primary"}}}},"output":{ACTION_OUT}}}"#
        );
        let err = do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect_err("tainted");
        assert_eq!(err.code, ErrorCode::UnsupportedKind);
        assert!(err.message.contains("normalize"), "got: {}", err.message);
    }

    #[test]
    fn x_prefixed_field_does_not_taint_at_resolve() {
        // The producer's `x-` opt-out: a marked-ignorable field resolves cleanly.
        let env_tags = format!(
            r#"{{"observation":{{"cam":{{"type":"image","role":"image/primary","x-note":"hi"}}}},"action":{ACTION_TAGS}}}"#
        );
        let obs_space = r#"{"kind":"dict","dtype":"unspecified","keys":["cam"],"children":[
            {"kind":"box","shape":[4,4,3],"dtype":"uint8"}]}"#;
        let model = format!(
            r#"{{"input":{{"pixels":{{"type":"image","role":"image/primary"}}}},"output":{ACTION_OUT}}}"#
        );
        do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect("x- field tolerated");
    }
}
