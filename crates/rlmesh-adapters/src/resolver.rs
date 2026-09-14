//! Resolve env and model IO specs into a concrete adapter plan.

mod action;
mod custom;
mod image;
mod state;
mod text;

use std::collections::{BTreeMap, BTreeSet};

use super::advisory::Advisory;
use super::error::{AdapterResolutionError, ErrorCode};
use super::fmt::quoted;
use super::join::join;
use super::path::NodePath;
use super::plans::{ObsPlan, ResolvedAdapter};
use super::space_view::SpaceView;
use super::spec::{
    Attr, EnvFeature, EnvImage, EnvState, EnvTags, EnvText, FrameRef, InputNode, ModelLeaf,
    ModelSpec,
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

/// The C14 geometry rules, shared verbatim by `frame` (state parts and action
/// components) and `reference` (delta action components) -- they differ only in
/// their vocabulary and wording, carried by [`Attr`].
///
/// Returns the *agreed* value to record on the plan, and pushes at most one
/// advisory. The five cases:
///
/// 1. both silent -- nothing to check, nothing said.
/// 2. env declares, model silent -- the env stated a fact the model has no
///    requirement against; the value carries through, silently.
/// 3. model declares, env silent -- the model states a requirement the env
///    cannot confirm: `caution` (it fires only when a model opts in).
/// 4. both declare, equal -- agreement, silently.
/// 5. both declare, differing -- a hard [`ErrorCode::FrameMismatch`]. So is an
///    unrecognized value on either side: a geometry this core cannot name is a
///    geometry it cannot verify (the tolerant codec still relays it).
pub(super) fn check_geometry(
    attr: Attr,
    role: &str,
    env: Option<&FrameRef>,
    model: Option<&FrameRef>,
    advisories: &mut Vec<Advisory>,
) -> Result<Option<FrameRef>> {
    let name = attr.name();
    for (side, declared) in [("env", env), ("model", model)] {
        if let Some(value) = declared
            && !attr.recognizes(value)
        {
            return Err(err(
                ErrorCode::FrameMismatch,
                format!(
                    "role {}: the {side} declares unrecognized {name} {}; this core knows {:?}",
                    quoted(role),
                    quoted(value.as_str()),
                    attr.vocabulary()
                ),
            ));
        }
    }
    match (env, model) {
        (None, None) => Ok(None),
        (Some(env), None) => Ok(Some(env.clone())),
        (None, Some(model)) => {
            advisories.push(Advisory::caution(format!(
                "role {}: the model declares {name} {} but the env declares none, so the \
                 {name} it was trained against cannot be verified -- declare it on the env \
                 to silence this",
                quoted(role),
                quoted(model.as_str()),
            )));
            Ok(Some(model.clone()))
        }
        (Some(env), Some(model)) if env == model => Ok(Some(env.clone())),
        (Some(env), Some(model)) => Err(err(
            ErrorCode::FrameMismatch,
            format!(
                "role {}: the model expects {name} {} but the env declares {}",
                quoted(role),
                quoted(model.as_str()),
                quoted(env.as_str()),
            ),
        )),
    }
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

    // The quiet channel: surfaced through `advisories()` but kept out of
    // `describe()`. The geometry cautions belong here -- nothing was dropped or
    // fabricated, so they have no business in the `dropped:` section describe()
    // builds from the other channel.
    let mut quiet: Vec<Advisory> = env_spec.advisories.clone();
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
                &mut quiet,
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
    let mut advisories: Vec<Advisory> = env_spec
        .unknown
        .iter()
        .map(|unknown| {
            Advisory::info(format!(
                "env feature {} (role {}): unrecognized kind {}; ignored (no model input requires it)",
                quoted(&unknown.source.to_string()),
                quoted(unknown.role.as_deref().unwrap_or("<none>")),
                quoted(&unknown.kind)
            ))
        })
        .collect();

    let action_plan = action::plan_action(&model_spec.output, &env_spec.action, &mut quiet)?;

    // Model-side ad-hoc roles the env does not answer. An ad-hoc role matches
    // only on the exact string, so a typo or a private name silently degrades
    // to a zero fill (an optional input) or to dropped output dims (an unmatched
    // model actuator) -- the required cases already hard-errored above. One
    // advisory per role, not per placement.
    let mut leaves: Vec<PlacedLeaf> = Vec::new();
    collect_leaves(&model_spec.input, NodePath::root(), &mut leaves);
    let mut ad_hoc: BTreeSet<&str> = BTreeSet::new();
    for PlacedLeaf { leaf, .. } in &leaves {
        match leaf {
            ModelLeaf::Image(input) if !images_by_role.contains_key(&input.role) => {
                ad_hoc.insert(&input.role);
            }
            ModelLeaf::State(input) => ad_hoc.extend(
                input
                    .components
                    .iter()
                    .filter_map(|part| part.role.as_deref())
                    .filter(|role| !states_by_role.contains_key(*role)),
            ),
            ModelLeaf::Text(input) if !texts_by_role.contains_key(&input.role) => {
                ad_hoc.insert(&input.role);
            }
            _ => {}
        }
    }
    let env_action_roles: BTreeSet<&str> = env_spec
        .action
        .components
        .iter()
        .filter_map(|actuator| actuator.role.as_deref())
        .collect();
    ad_hoc.extend(
        model_spec
            .output
            .components
            .iter()
            .filter_map(|actuator| actuator.role.as_deref())
            .filter(|role| !env_action_roles.contains(role)),
    );
    advisories.extend(
        ad_hoc
            .into_iter()
            .filter(|role| !crate::roles::registry::is_sanctioned_role(role))
            .map(|role| {
                Advisory::info(format!(
                    "model declares ad-hoc role {} that this env does not: an ad-hoc role \
                     matches only on the exact string, so it resolves to a fill here -- use \
                     a registered role, or the {} namespace to mark it intentionally \
                     non-standard",
                    quoted(role),
                    quoted("x/"),
                ))
            }),
    );

    // A model-declared range only fires as an affine map when the *env* side also
    // declares a range to bridge between; against an unbounded env feature the map
    // is skipped, so a lone model range silently does nothing (it remaps from/into
    // an env range, it does not clamp). Surface that so the author learns it will
    // not take effect. (State: env is the source; action: env is the destination.)
    for obs_plan in &obs_plans {
        if let ObsPlan::Image(image) = obs_plan
            && let Some((requested, bound)) = &image.role_rebound
        {
            advisories.push(Advisory::caution(format!(
                "model input {}: no env camera has role {}; bound to the env's                  only camera ({}) instead -- declare the matching role on one                  side to silence this",
                quoted(&image.placement.to_string()),
                quoted(requested),
                quoted(bound),
            )));
        }
        if let ObsPlan::State(state) = obs_plan
            && state.pieces.iter().any(|piece| {
                piece.fill.is_none() && piece.dst_range.is_some() && piece.src_range.is_none()
            })
        {
            advisories.push(Advisory::info(format!(
                "model input {}: a state range is set but the env feature is \
                     unbounded, so the range is a no-op (it remaps an env range, it \
                     does not clamp)",
                quoted(&state.placement.to_string()),
            )));
        }
        // Two rescalings on one part compose silently: `range` bridges the two
        // declared scales, then `scale`/`offset` applies on top of the result.
        if let ObsPlan::State(state) = obs_plan
            && state.pieces.iter().any(|piece| {
                piece.dst_range.is_some() && (piece.scale.is_some() || piece.offset.is_some())
            })
        {
            advisories.push(Advisory::info(format!(
                "model input {}: a state part sets both range and scale/offset; the \
                     range map runs first and the affine applies to its result",
                quoted(&state.placement.to_string()),
            )));
        }
    }
    for segment in &action_plan.segments {
        if segment.src_range.is_some() && segment.dst_range.is_none() {
            advisories.push(Advisory::info(format!(
                "model action (role {}): a range is set but the env actuator is \
                 unbounded, so the range is a no-op (it remaps into an env range, it \
                 does not clamp)",
                quoted(segment.role.as_deref().unwrap_or("?")),
            )));
        }
    }
    advisories.sort_by(|a, b| a.message.cmp(&b.message));
    quiet.sort_by(|a, b| a.message.cmp(&b.message));

    let resolved = ResolvedAdapter::new(obs_plans, action_plan, advisories, quiet);
    // The frame-stacking × action-chunk-replay guard used to live here, but the
    // execution horizon is no longer part of the spec — it is a runtime decision
    // (`execution_horizon` on ResolveAdapter). The guard moved to the engine's
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
                .any(|a| a.message.contains("mic") && a.message.contains("audio")),
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
    fn filled_text_referencing_unknown_kind_is_unsupported_kind() {
        // A text input with a fill still fails loud when the env provides its
        // role under an unrecognized kind -- the fill must not mask present-
        // but-unreadable data.
        let env_tags = format!(
            r#"{{"observation":{{"note":{{"type":"richtext","role":"instruction"}}}},"action":{ACTION_TAGS}}}"#
        );
        let obs_space = r#"{"kind":"dict","dtype":"unspecified","keys":["note"],"children":[
            {"kind":"box","shape":[4],"dtype":"float32"}]}"#;
        let model = format!(
            r#"{{"input":{{"t":{{"type":"text","role":"instruction","fill":"hi"}}}},"output":{ACTION_OUT}}}"#
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

    /// Deliberate invariant: join advisories (env-declaration hints like a
    /// mislabeled image layout) surface through `advisories()` but stay OUT of
    /// `describe()` — they are not dropped env modalities, and the conformance
    /// vectors pin describe()'s exact text including its `dropped:` section.
    #[test]
    fn image_layout_advisory_surfaces_in_advisories_but_not_describe() {
        let env_tags = format!(
            r#"{{"observation":{{"cam":{{"type":"image","role":"image/primary","layout":"hwc"}}}},"action":{ACTION_TAGS}}}"#
        );
        let obs_space = r#"{"kind":"dict","dtype":"unspecified","keys":["cam"],"children":[
            {"kind":"box","shape":[3,224,224],"dtype":"uint8"}]}"#;
        let model = format!(
            r#"{{"input":{{"pixels":{{"type":"image","role":"image/primary"}}}},"output":{ACTION_OUT}}}"#
        );
        let adapter = do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect("resolves");
        let advisories = adapter.advisories();
        assert!(
            advisories
                .iter()
                .any(|a| a.message.contains("layout=hwc") && a.message.contains("looks like chw")),
            "expected the join layout hint in advisories(), got: {advisories:?}"
        );
        let described = adapter.describe();
        assert!(
            !described.contains("looks like chw") && !described.contains("dropped:"),
            "join advisories must stay out of describe(), got:\n{described}"
        );
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

    #[test]
    fn ad_hoc_role_the_env_lacks_draws_one_info_advisory() {
        // The optional part zero-fills instead of hard-erroring, so without the
        // advisory the author never learns their private role matched nothing.
        let env_tags = format!(
            r#"{{"observation":{{"cam":{{"type":"image","role":"image/primary"}}}},"action":{ACTION_TAGS}}}"#
        );
        let obs_space = r#"{"kind":"dict","dtype":"unspecified","keys":["cam"],"children":[
            {"kind":"box","shape":[4,4,3],"dtype":"uint8"}]}"#;
        let model = format!(
            r#"{{"input":{{"s":{{"type":"state","components":[
                {{"role":"proprio/made_up","dim":3,"optional":true}}]}}}},"output":{ACTION_OUT}}}"#
        );
        let adapter = do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect("resolves");
        let advisories = adapter.advisories();
        let hits: Vec<&crate::advisory::Advisory> = advisories
            .iter()
            .filter(|a| a.message.contains("ad-hoc role"))
            .collect();
        assert_eq!(hits.len(), 1, "one advisory per role, got: {hits:?}");
        assert!(hits[0].message.contains("proprio/made_up"), "{hits:?}");
        // A registered role the env also lacks is a contract, not a typo: silent.
        let model = format!(
            r#"{{"input":{{"s":{{"type":"state","components":[
                {{"role":"proprio/eef_pos_2","dim":3,"optional":true}}]}}}},"output":{ACTION_OUT}}}"#
        );
        let adapter = do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect("resolves");
        assert!(
            !adapter
                .advisories()
                .iter()
                .any(|a| a.message.contains("ad-hoc role")),
            "{:?}",
            adapter.advisories()
        );
    }

    #[test]
    fn a_second_arm_camera_role_never_rebinds_to_the_only_camera() {
        let env_tags = format!(
            r#"{{"observation":{{"cam":{{"type":"image","role":"image/primary"}}}},"action":{ACTION_TAGS}}}"#
        );
        let obs_space = r#"{"kind":"dict","dtype":"unspecified","keys":["cam"],"children":[
            {"kind":"box","shape":[4,4,3],"dtype":"uint8"}]}"#;
        let model = format!(
            r#"{{"input":{{"pixels":{{"type":"image","role":"image/wrist_2"}}}},"output":{ACTION_OUT}}}"#
        );
        let err = do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect_err("no rebind");
        assert_eq!(err.code, ErrorCode::MissingRole);
        // The first-arm wrist still rebinds (with the caution) -- only `_2` is
        // barred, because a lone camera cannot be the second of a pair.
        let model = format!(
            r#"{{"input":{{"pixels":{{"type":"image","role":"image/wrist"}}}},"output":{ACTION_OUT}}}"#
        );
        let adapter = do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect("rebinds");
        assert!(
            adapter
                .advisories()
                .iter()
                .any(|a| a.message.contains("only camera")),
            "{:?}",
            adapter.advisories()
        );
    }
}

/// The C14 geometry rules, exercised directly on [`check_geometry`] so each of
/// the five cases is pinned once rather than through a full resolve.
#[cfg(test)]
mod geometry_rule_tests {
    use super::check_geometry;
    use crate::error::ErrorCode;
    use crate::spec::{Attr, FrameRef};

    fn check(
        attr: Attr,
        env: Option<&str>,
        model: Option<&str>,
    ) -> (
        Result<Option<FrameRef>, crate::error::AdapterResolutionError>,
        Vec<crate::advisory::Advisory>,
    ) {
        let mut notes = Vec::new();
        let result = check_geometry(
            attr,
            "proprio/eef_pos",
            env.map(FrameRef::from).as_ref(),
            model.map(FrameRef::from).as_ref(),
            &mut notes,
        );
        (result, notes)
    }

    #[test]
    fn agreement_and_env_only_are_silent() {
        // Cases 1, 2 and 4: nothing declared, only the env declared, both
        // declaring the same thing. Every one resolves without a word.
        for (env, model, expected) in [
            (None, None, None),
            (Some("robot_base"), None, Some("robot_base")),
            (Some("world"), Some("world"), Some("world")),
        ] {
            let (result, notes) = check(Attr::Frame, env, model);
            assert_eq!(
                result.expect("resolves").as_ref().map(FrameRef::as_str),
                expected
            );
            assert!(notes.is_empty(), "expected silence, got {notes:?}");
        }
    }

    #[test]
    fn a_model_only_declaration_is_a_caution() {
        // Case 6: the model states a requirement the env cannot confirm. It
        // resolves (the value carries to the plan) with one caution.
        let (result, notes) = check(Attr::Frame, None, Some("robot_base"));
        assert_eq!(
            result.expect("resolves").as_ref().map(FrameRef::as_str),
            Some("robot_base")
        );
        assert_eq!(notes.len(), 1);
        assert_eq!(
            notes[0].severity,
            crate::advisory::AdvisorySeverity::Caution
        );
        assert!(
            notes[0].message.contains("declares frame \"robot_base\"")
                && notes[0].message.contains("cannot be verified"),
            "got: {}",
            notes[0].message
        );
    }

    #[test]
    fn a_disagreement_is_a_hard_frame_mismatch() {
        // Case 4: the pairing the whole PR exists to catch.
        let (result, _) = check(Attr::Frame, Some("world"), Some("robot_base"));
        let error = result.expect_err("mismatch");
        assert_eq!(error.code, ErrorCode::FrameMismatch);
        assert_eq!(
            error.message,
            "role \"proprio/eef_pos\": the model expects frame \"robot_base\" \
             but the env declares \"world\""
        );

        // `reference` runs the same rule against its own vocabulary -- an
        // absolute-pose head bound to a delta controller, named.
        let (result, _) = check(Attr::Reference, Some("current"), Some("target"));
        let error = result.expect_err("mismatch");
        assert_eq!(error.code, ErrorCode::FrameMismatch);
        assert_eq!(
            error.message,
            "role \"proprio/eef_pos\": the model expects reference \"target\" \
             but the env declares \"current\""
        );
    }

    #[test]
    fn an_unrecognized_value_is_a_hard_frame_mismatch_on_either_side() {
        // Case 7. The codec relays an unknown value (a newer peer's vocabulary
        // survives round-trip); resolve refuses it, because a geometry this core
        // cannot name is a geometry it cannot verify. Both sides, both attrs.
        for (attr, env, model) in [
            (Attr::Frame, Some("gripper"), None),
            (Attr::Frame, None, Some("base")),
            (Attr::Reference, None, Some("previous")),
        ] {
            let (result, _) = check(attr, env, model);
            let error = result.expect_err("unrecognized");
            assert_eq!(error.code, ErrorCode::FrameMismatch);
            assert!(
                error.message.contains("unrecognized"),
                "got: {}",
                error.message
            );
        }
    }
}
