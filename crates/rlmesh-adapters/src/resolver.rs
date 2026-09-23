//! Resolve env and model IO specs into a concrete adapter plan.

mod action;
mod custom;
mod image;
mod state;
mod text;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use super::advisory::Advisory;
use super::error::{AdapterResolutionError, ErrorCode};
use super::fmt::quoted;
use super::join::join;
use super::path::NodePath;
use super::plans::{ObsPlan, ResolvedAdapter};
use super::space_view::SpaceView;
use super::spec::{
    AcceptSet, Attr, EnvFeature, EnvImage, EnvState, EnvTags, EnvText, FrameRef, InputNode,
    ModelLeaf, ModelSpec, Provenance,
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

/// The identity a leaf binds by: `(role, part, provenance)` exactly as
/// declared. The provenance lets a sim publish one role as its truth and as
/// its estimate; images and actions carry none (`None`).
pub(super) type LeafKey = (String, Option<String>, Option<String>);

fn leaf_key(role: &str, part: Option<&str>, provenance: Option<&str>) -> LeafKey {
    (
        role.to_owned(),
        part.map(str::to_owned),
        provenance.map(str::to_owned),
    )
}

/// The `provenance` rules, the frame table verbatim with the model side an
/// accept set: both silent, silent; env only, silent; model only, `caution`;
/// the env's value in the model's set, silent; anything else, including a
/// value outside the vocabulary on either side, a hard
/// [`ErrorCode::ProvenanceMismatch`]. Returns the value to record on the plan.
pub(super) fn check_provenance(
    role: &str,
    env: Option<&FrameRef>,
    model: Option<&AcceptSet<Provenance>>,
    advisories: &mut Vec<Advisory>,
) -> Result<Option<FrameRef>> {
    let attr = Attr::Provenance;
    if let Some(value) = env
        && !attr.recognizes(value)
    {
        return Err(err(
            ErrorCode::ProvenanceMismatch,
            format!(
                "role {}: the env declares unrecognized provenance {}; this core knows {:?}",
                quoted(role),
                quoted(value.as_str()),
                attr.vocabulary()
            ),
        ));
    }
    if let Some(set) = model
        && set.first_known().is_none()
    {
        return Err(err(
            ErrorCode::ProvenanceMismatch,
            format!(
                "role {}: the model declares unrecognized provenance {:?}; this core knows {:?}",
                quoted(role),
                set.wire_names(),
                attr.vocabulary()
            ),
        ));
    }
    match (env, model) {
        (None, None) => Ok(None),
        (Some(env), None) => Ok(Some(env.clone())),
        (None, Some(set)) => {
            let declared = set.wire_names().join("|");
            advisories.push(Advisory::caution(format!(
                "role {}: the model declares provenance {} but the env declares none, so where \
                 the values it was trained on came from cannot be verified -- declare it on the \
                 env to silence this",
                quoted(role),
                quoted(&declared),
            )));
            Ok(Some(FrameRef::from(declared.as_str())))
        }
        (Some(env), Some(set)) if set.wire_names().contains(&env.as_str()) => Ok(Some(env.clone())),
        (Some(env), Some(set)) => Err(err(
            ErrorCode::ProvenanceMismatch,
            format!(
                "role {}: the model expects provenance {:?} but the env declares {}",
                quoted(role),
                set.wire_names(),
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

/// Index leaves by `(role, part, provenance)`: a role may repeat across
/// parts (one `proprio/eef_pos` per arm) or provenances (a sim's truth beside
/// its estimate), never within one key.
pub(super) fn index_by_key<'spec, T>(
    features: impl Iterator<Item = (&'spec str, Option<&'spec str>, Option<&'spec str>, T)>,
    label: &str,
) -> Result<BTreeMap<LeafKey, T>> {
    let mut by_key: BTreeMap<LeafKey, T> = BTreeMap::new();
    for (role, part, provenance, feature) in features {
        let key = leaf_key(role, part, provenance);
        if by_key.contains_key(&key) {
            let mut message = format!("duplicate {label} role {}", quoted(&key.0));
            if let Some(part) = &key.1 {
                let _ = write!(message, " under part {}", quoted(part));
            }
            if let Some(provenance) = &key.2 {
                let _ = write!(message, " with provenance {}", quoted(provenance));
            }
            return Err(err(ErrorCode::Duplicate, message));
        }
        by_key.insert(key, feature);
    }
    Ok(by_key)
}

/// Whether any indexed leaf carries `role` (under any part).
pub(super) fn has_role<T>(by_key: &BTreeMap<LeafKey, T>, role: &str) -> bool {
    by_key.keys().any(|(indexed, _, _)| indexed == role)
}

/// What a seeker found for `(role, part)` among the other side's leaves.
pub(super) struct Bound<'a, T> {
    pub feature: &'a T,
    /// The part the leaf was bound under.
    pub part: Option<String>,
}

/// The `part` and `provenance` identity rules, shared by every planner.
/// `seeker` names who is looking (`model input "state"`, `env action`),
/// `what` the role's kind word (`state role`), `offerer` whose leaves are
/// searched (`env`, `model`).
///
/// The part rule picks the `(role, part)` the seeker binds under:
///
/// - A named part binds only that leaf; it never rebinds (`Ok(None)` when the
///   offerer lacks it, for the caller's optional/fill fallback or error).
/// - No part binds the offerer's part-less leaf when there is one; else its
///   *only* leaf of that role under any part, with an `info` naming the bind;
///   else, several candidates are a [`MissingRole`](ErrorCode::MissingRole)
///   that names the parts -- ambiguity is never guessed through.
///
/// The provenance rule then picks one leaf among those under that key (an
/// env may publish a role as its truth and as its estimate):
///
/// - A seeker that accepts provenances binds the first of them, in its own
///   order, that the offerer declares; when none matches and the offerer has
///   one leaf, that leaf binds and [`check_provenance`] reports the
///   disagreement (or the caution) precisely; with several it is a
///   [`ProvenanceMismatch`](ErrorCode::ProvenanceMismatch) naming them.
/// - A seeker that pins none binds the only leaf; several are an
///   [`Ambiguous`](ErrorCode::Ambiguous) naming their provenances.
#[allow(
    clippy::too_many_arguments,
    reason = "the seeker's identity (role, part, provenance) plus the three message words; \
              a struct for the words would name nothing the call sites do not already say"
)]
pub(super) fn bind<'a, T>(
    by_key: &'a BTreeMap<LeafKey, T>,
    role: &str,
    part: Option<&str>,
    provenance: Option<&AcceptSet<Provenance>>,
    seeker: &str,
    what: &str,
    offerer: &str,
    advisories: &mut Vec<Advisory>,
) -> Result<Option<Bound<'a, T>>> {
    let (role_key, part_key, _) = leaf_key(role, part, None);
    let under_part = |wanted: Option<&str>| -> Vec<(&LeafKey, &T)> {
        by_key
            .iter()
            .filter(|((indexed_role, indexed_part, _), _)| {
                *indexed_role == role_key && indexed_part.as_deref() == wanted
            })
            .collect()
    };
    let mut candidates = under_part(part_key.as_deref());
    if candidates.is_empty() {
        if part_key.is_some() {
            return Ok(None);
        }
        let parts: BTreeSet<&str> = by_key
            .keys()
            .filter(|(indexed_role, _, _)| *indexed_role == role_key)
            .filter_map(|(_, indexed_part, _)| indexed_part.as_deref())
            .collect();
        match parts.iter().copied().collect::<Vec<_>>().as_slice() {
            [] => return Ok(None),
            [only] => {
                advisories.push(Advisory::info(format!(
                    "{seeker}: {what} {} declares no part and the {offerer} declares it only \
                     under part {}; bound to that leaf (declare part= to pin it)",
                    quoted(role),
                    quoted(only)
                )));
                candidates = under_part(Some(only));
            }
            several => {
                return Err(err(
                    ErrorCode::MissingRole,
                    format!(
                        "{seeker} needs {what} {} but the {offerer} declares it under parts \
                         {several:?}; declare part=",
                        quoted(role),
                    ),
                ));
            }
        }
    }
    let chosen = match provenance {
        Some(set) => set
            .wire_names()
            .into_iter()
            .find_map(|wanted| {
                candidates
                    .iter()
                    .find(|((_, _, indexed), _)| indexed.as_deref() == Some(wanted))
                    .copied()
            })
            .or_else(|| (candidates.len() == 1).then(|| candidates[0])),
        None => (candidates.len() == 1).then(|| candidates[0]),
    };
    let Some(((_, bound_part, _), feature)) = chosen else {
        let offered: Vec<&str> = candidates
            .iter()
            .map(|((_, _, indexed), _)| indexed.as_deref().unwrap_or("none"))
            .collect();
        return Err(match provenance {
            None => err(
                ErrorCode::Ambiguous,
                format!(
                    "{seeker}: {what} {} declares no provenance and the {offerer} declares it \
                     under provenances {offered:?}; declare provenance= to pin one",
                    quoted(role),
                ),
            ),
            Some(set) => err(
                ErrorCode::ProvenanceMismatch,
                format!(
                    "{seeker}: {what} {} accepts provenance {:?} but the {offerer} declares it \
                     under {offered:?}",
                    quoted(role),
                    set.wire_names(),
                ),
            ),
        });
    };
    Ok(Some(Bound {
        feature,
        part: bound_part.clone(),
    }))
}

/// Enforce the closed kind set on the model side (the env side runs it at
/// `join`). A newer peer's kind still parses and relays; it fails here,
/// named by placement.
fn check_model_roles(model_spec: &ModelSpec) -> Result<()> {
    let check = |role: &str, locus: String| {
        crate::roles::registry::check_role(role).map_err(|reason| {
            err(
                ErrorCode::UnsupportedKind,
                format!("{locus}: {reason}; if a newer peer wrote it, upgrade the runtime"),
            )
        })
    };
    let mut leaves: Vec<PlacedLeaf> = Vec::new();
    collect_leaves(&model_spec.input, NodePath::root(), &mut leaves);
    for PlacedLeaf { leaf, placement } in &leaves {
        let locus = || format!("model input {}", quoted(&placement.to_string()));
        match leaf {
            ModelLeaf::Image(input) => check(&input.role, locus())?,
            ModelLeaf::State(input) => {
                for part in &input.components {
                    if let Some(role) = &part.role {
                        check(role, locus())?;
                    }
                }
            }
            ModelLeaf::Text(input) => check(&input.role, locus())?,
            ModelLeaf::Custom(_) | ModelLeaf::Unknown { .. } => {}
        }
    }
    for actuator in &model_spec.output.components {
        if let Some(role) = &actuator.role {
            check(role, "model action".to_owned())?;
        }
    }
    Ok(())
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
    check_model_roles(model_spec)?;
    // The model's own outputs, keyed once: the action planner binds env
    // actuators to them, and a state part reading its previous action does too.
    let outputs = action::index_outputs(&model_spec.output)?;

    let env_spec = join(env_tags, observation_space, action_space).map_err(|error| {
        let code = match error {
            crate::join::JoinError::ActionRoleOnObservation { .. } => {
                ErrorCode::ActionRoleOnObservation
            }
            _ => ErrorCode::InvalidTag,
        };
        err(code, error.to_string())
    })?;
    let images = env_spec
        .observation
        .iter()
        .filter_map(|feature| match feature {
            EnvFeature::Image(image) => {
                Some((image.role.as_str(), image.part.as_deref(), None, image))
            }
            _ => None,
        });
    let states = env_spec
        .observation
        .iter()
        .filter_map(|feature| match feature {
            EnvFeature::State(state) => Some((
                state.role.as_str(),
                state.part.as_deref(),
                state.provenance.as_ref().map(FrameRef::as_str),
                state,
            )),
            _ => None,
        });
    let texts = env_spec
        .observation
        .iter()
        .filter_map(|feature| match feature {
            EnvFeature::Text(text) => Some((&text.role, text)),
            _ => None,
        });
    let images_by_role: BTreeMap<LeafKey, &EnvImage> = index_by_key(images, "env image")?;
    let states_by_role: BTreeMap<LeafKey, &EnvState> = index_by_key(states, "env state")?;
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
                &mut quiet,
            )?),
            ModelLeaf::State(input) => ObsPlan::State(state::plan_state(
                input,
                placement,
                &states_by_role,
                &outputs,
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

    let action_plan =
        action::plan_action(&model_spec.output, &env_spec.action, &outputs, &mut quiet)?;

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
            ModelLeaf::Image(input) => {
                if !has_role(&images_by_role, &input.role) {
                    ad_hoc.insert(&input.role);
                }
            }
            ModelLeaf::State(input) => {
                // An action-source part is answered by the model's own
                // actuator (resolved above), never by an env leaf.
                ad_hoc.extend(
                    input
                        .components
                        .iter()
                        .filter(|part| part.source == crate::spec::PartSource::Observation)
                        .filter_map(|part| part.role.as_deref())
                        .filter(|role| !has_role(&states_by_role, role)),
                );
            }
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
                {{"role":"proprio/eef_pos","dim":3,"optional":true}}]}}}},"output":{ACTION_OUT}}}"#
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
    fn a_parted_camera_role_never_rebinds_to_the_only_camera() {
        let env_tags = format!(
            r#"{{"observation":{{"cam":{{"type":"image","role":"image/primary"}}}},"action":{ACTION_TAGS}}}"#
        );
        let obs_space = r#"{"kind":"dict","dtype":"unspecified","keys":["cam"],"children":[
            {"kind":"box","shape":[4,4,3],"dtype":"uint8"}]}"#;
        let model = format!(
            r#"{{"input":{{"pixels":{{"type":"image","role":"image/wrist","part":"right_arm"}}}},"output":{ACTION_OUT}}}"#
        );
        let err = do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect_err("no rebind");
        assert_eq!(err.code, ErrorCode::MissingRole);
        // The unparted wrist still rebinds (with the caution): only a named
        // part is barred, since a lone camera cannot be one arm of a pair.
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

/// The `part` identity rules (section 3 of the parts design), end to end.
#[cfg(test)]
mod part_tests {
    use super::resolve;
    use crate::advisory::AdvisorySeverity;
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

    const ACTION_SPACE: &str = r#"{"kind":"box","shape":[1],"dtype":"float32"}"#;
    const ACTION_OUT: &str = r#"{"components":[{"role":"action/gripper","dim":1}]}"#;
    const ACTION_TAGS: &str = r#"{"components":[{"role":"action/gripper","dim":1}]}"#;
    const TWO_POS: &str = r#"{"kind":"dict","dtype":"unspecified","keys":["l","r"],"children":[
        {"kind":"box","shape":[3],"dtype":"float32"},
        {"kind":"box","shape":[3],"dtype":"float32"}]}"#;
    const ONE_POS: &str = r#"{"kind":"dict","dtype":"unspecified","keys":["l"],"children":[
        {"kind":"box","shape":[3],"dtype":"float32"}]}"#;

    fn env_two_arms() -> String {
        format!(
            r#"{{"observation":{{
                "l":{{"type":"state","role":"proprio/eef_pos","part":"left_arm"}},
                "r":{{"type":"state","role":"proprio/eef_pos","part":"right_arm"}}}},
                "action":{ACTION_TAGS}}}"#
        )
    }

    fn env_one_arm(part: &str) -> String {
        format!(
            r#"{{"observation":{{"l":{{"type":"state","role":"proprio/eef_pos","part":{part:?}}}}},
                "action":{ACTION_TAGS}}}"#
        )
    }

    fn model_part(part: Option<&str>) -> String {
        let part = part.map_or(String::new(), |part| format!(r#","part":{part:?}"#));
        format!(
            r#"{{"input":{{"s":{{"type":"state","components":[{{"role":"proprio/eef_pos","dim":3{part}}}]}}}},"output":{ACTION_OUT}}}"#
        )
    }

    #[test]
    fn a_named_part_binds_only_its_leaf_and_prints_it() {
        let adapter = do_resolve(
            &env_two_arms(),
            TWO_POS,
            ACTION_SPACE,
            &model_part(Some("right_arm")),
        )
        .expect("resolves");
        let described = adapter.describe();
        assert!(
            described.contains(r##""s" <- concat(r[:3]#right_arm)"##),
            "got:\n{described}"
        );
        assert!(
            adapter.advisories().is_empty(),
            "{:?}",
            adapter.advisories()
        );
    }

    #[test]
    fn no_part_against_several_parts_is_missing_role_naming_them() {
        let err = do_resolve(&env_two_arms(), TWO_POS, ACTION_SPACE, &model_part(None))
            .expect_err("ambiguous");
        assert_eq!(err.code, ErrorCode::MissingRole);
        assert!(
            err.message
                .contains(r#"under parts ["left_arm", "right_arm"]; declare part="#),
            "got: {}",
            err.message
        );
        // `optional` does not turn ambiguity into a fill: the env has the data.
        let optional = format!(
            r#"{{"input":{{"s":{{"type":"state","components":[{{"role":"proprio/eef_pos","dim":3,"optional":true}}]}}}},"output":{ACTION_OUT}}}"#
        );
        let err = do_resolve(&env_two_arms(), TWO_POS, ACTION_SPACE, &optional).expect_err("err");
        assert_eq!(err.code, ErrorCode::MissingRole);
    }

    #[test]
    fn no_part_against_one_part_rebinds_with_an_info() {
        let adapter = do_resolve(
            &env_one_arm("left_arm"),
            ONE_POS,
            ACTION_SPACE,
            &model_part(None),
        )
        .expect("resolves");
        let notes = adapter.advisories();
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert_eq!(notes[0].severity, AdvisorySeverity::Info);
        assert!(
            notes[0]
                .message
                .contains(r#"declares it only under part "left_arm"; bound to that leaf"#),
            "{}",
            notes[0].message
        );
        assert!(
            adapter.describe().contains("l[:3]#left_arm"),
            "got:\n{}",
            adapter.describe()
        );
        assert!(!adapter.describe().contains("dropped:"));
    }

    #[test]
    fn a_named_part_never_rebinds() {
        // The env has the role under another part: a named part is missing,
        // not rebound -- fill if optional, else MissingRole naming what exists.
        let err = do_resolve(
            &env_one_arm("left_arm"),
            ONE_POS,
            ACTION_SPACE,
            &model_part(Some("torso")),
        )
        .expect_err("missing");
        assert_eq!(err.code, ErrorCode::MissingRole);
        assert!(
            err.message.contains(r#"(part "torso")"#)
                && err.message.contains(r#"["proprio/eef_pos#left_arm"]"#),
            "got: {}",
            err.message
        );
        let optional = format!(
            r#"{{"input":{{"s":{{"type":"state","components":[{{"role":"proprio/eef_pos","dim":3,"part":"torso","optional":true}}]}}}},"output":{ACTION_OUT}}}"#
        );
        let adapter =
            do_resolve(&env_one_arm("left_arm"), ONE_POS, ACTION_SPACE, &optional).expect("fills");
        assert!(
            adapter.describe().contains("zeros(3)#torso"),
            "got:\n{}",
            adapter.describe()
        );
    }

    #[test]
    fn an_unknown_kind_prefix_parses_and_fails_at_resolve() {
        // The kind set is closed; a newer peer's kind relays and dies here,
        // named. `x/` stays the whole-role escape.
        let model = format!(
            r#"{{"input":{{"s":{{"type":"state","components":[{{"role":"audio/mic","dim":3}}]}}}},"output":{ACTION_OUT}}}"#
        );
        let err =
            do_resolve(&env_one_arm("left_arm"), ONE_POS, ACTION_SPACE, &model).expect_err("err");
        assert_eq!(err.code, ErrorCode::UnsupportedKind);
        assert!(
            err.message.contains(r#"model input "s""#)
                && err.message.contains("kind this core does not define")
                && err.message.contains("upgrade the runtime"),
            "{}",
            err.message
        );
        let env = format!(
            r#"{{"observation":{{"l":{{"type":"state","role":"audio/mic"}}}},"action":{ACTION_TAGS}}}"#
        );
        let err = do_resolve(&env, ONE_POS, ACTION_SPACE, &model_part(None)).expect_err("err");
        assert_eq!(err.code, ErrorCode::InvalidTag);
        let escaped = format!(
            r#"{{"observation":{{"l":{{"type":"state","role":"x/mic"}}}},"action":{ACTION_TAGS}}}"#
        );
        let model = format!(
            r#"{{"input":{{"s":{{"type":"state","components":[{{"role":"x/mic","dim":3}}]}}}},"output":{ACTION_OUT}}}"#
        );
        do_resolve(&escaped, ONE_POS, ACTION_SPACE, &model).expect("x/ escapes");
    }

    #[test]
    fn an_action_role_split_by_part_binds_per_arm() {
        let env = r#"{"observation":{"l":{"type":"state","role":"proprio/eef_pos","part":"left_arm"}},
            "action":{"components":[
                {"role":"action/joint_pos","dim":2,"part":"left_arm"},
                {"role":"action/joint_pos","dim":2,"part":"right_arm"}]}}"#;
        let model = r#"{"input":{"s":{"type":"state","components":[{"role":"proprio/eef_pos","dim":3,"part":"left_arm"}]}},
            "output":{"components":[
                {"role":"action/joint_pos","dim":2,"part":"right_arm"},
                {"role":"action/joint_pos","dim":2,"part":"left_arm"}]}}"#;
        let action_space = r#"{"kind":"box","shape":[4],"dtype":"float32"}"#;
        let adapter = do_resolve(env, ONE_POS, action_space, model).expect("resolves");
        let described = adapter.describe();
        assert!(
            described.contains(r##""action/joint_pos" <- model[2:4]#left_arm"##)
                && described.contains(r##""action/joint_pos" <- model[0:2]#right_arm"##),
            "got:\n{described}"
        );
        // A part-less env actuator binds the model's only output of the role
        // under any part, with the same info; two candidates are an error.
        let env_bare = r#"{"observation":{"l":{"type":"state","role":"proprio/eef_pos","part":"left_arm"}},
            "action":{"components":[{"role":"action/joint_pos","dim":2}]}}"#;
        let model_one = r#"{"input":{"s":{"type":"state","components":[{"role":"proprio/eef_pos","dim":3,"part":"left_arm"}]}},
            "output":{"components":[{"role":"action/joint_pos","dim":2,"part":"left_arm"}]}}"#;
        let adapter = do_resolve(
            env_bare,
            ONE_POS,
            r#"{"kind":"box","shape":[2],"dtype":"float32"}"#,
            model_one,
        )
        .expect("resolves");
        assert!(
            adapter.advisories().iter().any(|note| note
                .message
                .contains(r#"env action: role "action/joint_pos" declares no part"#)),
            "{:?}",
            adapter.advisories()
        );
        let err = do_resolve(
            env_bare,
            ONE_POS,
            r#"{"kind":"box","shape":[2],"dtype":"float32"}"#,
            model,
        )
        .expect_err("ambiguous");
        assert_eq!(err.code, ErrorCode::MissingRole);
        assert!(err.message.contains("declare part="), "{}", err.message);
    }

    #[test]
    fn a_custom_part_binds_on_exact_agreement_with_no_advisory() {
        for part in ["franka", "x/franka"] {
            let adapter = do_resolve(
                &env_one_arm(part),
                ONE_POS,
                ACTION_SPACE,
                &model_part(Some(part)),
            )
            .expect("resolves on exact agreement");
            assert!(
                adapter.advisories().is_empty(),
                "{:?}",
                adapter.advisories()
            );
            assert!(!adapter.describe().contains("dropped:"));
        }
    }

    #[test]
    fn a_parted_camera_follows_the_same_rules() {
        let env = format!(
            r#"{{"observation":{{"cam":{{"type":"image","role":"image/wrist","part":"left_arm"}}}},"action":{ACTION_TAGS}}}"#
        );
        let obs = r#"{"kind":"dict","dtype":"unspecified","keys":["cam"],"children":[
            {"kind":"box","shape":[4,4,3],"dtype":"uint8"}]}"#;
        let model = format!(
            r#"{{"input":{{"pixels":{{"type":"image","role":"image/wrist"}}}},"output":{ACTION_OUT}}}"#
        );
        let adapter = do_resolve(&env, obs, ACTION_SPACE, &model).expect("rebinds by part");
        assert!(
            adapter
                .describe()
                .contains(r##"<- image "cam"#left_arm ("##),
            "{}",
            adapter.describe()
        );
        assert!(
            adapter
                .advisories()
                .iter()
                .all(|note| note.severity == AdvisorySeverity::Info),
            "{:?}",
            adapter.advisories()
        );
        // A named part never falls through to the lone-camera fallback.
        let model = format!(
            r#"{{"input":{{"pixels":{{"type":"image","role":"image/primary","part":"head"}}}},"output":{ACTION_OUT}}}"#
        );
        let err = do_resolve(&env, obs, ACTION_SPACE, &model).expect_err("no rebind");
        assert_eq!(err.code, ErrorCode::MissingRole);
    }
}

/// The `labels` layout rules (section 3 of the design), end to end, on the
/// Go2: SDK motor order on the env, Isaac order on a model.
#[cfg(test)]
mod labels_tests {
    use std::collections::BTreeMap;

    use super::resolve;
    use crate::apply::{NoCustoms, Value};
    use crate::error::ErrorCode;
    use crate::space_view::SpaceView;
    use crate::spec::{EnvTags, ModelSpec};

    /// Go2 joints in SDK motor order: FR, FL, RR, RL x hip, thigh, calf.
    const GO2_JOINTS: [&str; 12] = [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ];

    /// SDK order → Isaac order: FL,FR,RL,RR x hip,thigh,calf.
    const ISAAC: [usize; 12] = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8];

    fn labels(order: &[usize]) -> String {
        let names: Vec<String> = order
            .iter()
            .map(|&i| format!("{:?}", GO2_JOINTS[i]))
            .collect();
        format!("[{}]", names.join(","))
    }

    fn sdk() -> String {
        labels(&(0..12).collect::<Vec<_>>())
    }

    fn space(json: &str) -> SpaceView {
        serde_json::from_str(json).expect("parse space")
    }

    fn go2_env(actuator_extra: &str) -> String {
        let sdk = sdk();
        format!(
            r#"{{"observation":{{
                "joint_pos":{{"type":"state","role":"proprio/joint_pos","labels":{sdk}}},
                "joint_vel":{{"type":"state","role":"proprio/joint_vel","labels":{sdk}}}}},
                "action":{{"components":[{{"role":"action/joint_pos","dim":12,"labels":{sdk}{actuator_extra}}}]}}}}"#
        )
    }

    const GO2_OBS: &str = r#"{"kind":"dict","dtype":"unspecified","keys":["joint_pos","joint_vel"],"children":[
        {"kind":"box","shape":[12],"dtype":"float32"},
        {"kind":"box","shape":[12],"dtype":"float32"}]}"#;
    const GO2_ACT: &str = r#"{"kind":"box","shape":[12],"dtype":"float32"}"#;

    fn go2_model(model_labels: Option<&str>) -> String {
        let labels = model_labels.map_or(String::new(), |labels| format!(r#","labels":{labels}"#));
        let dim = if model_labels.is_some() {
            ""
        } else {
            r#","dim":12"#
        };
        format!(
            r#"{{"input":{{"obs":{{"type":"state","components":[
                {{"role":"proprio/joint_pos"{dim}{labels},"axis_offset":[0.0,-0.8,1.5,0.0,-0.8,1.5,0.0,-0.8,1.5,0.0,-0.8,1.5]}},
                {{"role":"proprio/joint_vel"{dim}{labels},"scale":0.05}}]}}}},
                "output":{{"components":[{{"role":"action/joint_pos","dim":12{labels},
                    "axis_scale":[0.125,0.25,0.25,0.125,0.25,0.25,0.125,0.25,0.25,0.125,0.25,0.25],
                    "axis_offset":[0.0,0.8,-1.5,0.0,0.8,-1.5,0.0,0.8,-1.5,0.0,0.8,-1.5]}}]}}}}"#
        )
    }

    fn do_resolve(
        env: &str,
        model: &str,
    ) -> Result<crate::plans::ResolvedAdapter, crate::error::AdapterResolutionError> {
        let tags: EnvTags = serde_json::from_str(env).expect("parse env tags");
        let spec: ModelSpec = serde_json::from_str(model).expect("parse model spec");
        resolve(&tags, &space(GO2_OBS), &space(GO2_ACT), &spec, false)
    }

    fn tensor(values: &[f32]) -> Value {
        Value::Tensor(crate::apply::value::tensor_from_f32(
            vec![values.len() as i64],
            values,
        ))
    }

    #[test]
    fn an_isaac_order_model_resolves_to_the_permutation_on_both_sides() {
        let adapter =
            do_resolve(&go2_env(""), &go2_model(Some(&labels(&ISAAC)))).expect("resolves");
        let described = adapter.describe();
        assert!(
            described.contains("joint_pos perm[3,4,5,0,1,2,9,10,11,6,7,8] (+[FL_hip_joint:0.0,FL_thigh_joint:-0.8,FL_calf_joint:1.5,")
                && described.contains("joint_vel perm[3,4,5,0,1,2,9,10,11,6,7,8] (*0.05)")
                && described.contains(
                    "\"action/joint_pos\" <- model[0:12] (model *[FL_hip_joint:0.125,FL_thigh_joint:0.25,FL_calf_joint:0.25,"
                )
                && described.ends_with("]) perm[3,4,5,0,1,2,9,10,11,6,7,8]"),
            "got:\n{described}"
        );
        // The permutation is its own inverse on the Go2, so the action-side
        // scatter lists the same indices; the values prove the direction.
        assert!(
            adapter.advisories().is_empty(),
            "{:?}",
            adapter.advisories()
        );
        let mut raw: BTreeMap<String, Value> = BTreeMap::new();
        let sdk_values: Vec<f32> = (0..12).map(|i| i as f32).collect();
        raw.insert("joint_pos".to_owned(), tensor(&sdk_values));
        raw.insert("joint_vel".to_owned(), tensor(&sdk_values));
        let Value::Map(payload) = adapter.transform_obs(&raw, &NoCustoms).expect("apply") else {
            panic!("expected a map");
        };
        let Value::Tensor(obs) = &payload["obs"] else {
            panic!("expected a tensor");
        };
        let obs = crate::apply::value::to_f32_vec(obs);
        // joint_pos: SDK value at ISAAC[i], minus the stand pose in Isaac order.
        let expected_pos: Vec<f32> = ISAAC
            .iter()
            .enumerate()
            .map(|(i, &j)| sdk_values[j] + [0.0, -0.8, 1.5][i % 3])
            .collect();
        for (got, want) in obs[..12].iter().zip(&expected_pos) {
            assert!((got - want).abs() < 1e-6, "{obs:?}");
        }
        assert!((obs[12] - 3.0 * 0.05).abs() < 1e-6, "{obs:?}");
        // Action: the model emits ones in Isaac order; each env axis reads its
        // own joint's scale and pose back in SDK order.
        let action = adapter
            .transform_action(&tensor(&[1.0; 12]))
            .expect("apply");
        let action = crate::apply::value::to_f32_vec(&action);
        let expected_act: Vec<f32> = (0..12)
            .map(|j| [0.125, 0.25, 0.25][j % 3] + [0.0, 0.8, -1.5][j % 3])
            .collect();
        for (got, want) in action.iter().zip(&expected_act) {
            assert!((got - want).abs() < 1e-6, "{action:?}");
        }
    }

    #[test]
    fn an_sdk_order_model_is_the_identity() {
        let adapter = do_resolve(&go2_env(""), &go2_model(Some(&sdk()))).expect("resolves");
        let described = adapter.describe();
        assert!(
            described.contains("joint_pos[:12] (+[FR_hip_joint:0.0,FR_thigh_joint:-0.8,")
                && !described.contains("perm")
                && !described.contains("select"),
            "got:\n{described}"
        );
        assert!(
            adapter.advisories().is_empty(),
            "{:?}",
            adapter.advisories()
        );
    }

    #[test]
    fn model_only_labels_are_a_mismatch_and_env_only_labels_are_silent() {
        let unlabeled_env = r#"{"observation":{
                "joint_pos":{"type":"state","role":"proprio/joint_pos"},
                "joint_vel":{"type":"state","role":"proprio/joint_vel"}},
                "action":{"components":[{"role":"action/joint_pos","dim":12}]}}"#;
        let err = do_resolve(unlabeled_env, &go2_model(Some(&sdk()))).expect_err("model-only");
        assert_eq!(err.code, ErrorCode::LabelMismatch);
        assert!(
            err.message.contains("the env leaf declares none"),
            "{}",
            err.message
        );
        // Env-only: silent, and the env's names annotate a per-axis vector the
        // model declared positionally.
        let adapter = do_resolve(&go2_env(""), &go2_model(None)).expect("env-only labels resolve");
        let described = adapter.describe();
        assert!(
            described.contains("joint_pos[:12] (+[FR_hip_joint:0.0,FR_thigh_joint:-0.8,")
                && described.contains("(model *[FR_hip_joint:0.125,"),
            "got:\n{described}"
        );
        assert!(
            adapter.advisories().is_empty(),
            "{:?}",
            adapter.advisories()
        );
    }

    #[test]
    fn a_subset_of_the_env_labels_is_a_mismatch_on_either_side() {
        // A front-legs-only checkpoint: six of the twelve labels.
        let front = labels(&[3, 4, 5, 0, 1, 2]);
        let model = format!(
            r#"{{"input":{{"obs":{{"type":"state","components":[{{"role":"proprio/joint_pos","labels":{front}}}]}}}},
                "output":{{"components":[{{"role":"action/joint_pos","dim":12}}]}}}}"#
        );
        let err = do_resolve(&go2_env(""), &model).expect_err("observation subset");
        assert_eq!(err.code, ErrorCode::LabelMismatch);
        assert!(
            err.message
                .contains(r#"the model input lacks ["RR_hip_joint""#)
                && !err.message.contains("the env leaf lacks"),
            "{}",
            err.message
        );
        let model = format!(
            r#"{{"input":{{"obs":{{"type":"state","components":[{{"role":"proprio/joint_pos","dim":12}}]}}}},
                "output":{{"components":[{{"role":"action/joint_pos","dim":6,"labels":{front}}}]}}}}"#
        );
        let env = go2_env(r#","optional":true,"fill":0.5"#);
        let err = do_resolve(&env, &model).expect_err("action subset, even optional");
        assert_eq!(err.code, ErrorCode::LabelMismatch);
        assert!(
            err.message.contains(r#"the model lacks ["RR_hip_joint""#),
            "{}",
            err.message
        );
    }

    #[test]
    fn a_label_the_env_lacks_is_a_mismatch_even_when_the_part_is_optional() {
        let mut model_labels: Vec<String> = GO2_JOINTS
            .iter()
            .map(|label| format!("{label:?}"))
            .collect();
        model_labels[1] = r#""FR_shin""#.to_owned();
        let model_labels = format!("[{}]", model_labels.join(","));
        for optional in ["", r#","optional":true"#] {
            let model = format!(
                r#"{{"input":{{"obs":{{"type":"state","components":[{{"role":"proprio/joint_pos","labels":{model_labels}{optional}}}]}}}},
                    "output":{{"components":[{{"role":"action/joint_pos","dim":12}}]}}}}"#
            );
            let err = do_resolve(&go2_env(""), &model).expect_err("mismatched label");
            assert_eq!(err.code, ErrorCode::LabelMismatch);
            assert!(
                err.message.contains(r#"the env leaf lacks ["FR_shin"]"#)
                    && err
                        .message
                        .contains(r#"the model input lacks ["FR_thigh_joint"]"#),
                "{}",
                err.message
            );
        }
    }

    #[test]
    fn a_three_cycle_of_custom_labels_permutes_each_direction_the_right_way() {
        // Not its own inverse, so a gather/scatter direction mistake shows;
        // the names are no robot's, and no registry is consulted.
        let env = r#"{"observation":{"q":{"type":"state","role":"proprio/joint_pos","labels":["tail_a","tail_b","tail_c"]}},
            "action":{"components":[{"role":"action/joint_pos","dim":3,"labels":["tail_a","tail_b","tail_c"]}]}}"#;
        let model = r#"{"input":{"q":{"type":"state","components":[{"role":"proprio/joint_pos","labels":["tail_b","tail_c","tail_a"]}]}},
            "output":{"components":[{"role":"action/joint_pos","dim":3,"labels":["tail_b","tail_c","tail_a"]}]}}"#;
        let tags: EnvTags = serde_json::from_str(env).expect("parse env tags");
        let spec: ModelSpec = serde_json::from_str(model).expect("parse model spec");
        let obs = space(
            r#"{"kind":"dict","dtype":"unspecified","keys":["q"],"children":[{"kind":"box","shape":[3],"dtype":"float32"}]}"#,
        );
        let act = space(r#"{"kind":"box","shape":[3],"dtype":"float32"}"#);
        let adapter = resolve(&tags, &obs, &act, &spec, false).expect("resolves");
        assert!(
            adapter.advisories().is_empty(),
            "{:?}",
            adapter.advisories()
        );
        let described = adapter.describe();
        assert!(
            described.contains("q perm[1,2,0]") && described.ends_with("perm[2,0,1]"),
            "got:\n{described}"
        );
        let mut raw: BTreeMap<String, Value> = BTreeMap::new();
        raw.insert("q".to_owned(), tensor(&[10.0, 20.0, 30.0]));
        let Value::Map(payload) = adapter.transform_obs(&raw, &NoCustoms).expect("apply") else {
            panic!("expected a map");
        };
        let Value::Tensor(q) = &payload["q"] else {
            panic!("expected a tensor");
        };
        assert_eq!(crate::apply::value::to_f32_vec(q), vec![20.0, 30.0, 10.0]);
        let action = adapter
            .transform_action(&tensor(&[20.0, 30.0, 10.0]))
            .expect("apply");
        assert_eq!(
            crate::apply::value::to_f32_vec(&action),
            vec![10.0, 20.0, 30.0]
        );
    }

    #[test]
    fn a_per_axis_vector_must_match_the_resolved_width() {
        let model = r#"{"input":{"obs":{"type":"state","components":[{"role":"proprio/joint_pos","dim":12,"axis_scale":[1.0,2.0]}]}},
            "output":{"components":[{"role":"action/joint_pos","dim":12}]}}"#;
        let err = do_resolve(&go2_env(""), model).expect_err("short vector");
        assert_eq!(err.code, ErrorCode::DimMismatch);
        assert!(
            err.message
                .contains("axis_scale has 2 values but the resolved width is 12"),
            "{}",
            err.message
        );
        let model = r#"{"input":{"obs":{"type":"state","components":[{"role":"proprio/joint_pos","dim":12}]}},
            "output":{"components":[{"role":"action/joint_pos","dim":12,"axis_offset":[1.0]}]}}"#;
        let err = do_resolve(&go2_env(""), model).expect_err("short vector");
        assert_eq!(err.code, ErrorCode::DimMismatch);
        assert!(
            err.message.contains("the model's axis_offset has 1 values"),
            "{}",
            err.message
        );
    }

    #[test]
    fn unfamiliar_env_labels_resolve_silently() {
        let env = r#"{"observation":{"j":{"type":"state","role":"proprio/joint_pos","labels":["j0","j1"]}},
            "action":{"components":[{"role":"action/joint_pos","dim":12}]}}"#;
        let obs = r#"{"kind":"dict","dtype":"unspecified","keys":["j"],"children":[{"kind":"box","shape":[2],"dtype":"float32"}]}"#;
        let model = r#"{"input":{"obs":{"type":"state","components":[{"role":"proprio/joint_pos","dim":2}]}},
            "output":{"components":[{"role":"action/joint_pos","dim":12}]}}"#;
        let tags: EnvTags = serde_json::from_str(env).unwrap();
        let spec: ModelSpec = serde_json::from_str(model).unwrap();
        let adapter = resolve(&tags, &space(obs), &space(GO2_ACT), &spec, false).expect("resolves");
        assert!(
            adapter.advisories().is_empty(),
            "{:?}",
            adapter.advisories()
        );
    }

    /// The Go2 body: gyro, IMU orientation and the velocity command beside the
    /// joints, with the model reading the orientation as projected gravity and
    /// clamping the assembled vector (rl_sar `robot_lab`, 45 - 12 for the
    /// previous action, which is not part of this pairing).
    fn go2_body_env(base_quat: &str) -> String {
        let sdk = sdk();
        format!(
            r#"{{"observation":{{
                "ang_vel":{{"type":"state","role":"proprio/base_ang_vel","frame":"robot_base","provenance":"sensed","range":[-20.0,20.0]}},
                {base_quat},
                "command":{{"type":"state","role":"command/base_vel","frame":"robot_base","range":[-3.0,3.0]}},
                "joint_pos":{{"type":"state","role":"proprio/joint_pos","labels":{sdk},"provenance":"sensed"}},
                "joint_vel":{{"type":"state","role":"proprio/joint_vel","labels":{sdk},"provenance":"sensed","range":[-30.0,30.0]}}}},
                "action":{{"components":[{{"role":"action/joint_pos","dim":12,"labels":{sdk}}}]}}}}"#
        )
    }

    const GO2_BODY_QUAT: &str = r#""base_quat":{"type":"state","role":"proprio/base_rot","encoding":"quat_wxyz","frame":"world","provenance":"sensed"}"#;

    fn go2_body_obs(keys: &[&str]) -> String {
        let children: Vec<&str> = keys
            .iter()
            .map(|key| match *key {
                "joint_pos" | "joint_vel" => r#"{"kind":"box","shape":[12],"dtype":"float32"}"#,
                "base_quat" | "base_quat_est" => r#"{"kind":"box","shape":[4],"dtype":"float32"}"#,
                _ => r#"{"kind":"box","shape":[3],"dtype":"float32"}"#,
            })
            .collect();
        let keys: Vec<String> = keys.iter().map(|key| format!("{key:?}")).collect();
        format!(
            r#"{{"kind":"dict","dtype":"unspecified","keys":[{}],"children":[{}]}}"#,
            keys.join(","),
            children.join(",")
        )
    }

    fn go2_body_model(provenance: &str) -> String {
        let sdk = sdk();
        format!(
            r#"{{"input":{{"obs":{{"type":"state","clip":[-100.0,100.0],"components":[
                {{"role":"proprio/base_ang_vel","frame":"robot_base","scale":0.25}},
                {{"role":"proprio/base_rot","encoding":"gravity_xyz","frame":"world"{provenance}}},
                {{"role":"command/base_vel","frame":"robot_base"}},
                {{"role":"proprio/joint_pos","labels":{sdk},"axis_offset":[0.0,-0.8,1.5,0.0,-0.8,1.5,0.0,-0.8,1.5,0.0,-0.8,1.5]}},
                {{"role":"proprio/joint_vel","labels":{sdk},"scale":0.05}}]}}}},
                "output":{{"components":[{{"role":"action/joint_pos","dim":12,"labels":{sdk},
                    "axis_scale":[0.125,0.25,0.25,0.125,0.25,0.25,0.125,0.25,0.25,0.125,0.25,0.25],
                    "axis_offset":[0.0,0.8,-1.5,0.0,0.8,-1.5,0.0,0.8,-1.5,0.0,0.8,-1.5]}}]}}}}"#
        )
    }

    #[test]
    fn the_go2_body_resolves_to_a_33_wide_clamped_observation_reading_gravity() {
        let tags: EnvTags = serde_json::from_str(&go2_body_env(GO2_BODY_QUAT)).unwrap();
        let spec: ModelSpec = serde_json::from_str(&go2_body_model("")).unwrap();
        let obs = go2_body_obs(&["ang_vel", "base_quat", "command", "joint_pos", "joint_vel"]);
        let adapter =
            resolve(&tags, &space(&obs), &space(GO2_ACT), &spec, false).expect("resolves");
        let described = adapter.describe();
        assert!(
            described.contains(
                "concat(ang_vel (*0.25)@robot_base#sensed, base_quat (quat_wxyz->gravity_xyz)@world#sensed, command@robot_base, joint_pos[:12] (+[FR_hip_joint:0.0,"
            ) && described.contains("#sensed, joint_vel[:12] (*0.05)#sensed) clip[-100.0,100.0]\n"),
            "got:\n{described}"
        );
        assert!(
            adapter.advisories().is_empty(),
            "{:?}",
            adapter.advisories()
        );
        let crate::plans::ObsPlan::State(plan) = &adapter.obs_plans[0] else {
            panic!("expected a state plan");
        };
        assert_eq!(plan.native_width, Some(33));
        let mut raw: BTreeMap<String, Value> = BTreeMap::new();
        raw.insert("ang_vel".to_owned(), tensor(&[4.0, 0.0, 0.0]));
        // Identity orientation, wxyz: gravity is straight down in the base.
        raw.insert("base_quat".to_owned(), tensor(&[1.0, 0.0, 0.0, 0.0]));
        raw.insert("command".to_owned(), tensor(&[0.5, 0.0, 0.0]));
        let mut joints = vec![0.0; 12];
        joints[0] = 1.0e6;
        raw.insert("joint_pos".to_owned(), tensor(&joints));
        raw.insert("joint_vel".to_owned(), tensor(&[0.0; 12]));
        let Value::Map(payload) = adapter.transform_obs(&raw, &NoCustoms).expect("apply") else {
            panic!("expected a map");
        };
        let Value::Tensor(obs) = &payload["obs"] else {
            panic!("expected a tensor");
        };
        let obs = crate::apply::value::to_f32_vec(obs);
        assert_eq!(obs.len(), 33);
        assert_eq!(&obs[..3], &[1.0, 0.0, 0.0]);
        assert_eq!(&obs[3..6], &[0.0, 0.0, -1.0]);
        assert_eq!(&obs[6..9], &[0.5, 0.0, 0.0]);
        // The container clamp caught the runaway encoder reading.
        assert_eq!(obs[9], 100.0);
        assert!((obs[10] + 0.8).abs() < 1e-6, "{obs:?}");
    }

    /// The design's Go2 pairing complete: the body model plus a part reading
    /// the model's own previous action, 45 wide.
    fn go2_full_model(previous: &str) -> String {
        go2_body_model("").replacen(
            r#""scale":0.05}]}"#,
            &format!(r#""scale":0.05}},{previous}]}}"#),
            1,
        )
    }

    const PREVIOUS: &str = r#"{"role":"action/joint_pos","source":"action"}"#;

    fn go2_full_resolve(
        env: &str,
        previous: &str,
    ) -> Result<crate::plans::ResolvedAdapter, crate::error::AdapterResolutionError> {
        let tags: EnvTags = serde_json::from_str(env).unwrap();
        let spec: ModelSpec = serde_json::from_str(&go2_full_model(previous)).unwrap();
        let obs = go2_body_obs(&["ang_vel", "base_quat", "command", "joint_pos", "joint_vel"]);
        resolve(&tags, &space(&obs), &space(GO2_ACT), &spec, false)
    }

    fn go2_body_raw() -> BTreeMap<String, Value> {
        let mut raw: BTreeMap<String, Value> = BTreeMap::new();
        raw.insert("ang_vel".to_owned(), tensor(&[4.0, 0.0, 0.0]));
        raw.insert("base_quat".to_owned(), tensor(&[1.0, 0.0, 0.0, 0.0]));
        raw.insert("command".to_owned(), tensor(&[0.5, 0.0, 0.0]));
        raw.insert("joint_pos".to_owned(), tensor(&[0.0; 12]));
        raw.insert("joint_vel".to_owned(), tensor(&[0.0; 12]));
        raw
    }

    /// The assembled `obs` vector at `step`, through the stateful seam.
    fn assembled(
        adapter: &crate::plans::ResolvedAdapter,
        buffers: &mut crate::stateful::FrameBuffers,
        step: i64,
    ) -> Vec<f32> {
        let payload = crate::stateful::assemble_obs(
            adapter,
            &go2_body_raw(),
            "ep",
            step,
            buffers,
            &NoCustoms,
            &crate::stateful::NoEncodings,
        )
        .expect("assemble");
        let Value::Map(payload) = payload else {
            panic!("expected a map");
        };
        let Value::Tensor(obs) = &payload["obs"] else {
            panic!("expected a tensor");
        };
        crate::apply::value::to_f32_vec(obs)
    }

    #[test]
    fn the_previous_action_completes_the_go2_observation_at_45_wide() {
        let adapter = go2_full_resolve(&go2_body_env(GO2_BODY_QUAT), PREVIOUS).expect("resolves");
        let described = adapter.describe();
        assert!(
            described.contains(
                "joint_vel[:12] (*0.05)#sensed, previous action/joint_pos (fill 0.0)) clip[-100.0,100.0]\n"
            ),
            "got:\n{described}"
        );
        // Answered by the model's own actuator: no ad-hoc-role nudge, and the
        // route holds history the way a stacked one does.
        assert!(
            adapter.advisories().is_empty(),
            "{:?}",
            adapter.advisories()
        );
        assert_eq!(adapter.history_keys(), vec!["obs".to_owned()]);
        assert!(adapter.reads_previous_action() && adapter.holds_history());
        assert!(adapter.history_windows().is_empty());
        let crate::plans::ObsPlan::State(plan) = &adapter.obs_plans[0] else {
            panic!("expected a state plan");
        };
        assert_eq!(plan.native_width, Some(45));
        // The stateless transform reads the fill; the stateful seam reads the
        // raw output executed at the previous step, before the actuator's
        // affine, and the fill at the episode's first step.
        let Value::Map(payload) = adapter
            .transform_obs(&go2_body_raw(), &NoCustoms)
            .expect("apply")
        else {
            panic!("expected a map");
        };
        let Value::Tensor(obs) = &payload["obs"] else {
            panic!("expected a tensor");
        };
        assert_eq!(&crate::apply::value::to_f32_vec(obs)[33..], &[0.0; 12]);
        let mut buffers = crate::stateful::FrameBuffers::new();
        assert_eq!(&assembled(&adapter, &mut buffers, 0)[33..], &[0.0; 12]);
        let raw: Vec<f32> = (0..12).map(|i| 0.1 * i as f32).collect();
        crate::stateful::record_action(&adapter, &tensor(&raw), &mut buffers, "ep", 0)
            .expect("record");
        let obs = assembled(&adapter, &mut buffers, 1);
        assert_eq!(obs.len(), 45);
        assert_eq!(&obs[33..], raw.as_slice());
    }

    #[test]
    fn a_previous_action_part_permutes_the_actuator_axes_by_label() {
        // The model reads back its own output in Isaac order, and a stand-in
        // of 0.5 before the first action.
        let isaac = labels(&ISAAC);
        let previous = format!(
            r#"{{"role":"action/joint_pos","source":"action","labels":{isaac},"fill":0.5}}"#
        );
        let adapter = go2_full_resolve(&go2_body_env(GO2_BODY_QUAT), &previous).expect("resolves");
        assert!(
            adapter.describe().contains(
                "previous action/joint_pos perm[3,4,5,0,1,2,9,10,11,6,7,8] (fill 0.5)) clip"
            ),
            "got:\n{}",
            adapter.describe()
        );
        let mut buffers = crate::stateful::FrameBuffers::new();
        assert_eq!(&assembled(&adapter, &mut buffers, 0)[33..], &[0.5; 12]);
        let raw: Vec<f32> = (0..12).map(|i| i as f32).collect();
        crate::stateful::record_action(&adapter, &tensor(&raw), &mut buffers, "ep", 0)
            .expect("record");
        let expected: Vec<f32> = ISAAC.iter().map(|&i| i as f32).collect();
        assert_eq!(
            &assembled(&adapter, &mut buffers, 1)[33..],
            expected.as_slice()
        );
        // A label set that differs from the actuator's is a mismatch naming it.
        let err = go2_full_resolve(
            &go2_body_env(GO2_BODY_QUAT),
            r#"{"role":"action/joint_pos","source":"action","labels":["FR_hip_joint","FR_shin"]}"#,
        )
        .expect_err("missing label");
        assert_eq!(err.code, ErrorCode::LabelMismatch);
        assert!(
            err.message.contains(r#"the actuator lacks ["FR_shin"]"#),
            "{}",
            err.message
        );
    }

    #[test]
    fn a_previous_action_part_binds_only_the_models_own_actuator() {
        let env = go2_body_env(GO2_BODY_QUAT);
        let err = go2_full_resolve(&env, r#"{"role":"action/gripper","source":"action"}"#)
            .expect_err("no actuator");
        assert_eq!(err.code, ErrorCode::MissingRole);
        assert!(
            err.message.contains(
                r#"reads the previous action for role "action/gripper" but no actuator emits it"#
            ),
            "{}",
            err.message
        );
        // A declared dim must be the actuator's.
        let err = go2_full_resolve(
            &env,
            r#"{"role":"action/joint_pos","source":"action","dim":6}"#,
        )
        .expect_err("dim");
        assert_eq!(err.code, ErrorCode::DimMismatch);
        assert!(
            err.message
                .contains("declares dim 6 but the actuator emits 12"),
            "{}",
            err.message
        );
        // Never an env role: an observation tag under `action/` is refused at
        // join, before any model part is planned.
        let env = env.replacen(
            r#""role":"command/base_vel""#,
            r#""role":"action/base_vel""#,
            1,
        );
        let err = go2_full_resolve(&env, PREVIOUS).expect_err("action role on an observation");
        assert_eq!(err.code, ErrorCode::ActionRoleOnObservation);
        assert!(
            err.message.contains(
                r#"observation "command" declares role "action/base_vel", an action kind"#
            ),
            "{}",
            err.message
        );
    }

    #[test]
    fn a_sim_publishing_base_rot_twice_needs_the_model_to_pin_a_provenance() {
        let two = format!(
            r#"{GO2_BODY_QUAT}, "base_quat_est":{{"type":"state","role":"proprio/base_rot","encoding":"quat_wxyz","frame":"world","provenance":"estimated"}}"#
        );
        let truth = GO2_BODY_QUAT.replace("sensed", "privileged");
        let two = two.replacen(GO2_BODY_QUAT, &truth, 1);
        let tags: EnvTags = serde_json::from_str(&go2_body_env(&two)).unwrap();
        let obs = go2_body_obs(&[
            "ang_vel",
            "base_quat",
            "base_quat_est",
            "command",
            "joint_pos",
            "joint_vel",
        ]);
        let spec: ModelSpec = serde_json::from_str(&go2_body_model("")).unwrap();
        let err =
            resolve(&tags, &space(&obs), &space(GO2_ACT), &spec, false).expect_err("ambiguous");
        assert_eq!(err.code, ErrorCode::Ambiguous);
        assert!(
            err.message
                .contains(r#"under provenances ["estimated", "privileged"]"#),
            "{}",
            err.message
        );
        let spec: ModelSpec =
            serde_json::from_str(&go2_body_model(r#","provenance":"estimated""#)).unwrap();
        let adapter = resolve(&tags, &space(&obs), &space(GO2_ACT), &spec, false).expect("pinned");
        assert!(
            adapter
                .describe()
                .contains("base_quat_est (quat_wxyz->gravity_xyz)@world#estimated"),
            "{}",
            adapter.describe()
        );
        assert!(
            adapter.advisories().is_empty(),
            "{:?}",
            adapter.advisories()
        );
        let spec: ModelSpec =
            serde_json::from_str(&go2_body_model(r#","provenance":"sensed""#)).unwrap();
        let err = resolve(&tags, &space(&obs), &space(GO2_ACT), &spec, false).expect_err("neither");
        assert_eq!(err.code, ErrorCode::ProvenanceMismatch);
        assert!(
            err.message.contains(r#"accepts provenance ["sensed"] but the env declares it under ["estimated", "privileged"]"#),
            "{}",
            err.message
        );
    }
}

#[cfg(test)]
mod provenance_tests {
    use super::resolve;
    use crate::advisory::AdvisorySeverity;
    use crate::error::ErrorCode;
    use crate::space_view::SpaceView;
    use crate::spec::{EnvTags, ModelSpec};

    const OBS: &str = r#"{"kind":"dict","dtype":"unspecified","keys":["rot"],"children":[
        {"kind":"box","shape":[4],"dtype":"float32"}]}"#;
    const ACT: &str = r#"{"kind":"box","shape":[1],"dtype":"float32"}"#;

    fn env(provenance: &str) -> String {
        format!(
            r#"{{"observation":{{"rot":{{"type":"state","role":"proprio/base_rot","encoding":"quat_wxyz"{provenance}}}}},
                "action":{{"components":[{{"role":"action/gripper","dim":1}}]}}}}"#
        )
    }

    fn model(provenance: &str) -> String {
        format!(
            r#"{{"input":{{"s":{{"type":"state","components":[{{"role":"proprio/base_rot","encoding":"quat_wxyz"{provenance}}}]}}}},
                "output":{{"components":[{{"role":"action/gripper","dim":1}}]}}}}"#
        )
    }

    fn do_resolve(
        env: &str,
        model: &str,
    ) -> Result<crate::plans::ResolvedAdapter, crate::error::AdapterResolutionError> {
        let tags: EnvTags = serde_json::from_str(env).expect("parse env tags");
        let spec: ModelSpec = serde_json::from_str(model).expect("parse model spec");
        let obs: SpaceView = serde_json::from_str(OBS).unwrap();
        let act: SpaceView = serde_json::from_str(ACT).unwrap();
        resolve(&tags, &obs, &act, &spec, false)
    }

    #[test]
    fn agreement_and_env_only_are_silent_and_print_the_value() {
        for model_side in [
            "",
            r#","provenance":"sensed""#,
            r#","provenance":["estimated","sensed"]"#,
        ] {
            let adapter = do_resolve(&env(r#","provenance":"sensed""#), &model(model_side))
                .expect("resolves");
            assert!(
                adapter.describe().contains("concat(rot#sensed)"),
                "{model_side}: {}",
                adapter.describe()
            );
            assert!(
                adapter.advisories().is_empty(),
                "{:?}",
                adapter.advisories()
            );
        }
        let adapter = do_resolve(&env(""), &model("")).expect("resolves");
        assert!(
            adapter.describe().contains("concat(rot)"),
            "{}",
            adapter.describe()
        );
    }

    #[test]
    fn a_model_only_declaration_is_a_caution() {
        let adapter = do_resolve(&env(""), &model(r#","provenance":["sensed","estimated"]"#))
            .expect("resolves");
        let notes = adapter.advisories();
        assert!(
            notes
                .iter()
                .any(|note| note.severity == AdvisorySeverity::Caution
                    && note.message.contains(
                        r#"declares provenance "sensed|estimated" but the env declares none"#
                    )),
            "{notes:?}"
        );
        assert!(
            adapter.describe().contains("rot#sensed|estimated"),
            "{}",
            adapter.describe()
        );
        // Quiet channel: never under `dropped:`.
        assert!(!adapter.describe().contains("dropped:"));
    }

    #[test]
    fn a_disagreement_or_an_unknown_value_is_a_provenance_mismatch() {
        let err = do_resolve(
            &env(r#","provenance":"privileged""#),
            &model(r#","provenance":"sensed""#),
        )
        .expect_err("mismatch");
        assert_eq!(err.code, ErrorCode::ProvenanceMismatch);
        assert!(
            err.message
                .contains(r#"expects provenance ["sensed"] but the env declares "privileged""#),
            "{}",
            err.message
        );
        let err = do_resolve(&env(r#","provenance":"guessed""#), &model("")).expect_err("unknown");
        assert_eq!(err.code, ErrorCode::ProvenanceMismatch);
        assert!(
            err.message.contains("unrecognized provenance \"guessed\""),
            "{}",
            err.message
        );
        let err = do_resolve(
            &env(r#","provenance":"sensed""#),
            &model(r#","provenance":"guessed""#),
        )
        .expect_err("unknown");
        assert_eq!(err.code, ErrorCode::ProvenanceMismatch);
        assert!(
            err.message
                .contains(r#"unrecognized provenance ["guessed"]"#),
            "{}",
            err.message
        );
    }
}
