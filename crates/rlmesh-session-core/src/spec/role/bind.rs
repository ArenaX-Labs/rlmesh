//! Binding a lowered model spec to a lowered Gym env by role: matching
//! each model leaf and each env actuator to its counterpart, then planning
//! them (`image`, `state`, `text`, `custom`, `action`).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use crate::spec::projection::Projection;
use crate::spec::role::error::{Advisory, Error, ErrorCode};
use crate::spec::role::fmt::quoted;
use crate::spec::role::frames::{Attr, FrameRef, Provenance};
use crate::spec::role::input::{
    Accept, Actuator, Env, EnvImage, EnvState, EnvText, LeafKind, Spec,
};
use crate::spec::role::path::NodePath;
use crate::spec::role::plan::{ActionPlan, AppliedProjection, Binding, ObsPlan};
use crate::spec::role::roles::registry;
use crate::spec::role::{action, custom, image, state, text};

pub(super) type Result<T> = std::result::Result<T, Error>;

pub(super) fn err(code: ErrorCode, message: String) -> Error {
    Error::new(code, message)
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
    model: Option<&Accept<Provenance>>,
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
                set.names,
                attr.vocabulary()
            ),
        ));
    }
    match (env, model) {
        (None, None) => Ok(None),
        (Some(env), None) => Ok(Some(env.clone())),
        (None, Some(set)) => {
            let declared = set.names.join("|");
            advisories.push(Advisory::caution(format!(
                "role {}: the model declares provenance {} but the env declares none, so where \
                 the values it was trained on came from cannot be verified -- declare it on the \
                 env to silence this",
                quoted(role),
                quoted(&declared),
            )));
            Ok(Some(FrameRef::from(declared.as_str())))
        }
        (Some(env), Some(set)) if set.names.iter().any(|name| name == env.as_str()) => {
            Ok(Some(env.clone()))
        }
        (Some(env), Some(set)) => Err(err(
            ErrorCode::ProvenanceMismatch,
            format!(
                "role {}: the model expects provenance {:?} but the env declares {}",
                quoted(role),
                set.names,
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
pub(super) fn seek<'a, T>(
    by_key: &'a BTreeMap<LeafKey, T>,
    role: &str,
    part: Option<&str>,
    provenance: Option<&Accept<Provenance>>,
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
            .names
            .iter()
            .find_map(|wanted| {
                candidates
                    .iter()
                    .find(|((_, _, indexed), _)| indexed.as_deref() == Some(wanted.as_str()))
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
                    set.names,
                ),
            ),
        });
    };
    Ok(Some(Bound {
        feature,
        part: bound_part.clone(),
    }))
}

/// The named projections a resolve may apply (the model spec's `allow`) and
/// the ones it has applied so far, each with what it applied to.
pub(super) struct Projections {
    allowed: Vec<Projection>,
    applied: Vec<AppliedProjection>,
}

impl Projections {
    /// A ledger allowing `allowed`, for tests that plan one leaf.
    #[cfg(test)]
    pub(super) fn allowing(allowed: &[Projection]) -> Self {
        Self {
            allowed: allowed.to_vec(),
            applied: Vec::new(),
        }
    }

    /// Records `projection` applied to `target`; refused with `code` and
    /// `refusal` (plus how to allow it) when the model spec does not allow it.
    pub(super) fn apply(
        &mut self,
        projection: Projection,
        target: String,
        code: ErrorCode,
        refusal: String,
    ) -> Result<()> {
        if !self.allowed.contains(&projection) {
            return Err(err(
                code,
                format!(
                    "{refusal}. Binding anyway is the named projection {}; add it to the \
                     model spec's allow to bind",
                    quoted(projection.name())
                ),
            ));
        }
        self.applied.push(AppliedProjection { projection, target });
        Ok(())
    }
}

/// Model outputs no env actuator reads: refused, unless the model spec allows
/// `drop_action`, which then records each one it discards.
fn project(
    projections: &mut Projections,
    outputs: &action::ModelOutputs<'_>,
    plan: &ActionPlan,
    env: &[Actuator],
) -> Result<()> {
    let unconsumed = action::unconsumed(outputs, plan);
    let Some((role, part, _, dim)) = unconsumed.first().copied() else {
        return Ok(());
    };
    if !projections.allowed.contains(&Projection::DropAction) {
        let takes: Vec<String> = env
            .iter()
            .filter_map(|actuator| {
                actuator.role.as_ref().map(|role| match &actuator.part {
                    Some(part) => format!("{role}#{part}"),
                    None => role.clone(),
                })
            })
            .collect();
        return Err(err(
            ErrorCode::MissingRole,
            format!(
                "model output role {}{} ({dim} dims) drives no env actuator: the env takes {takes:?}. \
                 Discarding it is the named projection {}; add it to the model spec's allow to \
                 bind anyway",
                quoted(role),
                state::part_suffix(part),
                quoted(Projection::DropAction.name()),
            ),
        ));
    }
    projections
        .applied
        .extend(
            unconsumed
                .into_iter()
                .map(|(role, part, start, dim)| AppliedProjection {
                    projection: Projection::DropAction,
                    target: format!(
                        "model output {}{} [{start}, {}) is discarded",
                        quoted(role),
                        state::part_suffix(part),
                        start + dim
                    ),
                }),
        );
    Ok(())
}

/// Binds `spec` to `env` by role: every model input leaf to the env
/// feature it reads (or its fill), every env actuator to the model output
/// that drives it (or its fill), refusing the first pairing that cannot be
/// made. Custom inputs need `trust_entrypoints`.
pub fn bind(spec: &Spec, env: &Env, trust_entrypoints: bool) -> Result<Binding> {
    let mut projections = Projections {
        allowed: spec.allow.clone(),
        applied: Vec::new(),
    };
    // The model's own outputs, keyed once: the action planner binds env
    // actuators to them, and a state part reading its previous action does too.
    let outputs = action::index_outputs(&spec.output)?;
    let images = env
        .images
        .iter()
        .map(|image| (image.role.as_str(), image.part.as_deref(), None, image));
    let states = env.states.iter().map(|state| {
        (
            state.role.as_str(),
            state.part.as_deref(),
            state.provenance.as_ref().map(FrameRef::as_str),
            state,
        )
    });
    let texts = env.texts.iter().map(|text| (&text.role, text));
    let images_by_role: BTreeMap<LeafKey, &EnvImage> = index_by_key(images, "env image")?;
    let states_by_role: BTreeMap<LeafKey, &EnvState> = index_by_key(states, "env state")?;
    let texts_by_role: BTreeMap<String, &EnvText> = index_by_role(texts, "env text")?;

    // Side table for referenced-unknown detection: a role the env declares only
    // as an unrecognized observation kind. A roleless unknown leaf is
    // unreferenceable, so it never enters here (drop-only).
    let unknown_roles: BTreeMap<String, String> = env
        .unknown
        .iter()
        .filter_map(|unknown| {
            unknown
                .role
                .clone()
                .map(|role| (role, unknown.kind.clone()))
        })
        .collect();

    // The quiet channel: surfaced through `advisories()` but kept out of
    // `describe()`. The geometry cautions belong here -- nothing was dropped or
    // fabricated, so they have no business in the `dropped:` section describe()
    // builds from the other channel.
    let mut quiet: Vec<Advisory> = env.advisories.clone();
    let mut obs_plans: Vec<ObsPlan> = Vec::with_capacity(spec.inputs.len());
    for leaf in &spec.inputs {
        let placement = leaf.at.clone();
        obs_plans.push(match &leaf.kind {
            LeafKind::Image(input) => ObsPlan::Image(image::plan_image(
                input,
                placement,
                &images_by_role,
                &unknown_roles,
                &mut quiet,
                &mut projections,
            )?),
            LeafKind::State(input) => ObsPlan::State(state::plan_state(
                input,
                placement,
                &states_by_role,
                &outputs,
                &unknown_roles,
                &mut quiet,
                &mut projections,
            )?),
            LeafKind::Text(input) => ObsPlan::Text(text::plan_text(
                input,
                placement,
                &texts_by_role,
                &unknown_roles,
            )?),
            LeafKind::Custom { transform } => ObsPlan::Custom(custom::plan_custom(
                transform,
                placement,
                trust_entrypoints,
            )?),
        });
    }

    // Reaching here means no model input referenced an unknown kind (a reference
    // hard-errors above), so every recorded unknown leaf was unreferenced: emit
    // one deterministic advisory per leaf, sorted by tree path. Run proceeds.
    let mut advisories: Vec<Advisory> = env
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

    let action_plan = action::plan_action(
        &env.action,
        env.clip,
        &outputs,
        &mut quiet,
        &mut projections,
    )?;
    project(&mut projections, &outputs, &action_plan, &env.action)?;

    // Model-side ad-hoc input roles the env does not answer. An ad-hoc role
    // matches only on the exact string, so a typo or a private name silently
    // degrades to a zero fill (an optional input) -- the required cases already
    // hard-errored above, and an unmatched model output is refused unless
    // `drop_action` is allowed. One advisory per role, not per placement.
    let mut ad_hoc: BTreeSet<&str> = BTreeSet::new();
    for leaf in &spec.inputs {
        match &leaf.kind {
            LeafKind::Image(input) => {
                if !has_role(&images_by_role, &input.role) {
                    ad_hoc.insert(&input.role);
                }
            }
            LeafKind::State(input) => {
                // An action-source part is answered by the model's own
                // actuator (resolved above), never by an env leaf.
                ad_hoc.extend(
                    input
                        .pieces
                        .iter()
                        .filter(|piece| !piece.previous)
                        .filter_map(|piece| piece.role.as_deref())
                        .filter(|role| !has_role(&states_by_role, role)),
                );
            }
            LeafKind::Text(input) if !texts_by_role.contains_key(&input.role) => {
                ad_hoc.insert(&input.role);
            }
            _ => {}
        }
    }
    advisories.extend(
        ad_hoc
            .into_iter()
            .filter(|role| !registry::is_sanctioned_role(role))
            .map(|role| {
                let mut message = format!(
                    "model declares ad-hoc role {} that this env does not: an ad-hoc role \
                     matches only on the exact string, so it resolves to a fill here -- use \
                     a registered role, or the {} namespace to mark it intentionally \
                     non-standard",
                    quoted(role),
                    quoted("x/"),
                );
                if let Some(hint) = registry::kindless_hint(role) {
                    message.push_str(". ");
                    message.push_str(&hint);
                }
                Advisory::info(message)
            }),
    );

    // A model-declared range only fires as an affine map when the *env* side also
    // declares a range to bridge between; against an unbounded env feature the map
    // is skipped, so a lone model range silently does nothing (it remaps from/into
    // an env range, it does not clamp). Surface that so the author learns it will
    // not take effect. (State: env is the source; action: env is the destination.)
    for obs_plan in &obs_plans {
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

    // The frame-stacking × action-chunk-replay guard is the host engine's:
    // the execution horizon is a runtime decision, so it is checked where the
    // resolved stacks and the horizon are both known.
    Ok(Binding::new(obs_plans, action_plan, advisories, quiet)
        .with_projections(projections.applied))
}

#[cfg(test)]
mod geometry_rule_tests {
    use super::check_geometry;
    use crate::spec::role::frames::{Attr, FrameRef};
    use crate::spec::role::{Advisory, Error, ErrorCode, Severity};

    fn check(
        attr: Attr,
        env: Option<&str>,
        model: Option<&str>,
    ) -> (Result<Option<FrameRef>, Error>, Vec<Advisory>) {
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
        assert_eq!(notes[0].severity, Severity::Caution);
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
