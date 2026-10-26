//! v1 declarations lowered into binder input, and [`resolve`]: an env's
//! tags and spaces and a model's spec, bound by the binder's Gym door
//! ([`rlmesh_session_core::spec::role`]).
//!
//! Lowering is where a declaration is held to what it alone can get wrong
//! (an unknown field, kind, projection, resample, crop mode, channel order
//! or dtype, a contradictory crop or fill); every check that needs the
//! other side is the binder's.

use rlmesh_session_core::spec::Projection;
use rlmesh_session_core::spec::plan::Fit;
use rlmesh_session_core::spec::policy::{ImageLayout as Layout, Resample};
use rlmesh_session_core::spec::role::input::{
    Accept, Actuator as RoleActuator, Crop, CustomArms, Env, EnvImage as RoleEnvImage,
    EnvState as RoleEnvState, EnvText as RoleEnvText, EnvUnknown, Image as RoleImage, Leaf,
    LeafKind, Piece, Spec, StackPad as RoleStackPad, State as RoleState, Text as RoleText,
};
use rlmesh_session_core::spec::role::{self, Severity};
use rlmesh_session_core::spec::rotation::Rotation;
use rlmesh_session_core::wire::DType;

use crate::advisory::{Advisory, AdvisorySeverity};
use crate::apply::RESAMPLES;
use crate::error::{AdapterResolutionError, ErrorCode};
use crate::join::{JoinError, join};
use crate::path::NodePath;
use crate::plans::Binding as ResolvedAdapter;
use crate::space_view::SpaceView;
use crate::spec::{
    AcceptSet, Actuator, CHANNEL_ORDERS, CROP_MODES, EnvFeature, EnvFeatures, FitMode, Image,
    ImageLayout, InputNode, ModelLeaf, ModelSpec, PartSource, RotationEncoding, StackPad, State,
    StateContainer, StateEncoding, TextContainer,
};

type Result<T> = std::result::Result<T, AdapterResolutionError>;

fn err(code: ErrorCode, message: String) -> AdapterResolutionError {
    AdapterResolutionError::new(code, message)
}

fn quoted(value: &str) -> String {
    format!("{value:?}")
}

/// Derive the binding of an env/model pair.
///
/// The env side is given as the [`EnvTags`](crate::spec::EnvTags)
/// observation/action tree over its observation and action spaces; [`join`]
/// derives the keyed env features (with widths and ranges) those plus the
/// spaces imply, and the binder matches each model input leaf to an env
/// feature by role, placed at its tree position. A model role may be reused
/// across leaves (one camera → several input slots).
pub fn resolve(
    env_tags: &crate::spec::EnvTags,
    observation_space: &SpaceView,
    action_space: &SpaceView,
    model_spec: &ModelSpec,
    trust_entrypoints: bool,
) -> Result<ResolvedAdapter> {
    // READ-door taint (§8): a *bare* (non-`x-`) unknown field on a recognized
    // kind is must-understand — an old core silently applying its own default
    // for a modifier it never parsed is worse than failing. Fail closed, on both
    // sides, before any plan is built. Unknown *kinds* are tolerated here (the
    // binder ignores or fails them precisely); only fields taint.
    crate::spec::reject_bare_fields_env(env_tags)
        .map_err(|message| err(ErrorCode::UnsupportedKind, message))?;
    crate::spec::reject_bare_fields_model(model_spec)
        .map_err(|message| err(ErrorCode::UnsupportedKind, message))?;
    let spec = lower_model(model_spec)?;
    let features = join(env_tags, observation_space, action_space).map_err(|error| {
        let code = match error {
            JoinError::ActionRoleOnObservation { .. } => ErrorCode::ActionRoleOnObservation,
            _ => ErrorCode::InvalidTag,
        };
        err(code, error.to_string())
    })?;
    role::bind(&spec, &lower_env(&features), trust_entrypoints)
}

/// The binding's advisories as the host's.
pub fn advisories(adapter: &ResolvedAdapter) -> Vec<Advisory> {
    adapter
        .advisories()
        .into_iter()
        .map(host_advisory)
        .collect()
}

fn host_advisory(advisory: role::Advisory) -> Advisory {
    Advisory {
        severity: match advisory.severity {
            Severity::Info => AdvisorySeverity::Info,
            Severity::Caution => AdvisorySeverity::Caution,
        },
        message: advisory.message,
    }
}

fn role_advisory(advisory: &Advisory) -> role::Advisory {
    match advisory.severity {
        AdvisorySeverity::Info => role::Advisory::info(advisory.message.clone()),
        AdvisorySeverity::Caution => role::Advisory::caution(advisory.message.clone()),
    }
}

/// The binder's encoding for a v1 one (the same names).
pub(crate) fn rotation(encoding: RotationEncoding) -> Rotation {
    match encoding {
        RotationEncoding::QuatXyzw => Rotation::QuatXyzw,
        RotationEncoding::QuatWxyz => Rotation::QuatWxyz,
        RotationEncoding::AxisAngle => Rotation::AxisAngle,
        RotationEncoding::Rot6d => Rotation::Rot6d,
        RotationEncoding::Rot6dRowMajor => Rotation::Rot6dRowMajor,
        RotationEncoding::EulerXyz => Rotation::EulerXyz,
        RotationEncoding::GravityXyz => Rotation::GravityXyz,
    }
}

fn layout(layout: ImageLayout) -> Layout {
    match layout {
        ImageLayout::Hwc => Layout::Hwc,
        ImageLayout::Chw => Layout::Chw,
    }
}

fn fit(mode: FitMode) -> Fit {
    match mode {
        FitMode::Stretch => Fit::Stretch,
        FitMode::Crop => Fit::Crop,
        FitMode::Pad => Fit::Pad,
    }
}

fn rotations(set: &AcceptSet<RotationEncoding>) -> Accept<Rotation> {
    Accept {
        names: set.wire_names().into_iter().map(str::to_owned).collect(),
        known: set.known().map(rotation).collect(),
    }
}

fn dtype(name: &str, at: &str) -> Result<DType> {
    rlmesh_spaces::DType::from_name(name)
        .and_then(|dtype| DType::from_code(dtype as u32))
        .ok_or_else(|| {
            err(
                ErrorCode::Unsupported,
                format!("model input {at}: unknown dtype {}", quoted(name)),
            )
        })
}

/// The model spec's input leaves in tree order, its action layout and its
/// allowed projections.
fn lower_model(model_spec: &ModelSpec) -> Result<Spec> {
    let allow = model_spec
        .allow
        .iter()
        .map(|name| {
            Projection::from_name(name).ok_or_else(|| {
                let known: Vec<&str> = Projection::ALL.iter().map(|p| p.name()).collect();
                err(
                    ErrorCode::Unsupported,
                    format!(
                        "model spec allows unknown projection {}; the named projections are {known:?}",
                        quoted(name)
                    ),
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut inputs = Vec::new();
    collect(&model_spec.input, NodePath::root(), &mut inputs)?;
    // `clip` is an env-side actuator clamp on the assembled action vector; a
    // clip declared on the *model* action layout would silently do nothing.
    // Reject it, mirroring the per-component guard that likewise rejects an
    // env-side knob on the model.
    if model_spec.output.clip.is_some() {
        return Err(err(
            ErrorCode::Unsupported,
            "clip is an env-side actuator clamp; the model action declaration must leave it unset"
                .to_owned(),
        ));
    }
    Ok(Spec {
        inputs,
        output: model_spec.output.components.iter().map(actuator).collect(),
        allow,
    })
}

/// Flatten the model input tree into leaves, each carrying the placement of
/// where its produced value lands in the payload. A `Dict` node recurses each
/// key, a `Tuple` each index, a `Leaf` lowers itself.
fn collect(node: &InputNode, placement: NodePath, out: &mut Vec<Leaf>) -> Result<()> {
    match node {
        InputNode::Leaf(leaf) => {
            let kind = leaf_kind(leaf, &placement)?;
            out.push(Leaf {
                at: placement,
                kind,
            });
        }
        InputNode::Dict(map) => {
            for (key, child) in map {
                collect(child, placement.push_key(key.clone()), out)?;
            }
        }
        InputNode::Tuple(items) => {
            for (index, item) in items.iter().enumerate() {
                collect(item, placement.push_index(index), out)?;
            }
        }
    }
    Ok(())
}

fn leaf_kind(leaf: &ModelLeaf, placement: &NodePath) -> Result<LeafKind> {
    let at = quoted(&placement.to_string());
    let check = |role: &str| {
        crate::roles::registry::check_role(role).map_err(|reason| {
            err(
                ErrorCode::UnsupportedKind,
                format!(
                    "model input {at}: {reason}; if a newer peer wrote it, upgrade the runtime"
                ),
            )
        })
    };
    Ok(match leaf {
        ModelLeaf::Image(input) => {
            check(&input.role)?;
            LeafKind::Image(image(input, &at)?)
        }
        ModelLeaf::State(input) => {
            for piece in &input.components {
                if let Some(role) = &piece.role {
                    check(role)?;
                }
            }
            LeafKind::State(state(input, &at)?)
        }
        ModelLeaf::Text(input) => {
            check(&input.role)?;
            LeafKind::Text(RoleText {
                role: input.role.clone(),
                fill: input.fill.clone(),
                list: input.container == TextContainer::List,
            })
        }
        ModelLeaf::Custom(input) => LeafKind::Custom {
            transform: input.transform.clone(),
        },
        // A model input of an unrecognized kind has no apply path on an old
        // core, even if the env offers a matching unknown feature — agreement
        // is irrelevant. Localized, named by placement.
        ModelLeaf::Unknown { kind, .. } => {
            return Err(err(
                ErrorCode::UnsupportedKind,
                format!(
                    "model input {at} is of unrecognized kind {}; upgrade the runtime",
                    quoted(kind)
                ),
            ));
        }
    })
}

fn image(input: &Image, at: &str) -> Result<RoleImage> {
    // `fill` only colors the zero-filled frame an `optional` camera
    // produces when the env lacks it; without `optional` it can never take
    // effect, so a set-but-inert fill is a spec error, not a silent no-op.
    if input.fill.is_some() && !input.optional {
        return Err(err(
            ErrorCode::Unsupported,
            format!(
                "model input {at}: fill only applies to an optional camera; \
                 set optional or drop fill"
            ),
        ));
    }
    let resample = Resample::parse(&input.resample)
        .filter(|_| RESAMPLES.contains(&input.resample.as_str()))
        .ok_or_else(|| {
            err(
                ErrorCode::Unsupported,
                format!(
                    "model input {at}: unsupported resample {}; expected one of {RESAMPLES:?}",
                    quoted(&input.resample),
                ),
            )
        })?;
    // The two other constrained-string vocabularies, checked like `resample`:
    // they are strings on the wire so a future additive value degrades to this
    // typed error instead of a parse failure.
    for (field, value, vocabulary) in [
        ("crop_mode", &input.crop_mode, CROP_MODES),
        ("channel_order", &input.channel_order, CHANNEL_ORDERS),
    ] {
        if !vocabulary.contains(&value.as_str()) {
            return Err(err(
                ErrorCode::Unsupported,
                format!(
                    "model input {at}: unsupported {field} {}; expected one of {vocabulary:?}",
                    quoted(value)
                ),
            ));
        }
    }
    // The two fractions say the same thing two ways — a side fraction and an
    // area fraction — so declaring both is a spec error rather than a silent
    // precedence rule.
    if input.crop.is_some() && input.crop_area.is_some() {
        return Err(err(
            ErrorCode::Unsupported,
            format!(
                "model input {at}: pass crop (a side fraction) or crop_area (an area \
                 fraction), not both"
            ),
        ));
    }
    let crop = input
        .crop
        .or_else(|| input.crop_area.map(f64::sqrt))
        .map(|fraction| Crop {
            fraction,
            area: input.crop_area,
            slice: input.crop_mode == "slice",
        });
    Ok(RoleImage {
        role: input.role.clone(),
        part: input.part.clone(),
        height: input.height,
        width: input.width,
        layout: layout(input.layout),
        channels: input.channels,
        dtype: dtype(&input.dtype, at)?,
        normalize: input.normalize.range(),
        lead_dims: input.lead_dims,
        upside_down: input.upside_down,
        resample,
        allow_upscale: input.allow_upscale,
        fit: input.fit.as_ref().map(|set| Accept {
            names: set.wire_names().into_iter().map(str::to_owned).collect(),
            known: set.known().map(fit).collect(),
        }),
        optional: input.optional,
        fill: input.fill,
        stack: input.stack,
        offsets: input.offsets.clone(),
        stack_pad: match input.stack_pad {
            StackPad::First => RoleStackPad::First,
            StackPad::Black => RoleStackPad::Black,
        },
        crop,
        jpeg_quality: input.jpeg_quality,
        bgr: input.channel_order == "bgr",
        render: input.render,
    })
}

fn state(input: &State, at: &str) -> Result<RoleState> {
    Ok(RoleState {
        pieces: input
            .components
            .iter()
            .map(|part| {
                let (encoding, custom) = match &part.encoding {
                    None => (None, None),
                    Some(StateEncoding::Native(set)) => (Some(rotations(set)), None),
                    Some(StateEncoding::Custom(custom)) => (
                        Some(rotations(&AcceptSet::single(custom.base))),
                        Some(CustomArms {
                            from_base: custom.from_base.is_some(),
                            to_base: custom.to_base.is_some(),
                        }),
                    ),
                };
                Piece {
                    role: part.role.clone(),
                    previous: part.source == PartSource::Action,
                    encoding,
                    custom,
                    dim: part.dim,
                    index: part.index,
                    range: part.range,
                    optional: part.optional,
                    fill: part.fill,
                    post_rotate: part.post_rotate.as_ref().map(|literal| literal.matrix()),
                    scale: part.scale,
                    offset: part.offset,
                    axis_scale: part.axis_scale.clone(),
                    axis_offset: part.axis_offset.clone(),
                    frame: part.frame.clone(),
                    provenance: part.provenance.as_ref().map(|set| Accept {
                        names: set.wire_names().into_iter().map(str::to_owned).collect(),
                        known: set.known().collect(),
                    }),
                    part: part.part.clone(),
                    labels: part.labels.clone(),
                    polarity: part.polarity,
                }
            })
            .collect(),
        pad_to: input.pad_to,
        dtype: dtype(&input.dtype, at)?,
        reshape: input.reshape.clone(),
        clip: input.clip,
        list: input.container == StateContainer::List,
    })
}

fn actuator(component: &Actuator) -> RoleActuator {
    let custom = component.encoding.as_ref().and_then(|encoding| {
        encoding.custom().map(|custom| CustomArms {
            from_base: custom.from_base.is_some(),
            to_base: custom.to_base.is_some(),
        })
    });
    RoleActuator {
        role: component.role.clone(),
        dim: component.dim,
        encoding: component
            .encoding
            .as_ref()
            .map(|encoding| rotation(encoding.base())),
        custom,
        range: component.range,
        binary: component.binary,
        scale: component.scale,
        offset: component.offset,
        axis_scale: component.axis_scale.clone(),
        axis_offset: component.axis_offset.clone(),
        invert: component.invert,
        threshold: component.threshold,
        clip: component.clip,
        fill: component.fill,
        optional: component.optional,
        frame: component.frame.clone(),
        reference: component.reference.clone(),
        part: component.part.clone(),
        labels: component.labels.clone(),
        polarity: component.polarity,
        unit: component.unit,
    }
}

/// The env's joined features as binder input.
fn lower_env(features: &EnvFeatures) -> Env {
    let mut env = Env {
        action: features.action.components.iter().map(actuator).collect(),
        clip: features.action.clip,
        advisories: features.advisories.iter().map(role_advisory).collect(),
        unknown: features
            .unknown
            .iter()
            .map(|unknown| EnvUnknown {
                source: unknown.source.clone(),
                role: unknown.role.clone(),
                kind: unknown.kind.clone(),
            })
            .collect(),
        ..Env::default()
    };
    for feature in &features.observation {
        match feature {
            EnvFeature::Image(image) => env.images.push(RoleEnvImage {
                source: image.source.clone(),
                role: image.role.clone(),
                part: image.part.clone(),
                layout: layout(image.layout),
                upside_down: image.upside_down,
                height: image.height,
                width: image.width,
                channels: image.channels,
                value_range: image.value_range,
            }),
            EnvFeature::State(state) => env.states.push(RoleEnvState {
                source: state.source.clone(),
                role: state.role.clone(),
                part: state.part.clone(),
                slice_offset: state.slice_offset,
                dim: state.dim,
                encoding: state.encoding.as_ref().map(rotations),
                range: state.range,
                frame: state.frame.clone(),
                provenance: state.provenance.clone(),
                labels: state.labels.clone(),
                polarity: state.polarity,
            }),
            EnvFeature::Text(text) => env.texts.push(RoleEnvText {
                source: text.source.clone(),
                role: text.role.clone(),
            }),
        }
    }
    env
}

#[cfg(test)]
mod tests;
