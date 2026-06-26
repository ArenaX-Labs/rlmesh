//! Pair each model state component with an env feature and derive the plan.

use std::collections::BTreeMap;

use super::{Result, err};
use crate::error::ErrorCode;
use crate::fmt::{quoted, quoted_accept_set, quoted_encoding, quoted_keys};
use crate::path::NodePath;
use crate::plans::{StatePiece, StatePlan};
use crate::spec::{AcceptSet, ConcatPart, EnvState, RotationEncoding, State};

/// Width of an optional component's zero fill when the env lacks it.
fn zero_fill_width(component: &ConcatPart, at: &str) -> Result<u32> {
    if component.index.is_some() {
        return Ok(1);
    }
    if let Some(dim) = component.dim {
        return Ok(dim);
    }
    if let Some(native) = component
        .encoding
        .as_ref()
        .and_then(|set| set.first_known())
    {
        return Ok(native.dims());
    }
    Err(err(
        ErrorCode::MissingWidth,
        format!(
            "model input {at}: optional state role {} needs dim, index, or encoding \
         to size its zero fill",
            quoted(&component.role)
        ),
    ))
}

/// Choose the (source, destination) rotation encodings for a state piece.
///
/// `env` is the producer: its first recognized entry is the native (raw)
/// encoding the runtime value is in. `model` is the consumer: its entries are
/// an accept-set in preference order. Prefer no conversion — when the model
/// accepts the env's native, use it on both sides; otherwise convert the native
/// into the model's most-preferred recognized encoding. `(None, None)` means
/// neither side declares a rotation. A side that declares only *unrecognized*
/// encodings is a typed resolve error (graceful degradation's loud edge), not a
/// silent pass.
fn select_state_encoding(
    role: &str,
    env: Option<&AcceptSet<RotationEncoding>>,
    model: Option<&AcceptSet<RotationEncoding>>,
) -> Result<(Option<RotationEncoding>, Option<RotationEncoding>)> {
    let (env_set, model_set) = match (env, model) {
        (None, None) => return Ok((None, None)),
        (Some(_), None) | (None, Some(_)) => {
            return Err(err(
                ErrorCode::EncodingMismatch,
                format!(
                    "state role {}: cannot convert encoding {} to {}; both sides \
                 must declare a rotation encoding",
                    quoted(role),
                    quoted_accept_set(env),
                    quoted_accept_set(model)
                ),
            ));
        }
        (Some(env_set), Some(model_set)) => (env_set, model_set),
    };
    let Some(native) = env_set.first_known() else {
        return Err(err(
            ErrorCode::EncodingMismatch,
            format!(
                "state role {}: env declares only unrecognized rotation encoding(s) {:?}; \
             upgrade the runtime to one that recognizes them",
                quoted(role),
                env_set.wire_names()
            ),
        ));
    };
    if model_set.first_known().is_none() {
        return Err(err(
            ErrorCode::EncodingMismatch,
            format!(
                "state role {}: model declares only unrecognized rotation encoding(s) {:?}; \
             upgrade the runtime to one that recognizes them",
                quoted(role),
                model_set.wire_names()
            ),
        ));
    }
    let dst = if model_set.accepts(native) {
        native
    } else {
        model_set
            .first_known()
            .expect("model has a recognized encoding")
    };
    Ok((Some(native), Some(dst)))
}

pub(super) fn plan_state(
    model_input: &State,
    placement: NodePath,
    states_by_role: &BTreeMap<String, &EnvState>,
) -> Result<StatePlan> {
    let at = quoted(&placement.to_string());
    let mut pieces: Vec<StatePiece> = Vec::with_capacity(model_input.components.len());
    for component in &model_input.components {
        let Some(env_state) = states_by_role.get(&component.role).copied() else {
            if component.optional {
                pieces.push(StatePiece {
                    source: NodePath::root(),
                    src_offset: None,
                    src_dim: None,
                    src_encoding: None,
                    dst_encoding: None,
                    dim: Some(zero_fill_width(component, &at)?),
                    index: None,
                    src_range: None,
                    dst_range: None,
                    zero_fill: true,
                });
                continue;
            }
            return Err(err(
                ErrorCode::MissingRole,
                format!(
                    "model input {at} needs state role {} but the env offers {}",
                    quoted(&component.role),
                    quoted_keys(states_by_role)
                ),
            ));
        };
        let (src_encoding, dst_encoding) = select_state_encoding(
            &component.role,
            env_state.encoding.as_ref(),
            component.encoding.as_ref(),
        )?;
        // When converting, the env feature's declared width must match the
        // native (source) encoding the raw value is in.
        if let (Some(src), Some(dst)) = (src_encoding, dst_encoding)
            && src != dst
            && let Some(env_dim) = env_state.dim
            && env_dim != src.dims()
        {
            return Err(err(
                ErrorCode::DimMismatch,
                format!(
                    "state role {}: env feature {} declares {env_dim} dims but \
                 encoding {} has {}",
                    quoted(&component.role),
                    quoted(&env_state.source.to_string()),
                    quoted_encoding(Some(src)),
                    src.dims()
                ),
            ));
        }
        // Bounds-check the requested slice against the source width. The
        // width is the env feature's, unless a rotation conversion reshapes it
        // first (in which case the converted width applies). Without this an
        // out-of-range index or dim silently yields fewer values.
        let converts = matches!(
            (src_encoding, dst_encoding),
            (Some(src), Some(dst)) if src != dst
        );
        let source_width = if converts {
            dst_encoding.map(|encoding| encoding.dims())
        } else {
            env_state.dim
        };
        if let Some(width) = source_width {
            if let Some(index) = component.index {
                if index >= width {
                    return Err(err(
                        ErrorCode::SliceOutOfRange,
                        format!(
                            "state role {}: index {index} is out of range for the \
                         width-{width} source feature {}",
                            quoted(&component.role),
                            quoted(&env_state.source.to_string())
                        ),
                    ));
                }
            } else if let Some(dim) = component.dim
                && dim > width
            {
                return Err(err(
                    ErrorCode::SliceOutOfRange,
                    format!(
                        "state role {}: requested {dim} dims but the source feature \
                     {} has width {width}",
                        quoted(&component.role),
                        quoted(&env_state.source.to_string())
                    ),
                ));
            }
        }
        pieces.push(StatePiece {
            source: env_state.source.clone(),
            src_offset: env_state.slice_offset,
            // src_dim is the slice width, meaningful only for a layout field
            // (where slice_offset is set); a whole-leaf state leaves it None so
            // the documented "used only when src_offset is set" invariant holds
            // (env_state.dim there is the advisory space width, not a slice).
            src_dim: env_state.slice_offset.and(env_state.dim),
            src_encoding,
            dst_encoding,
            dim: component.dim,
            index: component.index,
            src_range: env_state.range,
            dst_range: component.range,
            zero_fill: false,
        });
    }
    Ok(StatePlan {
        placement,
        pieces,
        pad_to: model_input.pad_to,
        dtype: model_input.dtype.clone(),
        reshape: model_input.reshape.clone(),
        container: model_input.container,
    })
}

#[cfg(test)]
mod encoding_selection_tests {
    use super::*;

    fn set(json: &str) -> AcceptSet<RotationEncoding> {
        serde_json::from_str(json).expect("parse accept-set")
    }

    #[test]
    fn matching_single_encoding_does_not_convert() {
        let (env, model) = (set(r#""quat_xyzw""#), set(r#""quat_xyzw""#));
        let (src, dst) =
            select_state_encoding("proprio/rot", Some(&env), Some(&model)).expect("ok");
        assert_eq!(src, Some(RotationEncoding::QuatXyzw));
        assert_eq!(dst, Some(RotationEncoding::QuatXyzw));
    }

    #[test]
    fn differing_single_encodings_convert_native_to_target() {
        let (env, model) = (set(r#""quat_xyzw""#), set(r#""rot6d""#));
        let (src, dst) =
            select_state_encoding("proprio/rot", Some(&env), Some(&model)).expect("ok");
        assert_eq!(src, Some(RotationEncoding::QuatXyzw)); // env native = source
        assert_eq!(dst, Some(RotationEncoding::Rot6d)); // model target
    }

    #[test]
    fn prefers_no_conversion_when_model_accepts_the_native() {
        // The model lists rot6d first but also accepts the env's native quat:
        // take the native (no conversion), even though rot6d is preferred.
        let (env, model) = (set(r#""quat_xyzw""#), set(r#"["rot6d", "quat_xyzw"]"#));
        let (src, dst) =
            select_state_encoding("proprio/rot", Some(&env), Some(&model)).expect("ok");
        assert_eq!(src, Some(RotationEncoding::QuatXyzw));
        assert_eq!(dst, Some(RotationEncoding::QuatXyzw));
    }

    #[test]
    fn falls_back_past_an_unrecognized_preference() {
        // Scenario 1: a model trained on a future encoding lists it first but
        // accepts the frozen env's native as a fallback — the unknown is
        // skipped at resolve, the native chosen, no runtime error.
        let (env, model) = (set(r#""quat_xyzw""#), set(r#"["rot10d", "quat_xyzw"]"#));
        let (src, dst) =
            select_state_encoding("proprio/rot", Some(&env), Some(&model)).expect("ok");
        assert_eq!(src, Some(RotationEncoding::QuatXyzw));
        assert_eq!(dst, Some(RotationEncoding::QuatXyzw));
    }

    #[test]
    fn wholly_unrecognized_declaration_errors_at_resolve() {
        // Graceful degradation's loud edge: a side that names only an unknown
        // encoding parses, but resolves to a typed error (not a silent pass).
        let (env, model) = (set(r#""quat_xyzw""#), set(r#""rot10d""#));
        let error =
            select_state_encoding("proprio/rot", Some(&env), Some(&model)).expect_err("err");
        assert_eq!(error.code, ErrorCode::EncodingMismatch);
        assert!(
            error.message.contains("unrecognized"),
            "got: {}",
            error.message
        );
    }

    #[test]
    fn one_sided_declaration_is_a_mismatch() {
        let model = set(r#""quat_xyzw""#);
        let error = select_state_encoding("proprio/rot", None, Some(&model)).expect_err("err");
        assert_eq!(error.code, ErrorCode::EncodingMismatch);
        assert!(
            error.message.contains("both sides"),
            "got: {}",
            error.message
        );
    }

    #[test]
    fn neither_side_declares_means_no_rotation() {
        let (src, dst) = select_state_encoding("proprio/rot", None, None).expect("ok");
        assert_eq!(src, None);
        assert_eq!(dst, None);
    }
}
