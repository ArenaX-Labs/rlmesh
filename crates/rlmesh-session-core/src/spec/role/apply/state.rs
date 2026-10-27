//! A model state input from its sources.

#[allow(unused_imports)]
use crate::prelude::*;

use crate::spec::apply::{ApplyError, Tensor};
use crate::spec::convert;
use crate::spec::role::apply::geometry::convert_rotation_with;
use crate::spec::role::apply::{cast, err};
use crate::spec::role::plan::{StatePiece, StatePlan};

fn reshape(shape_spec: &[i64], len: usize) -> Result<Vec<usize>, ApplyError> {
    let mut inferred: Option<usize> = None;
    let mut known: usize = 1;
    for (position, &dim) in shape_spec.iter().enumerate() {
        if dim == -1 {
            if inferred.is_some() {
                return Err(err("reshape allows at most one -1 dimension"));
            }
            inferred = Some(position);
        } else if dim < 0 {
            return Err(err(format!("invalid reshape dimension {dim}")));
        } else {
            // checked: reshape elements come from the wire uncapped (unlike the
            // MAX_DIM-bounded count fields), so a pathological spec can overflow
            // usize here. With the workspace's overflow-checks=true that would
            // panic the per-step apply path; a clean error beats an abort.
            known = known
                .checked_mul(dim as usize)
                .ok_or_else(|| err(format!("reshape dimensions {shape_spec:?} overflow")))?;
        }
    }
    let mut out: Vec<usize> = shape_spec
        .iter()
        .map(|&dim| if dim < 0 { 0 } else { dim as usize })
        .collect();
    if let Some(position) = inferred {
        if known == 0 || !len.is_multiple_of(known) {
            return Err(err(format!(
                "cannot reshape {len} elements into {shape_spec:?}"
            )));
        }
        out[position] = len / known;
    } else if known != len {
        return Err(err(format!(
            "cannot reshape {len} elements into {shape_spec:?}"
        )));
    }
    Ok(out)
}

/// `value * scale + offset` per element ([`convert::affine`]), each of scale
/// and offset a scalar times/plus an optional per-axis value, then negated
/// when `invert` and less `threshold`. The per-axis widths were checked at
/// resolve; a runtime value of another width already failed the piece-width
/// assertion before this runs.
#[allow(
    clippy::too_many_arguments,
    reason = "a side's correction fields, in the order they apply"
)]
pub(super) fn apply_affine(
    value: &mut [f32],
    scale: Option<f64>,
    axis_scale: Option<&[f64]>,
    offset: Option<f64>,
    axis_offset: Option<&[f64]>,
    invert: bool,
    threshold: Option<f64>,
) {
    if scale.is_none()
        && axis_scale.is_none()
        && offset.is_none()
        && axis_offset.is_none()
        && !invert
        && threshold.is_none()
    {
        return;
    }
    let axis = |values: Option<&[f64]>, i: usize, default: f64| {
        values.and_then(|v| v.get(i).copied()).unwrap_or(default)
    };
    let sign = if invert { -1.0 } else { 1.0 };
    for (i, entry) in value.iter_mut().enumerate() {
        let s = sign * scale.unwrap_or(1.0) * axis(axis_scale, i, 1.0);
        let o =
            sign * (offset.unwrap_or(0.0) + axis(axis_offset, i, 0.0)) - threshold.unwrap_or(0.0);
        *entry = convert::affine(f64::from(*entry), 0, &[s], &[o]) as f32;
    }
}

impl StatePlan {
    /// The state vector: `read` gives an env-sourced piece's raw value as a
    /// flat `f32` vector, and `previous` is the raw action the model executed
    /// at the previous step (in model order, the action plan's width), read
    /// by any action-source piece; `None` is the episode's first step, where
    /// such a piece reads its `fill`.
    pub fn apply(
        &self,
        read: &mut dyn FnMut(&StatePiece) -> Result<Vec<f32>, ApplyError>,
        previous: Option<&[f32]>,
    ) -> Result<Tensor, ApplyError> {
        let plan = self;
        let mut state: Vec<f32> = Vec::new();
        for piece in &plan.pieces {
            if let (Some(action), Some(row)) = (&piece.previous, previous) {
                let slice = row
                    .get(action.start as usize..action.stop as usize)
                    .ok_or_else(|| {
                        err(format!(
                            "previous action for '{}' reads [{}, {}) but the recorded action has \
                             {} elements",
                            action.role,
                            action.start,
                            action.stop,
                            row.len()
                        ))
                    })?;
                match &piece.gather {
                    Some(gather) => state.extend(gather.iter().map(|&index| slice[index as usize])),
                    None => state.extend_from_slice(slice),
                }
                continue;
            }
            if let Some(fill) = piece.fill {
                let mut value = vec![fill as f32; piece.dim.unwrap_or(0) as usize];
                // The scalar affine was folded into `fill` at resolve; a per-axis
                // one has no scalar to fold into and applies here.
                apply_affine(
                    &mut value,
                    None,
                    piece.axis_scale.as_deref(),
                    None,
                    piece.axis_offset.as_deref(),
                    false,
                    None,
                );
                state.extend(value);
                continue;
            }
            let mut value = read(piece)?;
            // Slice the env field out of its leaf, before any rotation or
            // model-side index/dim. Only a StateLayout field sets src_offset; a
            // whole-leaf state leaves it None and reads the entire runtime value.
            if let Some(offset) = piece.src_offset {
                let start = offset as usize;
                let end = piece
                    .src_dim
                    .map_or(value.len(), |dim| start + dim as usize);
                // The slice was validated against the space width at resolve time;
                // a runtime value shorter than that is an env contract violation.
                // Erroring beats silently feeding the model a truncated field.
                if end > value.len() {
                    return Err(err(format!(
                        "state layout field '{}' needs indices [{start}, {end}) but the \
                         runtime observation has only {} elements",
                        piece.source,
                        value.len()
                    )));
                }
                value = value[start..end].to_vec();
            }
            // Permute the named axes into the model's order. The env leaf's width
            // was label-checked at join, so an index past the runtime value is an
            // env contract violation.
            if let Some(gather) = &piece.gather {
                let mut gathered = Vec::with_capacity(gather.len());
                for &index in gather {
                    let Some(entry) = value.get(index as usize) else {
                        return Err(err(format!(
                            "state piece '{}' gathers axis {index} but the runtime observation \
                             has only {} elements",
                            piece.source,
                            value.len()
                        )));
                    };
                    gathered.push(*entry);
                }
                value = gathered;
            }
            if let (Some(src), Some(dst)) = (piece.src_encoding, piece.dst_encoding)
                && (src != dst || piece.post_rotate.is_some())
            {
                value = convert_rotation_with(&value, src, dst, piece.post_rotate.as_ref())?;
            }
            if let Some(index) = piece.index {
                let index = index as usize;
                value = value
                    .get(index..value.len().min(index + 1))
                    .unwrap_or_default()
                    .to_vec();
            } else if let Some(dim) = piece.dim {
                value.truncate(dim as usize);
            }
            if let (Some(src), Some(dst)) = (piece.src_range, piece.dst_range)
                && src != dst
            {
                map_range(&mut value, src, dst)?;
            }
            if let Some(sum) = piece.flip {
                for entry in &mut value {
                    *entry = (sum - f64::from(*entry)) as f32;
                }
            }
            // The resolved widths are the state's layout: a host-side custom
            // encoding slices itself out of the assembled vector by them. A runtime
            // value of another width would shift every later piece, so say so here
            // instead of handing the model a quietly re-laid-out state (and
            // before a per-axis affine is zipped against it).
            if let Some(width) = piece.width
                && value.len() != width as usize
            {
                return Err(err(format!(
                    "state piece '{}' resolved to {width} dims but the runtime \
                     observation yields {}",
                    piece.source,
                    value.len()
                )));
            }
            // The model-side affine sits after the range map: `range` bridges two
            // declared scales, `scale`/`offset` is the model's own convention
            // (e.g. RoboTwin's `1 - 2g` gripper, a per-joint stand pose).
            apply_affine(
                &mut value,
                piece.scale,
                piece.axis_scale.as_deref(),
                piece.offset,
                piece.axis_offset.as_deref(),
                false,
                None,
            );
            state.extend(value);
        }
        // The container clamp sits after every part's own transforms and before
        // the pad, so a pad slot is never clamped into a non-zero value.
        if let Some((low, high)) = plan.clip {
            let (low, high) = (low as f32, high as f32);
            for entry in &mut state {
                *entry = entry.clamp(low, high);
            }
        }
        if let Some(pad_to) = plan.pad_to {
            let pad_to = pad_to as usize;
            if state.len() > pad_to {
                return Err(err(format!(
                    "state for '{}' has {} dims, more than pad_to={pad_to}",
                    plan.placement,
                    state.len()
                )));
            }
            state.resize(pad_to, 0.0);
        }
        let len = state.len();
        let shape = match &plan.reshape {
            Some(shape_spec) => reshape(shape_spec, len)?
                .into_iter()
                .map(|dim| dim as u32)
                .collect(),
            None => vec![len as u32],
        };
        let values: Vec<f64> = state.iter().map(|&v| f64::from(v)).collect();
        cast(&values, shape, plan.dtype)
    }
}

/// Affinely map values from the `src` range into the `dst` range, in `f32`.
pub(super) fn map_range(
    value: &mut [f32],
    src: (f64, f64),
    dst: (f64, f64),
) -> Result<(), ApplyError> {
    let (src_low, src_high) = (src.0 as f32, src.1 as f32);
    let (dst_low, dst_high) = (dst.0 as f32, dst.1 as f32);
    let span = src_high - src_low;
    if span == 0.0 {
        return Err(err("source range for an affine map has zero width"));
    }
    for entry in value {
        *entry = (*entry - src_low) / span * (dst_high - dst_low) + dst_low;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::role::path::NodePath;
    use crate::wire::part::DType;

    /// The state of `plan` over one env leaf of `values`.
    fn state(plan: &StatePlan, values: &[f32]) -> Vec<f32> {
        let tensor = plan
            .apply(&mut |_| Ok(values.to_vec()), None)
            .expect("apply");
        tensor
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    #[test]
    fn maps_state_values_from_env_range_into_model_range() {
        let plan = StatePlan {
            placement: NodePath::root().push_key("state"),
            pieces: vec![StatePiece {
                source: NodePath::root().push_key("gripper"),
                src_offset: None,
                src_dim: None,
                src_encoding: None,
                dst_encoding: None,
                post_rotate: None,
                dim: None,
                index: None,
                src_range: Some((0.0, 255.0)),
                dst_range: Some((-1.0, 1.0)),
                scale: None,
                offset: None,
                axis_scale: None,
                axis_offset: None,
                gather: None,
                labels: None,
                fill: None,
                absent_role: None,
                previous: None,
                width: Some(3),
                frame: None,
                provenance: None,
                part: None,
                flip: None,
            }],
            pad_to: None,
            clip: None,
            native_width: Some(3),
            dtype: DType::F32,
            reshape: None,
            list: false,
        };
        let values = state(&plan, &[0.0, 127.5, 255.0]);
        assert!((values[0] + 1.0).abs() < 1e-6, "{values:?}");
        assert!(values[1].abs() < 1e-6, "{values:?}");
        assert!((values[2] - 1.0).abs() < 1e-6, "{values:?}");
    }

    #[test]
    fn reshape_dim_product_overflow_is_a_clean_error() {
        // A pathological reshape spec overflows the usize product; it must
        // return an ApplyError, not panic (overflow-checks=true) on the hot path.
        let err = reshape(&[i64::MAX, i64::MAX], 4).unwrap_err();
        assert!(err.to_string().contains("overflow"), "got: {err}");
    }

    #[test]
    fn slices_a_layout_field_out_of_a_flat_leaf() {
        // A flat width-8 obs; the gripper field is at offset 3, width 1, then a
        // model index picks element 0 of that field.
        let plan = StatePlan {
            placement: NodePath::root().push_key("state"),
            pieces: vec![StatePiece {
                // A root (single-leaf) source: keyed under the reserved envelope
                // key, with the layout slice offset applied below.
                source: NodePath::root(),
                src_offset: Some(3),
                src_dim: Some(1),
                src_encoding: None,
                dst_encoding: None,
                post_rotate: None,
                dim: None,
                index: Some(0),
                src_range: None,
                dst_range: None,
                scale: None,
                offset: None,
                axis_scale: None,
                axis_offset: None,
                gather: None,
                labels: None,
                fill: None,
                absent_role: None,
                previous: None,
                width: Some(1),
                frame: None,
                provenance: None,
                part: None,
                flip: None,
            }],
            pad_to: None,
            clip: None,
            native_width: Some(1),
            dtype: DType::F32,
            reshape: None,
            list: false,
        };
        assert_eq!(
            state(&plan, &[0.0, 0.1, 0.2, 0.9, 0.4, 0.5, 0.6, 0.7]),
            vec![0.9]
        );
    }
}
