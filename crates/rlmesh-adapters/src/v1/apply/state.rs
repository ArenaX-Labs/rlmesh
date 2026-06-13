//! Produce one model state input from a raw observation.

use std::collections::BTreeMap;

use rlmesh_spaces::{DType, Tensor};

use super::super::plans::StatePlan;
use super::super::spec::StateContainer;
use super::error::ApplyError;
use super::geometry::convert_rotation;
use super::lookup::{lookup, map_range, numeric_vector};
use super::value::{self, Value};

fn reshape(shape_spec: &[i64], len: usize) -> Result<Vec<usize>, ApplyError> {
    let mut inferred: Option<usize> = None;
    let mut known: usize = 1;
    for (position, &dim) in shape_spec.iter().enumerate() {
        if dim == -1 {
            if inferred.is_some() {
                return Err(ApplyError::new(
                    "reshape allows at most one -1 dimension".to_owned(),
                ));
            }
            inferred = Some(position);
        } else if dim < 0 {
            return Err(ApplyError::new(format!("invalid reshape dimension {dim}")));
        } else {
            known *= dim as usize;
        }
    }
    let mut out: Vec<usize> = shape_spec
        .iter()
        .map(|&dim| if dim < 0 { 0 } else { dim as usize })
        .collect();
    if let Some(position) = inferred {
        if known == 0 || !len.is_multiple_of(known) {
            return Err(ApplyError::new(format!(
                "cannot reshape {len} elements into {shape_spec:?}"
            )));
        }
        out[position] = len / known;
    } else if known != len {
        return Err(ApplyError::new(format!(
            "cannot reshape {len} elements into {shape_spec:?}"
        )));
    }
    Ok(out)
}

fn list_value(tensor: &Tensor) -> Vec<Value> {
    value::to_f64_vec(tensor)
        .into_iter()
        .map(Value::Number)
        .collect()
}

/// Produce one model state input from a raw observation.
pub(super) fn apply_state(
    plan: &StatePlan,
    raw_obs: &BTreeMap<String, Value>,
) -> Result<Value, ApplyError> {
    let mut state: Vec<f32> = Vec::new();
    for piece in &plan.pieces {
        if piece.zero_fill {
            state.extend(std::iter::repeat_n(0.0f32, piece.dim.unwrap_or(0) as usize));
            continue;
        }
        let mut value = numeric_vector(lookup(raw_obs, &piece.env_key)?)?;
        if let (Some(src), Some(dst)) = (piece.src_encoding, piece.dst_encoding)
            && src != dst
        {
            value = convert_rotation(&value, src, dst)?;
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
        state.extend(value);
    }
    if let Some(pad_to) = plan.pad_to {
        let pad_to = pad_to as usize;
        if state.len() > pad_to {
            return Err(ApplyError::new(format!(
                "state for '{}' has {} dims, more than pad_to={pad_to}",
                plan.model_key,
                state.len()
            )));
        }
        state.resize(pad_to, 0.0);
    }
    let len = state.len();
    let target = DType::from_name(&plan.dtype)
        .ok_or_else(|| ApplyError::new(format!("unsupported state dtype {:?}", plan.dtype)))?;
    let mut tensor = value::cast(&value::tensor_from_f32(vec![len as i64], &state), target)?;
    if let Some(shape_spec) = &plan.reshape {
        let shape = reshape(shape_spec, len)?;
        tensor = tensor
            .reshape(&value::shape_i64(&shape))
            .map_err(|err| ApplyError::new(err.to_string()))?;
    }
    if plan.container == StateContainer::List {
        return Ok(Value::List(list_value(&tensor)));
    }
    Ok(Value::Tensor(tensor))
}

#[cfg(test)]
mod tests {
    use super::super::super::plans::StatePiece;
    use super::*;

    #[test]
    fn maps_state_values_from_env_range_into_model_range() {
        let plan = StatePlan {
            model_key: "state".to_owned(),
            pieces: vec![StatePiece {
                env_key: "gripper".to_owned(),
                src_encoding: None,
                dst_encoding: None,
                dim: None,
                index: None,
                src_range: Some((0.0, 255.0)),
                dst_range: Some((-1.0, 1.0)),
                zero_fill: false,
            }],
            pad_to: None,
            dtype: "float32".to_owned(),
            reshape: None,
            container: StateContainer::Array,
        };
        let mut raw = BTreeMap::new();
        raw.insert(
            "gripper".to_owned(),
            Value::Tensor(value::tensor_from_f32(vec![3], &[0.0, 127.5, 255.0])),
        );
        let Value::Tensor(out) = apply_state(&plan, &raw).expect("apply") else {
            panic!("expected a tensor");
        };
        let values = value::to_f32_vec(&out);
        assert!((values[0] + 1.0).abs() < 1e-6, "{values:?}");
        assert!(values[1].abs() < 1e-6, "{values:?}");
        assert!((values[2] - 1.0).abs() < 1e-6, "{values:?}");
    }
}
