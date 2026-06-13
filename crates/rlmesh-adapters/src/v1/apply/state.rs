//! Produce one model state input from a raw observation.

use std::collections::BTreeMap;

use super::super::plans::StatePlan;
use super::super::spec::StateContainer;
use super::error::ApplyError;
use super::geometry::convert_rotation;
use super::lookup::{lookup, numeric_vector};
use super::value::{Array, ArrayData, Dtype, Value};

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

fn list_value(array: &Array) -> Vec<Value> {
    match &array.data {
        ArrayData::U8(data) => data.iter().map(|&x| Value::Number(f64::from(x))).collect(),
        ArrayData::I32(data) => data.iter().map(|&x| Value::Number(f64::from(x))).collect(),
        ArrayData::I64(data) => data.iter().map(|&x| Value::Number(x as f64)).collect(),
        ArrayData::F32(data) => data.iter().map(|&x| Value::Number(f64::from(x))).collect(),
        ArrayData::F64(data) => data.iter().map(|&x| Value::Number(x)).collect(),
    }
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
    let mut array = Array::from_f32(vec![len], state).cast(Dtype::parse(&plan.dtype)?);
    if let Some(shape_spec) = &plan.reshape {
        array.shape = reshape(shape_spec, len)?;
    }
    if plan.container == StateContainer::List {
        return Ok(Value::List(list_value(&array)));
    }
    Ok(Value::Array(array))
}
