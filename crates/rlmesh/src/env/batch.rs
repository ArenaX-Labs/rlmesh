//! The batched value layout a vector env sees, lane projections, and render
//! frame encoding.

use image::ImageEncoder;
use image::codecs::png::PngEncoder;
use rlmesh_spaces::types::SpaceKind as WireKind;
use rlmesh_spaces::{DType as CoreDType, Scalar, SpaceSpec as WireSpec, decode_scalars};

use crate::ext::TensorExt;
use crate::{DType, Error, Metadata, Result, Tensor, Value};

fn is_stacked(spec: &WireSpec) -> bool {
    matches!(
        spec.spec,
        Some(WireKind::Box(_) | WireKind::MultiBinary(_) | WireKind::MultiDiscrete(_))
    )
}

fn batch_error(message: impl Into<String>) -> Error {
    Error::Type(message.into())
}

/// Stack per-lane values into the batched layout of `spec` (Python's
/// default numpy path): Box / MultiBinary / MultiDiscrete one stacked tensor,
/// Discrete one int64 tensor, Dict / Tuple per child, anything else a List.
pub(crate) fn stack_lanes(spec: &WireSpec, lanes: Vec<Value>) -> Result<Value> {
    match &spec.spec {
        Some(WireKind::Dict(dict)) => {
            let mut columns: Vec<Vec<Value>> =
                vec![Vec::with_capacity(lanes.len()); dict.keys.len()];
            for lane in lanes {
                let Value::Dict(mut map) = lane else {
                    return Err(batch_error("a Dict space's action must be a dict"));
                };
                for (column, key) in columns.iter_mut().zip(&dict.keys) {
                    column.push(map.remove(key).unwrap_or(Value::None));
                }
            }
            let mut out = Metadata::new();
            for ((key, child), column) in dict.keys.iter().zip(&dict.spaces).zip(columns) {
                out.insert(key.clone(), stack_lanes(child, column)?);
            }
            Ok(Value::Dict(out))
        }
        Some(WireKind::Tuple(tuple)) => {
            let mut columns: Vec<Vec<Value>> =
                vec![Vec::with_capacity(lanes.len()); tuple.spaces.len()];
            for lane in lanes {
                let (Value::Tuple(items) | Value::List(items)) = lane else {
                    return Err(batch_error("a Tuple space's action must be a tuple"));
                };
                for (column, item) in columns.iter_mut().zip(items) {
                    column.push(item);
                }
            }
            Ok(Value::Tuple(
                tuple
                    .spaces
                    .iter()
                    .zip(columns)
                    .map(|(child, column)| stack_lanes(child, column))
                    .collect::<Result<_>>()?,
            ))
        }
        Some(WireKind::Discrete(_)) => {
            let ints = lanes
                .iter()
                .map(|lane| match lane {
                    Value::Int(int) => Ok(*int),
                    other => Err(batch_error(format!(
                        "a Discrete action must be an int, got {}",
                        other.kind().python_name()
                    ))),
                })
                .collect::<Result<Vec<i64>>>()?;
            let bytes = ints.iter().flat_map(|int| int.to_le_bytes()).collect();
            Ok(Value::Tensor(Tensor::from_vec(
                bytes,
                &[ints.len()],
                DType::Int64,
            )?))
        }
        _ if is_stacked(spec) => {
            let tensors = lanes
                .into_iter()
                .map(|lane| match lane {
                    Value::Tensor(tensor) => Ok(tensor),
                    other => Err(batch_error(format!(
                        "a {} action must be a tensor, got {}",
                        crate::spaces::SpaceKind::of(spec),
                        other.kind().python_name()
                    ))),
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Value::Tensor(Tensor::stack(&tensors)?))
        }
        _ => Ok(Value::List(lanes)),
    }
}

/// Split a batched value of `spec` into `num_envs` per-lane values. A List of
/// per-lane values is accepted for any kind, as Python accepts any sequence.
pub(crate) fn unstack_lanes(spec: &WireSpec, batch: Value, num_envs: usize) -> Result<Vec<Value>> {
    let lanes = unstack_lanes_unchecked(spec, batch, num_envs)?;
    for lane in &lanes {
        crate::value::space_codec::value_to_space(spec, lane)?;
    }
    Ok(lanes)
}

fn unstack_lanes_unchecked(spec: &WireSpec, batch: Value, num_envs: usize) -> Result<Vec<Value>> {
    if let Value::List(items) = batch {
        return check_width(items, num_envs);
    }
    match (&spec.spec, batch) {
        (Some(WireKind::Dict(dict)), Value::Dict(mut map)) => {
            let mut lanes = vec![Metadata::new(); num_envs];
            for (key, child) in dict.keys.iter().zip(&dict.spaces) {
                let Some(column) = map.remove(key) else {
                    continue;
                };
                for (lane, value) in lanes
                    .iter_mut()
                    .zip(unstack_lanes_unchecked(child, column, num_envs)?)
                {
                    lane.insert(key.clone(), value);
                }
            }
            Ok(lanes.into_iter().map(Value::Dict).collect())
        }
        (Some(WireKind::Tuple(tuple)), Value::Tuple(children)) => {
            let mut lanes = vec![Vec::with_capacity(children.len()); num_envs];
            for (child, column) in tuple.spaces.iter().zip(children) {
                for (lane, value) in lanes
                    .iter_mut()
                    .zip(unstack_lanes_unchecked(child, column, num_envs)?)
                {
                    lane.push(value);
                }
            }
            Ok(lanes.into_iter().map(Value::Tuple).collect())
        }
        (Some(WireKind::Discrete(_)), Value::Tensor(tensor)) => {
            let ints = tensor_ints(&tensor)?;
            check_width(ints.into_iter().map(Value::Int).collect(), num_envs)
        }
        (_, Value::Tensor(tensor)) if is_stacked(spec) => check_width(
            tensor.unstack()?.into_iter().map(Value::Tensor).collect(),
            num_envs,
        ),
        (_, other) => Err(batch_error(format!(
            "a batched {} value cannot be a {}",
            crate::spaces::SpaceKind::of(spec),
            other.kind().python_name()
        ))),
    }
}

fn check_width(items: Vec<Value>, num_envs: usize) -> Result<Vec<Value>> {
    if items.len() != num_envs {
        return Err(batch_error(format!(
            "a batch of {} values for {num_envs} envs",
            items.len()
        )));
    }
    Ok(items)
}

fn tensor_ints(tensor: &Tensor) -> Result<Vec<i64>> {
    let core = tensor.core();
    decode_scalars(&core.to_contiguous_bytes(), core.dtype())
        .map_err(|err| batch_error(err.to_string()))?
        .into_iter()
        .map(|scalar| match scalar {
            Scalar::Int(int) => Ok(int),
            Scalar::Float(float) if float.fract() == 0.0 => Ok(float as i64),
            _ => Err(batch_error("a Discrete batch must hold integers")),
        })
        .collect()
}

/// Lane 0 of a batched value, walking the per-lane `spec` (Python's
/// `_lane0`): Dict and Tuple per child, anything else the first item of a
/// List or the first slice of a Tensor.
pub(crate) fn lane0(spec: &WireSpec, value: Value) -> Result<Value> {
    let lane = lane0_unchecked(spec, value)?;
    crate::value::space_codec::value_to_space(spec, &lane)?;
    Ok(lane)
}

fn lane0_unchecked(spec: &WireSpec, value: Value) -> Result<Value> {
    match (&spec.spec, value) {
        (Some(WireKind::Dict(dict)), Value::Dict(mut map)) => {
            let mut out = Metadata::new();
            for (key, child) in dict.keys.iter().zip(&dict.spaces) {
                if let Some(column) = map.remove(key) {
                    out.insert(key.clone(), lane0_unchecked(child, column)?);
                }
            }
            Ok(Value::Dict(out))
        }
        (Some(WireKind::Tuple(tuple)), Value::Tuple(items) | Value::List(items)) => {
            if items.len() != tuple.spaces.len() {
                return Err(batch_error("a Tuple observation has the wrong arity"));
            }
            Ok(Value::Tuple(
                tuple
                    .spaces
                    .iter()
                    .zip(items)
                    .map(|(child, item)| lane0_unchecked(child, item))
                    .collect::<Result<_>>()?,
            ))
        }
        (_, value) => {
            let first = first_item(value)?;
            // A Discrete lane is an int, as NumPy's int64 scalar encodes.
            match (&spec.spec, first) {
                (Some(WireKind::Discrete(_)), Value::Tensor(tensor)) if tensor.size() == 1 => {
                    Ok(Value::Int(tensor_ints(&tensor)?[0]))
                }
                (_, first) => Ok(first),
            }
        }
    }
}

/// `value[0]`: a List's or Tuple's first item, a Tensor's first slice.
pub(crate) fn first_item(value: Value) -> Result<Value> {
    match value {
        Value::List(items) | Value::Tuple(items) => items
            .into_iter()
            .next()
            .ok_or_else(|| batch_error("an empty batch has no lane 0")),
        Value::Tensor(tensor) => tensor
            .unstack()?
            .into_iter()
            .next()
            .map(Value::Tensor)
            .ok_or_else(|| batch_error("an empty batch has no lane 0")),
        other => Err(batch_error(format!(
            "a batched value cannot be a {}",
            other.kind().python_name()
        ))),
    }
}

/// A one-lane batch of `action` (Python's `_batch1`): Dict and Tuple per
/// child; a tensor gains a leading axis of 1; anything else, a Discrete int
/// included, is a one-item List.
pub(crate) fn batch1(spec: &WireSpec, action: Value) -> Result<Value> {
    crate::value::space_codec::value_to_space(spec, &action)?;
    batch1_unchecked(spec, action)
}

fn batch1_unchecked(spec: &WireSpec, action: Value) -> Result<Value> {
    match &spec.spec {
        Some(WireKind::Dict(dict)) => {
            let Value::Dict(mut map) = action else {
                return Err(batch_error(format!(
                    "'{}' object is not subscriptable",
                    action.kind().python_name()
                )));
            };
            let mut out = Metadata::new();
            for (key, child) in dict.keys.iter().zip(&dict.spaces) {
                if let Some(item) = map.remove(key) {
                    out.insert(key.clone(), batch1_unchecked(child, item)?);
                }
            }
            Ok(Value::Dict(out))
        }
        Some(WireKind::Tuple(tuple)) => {
            let (Value::Tuple(items) | Value::List(items)) = action else {
                return Err(batch_error(format!(
                    "'{}' object is not iterable",
                    action.kind().python_name()
                )));
            };
            Ok(Value::Tuple(
                tuple
                    .spaces
                    .iter()
                    .zip(items)
                    .map(|(child, item)| batch1_unchecked(child, item))
                    .collect::<Result<_>>()?,
            ))
        }
        _ => match action {
            Value::Tensor(tensor) => {
                let mut shape: Vec<i64> = vec![1];
                shape.extend(tensor.shape().iter().map(|&dim| dim as i64));
                Ok(Value::Tensor(tensor.copy().reshape(&shape)?))
            }
            other => Ok(Value::List(vec![other])),
        },
    }
}

/// Lane 0 of a vector info (gymnasium's `key` array plus `_key` mask).
pub(crate) fn info0(info: &Metadata) -> Metadata {
    let mut lane = Metadata::new();
    for (key, value) in info {
        if key.starts_with('_') {
            continue;
        }
        if let Some(mask) = info.get(&format!("_{key}"))
            && !mask_lane0(mask)
        {
            continue;
        }
        let value = match value {
            Value::Dict(map) => Value::Dict(info0(map)),
            Value::List(items) | Value::Tuple(items) if items.len() == 1 => items[0].clone(),
            Value::Tensor(tensor) if tensor.shape().first() == Some(&1) => {
                first_item(value.clone()).unwrap_or_else(|_| value.clone())
            }
            other => other.clone(),
        };
        lane.insert(key.to_string(), value);
    }
    lane
}

/// Python's `bool(mask[0])` for a mask value.
fn mask_lane0(mask: &Value) -> bool {
    match first_item(mask.clone()) {
        Ok(Value::Bool(flag)) => flag,
        Ok(Value::Int(int)) => int != 0,
        Ok(Value::Float(float)) => float != 0.0,
        Ok(Value::Tensor(tensor)) => tensor.tobytes().iter().any(|&byte| byte != 0),
        Ok(Value::None) => false,
        _ => true,
    }
}

/// PNG-encode a render frame by Python's lenient rule (ENV-5): any numeric
/// image value cast to uint8; `[H, W, 3]`, `[H, W, 4]`, `[3, H, W]`,
/// `[4, H, W]` and `[H, W]` encode, any other shape gives no frame.
pub(crate) fn encode_frame(frame: &Value) -> Result<Option<Vec<u8>>> {
    let Some((shape, bytes)) = frame_u8(frame)? else {
        return Ok(None);
    };
    let dim = |d: usize| u32::try_from(d).unwrap_or(0);
    let (width, height, color, data) = match shape.as_slice() {
        [h, w, 3] => (dim(*w), dim(*h), image::ExtendedColorType::Rgb8, bytes),
        [h, w, 4] => (dim(*w), dim(*h), image::ExtendedColorType::Rgba8, bytes),
        [3, h, w] => (
            dim(*w),
            dim(*h),
            image::ExtendedColorType::Rgb8,
            chw_to_hwc(&bytes, 3, *h, *w),
        ),
        [4, h, w] => (
            dim(*w),
            dim(*h),
            image::ExtendedColorType::Rgba8,
            chw_to_hwc(&bytes, 4, *h, *w),
        ),
        [h, w] => (dim(*w), dim(*h), image::ExtendedColorType::L8, bytes),
        _ => return Ok(None),
    };
    let mut encoded = Vec::new();
    PngEncoder::new(&mut encoded)
        .write_image(&data, width, height, color)
        .map_err(|err| Error::Internal(format!("encode render frame: {err}")))?;
    Ok(Some(encoded))
}

fn chw_to_hwc(bytes: &[u8], channels: usize, height: usize, width: usize) -> Vec<u8> {
    let plane = height * width;
    let mut out = Vec::with_capacity(bytes.len());
    for pixel in 0..plane {
        for channel in 0..channels {
            out.push(bytes.get(channel * plane + pixel).copied().unwrap_or(0));
        }
    }
    out
}

/// A numeric frame as `(shape, uint8 elements)`, as NumPy's `astype(uint8)`.
fn frame_u8(frame: &Value) -> Result<Option<(Vec<usize>, Vec<u8>)>> {
    match frame {
        Value::Tensor(tensor) => {
            let core = tensor.core();
            let bytes = core.to_contiguous_bytes();
            let data = if core.dtype() == CoreDType::Uint8 {
                bytes.into_owned()
            } else {
                decode_scalars(&bytes, core.dtype())
                    .map_err(|err| Error::Type(format!("render frame: {err}")))?
                    .into_iter()
                    .map(scalar_u8)
                    .collect()
            };
            Ok(Some((tensor.shape().to_vec(), data)))
        }
        Value::List(_) | Value::Tuple(_) => {
            let mut shape = Vec::new();
            let mut data = Vec::new();
            nested_u8(frame, 0, &mut shape, &mut data)?;
            Ok(Some((shape, data)))
        }
        _ => Ok(None),
    }
}

fn scalar_u8(scalar: Scalar) -> u8 {
    match scalar {
        Scalar::Bool(flag) => u8::from(flag),
        Scalar::Int(int) => int as u8,
        Scalar::Float(float) => (float as i64) as u8,
    }
}

fn nested_u8(value: &Value, depth: usize, shape: &mut Vec<usize>, out: &mut Vec<u8>) -> Result<()> {
    let ragged = || Error::Type("render frame is a ragged nested list".to_string());
    match value {
        Value::List(items) | Value::Tuple(items) => {
            match shape.get(depth) {
                Some(&len) if len != items.len() => return Err(ragged()),
                Some(_) => {}
                None if depth == shape.len() => shape.push(items.len()),
                None => return Err(ragged()),
            }
            for item in items {
                nested_u8(item, depth + 1, shape, out)?;
            }
            Ok(())
        }
        Value::Int(int) => {
            out.push(*int as u8);
            Ok(())
        }
        Value::Float(float) => {
            out.push((*float as i64) as u8);
            Ok(())
        }
        Value::Bool(flag) => {
            out.push(u8::from(*flag));
            Ok(())
        }
        other => Err(Error::Type(format!(
            "render frame holds a {}, not a number",
            other.kind().python_name()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::{BoxOptions, Discrete, DiscreteOptions, Space};

    fn wire(space: impl Into<Space>) -> WireSpec {
        space.into().spec().wire().clone()
    }

    #[test]
    fn missing_batched_dict_keys_use_native_conformance_text() {
        use rlmesh_spaces::spaces::{DictSpaceBuilder, DiscreteBuilder};
        let space = DictSpaceBuilder::new()
            .insert(
                "nested",
                DictSpaceBuilder::new()
                    .insert("choice", DiscreteBuilder::new(3).build().unwrap())
                    .build()
                    .unwrap(),
            )
            .build()
            .unwrap();
        let value = Value::Dict(
            [("nested", Value::Dict(Metadata::new()))]
                .into_iter()
                .collect(),
        );
        for err in [
            unstack_lanes(&space, value.clone(), 2).unwrap_err(),
            lane0(&space, value.clone()).unwrap_err(),
            batch1(&space, value).unwrap_err(),
        ] {
            assert_eq!(err.class(), crate::ErrorClass::ValueError);
            assert_eq!(err.to_string(), "$.nested: missing key 'choice'");
        }
    }

    #[test]
    fn discrete_lanes_stack_into_an_int64_tensor_and_back() {
        let spec = wire(Discrete::new(4, DiscreteOptions::default()).unwrap());
        let batch = stack_lanes(&spec, vec![Value::Int(1), Value::Int(3)]).unwrap();
        let Value::Tensor(tensor) = &batch else {
            panic!("{batch:?}")
        };
        assert_eq!(tensor.shape(), &[2]);
        assert_eq!(tensor.dtype(), DType::Int64);
        let lanes = unstack_lanes(&spec, batch, 2).unwrap();
        assert_eq!(lanes, vec![Value::Int(1), Value::Int(3)]);
        assert_eq!(
            batch1(&spec, Value::Int(2)).unwrap(),
            Value::List(vec![Value::Int(2)])
        );
    }

    #[test]
    fn box_lanes_stack_and_lane0_slices() {
        let spec = wire(crate::spaces::Box::new(-1.0, 1.0, [2], BoxOptions::default()).unwrap());
        let lane = Tensor::from_vec(
            [0.5f32, -0.5]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect(),
            &[2],
            DType::Float32,
        )
        .unwrap();
        let batch = stack_lanes(&spec, vec![Value::Tensor(lane.clone()); 3]).unwrap();
        assert_eq!(batch.as_tensor().unwrap().shape(), &[3, 2]);
        assert_eq!(lane0(&spec, batch).unwrap(), Value::Tensor(lane.clone()));
        let one = batch1(&spec, Value::Tensor(lane)).unwrap();
        assert_eq!(one.as_tensor().unwrap().shape(), &[1, 2]);
    }

    #[test]
    fn info0_drops_masked_and_private_keys() {
        let mut info = Metadata::new();
        info.insert("a", Value::List(vec![Value::Int(5)]));
        info.insert("_a", Value::List(vec![Value::Bool(true)]));
        info.insert("b", Value::List(vec![Value::Int(6)]));
        info.insert("_b", Value::List(vec![Value::Bool(false)]));
        info.insert("text", "kept");
        let lane = info0(&info);
        assert_eq!(lane.get("a"), Some(&Value::Int(5)));
        assert!(!lane.contains_key("b"));
        assert!(!lane.contains_key("_a"));
        assert_eq!(lane.get("text"), Some(&Value::Text("kept".into())));
    }

    #[test]
    fn frames_encode_by_python_s_lenient_rule() {
        let frame = Tensor::from_vec(vec![0u8; 2 * 3 * 3], &[2, 3, 3], DType::Uint8).unwrap();
        assert!(encode_frame(&Value::Tensor(frame)).unwrap().is_some());
        let chw = Tensor::from_vec(
            [1.0f64; 3 * 2 * 2]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect(),
            &[3, 2, 2],
            DType::Float64,
        )
        .unwrap();
        assert!(encode_frame(&Value::Tensor(chw)).unwrap().is_some());
        let odd = Tensor::from_vec(vec![0u8; 5], &[5], DType::Uint8).unwrap();
        assert!(encode_frame(&Value::Tensor(odd)).unwrap().is_none());
        let nested = Value::List(vec![Value::List(vec![Value::Int(1), Value::Int(300)])]);
        assert!(encode_frame(&nested).unwrap().is_some());
    }
}
