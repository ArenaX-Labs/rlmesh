//! Thin-bytes leaf codec: the replacement for the recursive `SpaceValueNode`
//! tree. A value's bytes are a pure function of its route `SpaceSpec` —
//! [`flatten_leaves`]/[`leaf_specs`]/[`assemble_value`] (in `rlmesh-spaces`)
//! give the canonical pre-order; this module turns each fundamental leaf into
//! raw little-endian bytes and back. See `value-transport-redesign.md` §3–§5.
//!
//! Single value: one `Bytes` per leaf, each the raw leaf bytes (Text = raw
//! UTF-8, self-framed by the element length). Batch: one `Bytes` per leaf
//! holding the row-major `(N, *shape)` slab — for fixed-stride leaves that is
//! just the per-lane bytes concatenated (lane-contiguous); Text carries an
//! explicit `[u64 count][u64 byte_len…]` header. **N is an authoritative input
//! on decode, never recovered by division.**

use prost::bytes::Bytes;
use rlmesh_spaces as native;
use rlmesh_spaces::{SpaceKind, SpaceSpec, SpaceValue};

use crate::error::ProtocolError;

use super::codec::tensor_wire_bytes;
use super::scalars::{decode_int_sequence, encode_int_sequence};

// ---- single value -------------------------------------------------------

/// Encode a single typed value into one `Bytes` per canonical leaf.
pub fn encode_leaves(value: &SpaceValue, spec: &SpaceSpec) -> Result<Vec<Bytes>, ProtocolError> {
    let values = native::flatten_leaves(spec, value).map_err(structural_encode)?;
    native::leaf_specs(spec)
        .into_iter()
        .zip(values)
        .map(|(leaf_spec, leaf_value)| encode_leaf(leaf_value, leaf_spec))
        .collect()
}

/// Rebuild a single typed value from its canonical leaf bytes.
pub fn decode_leaves(leaves: &[Bytes], spec: &SpaceSpec) -> Result<SpaceValue, ProtocolError> {
    let specs = native::leaf_specs(spec);
    if leaves.len() != specs.len() {
        return Err(ProtocolError::LengthMismatch(format!(
            "got {} leaves, spec expects {}",
            leaves.len(),
            specs.len()
        )));
    }
    let values = leaves
        .iter()
        .zip(specs)
        .map(|(bytes, leaf_spec)| decode_leaf(bytes, leaf_spec))
        .collect::<Result<Vec<_>, _>>()?;
    native::assemble_value(spec, values).map_err(structural_decode)
}

// ---- batch (row-major slab) ---------------------------------------------

/// Encode `values.len()` lanes into one slab `Bytes` per canonical leaf.
pub fn encode_leaf_slab(
    values: &[SpaceValue],
    spec: &SpaceSpec,
) -> Result<Vec<Bytes>, ProtocolError> {
    let specs = native::leaf_specs(spec);
    // Per lane, canonical leaf values; then stack column-wise per leaf.
    let lanes = values
        .iter()
        .map(|value| native::flatten_leaves(spec, value))
        .collect::<Result<Vec<_>, _>>()
        .map_err(structural_encode)?;

    specs
        .iter()
        .enumerate()
        .map(|(p, &leaf_spec)| {
            let column = lanes.iter().map(|lane| lane[p]);
            if matches!(leaf_spec.spec.as_ref(), Some(SpaceKind::Text(_))) {
                encode_text_slab(column, lanes.len())
            } else {
                // Each lane MUST contribute exactly `stride` bytes. A fixed-stride
                // slab carries no per-lane framing, so a wrong-arity/shape lane
                // would otherwise be silently re-sliced across lane boundaries on
                // decode (the aggregate length can still match when sizes
                // compensate). Reject it here rather than corrupt the batch.
                let stride = leaf_stride(leaf_spec)?;
                let mut buf = Vec::with_capacity(stride.saturating_mul(lanes.len()));
                for (lane, leaf_value) in column.enumerate() {
                    let leaf = encode_leaf(leaf_value, leaf_spec)?;
                    if leaf.len() != stride {
                        return Err(ProtocolError::LengthMismatch(format!(
                            "batch lane {lane} leaf: got {} bytes, expected stride {stride}",
                            leaf.len()
                        )));
                    }
                    buf.extend_from_slice(&leaf);
                }
                Ok(Bytes::from(buf))
            }
        })
        .collect()
}

/// Split each leaf slab into `n` lanes and reassemble `n` typed values.
pub fn decode_leaf_slab(
    leaves: &[Bytes],
    spec: &SpaceSpec,
    n: usize,
) -> Result<Vec<SpaceValue>, ProtocolError> {
    let specs = native::leaf_specs(spec);
    if leaves.len() != specs.len() {
        return Err(ProtocolError::LengthMismatch(format!(
            "got {} leaf slabs, spec expects {}",
            leaves.len(),
            specs.len()
        )));
    }
    // per_lane[lane] collects this lane's leaf value for each leaf position.
    let mut per_lane: Vec<Vec<SpaceValue>> =
        (0..n).map(|_| Vec::with_capacity(specs.len())).collect();
    for (slab, &leaf_spec) in leaves.iter().zip(&specs) {
        for (lane, value) in decode_leaf_column(slab, leaf_spec, n)?
            .into_iter()
            .enumerate()
        {
            per_lane[lane].push(value);
        }
    }
    per_lane
        .into_iter()
        .map(|values| native::assemble_value(spec, values).map_err(structural_decode))
        .collect()
}

// ---- per-leaf -----------------------------------------------------------

fn encode_leaf(value: &SpaceValue, spec: &SpaceSpec) -> Result<Bytes, ProtocolError> {
    match (spec.spec.as_ref(), value) {
        (Some(SpaceKind::Box(_)), SpaceValue::Box(tensor)) => {
            // The wire carries only raw bytes; shape/dtype are recovered from the
            // spec on decode. A tensor that matches the spec by byte count but not
            // by shape/dtype would silently reinterpret on decode (the same-byte
            // class the spec-directed codec cannot catch), so pin both here.
            if tensor.dtype() != spec.dtype || tensor.shape() != spec.shape.as_slice() {
                return Err(ProtocolError::EncodeError(format!(
                    "box leaf {:?}{:?} does not match spec {:?}{:?}",
                    tensor.dtype(),
                    tensor.shape(),
                    spec.dtype,
                    spec.shape
                )));
            }
            Ok(tensor_wire_bytes(tensor))
        }
        (Some(SpaceKind::Discrete(_)), SpaceValue::Discrete(index)) => {
            // THE FIX: encode at the declared dtype width, not a fixed 8 bytes.
            range_check(*index, spec.dtype)?;
            Ok(Bytes::from(encode_int_sequence(&[*index], spec.dtype)?))
        }
        (Some(SpaceKind::MultiBinary(_)), SpaceValue::MultiBinary(flags)) => Ok(Bytes::from(
            flags.iter().map(|f| u8::from(*f)).collect::<Vec<u8>>(),
        )),
        (Some(SpaceKind::MultiDiscrete(_)), SpaceValue::MultiDiscrete(values)) => {
            for value in values {
                range_check(*value, spec.dtype)?;
            }
            Ok(Bytes::from(encode_int_sequence(values, spec.dtype)?))
        }
        (Some(SpaceKind::Text(_)), SpaceValue::Text(text)) => {
            Ok(Bytes::from(text.clone().into_bytes()))
        }
        _ => Err(ProtocolError::EncodeError(format!(
            "value kind did not match leaf space {:?}",
            spec.space_type()
        ))),
    }
}

fn decode_leaf(bytes: &[u8], spec: &SpaceSpec) -> Result<SpaceValue, ProtocolError> {
    match spec.spec.as_ref() {
        Some(SpaceKind::Box(_)) => Ok(SpaceValue::Box(
            // from_slice enforces byte_count == numel * dtype_size.
            native::Tensor::from_slice(bytes, &spec.shape, spec.dtype)
                .map_err(|err| ProtocolError::DecodeError(format!("invalid box payload: {err}")))?,
        )),
        Some(SpaceKind::Discrete(_)) => {
            exact_len(bytes.len(), native::dtype_size(spec.dtype), "discrete")?;
            let value = *decode_int_sequence(bytes, spec.dtype)?
                .first()
                .ok_or_else(|| ProtocolError::LengthMismatch("empty discrete leaf".into()))?;
            range_check(value, spec.dtype)?;
            Ok(SpaceValue::Discrete(value))
        }
        Some(SpaceKind::MultiBinary(_)) => {
            let numel = numel(spec)?;
            exact_len(bytes.len(), numel, "multibinary")?; // one byte per element
            Ok(SpaceValue::MultiBinary(
                bytes.iter().map(|b| *b != 0).collect(),
            ))
        }
        Some(SpaceKind::MultiDiscrete(_)) => {
            exact_len(
                bytes.len(),
                numel(spec)? * native::dtype_size(spec.dtype),
                "multidiscrete",
            )?;
            Ok(SpaceValue::MultiDiscrete(decode_int_sequence(
                bytes, spec.dtype,
            )?))
        }
        Some(SpaceKind::Text(_)) => Ok(SpaceValue::Text(decode_utf8(bytes)?)),
        _ => Err(ProtocolError::DecodeError(format!(
            "not a fundamental leaf: {:?}",
            spec.space_type()
        ))),
    }
}

/// Split one leaf's slab into its `n` per-lane values.
fn decode_leaf_column(
    slab: &[u8],
    spec: &SpaceSpec,
    n: usize,
) -> Result<Vec<SpaceValue>, ProtocolError> {
    if matches!(spec.spec.as_ref(), Some(SpaceKind::Text(_))) {
        return decode_text_slab(slab, n);
    }
    let stride = leaf_stride(spec)?;
    let want = stride
        .checked_mul(n)
        .ok_or_else(|| ProtocolError::DecodeError("slab size overflow".into()))?;
    if slab.len() != want {
        return Err(ProtocolError::LengthMismatch(format!(
            "slab: got {} bytes, expected {n} lanes * {stride}",
            slab.len()
        )));
    }
    if stride == 0 {
        // Zero-numel leaf: every lane is an empty leaf; N comes from the carrier.
        return (0..n).map(|_| decode_leaf(&[], spec)).collect();
    }
    slab.chunks_exact(stride)
        .map(|chunk| decode_leaf(chunk, spec))
        .collect()
}

// ---- Text batch framing: [u64 count][u64 byte_len…] + concatenated UTF-8 --

fn encode_text_slab<'a>(
    column: impl Iterator<Item = &'a SpaceValue>,
    n: usize,
) -> Result<Bytes, ProtocolError> {
    let mut bodies = Vec::with_capacity(n);
    for value in column {
        match value {
            SpaceValue::Text(text) => bodies.push(text.as_str()),
            _ => {
                return Err(ProtocolError::EncodeError(
                    "expected Text leaf in batch".into(),
                ));
            }
        }
    }
    let mut buf = Vec::new();
    buf.extend_from_slice(&(n as u64).to_le_bytes());
    for text in &bodies {
        buf.extend_from_slice(&(text.len() as u64).to_le_bytes());
    }
    for text in &bodies {
        buf.extend_from_slice(text.as_bytes());
    }
    Ok(Bytes::from(buf))
}

fn decode_text_slab(slab: &[u8], n: usize) -> Result<Vec<SpaceValue>, ProtocolError> {
    let mut cur = 0;
    let count = usize_from_u64(read_u64(slab, &mut cur)?)?;
    if count != n {
        return Err(ProtocolError::LengthMismatch(format!(
            "text slab count {count} != N {n}"
        )));
    }
    let lens = (0..n)
        .map(|_| read_u64(slab, &mut cur).and_then(usize_from_u64))
        .collect::<Result<Vec<_>, _>>()?;
    let mut out = Vec::with_capacity(n);
    for len in lens {
        let end = cur
            .checked_add(len)
            .ok_or_else(|| ProtocolError::DecodeError("text length overflow".into()))?;
        let bytes = slab
            .get(cur..end)
            .ok_or_else(|| ProtocolError::LengthMismatch("text slab body truncated".into()))?;
        out.push(SpaceValue::Text(decode_utf8(bytes)?));
        cur = end;
    }
    if cur != slab.len() {
        return Err(ProtocolError::LengthMismatch(
            "trailing bytes in text slab".into(),
        ));
    }
    Ok(out)
}

/// Narrow a wire u64 length to `usize`, erroring (not truncating) when it does
/// not fit — a remote peer can declare a length above `usize::MAX` on a 32-bit
/// host, where a bare `as usize` would silently read the wrong window.
fn usize_from_u64(v: u64) -> Result<usize, ProtocolError> {
    usize::try_from(v)
        .map_err(|_| ProtocolError::LengthMismatch("text slab length exceeds usize".into()))
}

fn read_u64(buf: &[u8], cur: &mut usize) -> Result<u64, ProtocolError> {
    let end = *cur + 8;
    let bytes = buf
        .get(*cur..end)
        .ok_or_else(|| ProtocolError::LengthMismatch("text slab header truncated".into()))?;
    *cur = end;
    Ok(u64::from_le_bytes(bytes.try_into().expect("8-byte slice")))
}

// ---- small helpers ------------------------------------------------------

fn range_check(value: i64, dtype: native::DType) -> Result<(), ProtocolError> {
    native::check_int_in_dtype_range(value, dtype)
        .map_err(|err| ProtocolError::OutOfRange(err.to_string()))
}

fn exact_len(got: usize, want: usize, what: &str) -> Result<(), ProtocolError> {
    if got != want {
        return Err(ProtocolError::LengthMismatch(format!(
            "{what} leaf: got {got} bytes, expected {want}"
        )));
    }
    Ok(())
}

/// Per-lane byte length of a fixed-stride (non-Text) leaf: `numel * per_elem`.
/// The single authority for both slab encode (length check) and slab decode
/// (chunk size).
fn leaf_stride(spec: &SpaceSpec) -> Result<usize, ProtocolError> {
    let per_elem = if matches!(spec.spec.as_ref(), Some(SpaceKind::MultiBinary(_))) {
        1
    } else {
        native::dtype_size(spec.dtype)
    };
    numel(spec)?
        .checked_mul(per_elem)
        .ok_or_else(|| ProtocolError::DecodeError("leaf stride overflow".into()))
}

/// Element count for a fixed-stride leaf (Text/composite have none).
fn numel(spec: &SpaceSpec) -> Result<usize, ProtocolError> {
    let count = match spec.spec.as_ref() {
        Some(SpaceKind::Box(_) | SpaceKind::MultiBinary(_)) => shape_numel(&spec.shape),
        Some(SpaceKind::Discrete(_)) => Some(1),
        Some(SpaceKind::MultiDiscrete(m)) => Some(m.nvec.len()),
        _ => None,
    };
    count.ok_or_else(|| {
        ProtocolError::DecodeError(format!("no fixed numel for {:?}", spec.space_type()))
    })
}

fn shape_numel(shape: &[i64]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, dim| {
        usize::try_from(*dim).ok().and_then(|d| acc.checked_mul(d))
    })
}

fn decode_utf8(bytes: &[u8]) -> Result<String, ProtocolError> {
    Ok(std::str::from_utf8(bytes)
        .map_err(|err| {
            ProtocolError::DecodeError(format!("text payload is not valid UTF-8: {err}"))
        })?
        .to_owned())
}

fn structural_encode(err: native::errors::SpaceError) -> ProtocolError {
    ProtocolError::EncodeError(err.to_string())
}

fn structural_decode(err: native::errors::SpaceError) -> ProtocolError {
    ProtocolError::DecodeError(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlmesh_spaces::{
        DType, DictSpec, DiscreteSpec, MultiDiscreteSpec, SpaceKind, TextSpec, TupleSpec,
    };
    use std::collections::BTreeMap;

    fn discrete(dtype: DType) -> SpaceSpec {
        SpaceSpec {
            dtype,
            spec: Some(SpaceKind::Discrete(DiscreteSpec { n: 1000, start: 0 })),
            ..Default::default()
        }
    }
    fn text() -> SpaceSpec {
        SpaceSpec {
            spec: Some(SpaceKind::Text(TextSpec::default())),
            ..Default::default()
        }
    }
    fn multidiscrete(dtype: DType) -> SpaceSpec {
        SpaceSpec {
            shape: vec![2],
            dtype,
            spec: Some(SpaceKind::MultiDiscrete(MultiDiscreteSpec {
                nvec: vec![4, 4],
            })),
        }
    }

    // Dict{ z: Discrete(int32), a: Tuple(MultiDiscrete(uint8), Text) } — keys
    // deliberately non-sorted to lock declared-order traversal end to end.
    fn nested() -> SpaceSpec {
        SpaceSpec {
            spec: Some(SpaceKind::Dict(DictSpec {
                keys: vec!["z".into(), "a".into()],
                spaces: vec![
                    discrete(DType::Int32),
                    SpaceSpec {
                        spec: Some(SpaceKind::Tuple(TupleSpec {
                            spaces: vec![multidiscrete(DType::Uint8), text()],
                        })),
                        ..Default::default()
                    },
                ],
            })),
            ..Default::default()
        }
    }

    fn nested_value(d: i64, m: (i64, i64), s: &str) -> SpaceValue {
        let mut map = BTreeMap::new();
        map.insert("z".to_string(), SpaceValue::Discrete(d));
        map.insert(
            "a".to_string(),
            SpaceValue::Tuple(vec![
                SpaceValue::MultiDiscrete(vec![m.0, m.1]),
                SpaceValue::Text(s.into()),
            ]),
        );
        SpaceValue::Dict(map)
    }

    #[test]
    fn single_roundtrip_declared_order_and_dtype_width() {
        let spec = nested();
        let value = nested_value(300, (1, 2), "hi");
        let leaves = encode_leaves(&value, &spec).unwrap();
        // z=Discrete(int32) -> 4 bytes (the dtype-width fix, not a fixed 8).
        assert_eq!(
            leaves[0].len(),
            4,
            "Discrete encodes at declared dtype width"
        );
        assert_eq!(decode_leaves(&leaves, &spec).unwrap(), value);
    }

    #[test]
    fn batch_slab_roundtrip_and_n_is_authoritative() {
        let spec = nested();
        let lanes = vec![
            nested_value(1, (0, 1), "a"),
            nested_value(2, (3, 3), "bb"),
            nested_value(3, (1, 1), ""),
        ];
        let slab = encode_leaf_slab(&lanes, &spec).unwrap();
        assert_eq!(decode_leaf_slab(&slab, &spec, 3).unwrap(), lanes);
        // Wrong N must hard-error, never silently re-slice a fixed-stride slab.
        assert!(decode_leaf_slab(&slab, &spec, 2).is_err());
    }

    #[test]
    fn batch_encode_rejects_wrong_arity_lane() {
        // A lane whose arity doesn't match the spec stride must hard-error at
        // encode, never silently concatenate into a slab the decoder re-slices
        // across lane boundaries (the aggregate length can still match when two
        // lanes' wrong sizes compensate, e.g. 3 + 1 == 2 + 2).
        let spec = multidiscrete(DType::Uint8); // nvec=[4,4] -> stride 2
        let lanes = vec![
            SpaceValue::MultiDiscrete(vec![1, 2, 3]), // 3 bytes, wrong arity
            SpaceValue::MultiDiscrete(vec![0]),       // 1 byte
        ];
        assert!(matches!(
            encode_leaf_slab(&lanes, &spec),
            Err(ProtocolError::LengthMismatch(_))
        ));
    }

    #[test]
    fn encode_rejects_out_of_dtype_range() {
        // MultiDiscrete(uint8) with value 300 used to silently wrap 300 -> 44.
        let spec = multidiscrete(DType::Uint8);
        let value = SpaceValue::MultiDiscrete(vec![300, 0]);
        assert!(matches!(
            encode_leaves(&value, &spec),
            Err(ProtocolError::OutOfRange(_))
        ));
    }

    #[test]
    fn encode_rejects_box_shape_mismatch() {
        // Same byte count (2 u8 elements) but a different shape: [1,2] tensor vs
        // [2,1] spec. The spec-directed decode reads back at the spec shape and
        // cannot catch this, so the encode must reject it rather than silently
        // reinterpret the layout on the far side.
        let spec = native::spaces::BoxSpaceBuilder::scalar(0.0, 255.0, vec![2, 1])
            .dtype(DType::Uint8)
            .build()
            .unwrap();
        let value = SpaceValue::Box(
            native::Tensor::from_vec(vec![0u8, 0u8], vec![1, 2], DType::Uint8).unwrap(),
        );
        assert!(matches!(
            encode_leaves(&value, &spec),
            Err(ProtocolError::EncodeError(_))
        ));
    }

    #[test]
    fn decode_rejects_wrong_leaf_length() {
        let spec = discrete(DType::Int32); // expects exactly 4 bytes
        assert!(matches!(
            decode_leaves(&[Bytes::from_static(&[0u8; 3])], &spec),
            Err(ProtocolError::LengthMismatch(_))
        ));
        // MultiBinary must reject extra/truncated bytes (old decode ignored length).
        let mb = SpaceSpec {
            shape: vec![3],
            spec: Some(SpaceKind::MultiBinary(Default::default())),
            ..Default::default()
        };
        assert!(decode_leaves(&[Bytes::from_static(&[1u8, 0, 1, 1])], &mb).is_err());

        // MultiDiscrete: 2 elems x int16 = exactly 4 bytes; extra and truncated reject.
        let md = multidiscrete(DType::Int16);
        assert!(matches!(
            decode_leaves(&[Bytes::from_static(&[0u8; 6])], &md),
            Err(ProtocolError::LengthMismatch(_))
        ));
        assert!(decode_leaves(&[Bytes::from_static(&[0u8; 2])], &md).is_err());
    }
}
