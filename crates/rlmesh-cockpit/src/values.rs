//! A sample's value as the cockpit shows it, decoded generically from the
//! stream's declared layout: numbers (named by the part's `names` when they
//! fit), a summary for large tensors, text, or a camera thumbnail. Nothing
//! here knows what a robot is (§ "Scope rule").

use std::sync::Arc;

use half::f16;
use rlmesh_session::media::decode_payload;
use rlmesh_wire::{DType, Encoding, StreamSpec};

/// The most elements shown one by one; a larger value is summarised.
pub const MAX_ELEMENTS: usize = 32;
/// The widest thumbnail, in pixels.
pub const THUMB_WIDTH: usize = 192;

/// A decoded value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// Every element, labelled.
    Numbers(Vec<(String, f64)>),
    /// Too many elements to list.
    Summary {
        /// Elements.
        count: usize,
        /// Smallest finite element.
        min: f64,
        /// Largest finite element.
        max: f64,
        /// Mean of the finite elements.
        mean: f64,
        /// NaN or infinite elements.
        nonfinite: usize,
    },
    /// UTF-8 text.
    Text(String),
    /// A camera frame, downsampled.
    Image(Arc<Thumbnail>),
    /// An applied tick that dispatched nothing (`present = false`).
    Absent,
    /// The payload does not decode against the declaration.
    Undecodable(String),
}

/// A downsampled RGB frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Thumbnail {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// `height × width × 3` bytes.
    pub rgb: Vec<u8>,
    /// The frame's source dimensions, `(height, width)`.
    pub source: (u32, u32),
    /// The sample's seq: the UI uploads a texture only when it changes.
    pub seq: u64,
}

/// Whether a stream carries camera frames.
pub fn is_image(spec: &StreamSpec) -> bool {
    matches!(spec.encoding, Encoding::Jpeg { .. })
        || (spec.fields.len() == 1 && spec.fields[0].rgb8_dims().is_some())
}

/// Decodes a camera frame into a thumbnail.
pub fn thumbnail(spec: &StreamSpec, payload: &[u8], seq: u64) -> Value {
    let Some((h, w)) = spec.fields.first().and_then(|f| f.rgb8_dims()) else {
        return Value::Undecodable("not an rgb8 image".into());
    };
    let rgb = match decode_payload(spec, payload) {
        Ok(rgb) => rgb,
        Err(err) => return Value::Undecodable(err.to_string()),
    };
    let (sh, sw) = (h as usize, w as usize);
    let step = sw.div_ceil(THUMB_WIDTH).max(1);
    let (width, height) = (sw.div_ceil(step), sh.div_ceil(step));
    let mut out = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let at = ((y * step).min(sh - 1) * sw + (x * step).min(sw - 1)) * 3;
            out.extend_from_slice(&rgb[at..at + 3]);
        }
    }
    Value::Image(Arc::new(Thumbnail {
        width,
        height,
        rgb: out,
        source: (h, w),
        seq,
    }))
}

/// Decodes a raw packed value (§10.6): every field, element by element.
/// `names` label the elements of a single-field value when they match its
/// length (joint names).
pub fn decode(spec: &StreamSpec, names: &[String], payload: &[u8]) -> Value {
    if spec.fields.len() == 1 && spec.fields[0].dtype == DType::Utf8 {
        return match std::str::from_utf8(payload) {
            Ok(text) => Value::Text(text.to_string()),
            Err(err) => Value::Undecodable(err.to_string()),
        };
    }
    let mut values = Vec::new();
    for (i, field) in spec.fields.iter().enumerate() {
        let Some(range) = spec.layout.field(i) else {
            return Value::Undecodable(format!("field {} has no place in the layout", field.name));
        };
        let Some(bytes) = payload.get(range) else {
            return Value::Undecodable(format!(
                "{} bytes, field {} needs more",
                payload.len(),
                field.name
            ));
        };
        let size = usize::try_from(field.dtype.size()).unwrap_or(0).max(1);
        for (j, chunk) in bytes.chunks_exact(size).enumerate() {
            let label = if spec.fields.len() == 1 && names.len() * size == bytes.len() {
                names[j].clone()
            } else if bytes.len() == size {
                field.name.clone()
            } else {
                format!("{}[{j}]", field.name)
            };
            values.push((label, element(field.dtype, chunk)));
        }
    }
    if values.len() <= MAX_ELEMENTS {
        return Value::Numbers(values);
    }
    let finite: Vec<f64> = values
        .iter()
        .map(|(_, v)| *v)
        .filter(|v| v.is_finite())
        .collect();
    let (min, max) = finite
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    Value::Summary {
        count: values.len(),
        min,
        max,
        mean: finite.iter().sum::<f64>() / finite.len().max(1) as f64,
        nonfinite: values.len() - finite.len(),
    }
}

/// One little-endian element as `f64`.
fn element(dtype: DType, b: &[u8]) -> f64 {
    let a = |n: usize| -> [u8; 8] {
        let mut out = [0; 8];
        out[..n].copy_from_slice(&b[..n]);
        out
    };
    match dtype {
        DType::Bool | DType::U8 => f64::from(b[0]),
        DType::I8 => f64::from(b[0] as i8),
        DType::U16 => f64::from(u16::from_le_bytes([b[0], b[1]])),
        DType::I16 => f64::from(i16::from_le_bytes([b[0], b[1]])),
        DType::F16 => f64::from(f16::from_le_bytes([b[0], b[1]])),
        DType::U32 => f64::from(u32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        DType::I32 => f64::from(i32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        DType::F32 => f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        DType::U64 => u64::from_le_bytes(a(8)) as f64,
        DType::I64 => i64::from_le_bytes(a(8)) as f64,
        DType::F64 => f64::from_le_bytes(a(8)),
        DType::Utf8 => f64::NAN,
    }
}

impl Value {
    /// The first `n` numbers, for plotting.
    pub fn series(&self, n: usize) -> Vec<f64> {
        match self {
            Self::Numbers(values) => values.iter().take(n).map(|(_, v)| *v).collect(),
            Self::Summary { mean, .. } => vec![*mean],
            _ => Vec::new(),
        }
    }

    /// One line of text.
    pub fn brief(&self) -> String {
        match self {
            Self::Numbers(values) => values
                .iter()
                .map(|(name, v)| format!("{name} {}", number(*v)))
                .collect::<Vec<_>>()
                .join("  "),
            Self::Summary {
                count,
                min,
                max,
                mean,
                nonfinite,
            } => {
                let bad = if *nonfinite > 0 {
                    format!(", {nonfinite} non-finite")
                } else {
                    String::new()
                };
                format!(
                    "{count} values, {} … {}, mean {}{bad}",
                    number(*min),
                    number(*max),
                    number(*mean)
                )
            }
            Self::Text(text) => format!("{text:?}"),
            Self::Image(t) => format!("{}x{} frame", t.source.1, t.source.0),
            Self::Absent => "nothing dispatched".into(),
            Self::Undecodable(why) => format!("undecodable: {why}"),
        }
    }
}

/// A number in at most about seven characters.
pub fn number(v: f64) -> String {
    if !v.is_finite() {
        return format!("{v}");
    }
    let a = v.abs();
    if a != 0.0 && !(1e-3..1e5).contains(&a) {
        format!("{v:.2e}")
    } else if a >= 100.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.3}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlmesh_wire::{ChannelId, ChannelPriority, FieldSpec, Layout, ParticipantId};

    fn spec(fields: Vec<FieldSpec>, encoding: Encoding) -> StreamSpec {
        StreamSpec {
            channel: ChannelId(1),
            owner: ParticipantId::new("ur10e").expect("id"),
            path: "arm".into(),
            layout: Layout::of(&fields).expect("layout"),
            fields,
            encoding,
            priority: ChannelPriority::Realtime,
            event: false,
            max_age: None,
            applied: false,
        }
    }

    #[test]
    fn joints_take_their_names_and_other_fields_their_indices() {
        let s = spec(
            vec![
                FieldSpec::new("pos", DType::F32, &[2]),
                FieldSpec::new("t", DType::F64, &[]),
            ],
            Encoding::Raw,
        );
        let pos = [1.5_f32.to_le_bytes(), (-2.0_f32).to_le_bytes()].concat();
        let payload = s
            .layout
            .pack(&[&pos, &7.25_f64.to_le_bytes()])
            .expect("pack");
        assert_eq!(
            decode(&s, &[], &payload),
            Value::Numbers(vec![
                ("pos[0]".into(), 1.5),
                ("pos[1]".into(), -2.0),
                ("t".into(), 7.25)
            ])
        );
        let one = spec(vec![FieldSpec::new("pos", DType::I16, &[2])], Encoding::Raw);
        let names = ["shoulder".to_string(), "elbow".to_string()];
        let payload = one.layout.pack(&[&[3, 0, 0xff, 0xff]]).expect("pack");
        assert_eq!(
            decode(&one, &names, &payload),
            Value::Numbers(vec![("shoulder".into(), 3.0), ("elbow".into(), -1.0)])
        );
        assert!(matches!(decode(&one, &[], &[1]), Value::Undecodable(_)));
    }

    #[test]
    fn large_tensors_are_summarised_and_count_non_finite_values() {
        let s = spec(
            vec![FieldSpec::new("depth", DType::F32, &[8, 8])],
            Encoding::Raw,
        );
        let mut values: Vec<f32> = (0..64).map(|i| i as f32).collect();
        values[3] = f32::NAN;
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let Value::Summary {
            count,
            min,
            max,
            nonfinite,
            ..
        } = decode(&s, &[], &bytes)
        else {
            panic!("a summary");
        };
        assert_eq!((count, min, max, nonfinite), (64, 0.0, 63.0, 1));
    }

    #[test]
    fn raw_frames_become_thumbnails_no_wider_than_the_limit() {
        let s = spec(
            vec![FieldSpec::new("rgb", DType::U8, &[480, 640, 3])],
            Encoding::Raw,
        );
        assert!(is_image(&s));
        let frame = vec![200_u8; 480 * 640 * 3];
        let Value::Image(t) = thumbnail(&s, &frame, 9) else {
            panic!("a thumbnail");
        };
        assert!(t.width <= THUMB_WIDTH, "{}", t.width);
        assert_eq!(
            (t.width, t.height, t.seq, t.source),
            (160, 120, 9, (480, 640))
        );
        assert_eq!(t.rgb.len(), 160 * 120 * 3);
        assert!(matches!(thumbnail(&s, &[0; 3], 1), Value::Undecodable(_)));
    }

    #[test]
    fn numbers_stay_short() {
        assert_eq!(number(0.0), "0.000");
        assert_eq!(number(-1.23456), "-1.235");
        assert_eq!(number(512.25), "512.2");
        assert_eq!(number(2.5e7), "2.50e7");
    }
}
