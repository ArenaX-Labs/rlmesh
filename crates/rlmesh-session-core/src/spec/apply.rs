//! Applying a binding's plans (§7.4): a bound input's samples to what
//! `predict` receives, and what `predict` returns to a part's packed
//! command values.
//!
//! - [`InputRow::apply`] takes the input's samples, oldest first (one, or
//!   the window the cache assembled), as packed payloads **after transport
//!   decoding** (a JPEG part's payload is its decoded `u8 [H, W, 3]` image;
//!   the session's subscriber decodes it, so this crate needs no codec), and
//!   returns a [`Value`]: one array, 1-D fields of one dtype concatenated,
//!   a dict keyed by field name, or text.
//! - [`ActionRow::pack`] takes the model's chunk (`[horizon, ...]`) and
//!   returns `horizon` actions packed at the part's command stride, ready for
//!   an `ActionChunk`. The edge then validates and clamps.
//!
//! Arithmetic runs in `f64`. A field no arithmetic step touches keeps its
//! bytes, so integer fields of any width pass through exactly; casts to an
//! integer round half away from zero and saturate, float narrowing rounds to
//! nearest (§7.2 `dtype`).

use std::fmt;

use crate::wire::part::{DType, FieldSpec, Layout, PartSpec};

use crate::spec::bind::{ActionRow, InputRow};
use crate::spec::convert;
use crate::spec::image::{Image, crop_cut, fit_resize, fit_resize_depth, normalise_table};
use crate::spec::kinds::Kind;
use crate::spec::plan::{Delivered, Delivery, Order, Step};
use crate::spec::rotation::{self, Rotation};

/// A dense little-endian, C-contiguous array.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    /// Its element type.
    pub dtype: DType,
    /// Its shape.
    pub shape: Vec<u32>,
    /// `product(shape) × dtype size` bytes.
    pub data: Vec<u8>,
}

impl Tensor {
    /// The elements as `f64`.
    pub fn to_f64(&self) -> Vec<f64> {
        decode(self.dtype, &self.data)
    }

    /// A tensor of `values` cast to `dtype`.
    pub fn from_f64(dtype: DType, shape: Vec<u32>, values: &[f64]) -> Self {
        Self {
            dtype,
            shape,
            data: encode(dtype, values),
        }
    }
}

/// What `predict` receives for an input, or returns for an action (§6.1).
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// One array.
    Tensor(Tensor),
    /// Arrays by field name, in delivery order.
    Dict(Vec<(String, Tensor)>),
    /// A `str`.
    Text(String),
}

/// A value that does not fit its plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApplyError(pub String);

impl fmt::Display for ApplyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ApplyError {}

fn err(message: impl Into<String>) -> ApplyError {
    ApplyError(message.into())
}

fn f16_to_f64(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((bits >> 10) & 0x1f);
    let frac = f64::from(bits & 0x3ff);
    match exp {
        0 => sign * frac * 2f64.powi(-24),
        31 if frac == 0.0 => sign * f64::INFINITY,
        31 => f64::NAN,
        e => sign * (1.0 + frac / 1024.0) * 2f64.powi(e - 15),
    }
}

/// Round to nearest even, through `f32` (exact for every `f16`).
fn f64_to_f16(value: f64) -> u16 {
    let bits = (value as f32).to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    let round = |m: u32, shift: u32| {
        let half = 1_u32 << (shift - 1);
        let rest = m & ((1 << shift) - 1);
        let r = m >> shift;
        if rest > half || (rest == half && r & 1 == 1) {
            r + 1
        } else {
            r
        }
    };
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        return sign | round(mant | 0x80_0000, (14 - e) as u32) as u16;
    }
    // A carry out of the mantissa rounds into the exponent (and to inf).
    sign | round(((e as u32) << 23) | mant, 13) as u16
}

/// Little-endian elements as `f64`.
fn decode(dtype: DType, bytes: &[u8]) -> Vec<f64> {
    macro_rules! each {
        ($n:literal, $f:expr) => {
            bytes
                .chunks_exact($n)
                .map(|c| $f(c.try_into().unwrap_or([0; $n])))
                .collect()
        };
    }
    match dtype {
        DType::Bool | DType::U8 => bytes.iter().map(|b| f64::from(*b)).collect(),
        DType::I8 => bytes.iter().map(|b| f64::from(*b as i8)).collect(),
        DType::U16 => each!(2, |b| f64::from(u16::from_le_bytes(b))),
        DType::I16 => each!(2, |b| f64::from(i16::from_le_bytes(b))),
        DType::F16 => each!(2, |b| f16_to_f64(u16::from_le_bytes(b))),
        DType::U32 => each!(4, |b| f64::from(u32::from_le_bytes(b))),
        DType::I32 => each!(4, |b| f64::from(i32::from_le_bytes(b))),
        DType::F32 => each!(4, |b| f64::from(f32::from_le_bytes(b))),
        DType::U64 => each!(8, |b| u64::from_le_bytes(b) as f64),
        DType::I64 => each!(8, |b| i64::from_le_bytes(b) as f64),
        DType::F64 => each!(8, f64::from_le_bytes),
        DType::Utf8 => Vec::new(),
    }
}

/// Integer rounding for casts: half away from zero, saturating, NaN to 0.
fn int(value: f64, low: f64, high: f64) -> f64 {
    if value.is_nan() {
        0.0
    } else {
        value.round().clamp(low, high)
    }
}

/// `values` as little-endian `dtype` elements.
fn encode(dtype: DType, values: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * dtype.size() as usize);
    for &v in values {
        match dtype {
            DType::Bool => out.push(u8::from(v != 0.0 && !v.is_nan())),
            DType::U8 => out.push(int(v, 0.0, 255.0) as u8),
            DType::I8 => out.push(int(v, -128.0, 127.0) as i8 as u8),
            DType::U16 => out.extend((int(v, 0.0, 65535.0) as u16).to_le_bytes()),
            DType::I16 => out.extend((int(v, -32768.0, 32767.0) as i16).to_le_bytes()),
            DType::F16 => out.extend(f64_to_f16(v).to_le_bytes()),
            DType::U32 => out.extend((int(v, 0.0, f64::from(u32::MAX)) as u32).to_le_bytes()),
            DType::I32 => {
                out.extend((int(v, f64::from(i32::MIN), f64::from(i32::MAX)) as i32).to_le_bytes());
            }
            DType::F32 => out.extend((v as f32).to_le_bytes()),
            DType::U64 => out.extend((int(v, 0.0, u64::MAX as f64) as u64).to_le_bytes()),
            DType::I64 => {
                out.extend((int(v, i64::MIN as f64, i64::MAX as f64) as i64).to_le_bytes())
            }
            DType::F64 => out.extend(v.to_le_bytes()),
            DType::Utf8 => {}
        }
    }
    out
}

/// One field's elements on their way through a plan.
#[derive(Clone, Debug)]
struct Field {
    name: String,
    shape: Vec<u32>,
    /// Untouched bytes of `dtype`, until a step needs arithmetic.
    raw: Option<(DType, Vec<u8>)>,
    values: Vec<f64>,
}

impl Field {
    fn values(&mut self) -> &mut Vec<f64> {
        if let Some((dtype, bytes)) = self.raw.take() {
            self.values = decode(dtype, &bytes);
        }
        &mut self.values
    }

    /// Elements along the leading axis.
    fn rows(&self) -> usize {
        self.shape.first().copied().unwrap_or(1) as usize
    }

    fn row_len(&self) -> usize {
        self.shape.iter().skip(1).map(|d| *d as usize).product()
    }

    /// Permutes the leading axis: row `i` takes source row `index[i]`.
    fn gather(&mut self, index: &[usize]) {
        let row = self.row_len();
        match &mut self.raw {
            Some((dtype, bytes)) => {
                let size = row * dtype.size() as usize;
                let mut out = Vec::with_capacity(index.len() * size);
                for &i in index {
                    out.extend_from_slice(&bytes[i * size..(i + 1) * size]);
                }
                *bytes = out;
            }
            None => {
                let mut out = Vec::with_capacity(index.len() * row);
                for &i in index {
                    out.extend_from_slice(&self.values[i * row..(i + 1) * row]);
                }
                self.values = out;
            }
        }
        if let Some(first) = self.shape.first_mut() {
            *first = index.len() as u32;
        }
    }

    /// Scatters the leading axis: source row `i` lands at `index[i]` of
    /// `rows` rows.
    fn scatter(&mut self, index: &[usize], rows: usize) {
        let row = self.row_len();
        let values = self.values();
        let mut out = vec![0.0; rows * row];
        for (i, &to) in index.iter().enumerate() {
            out[to * row..(to + 1) * row].copy_from_slice(&values[i * row..(i + 1) * row]);
        }
        *values = out;
        if let Some(first) = self.shape.first_mut() {
            *first = rows as u32;
        }
    }

    fn encoded(&mut self, dtype: DType) -> Vec<u8> {
        match &self.raw {
            Some((have, bytes)) if *have == dtype => bytes.clone(),
            _ => encode(dtype, self.values()),
        }
    }
}

/// The fields of one packed value.
fn unpack(fields: &[FieldSpec], payload: &[u8]) -> Result<Vec<Field>, ApplyError> {
    let layout = Layout::of(fields).map_err(err)?;
    if let Layout::Packed { stride, .. } = &layout
        && payload.len() != *stride
    {
        return Err(err(format!("{} bytes, stride {stride}", payload.len())));
    }
    fields
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let range = layout.field(i).ok_or_else(|| err("layout"))?;
            Ok(Field {
                name: f.name.clone(),
                shape: f.shape.clone(),
                raw: Some((f.dtype, payload[range].to_vec())),
                values: Vec::new(),
            })
        })
        .collect()
}

/// The declared affine over the delivered vector, one element index across
/// its fields.
fn affine(fields: &mut [Field], scale: &[f64], offset: &[f64]) {
    let mut index = 0;
    for field in fields {
        for x in field.values() {
            *x = convert::affine(*x, index, scale, offset);
            index += 1;
        }
    }
}

fn rescale(values: &mut [f64], from: (f64, f64), to: (f64, f64)) {
    for x in values {
        *x = to.0 + (*x - from.0) / (from.1 - from.0) * (to.1 - to.0);
    }
}

fn convert_rotation(field: &mut Field, from: Rotation, to: Rotation) {
    let mut out = Vec::new();
    rotation::convert(field.values(), from, to, &mut out);
    field.values = out;
    if let Some(last) = field.shape.last_mut() {
        *last = to.width();
    }
}

/// Fields as the delivery shape says (one array, a concatenation or a
/// dict), with a leading axis of `lead` stacked samples.
fn deliver(delivers: &Delivery, samples: Vec<Vec<Field>>) -> Value {
    match delivers {
        Delivery::Text => Value::Text(String::new()),
        Delivery::Array { dtype, shape, .. } => {
            let mut data = Vec::new();
            for mut fields in samples {
                for field in &mut fields {
                    data.extend(field.encoded(*dtype));
                }
            }
            Value::Tensor(Tensor {
                dtype: *dtype,
                shape: shape.clone(),
                data,
            })
        }
        Delivery::Dict(delivered) => {
            let mut out: Vec<(String, Tensor)> = delivered
                .iter()
                .map(|d| {
                    (
                        d.name.clone(),
                        Tensor {
                            dtype: d.dtype,
                            shape: d.shape.clone(),
                            data: Vec::new(),
                        },
                    )
                })
                .collect();
            for fields in samples {
                for mut field in fields {
                    if let Some((_, t)) = out.iter_mut().find(|(n, _)| *n == field.name) {
                        let bytes = field.encoded(t.dtype);
                        t.data.extend(bytes);
                    }
                }
            }
            Value::Dict(out)
        }
    }
}

impl InputRow {
    /// Applies the plan to the input's samples (oldest first; one, or the
    /// window), given as transport-decoded packed payloads. A blank input
    /// takes none.
    pub fn apply(&self, samples: &[&[u8]]) -> Result<Value, ApplyError> {
        let Some(part) = &self.part else {
            return Ok(self.blank_value());
        };
        let want = self.history.max(1) as usize;
        if samples.len() != want {
            return Err(err(format!(
                "{}: {} samples for a window of {want}",
                self.entry,
                samples.len()
            )));
        }
        if let Delivery::Text = self.delivers {
            let text = std::str::from_utf8(samples[want - 1]).map_err(|e| err(e.to_string()))?;
            return Ok(Value::Text(text.to_string()));
        }
        let kind = Kind::parse(&part.kind).unwrap_or(Kind::Custom);
        if kind.is_image() {
            let mut out = Vec::new();
            for sample in samples {
                out.extend(self.apply_image(part, kind, sample)?);
            }
            let Delivery::Array { dtype, shape, .. } = &self.delivers else {
                return Err(err("an image delivers one array"));
            };
            return Ok(Value::Tensor(Tensor {
                dtype: *dtype,
                shape: shape.clone(),
                data: out,
            }));
        }
        let source: &[FieldSpec] = match (&part.command, self.applied) {
            (Some(command), true) => &command.fields,
            _ => &part.fields,
        };
        let mut stacked = Vec::with_capacity(want);
        for sample in samples {
            let mut fields = unpack(source, sample)?;
            for step in &self.plan.steps {
                match step {
                    Step::Fields(names) => {
                        fields = names
                            .iter()
                            .filter_map(|n| fields.iter().find(|f| &f.name == n).cloned())
                            .collect();
                    }
                    Step::Names(Order::Reorder(index) | Order::Select(index)) => {
                        for f in &mut fields {
                            f.gather(index);
                        }
                    }
                    Step::Units { field, factor, .. } => {
                        for f in fields.iter_mut().filter(|f| &f.name == field) {
                            for x in f.values() {
                                *x *= factor;
                            }
                        }
                    }
                    Step::GripperWidth {
                        to_normalized,
                        low,
                        high,
                    } => {
                        for f in &mut fields {
                            let (from, to) = if *to_normalized {
                                ((*low, *high), (0.0, 1.0))
                            } else {
                                ((0.0, 1.0), (*low, *high))
                            };
                            rescale(f.values(), from, to);
                        }
                    }
                    Step::Rotation {
                        field, from, to, ..
                    } => {
                        for f in fields.iter_mut().filter(|f| &f.name == field) {
                            convert_rotation(f, *from, *to);
                        }
                    }
                    Step::Rescale { from, to } => {
                        for f in &mut fields {
                            rescale(f.values(), *from, *to);
                        }
                    }
                    Step::Affine { scale, offset } => affine(&mut fields, scale, offset),
                    _ => {}
                }
            }
            stacked.push(fields);
        }
        Ok(deliver(&self.delivers, stacked))
    }

    /// One camera or depth sample through the plan, as delivered bytes.
    fn apply_image(
        &self,
        part: &PartSpec,
        kind: Kind,
        payload: &[u8],
    ) -> Result<Vec<u8>, ApplyError> {
        let field = &part.fields[0];
        let (height, width) = (field.shape[0] as usize, field.shape[1] as usize);
        let channels = field.shape.get(2).map_or(1, |c| *c as usize);
        let expected = height * width * channels * field.dtype.size() as usize;
        if payload.len() != expected {
            return Err(err(format!(
                "{}: an image of {} bytes, expected {expected}",
                self.entry,
                payload.len()
            )));
        }
        let Delivery::Array { dtype, .. } = &self.delivers else {
            return Err(err("an image delivers one array"));
        };
        if kind == Kind::Depth {
            let mut image = Image {
                height,
                width,
                channels: 1,
                data: decode(field.dtype, payload),
            };
            let mut metres = field.unit == "m";
            for step in &self.plan.steps {
                match step {
                    Step::DepthMetres => {
                        for v in &mut image.data {
                            *v = if *v == 0.0 { f64::NAN } else { *v * 1e-3 };
                        }
                        metres = true;
                    }
                    Step::Rotate180 => image = image.rotate180(),
                    Step::CenterCrop { fraction, .. } => {
                        image = image.crop_center(
                            crop_cut(image.height, *fraction),
                            crop_cut(image.width, *fraction),
                        );
                    }
                    Step::Resize { to, fit, .. } => {
                        let fill = if metres { f64::NAN } else { 0.0 };
                        image = fit_resize_depth(
                            &image,
                            (to.height as usize, to.width as usize),
                            *fit,
                            fill,
                        );
                    }
                    Step::Affine { scale, offset } => {
                        for v in &mut image.data {
                            *v = convert::affine(*v, 0, scale, offset);
                        }
                    }
                    _ => {}
                }
            }
            return Ok(encode(*dtype, &image.data));
        }
        let mut image = Image {
            height,
            width,
            channels,
            data: payload.to_vec(),
        };
        let mut floats: Option<Vec<f64>> = None;
        let mut chw = false;
        for step in &self.plan.steps {
            match step {
                Step::Rotate180 => image = image.rotate180(),
                Step::CenterCrop { fraction, .. } => {
                    image = image.crop_center(
                        crop_cut(image.height, *fraction),
                        crop_cut(image.width, *fraction),
                    );
                }
                Step::Resize {
                    kernel, to, fit, ..
                } => {
                    image = fit_resize(
                        &image,
                        (to.height as usize, to.width as usize),
                        *fit,
                        *kernel,
                        None,
                    );
                }
                Step::Bgr => image = image.bgr(),
                Step::Chw => chw = true,
                Step::Normalise { low, high, .. } => {
                    let table = normalise_table(*low, *high);
                    floats = Some(
                        image
                            .data
                            .iter()
                            .map(|v| f64::from(table[usize::from(*v)]))
                            .collect(),
                    );
                }
                Step::Affine { scale, offset } => {
                    let values = floats
                        .get_or_insert_with(|| image.data.iter().map(|v| f64::from(*v)).collect());
                    let c = image.channels;
                    for (i, x) in values.iter_mut().enumerate() {
                        *x = convert::affine(*x, i % c, scale, offset);
                    }
                }
                _ => {}
            }
        }
        // Pixels no arithmetic touched stay bytes: a u8 image is delivered
        // without a float round trip.
        if floats.is_none() && *dtype == DType::U8 {
            if !chw {
                return Ok(image.data);
            }
            let (plane, c) = (image.height * image.width, image.channels);
            let mut out = vec![0_u8; image.data.len()];
            for (i, v) in image.data.iter().enumerate() {
                out[(i % c) * plane + i / c] = *v;
            }
            return Ok(out);
        }
        let values = floats.unwrap_or_else(|| image.data.iter().map(|v| f64::from(*v)).collect());
        let values = if chw {
            let (plane, c) = (image.height * image.width, image.channels);
            let mut out = vec![0.0; values.len()];
            for (i, v) in values.iter().enumerate() {
                out[(i % c) * plane + i / c] = *v;
            }
            out
        } else {
            values
        };
        if *dtype == DType::U8 && values.iter().all(|v| v.fract() == 0.0) {
            return Ok(values.iter().map(|v| *v as u8).collect());
        }
        Ok(encode(*dtype, &values))
    }

    fn blank_value(&self) -> Value {
        let fill = self.plan.steps.iter().find_map(|s| match s {
            Step::Blank { fill, .. } => Some(fill.clone()),
            _ => None,
        });
        let fill = fill.unwrap_or_else(|| vec![0.0]);
        let tensor = |d: &Delivered| -> Tensor {
            let n: usize = d.shape.iter().map(|x| *x as usize).product();
            let values: Vec<f64> = match fill.as_slice() {
                [one] => vec![*one; n],
                per => {
                    // Per channel: the last axis (HWC) or the first (CHW).
                    let c = per.len();
                    let chw = d.shape.len() >= 3 && d.shape[d.shape.len() - 3] as usize == c;
                    let plane = if chw { n / c } else { 1 };
                    (0..n)
                        .map(|i| {
                            if chw {
                                per[(i / plane) % c]
                            } else {
                                per[i % c]
                            }
                        })
                        .collect()
                }
            };
            Tensor::from_f64(d.dtype, d.shape.clone(), &values)
        };
        match &self.delivers {
            Delivery::Text => Value::Text(String::new()),
            Delivery::Array { dtype, shape, .. } => Value::Tensor(tensor(&Delivered {
                name: String::new(),
                dtype: *dtype,
                shape: shape.clone(),
            })),
            Delivery::Dict(fields) => {
                Value::Dict(fields.iter().map(|d| (d.name.clone(), tensor(d))).collect())
            }
        }
    }
}

impl ActionRow {
    /// Packs a model's chunk (`[horizon, ...]`, as [`Self::delivers`]
    /// describes) into `horizon` actions at the part's command stride.
    pub fn pack(&self, value: &Value) -> Result<Vec<u8>, ApplyError> {
        let Some(part) = &self.part else {
            return Err(err(format!(
                "{}: a dropped action packs nothing",
                self.entry
            )));
        };
        let command = part
            .command
            .as_ref()
            .ok_or_else(|| err(format!("{} is not commandable", part.path)))?;
        let layout = Layout::of(&command.fields).map_err(err)?;
        let stride = layout.stride().ok_or_else(|| err("a text command"))?;
        let horizon = self.horizon as usize;
        // The model's per-action fields, in delivery order.
        let model_fields: Vec<(String, Vec<u32>)> = match &self.delivers {
            Delivery::Array { shape, concat, .. } if !concat.is_empty() => {
                let _ = shape;
                command
                    .fields
                    .iter()
                    .map(|f| (f.name.clone(), f.shape.clone()))
                    .collect()
            }
            Delivery::Array { shape, .. } => {
                vec![(command.fields[0].name.clone(), shape[1..].to_vec())]
            }
            Delivery::Dict(fields) => fields
                .iter()
                .map(|d| (d.name.clone(), d.shape[1..].to_vec()))
                .collect(),
            Delivery::Text => return Err(err("a text action")),
        };
        // Split the model's value into per-action fields.
        let per_action: Vec<Vec<Field>> = match (value, &self.delivers) {
            (Value::Tensor(t), Delivery::Array { shape, .. }) => {
                if &t.shape != shape {
                    return Err(err(format!(
                        "{}: shape {:?}, expected {shape:?}",
                        self.entry, t.shape
                    )));
                }
                let all = t.to_f64();
                let row = all.len() / horizon.max(1);
                (0..horizon)
                    .map(|i| {
                        let mut at = i * row;
                        model_fields
                            .iter()
                            .map(|(name, shape)| {
                                let n: usize = shape.iter().map(|d| *d as usize).product();
                                let values = all[at..at + n].to_vec();
                                at += n;
                                Field {
                                    name: name.clone(),
                                    shape: shape.clone(),
                                    raw: None,
                                    values,
                                }
                            })
                            .collect()
                    })
                    .collect()
            }
            (Value::Dict(items), Delivery::Dict(fields)) => {
                let mut per: Vec<Vec<Field>> = vec![Vec::new(); horizon];
                for d in fields {
                    let t = items
                        .iter()
                        .find(|(n, _)| *n == d.name)
                        .map(|(_, t)| t)
                        .ok_or_else(|| {
                            err(format!("{}: no {:?} in the chunk", self.entry, d.name))
                        })?;
                    if t.shape != d.shape {
                        return Err(err(format!(
                            "{}.{}: shape {:?}, expected {:?}",
                            self.entry, d.name, t.shape, d.shape
                        )));
                    }
                    let all = t.to_f64();
                    let n = all.len() / horizon.max(1);
                    for (i, slot) in per.iter_mut().enumerate() {
                        slot.push(Field {
                            name: d.name.clone(),
                            shape: d.shape[1..].to_vec(),
                            raw: None,
                            values: all[i * n..(i + 1) * n].to_vec(),
                        });
                    }
                }
                per
            }
            _ => {
                return Err(err(format!(
                    "{}: the chunk does not match {}",
                    self.entry, self.delivers
                )));
            }
        };
        let mut out = Vec::with_capacity(horizon * stride);
        for mut fields in per_action {
            for step in &self.plan.steps {
                match step {
                    Step::Affine { scale, offset } => affine(&mut fields, scale, offset),
                    Step::Rescale { from, to } => {
                        for f in &mut fields {
                            rescale(f.values(), *from, *to);
                        }
                    }
                    Step::Rotation {
                        field, from, to, ..
                    } => {
                        for f in fields.iter_mut().filter(|f| &f.name == field) {
                            convert_rotation(f, *from, *to);
                        }
                    }
                    Step::Units { field, factor, .. } => {
                        for f in fields.iter_mut().filter(|f| &f.name == field) {
                            for x in f.values() {
                                *x *= factor;
                            }
                        }
                    }
                    Step::GripperWidth { low, high, .. } => {
                        for f in &mut fields {
                            rescale(f.values(), (0.0, 1.0), (*low, *high));
                        }
                    }
                    Step::Names(Order::Reorder(index)) => {
                        for f in &mut fields {
                            let rows = f.rows();
                            f.scatter(index, rows);
                        }
                    }
                    _ => {}
                }
            }
            let mut action = vec![0_u8; stride];
            for (i, spec) in command.fields.iter().enumerate() {
                let field = fields
                    .iter_mut()
                    .find(|f| f.name == spec.name)
                    .ok_or_else(|| err(format!("{}: no {:?}", self.entry, spec.name)))?;
                let bytes = field.encoded(spec.dtype);
                let range = layout.field(i).ok_or_else(|| err("layout"))?;
                if bytes.len() != range.len() {
                    return Err(err(format!(
                        "{}.{}: {} bytes, the command field takes {}",
                        self.entry,
                        spec.name,
                        bytes.len(),
                        range.len()
                    )));
                }
                action[range].copy_from_slice(&bytes);
            }
            out.extend(action);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn casts_round_half_away_from_zero_and_saturate() {
        assert_eq!(
            encode(DType::I8, &[2.5, -2.5, 300.0, -300.0, f64::NAN]),
            [3, 253, 127, 128, 0]
        );
        assert_eq!(encode(DType::U8, &[254.5, -1.0]), [255, 0]);
        for x in [0.0, 1.0, -2.5, 65504.0, 0.000_061_035_156_25] {
            assert_eq!(f16_to_f64(f64_to_f16(x)), x, "{x}");
        }
        assert!(f16_to_f64(f64_to_f16(1e6)).is_infinite());
        assert_eq!(decode(DType::I16, &(-7_i16).to_le_bytes()), [-7.0]);
    }
}
