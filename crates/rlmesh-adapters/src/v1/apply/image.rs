//! Image operations used by resolved adapters.
//!
//! Resizing implements the two algorithms pinned by the v1 conformance
//! vectors: `bilinear` (4-tap, half-pixel centers) and `bilinear_aa`
//! (antialiased separable triangle filter, PIL-compatible within +-1
//! uint8 step). Both compute in float64 with one final round-half-to-even
//! like the reference.

use std::collections::BTreeMap;

use super::super::plans::ImagePlan;
use super::super::spec::ImageLayout;
use super::error::ApplyError;
use super::lookup::lookup;
use super::value::{Array, ArrayData, Dtype, Value};

/// Produce one model image input from a raw observation.
pub(super) fn apply_image(
    plan: &ImagePlan,
    raw_obs: &BTreeMap<String, Value>,
) -> Result<Value, ApplyError> {
    let mut array = decode_image(lookup(raw_obs, &plan.env_key)?)?;
    array = to_layout(&array, plan.src_layout, ImageLayout::Hwc)?;
    if plan.flip {
        array = flip_180(&array)?;
    }
    if let Some((height, width)) = plan.size {
        array = resize_image(&array, height, width, &plan.resample)?;
    }
    array = finalize_dtype(&array, &plan.dtype, plan.normalize)?;
    array = to_layout(&array, ImageLayout::Hwc, plan.dst_layout)?;
    Ok(Value::Array(add_lead_dims(array, plan.lead_dims)))
}

/// Return an HWC uint8 array from a raw observation value.
pub fn decode_image(value: &Value) -> Result<Array, ApplyError> {
    let Value::Array(array) = value else {
        return Err(ApplyError::new(
            "expected an image array observation value (encoded image bytes \
             are a binding-level concern)"
                .to_owned(),
        ));
    };
    Ok(array.cast(Dtype::U8))
}

fn image_dims(array: &Array) -> Result<(usize, usize, usize), ApplyError> {
    if array.shape.len() != 3 {
        return Err(ApplyError::new(format!(
            "expected an HWC image with 3 axes, got shape {:?}",
            array.shape
        )));
    }
    Ok((array.shape[0], array.shape[1], array.shape[2]))
}

/// Reorder array elements so `out[i] = data[indices[i]]`, dtype-generic.
fn gather(data: &ArrayData, indices: &[usize]) -> ArrayData {
    match data {
        ArrayData::U8(values) => ArrayData::U8(indices.iter().map(|&i| values[i]).collect()),
        ArrayData::I32(values) => ArrayData::I32(indices.iter().map(|&i| values[i]).collect()),
        ArrayData::I64(values) => ArrayData::I64(indices.iter().map(|&i| values[i]).collect()),
        ArrayData::F32(values) => ArrayData::F32(indices.iter().map(|&i| values[i]).collect()),
        ArrayData::F64(values) => ArrayData::F64(indices.iter().map(|&i| values[i]).collect()),
    }
}

/// Rotate an HWC image by 180 degrees.
pub fn flip_180(array: &Array) -> Result<Array, ApplyError> {
    let (height, width, channels) = image_dims(array)?;
    let mut indices = Vec::with_capacity(array.len());
    for row in 0..height {
        for col in 0..width {
            let src = ((height - 1 - row) * width + (width - 1 - col)) * channels;
            indices.extend(src..src + channels);
        }
    }
    Ok(Array {
        dtype: array.dtype,
        shape: array.shape.clone(),
        data: gather(&array.data, &indices),
    })
}

/// Transpose an image between `hwc` and `chw` layouts (any dtype).
pub fn to_layout(
    array: &Array,
    source: ImageLayout,
    target: ImageLayout,
) -> Result<Array, ApplyError> {
    if source == target {
        return Ok(array.clone());
    }
    let mut indices = Vec::with_capacity(array.len());
    let shape = match target {
        ImageLayout::Chw => {
            let (height, width, channels) = image_dims(array)?;
            for channel in 0..channels {
                for row in 0..height {
                    for col in 0..width {
                        indices.push((row * width + col) * channels + channel);
                    }
                }
            }
            vec![channels, height, width]
        }
        ImageLayout::Hwc => {
            let (channels, height, width) = image_dims(array)?;
            for row in 0..height {
                for col in 0..width {
                    for channel in 0..channels {
                        indices.push(channel * height * width + row * width + col);
                    }
                }
            }
            vec![height, width, channels]
        }
    };
    Ok(Array {
        dtype: array.dtype,
        shape,
        data: gather(&array.data, &indices),
    })
}

fn u8_pixels(array: &Array) -> Result<&[u8], ApplyError> {
    match &array.data {
        ArrayData::U8(data) => Ok(data),
        _ => Err(ApplyError::new(
            "resize expects uint8 image data".to_owned(),
        )),
    }
}

fn finish_pixels(blended: Vec<f64>, shape: Vec<usize>) -> Array {
    let data: Vec<u8> = blended
        .into_iter()
        .map(|value| value.round_ties_even().clamp(0.0, 255.0) as u8)
        .collect();
    Array {
        dtype: Dtype::U8,
        shape,
        data: ArrayData::U8(data),
    }
}

/// 4-tap half-pixel-center bilinear resize (OpenCV/torch-compatible).
fn resize_bilinear(array: &Array, height: usize, width: usize) -> Result<Array, ApplyError> {
    let (src_height, src_width, channels) = image_dims(array)?;
    let data = u8_pixels(array)?;
    let coords = |dst: usize, src: usize| -> Vec<(usize, usize, f64)> {
        (0..dst)
            .map(|i| {
                let pos = (i as f64 + 0.5) * src as f64 / dst as f64 - 0.5;
                let lo = (pos.floor() as i64).clamp(0, src as i64 - 1) as usize;
                let hi = (lo + 1).min(src - 1);
                (lo, hi, (pos - lo as f64).clamp(0.0, 1.0))
            })
            .collect()
    };
    let rows = coords(height, src_height);
    let cols = coords(width, src_width);
    let pixel = |row: usize, col: usize, channel: usize| -> f64 {
        f64::from(data[(row * src_width + col) * channels + channel])
    };
    let mut blended = Vec::with_capacity(height * width * channels);
    for &(row0, row1, row_w) in &rows {
        for &(col0, col1, col_w) in &cols {
            for channel in 0..channels {
                let top =
                    pixel(row0, col0, channel) * (1.0 - col_w) + pixel(row0, col1, channel) * col_w;
                let bottom =
                    pixel(row1, col0, channel) * (1.0 - col_w) + pixel(row1, col1, channel) * col_w;
                blended.push(top * (1.0 - row_w) + bottom * row_w);
            }
        }
    }
    Ok(finish_pixels(blended, vec![height, width, channels]))
}

/// Per-output-pixel normalized triangle-filter weights, PIL-style.
fn triangle_weights(src: usize, dst: usize) -> Vec<(usize, Vec<f64>)> {
    let scale = src as f64 / dst as f64;
    let filterscale = scale.max(1.0);
    let support = filterscale;
    (0..dst)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let lo = ((center - support + 0.5) as i64).max(0) as usize;
            let hi = ((center + support + 0.5) as i64).min(src as i64) as usize;
            let mut row: Vec<f64> = (lo..hi)
                .map(|tap| (1.0 - ((tap as f64 + 0.5 - center) / filterscale).abs()).max(0.0))
                .collect();
            let total: f64 = row.iter().sum();
            if total > 0.0 {
                for weight in &mut row {
                    *weight /= total;
                }
            }
            (lo, row)
        })
        .collect()
}

/// Antialiased separable triangle-filter resize (PIL-compatible).
fn resize_bilinear_aa(array: &Array, height: usize, width: usize) -> Result<Array, ApplyError> {
    let (src_height, src_width, channels) = image_dims(array)?;
    let data = u8_pixels(array)?;
    let col_weights = triangle_weights(src_width, width);
    let row_weights = triangle_weights(src_height, height);

    // Horizontal pass: (src_height, width, channels) in float64.
    let mut horizontal = vec![0.0f64; src_height * width * channels];
    for row in 0..src_height {
        for (out_col, (lo, weights)) in col_weights.iter().enumerate() {
            for channel in 0..channels {
                let mut acc = 0.0;
                for (offset, weight) in weights.iter().enumerate() {
                    acc += weight
                        * f64::from(data[(row * src_width + lo + offset) * channels + channel]);
                }
                horizontal[(row * width + out_col) * channels + channel] = acc;
            }
        }
    }

    // Vertical pass: (height, width, channels).
    let mut blended = vec![0.0f64; height * width * channels];
    for (out_row, (lo, weights)) in row_weights.iter().enumerate() {
        for col in 0..width {
            for channel in 0..channels {
                let mut acc = 0.0;
                for (offset, weight) in weights.iter().enumerate() {
                    acc += weight * horizontal[((lo + offset) * width + col) * channels + channel];
                }
                blended[(out_row * width + col) * channels + channel] = acc;
            }
        }
    }
    Ok(finish_pixels(blended, vec![height, width, channels]))
}

/// Resize an HWC uint8 image with the declared resample algorithm.
pub fn resize_image(
    array: &Array,
    height: u32,
    width: u32,
    resample: &str,
) -> Result<Array, ApplyError> {
    let (src_height, src_width, _) = image_dims(array)?;
    let (height, width) = (height as usize, width as usize);
    if (src_height, src_width) == (height, width) {
        return Ok(array.clone());
    }
    match resample {
        "bilinear" => resize_bilinear(array, height, width),
        "bilinear_aa" => resize_bilinear_aa(array, height, width),
        other => Err(ApplyError::new(format!("unsupported resample {other:?}"))),
    }
}

/// Cast an image to `dtype`, optionally scaling 8-bit values into [0, 1].
pub fn finalize_dtype(array: &Array, dtype: &str, normalize: bool) -> Result<Array, ApplyError> {
    let dtype = Dtype::parse(dtype)?;
    if normalize {
        let scaled: Vec<f32> = array
            .to_f32_vec()
            .into_iter()
            .map(|value| value / 255.0)
            .collect();
        let scaled = Array {
            dtype: Dtype::F32,
            shape: array.shape.clone(),
            data: ArrayData::F32(scaled),
        };
        return Ok(scaled.cast(dtype));
    }
    Ok(array.cast(dtype))
}

/// Prepend `count` singleton axes to an image.
pub fn add_lead_dims(array: Array, count: u32) -> Array {
    if count == 0 {
        return array;
    }
    let mut shape = vec![1usize; count as usize];
    shape.extend(&array.shape);
    Array { shape, ..array }
}
