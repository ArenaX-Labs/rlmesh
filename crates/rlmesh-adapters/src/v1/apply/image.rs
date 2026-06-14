//! Image operations used by resolved adapters.
//!
//! Resizing implements the two algorithms pinned by the v1 conformance
//! vectors: `bilinear` (4-tap, half-pixel centers) and `bilinear_aa`
//! (antialiased separable triangle filter, PIL-compatible within +-1
//! uint8 step). Both compute in float64 with one final round-half-to-even
//! like the reference. Pixels are carried in `rlmesh_spaces::Tensor`.

use std::collections::BTreeMap;

use rlmesh_spaces::{DType, Tensor};

use super::super::plans::ImagePlan;
use super::super::spec::ImageLayout;
use super::error::ApplyError;
use super::lookup::lookup;
use super::value::{self, Value};

/// Produce one model image input from a raw observation.
pub(super) fn apply_image(
    plan: &ImagePlan,
    raw_obs: &BTreeMap<String, Value>,
) -> Result<Value, ApplyError> {
    let mut image = decode_image(lookup(raw_obs, &plan.env_key)?, plan.src_range)?;
    image = to_layout(&image, plan.src_layout, ImageLayout::Hwc)?;
    if plan.flip {
        image = flip_180(&image)?;
    }
    if let Some((height, width)) = plan.size {
        image = resize_image(&image, height, width, &plan.resample)?;
    }
    image = finalize_dtype(&image, &plan.dtype, plan.normalize)?;
    image = to_layout(&image, ImageLayout::Hwc, plan.dst_layout)?;
    Ok(Value::Tensor(add_lead_dims(image, plan.lead_dims)))
}

/// Return an HWC uint8 tensor from a raw observation value.
///
/// The pipeline operates on 8-bit pixels; the source is converted without
/// truncation-casting. A float image is mapped from its declared `src_range`
/// into `[0, 255]` (a `[0, 1]` image is scaled, a `[0, 255]` image passes
/// through), so it is neither floored to black nor saturated to white.
pub fn decode_image(value: &Value, src_range: Option<(f64, f64)>) -> Result<Tensor, ApplyError> {
    let Value::Tensor(tensor) = value else {
        return Err(ApplyError::new(
            "expected an image array observation value (encoded image bytes \
             are a binding-level concern)"
                .to_owned(),
        ));
    };
    Ok(value::tensor_from_u8(
        tensor.shape().to_vec(),
        value::to_u8_pixels(tensor, src_range),
    ))
}

fn image_dims(tensor: &Tensor) -> Result<(usize, usize, usize), ApplyError> {
    let shape = value::shape_usize(tensor);
    if shape.len() != 3 {
        return Err(ApplyError::new(format!(
            "expected an HWC image with 3 axes, got shape {:?}",
            tensor.shape()
        )));
    }
    Ok((shape[0], shape[1], shape[2]))
}

/// Rotate an HWC image by 180 degrees.
pub fn flip_180(tensor: &Tensor) -> Result<Tensor, ApplyError> {
    let (height, width, channels) = image_dims(tensor)?;
    let mut indices = Vec::with_capacity(tensor.numel());
    for row in 0..height {
        for col in 0..width {
            let src = ((height - 1 - row) * width + (width - 1 - col)) * channels;
            indices.extend(src..src + channels);
        }
    }
    Ok(value::gather(tensor, &indices, tensor.shape().to_vec()))
}

/// Transpose an image between `hwc` and `chw` layouts (any dtype).
pub fn to_layout(
    tensor: &Tensor,
    source: ImageLayout,
    target: ImageLayout,
) -> Result<Tensor, ApplyError> {
    if source == target {
        return Ok(tensor.clone());
    }
    let mut indices = Vec::with_capacity(tensor.numel());
    let shape = match target {
        ImageLayout::Chw => {
            let (height, width, channels) = image_dims(tensor)?;
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
            let (channels, height, width) = image_dims(tensor)?;
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
    Ok(value::gather(tensor, &indices, value::shape_i64(&shape)))
}

fn finish_pixels(blended: Vec<f64>, shape: Vec<usize>) -> Tensor {
    let data: Vec<u8> = blended
        .into_iter()
        .map(|value| value.round_ties_even().clamp(0.0, 255.0) as u8)
        .collect();
    value::tensor_from_u8(value::shape_i64(&shape), data)
}

/// 4-tap half-pixel-center bilinear resize (OpenCV/torch-compatible).
fn resize_bilinear(tensor: &Tensor, height: usize, width: usize) -> Result<Tensor, ApplyError> {
    let (src_height, src_width, channels) = image_dims(tensor)?;
    let data = value::u8_pixels(tensor)?;
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
fn resize_bilinear_aa(tensor: &Tensor, height: usize, width: usize) -> Result<Tensor, ApplyError> {
    let (src_height, src_width, channels) = image_dims(tensor)?;
    let data = value::u8_pixels(tensor)?;
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
    tensor: &Tensor,
    height: u32,
    width: u32,
    resample: &str,
) -> Result<Tensor, ApplyError> {
    let (src_height, src_width, _) = image_dims(tensor)?;
    let (height, width) = (height as usize, width as usize);
    if height == 0 || width == 0 || src_height == 0 || src_width == 0 {
        return Err(ApplyError::new(format!(
            "cannot resize an image with a zero dimension (source \
             {src_height}x{src_width}, target {height}x{width})"
        )));
    }
    if (src_height, src_width) == (height, width) {
        return Ok(tensor.clone());
    }
    match resample {
        "bilinear" => resize_bilinear(tensor, height, width),
        "bilinear_aa" => resize_bilinear_aa(tensor, height, width),
        other => Err(ApplyError::new(format!("unsupported resample {other:?}"))),
    }
}

/// Cast an image to `dtype`, optionally scaling 8-bit values into [0, 1].
pub fn finalize_dtype(tensor: &Tensor, dtype: &str, normalize: bool) -> Result<Tensor, ApplyError> {
    let target = DType::from_name(dtype).ok_or_else(|| {
        ApplyError::new(format!("unsupported dtype {dtype:?} for an image output"))
    })?;
    if normalize {
        let scaled: Vec<f32> = value::to_f32_vec(tensor)
            .into_iter()
            .map(|value| value / 255.0)
            .collect();
        let scaled = value::tensor_from_f32(tensor.shape().to_vec(), &scaled);
        return value::cast(&scaled, target);
    }
    value::cast(tensor, target)
}

/// Prepend `count` singleton axes to an image.
pub fn add_lead_dims(tensor: Tensor, count: u32) -> Tensor {
    if count == 0 {
        return tensor;
    }
    let mut shape = vec![1i64; count as usize];
    shape.extend_from_slice(tensor.shape());
    tensor
        .reshape(&shape)
        .expect("adding unit axes preserves the element count")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_scales_unit_float_images_instead_of_truncating() {
        // A normalized [0, 1] float image must scale into 8-bit, not floor to
        // an all-black image.
        let image = Value::Tensor(value::tensor_from_f32(vec![2, 2, 1], &[0.0, 0.4, 0.6, 1.0]));
        let decoded = decode_image(&image, Some((0.0, 1.0))).expect("decode");
        assert_eq!(decoded.dtype(), DType::Uint8);
        assert_eq!(decoded.to_contiguous_bytes().as_ref(), [0u8, 102, 153, 255]);
    }

    #[test]
    fn decode_passes_through_byte_range_float_images() {
        // A float image already in [0, 255] must NOT be scaled by 255 (which
        // would saturate every pixel > 1 to white); the declared byte range
        // maps through unchanged.
        let image = Value::Tensor(value::tensor_from_f32(
            vec![2, 2, 1],
            &[0.0, 64.0, 200.0, 255.0],
        ));
        let decoded = decode_image(&image, Some((0.0, 255.0))).expect("decode");
        assert_eq!(decoded.to_contiguous_bytes().as_ref(), [0u8, 64, 200, 255]);
    }

    #[test]
    fn decode_assumes_normalized_for_unbounded_float_images() {
        // With no declared range the [0, 1] assumption is kept (back-compat).
        let image = Value::Tensor(value::tensor_from_f32(vec![1, 1, 2], &[0.0, 1.0]));
        let decoded = decode_image(&image, None).expect("decode");
        assert_eq!(decoded.to_contiguous_bytes().as_ref(), [0u8, 255]);
    }

    #[test]
    fn decode_passes_through_uint8_images() {
        let image = Value::Tensor(value::tensor_from_u8(vec![1, 1, 3], vec![10, 20, 30]));
        let decoded = decode_image(&image, Some((0.0, 255.0))).expect("decode");
        assert_eq!(decoded.to_contiguous_bytes().as_ref(), [10u8, 20, 30]);
    }

    #[test]
    fn resize_rejects_zero_dimensions() {
        let image = value::tensor_from_u8(vec![0, 2, 3], Vec::new());
        let error = resize_image(&image, 4, 4, "bilinear").expect_err("zero dim");
        assert!(error.to_string().contains("zero dimension"), "{error}");
    }
}
