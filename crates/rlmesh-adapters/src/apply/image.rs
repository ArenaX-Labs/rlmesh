//! Image operations used by resolved adapters.
//!
//! Resizing implements the two algorithms pinned by the v1 conformance
//! vectors: `bilinear` (4-tap, half-pixel centers) and `bilinear_aa`
//! (antialiased separable triangle filter, PIL-compatible within +-1
//! uint8 step). Both compute in float64 with one final round-half-to-even
//! like the reference. Pixels are carried in `rlmesh_spaces::Tensor`.

use std::collections::BTreeMap;

use rlmesh_spaces::{DType, Tensor};

use super::lookup::resolve_in_obs;
use super::value::{self, Value};
use crate::error::ApplyError;
use crate::plans::ImagePlan;
use crate::spec::{FitMode, ImageLayout};

/// Produce one model image input from a raw observation.
pub(super) fn apply_image(
    plan: &ImagePlan,
    raw_obs: &BTreeMap<String, Value>,
) -> Result<Value, ApplyError> {
    if let Some((height, width, channels)) = plan.zero_fill {
        return apply_zero_fill_image(plan, height, width, channels);
    }
    let mut image = decode_image(resolve_in_obs(raw_obs, &plan.source)?, plan.src_range)?;
    image = to_layout(&image, plan.src_layout, ImageLayout::Hwc)?;
    if plan.flip {
        image = flip_180(&image)?;
    }
    if let Some((height, width)) = plan.size {
        image = fit_resize(&image, height, width, &plan.resample, plan.fit)?;
    }
    finalize_image(image, plan)
}

/// The shared tail both image paths (real and zero-fill) end with: map the HWC
/// uint8 frame into the model's dtype/range, transpose to the model's layout, and
/// prepend any leading axes. Kept in one place so a normalize/layout/lead policy
/// change cannot silently diverge between the two paths.
fn finalize_image(image: Tensor, plan: &ImagePlan) -> Result<Value, ApplyError> {
    let image = finalize_dtype(&image, &plan.dtype, plan.normalize)?;
    let image = to_layout(&image, ImageLayout::Hwc, plan.dst_layout)?;
    Ok(Value::Tensor(add_lead_dims(image, plan.lead_dims)))
}

/// Synthesize the model input for an optional image the env did not provide: a
/// frame filled with the spec's `absent_fill` level (black by default), run
/// through the same normalize/dtype/layout/lead steps as a real frame so it is
/// indistinguishable from an actual flat observation at that level.
fn apply_zero_fill_image(
    plan: &ImagePlan,
    height: u32,
    width: u32,
    channels: u32,
) -> Result<Value, ApplyError> {
    let (height, width, channels) = (height as usize, width as usize, channels as usize);
    let fill = value::tensor_from_u8(
        value::shape_i64(&[height, width, channels]),
        vec![plan.absent_fill; height * width * channels],
    );
    finalize_image(fill, plan)
}

/// Return an HWC uint8 tensor from a raw observation value.
///
/// The pipeline operates on 8-bit pixels; the source is converted without
/// truncation-casting. A float image is mapped from its declared `src_range`
/// into `[0, 255]` (a `[0, 1]` image is scaled, a `[0, 255]` image passes
/// through), so it is neither floored to black nor saturated to white.
pub fn decode_image(value: &Value, src_range: Option<(f64, f64)>) -> Result<Tensor, ApplyError> {
    let tensor = match value {
        Value::Tensor(tensor) => tensor.clone(),
        Value::Bytes(raw) => {
            // Preserve the encoded image's native channel count (grayscale -> 1,
            // luma+alpha -> 2, RGB -> 3, RGBA -> 4) rather than forcing RGB: a
            // forced 3-channel decode silently feeds a grayscale (channels=1) or
            // RGBA (channels=4) model a wrong-shaped tensor that the resolver's
            // declared-channel check cannot catch. Now the byte path matches the
            // array path -- both carry the env's actual channels.
            let decoded = image::load_from_memory(raw)
                .map_err(|err| ApplyError::new(format!("could not decode image bytes: {err}")))?;
            let (width, height) = (decoded.width(), decoded.height());
            let (pixels, channels) = match decoded.color().channel_count() {
                1 => (decoded.to_luma8().into_raw(), 1i64),
                2 => (decoded.to_luma_alpha8().into_raw(), 2i64),
                4 => (decoded.to_rgba8().into_raw(), 4i64),
                _ => (decoded.to_rgb8().into_raw(), 3i64),
            };
            value::tensor_from_u8(vec![i64::from(height), i64::from(width), channels], pixels)
        }
        _ => {
            return Err(ApplyError::new(
                "expected an image array observation value or encoded image bytes".to_owned(),
            ));
        }
    };
    Ok(value::tensor_from_u8(
        tensor.shape().to_vec(),
        value::to_u8_pixels(&tensor, src_range),
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

/// Resize an HWC uint8 image to `(height, width)`, reconciling an aspect-ratio
/// change per `fit`. When the aspects already match, every mode is the same
/// uniform scale as a plain stretch.
fn fit_resize(
    tensor: &Tensor,
    height: u32,
    width: u32,
    resample: &str,
    fit: FitMode,
) -> Result<Tensor, ApplyError> {
    let (src_height, src_width, _) = image_dims(tensor)?;
    let (target_height, target_width) = (height as usize, width as usize);
    match fit {
        FitMode::Stretch => resize_image(tensor, height, width, resample),
        FitMode::Crop => {
            // Cover: scale uniformly so both axes reach the target, then crop.
            let scale = (target_height as f64 / src_height as f64)
                .max(target_width as f64 / src_width as f64);
            let scaled_height = ((src_height as f64 * scale).round() as usize).max(target_height);
            let scaled_width = ((src_width as f64 * scale).round() as usize).max(target_width);
            let scaled = resize_image(tensor, scaled_height as u32, scaled_width as u32, resample)?;
            crop_center(&scaled, target_height, target_width)
        }
        FitMode::Pad => {
            // Contain: scale uniformly so both axes fit within the target, then pad.
            let scale = (target_height as f64 / src_height as f64)
                .min(target_width as f64 / src_width as f64);
            let scaled_height =
                ((src_height as f64 * scale).round() as usize).clamp(1, target_height);
            let scaled_width = ((src_width as f64 * scale).round() as usize).clamp(1, target_width);
            let scaled = resize_image(tensor, scaled_height as u32, scaled_width as u32, resample)?;
            pad_center(&scaled, target_height, target_width)
        }
    }
}

/// Center-crop an HWC uint8 image to `(height, width)`; the source must be at
/// least that size in both axes (the cover-scale in [`fit_resize`] guarantees it).
fn crop_center(tensor: &Tensor, height: usize, width: usize) -> Result<Tensor, ApplyError> {
    let (src_height, src_width, channels) = image_dims(tensor)?;
    let off_row = (src_height - height) / 2;
    let off_col = (src_width - width) / 2;
    let mut indices = Vec::with_capacity(height * width * channels);
    for row in 0..height {
        for col in 0..width {
            let src = ((off_row + row) * src_width + (off_col + col)) * channels;
            indices.extend(src..src + channels);
        }
    }
    Ok(value::gather(
        tensor,
        &indices,
        value::shape_i64(&[height, width, channels]),
    ))
}

/// Center-pad an HWC uint8 image to `(height, width)` with zeros (black bars);
/// the source must fit within the target (the contain-scale guarantees it).
fn pad_center(tensor: &Tensor, height: usize, width: usize) -> Result<Tensor, ApplyError> {
    let (src_height, src_width, channels) = image_dims(tensor)?;
    let off_row = (height - src_height) / 2;
    let off_col = (width - src_width) / 2;
    let data = value::u8_pixels(tensor)?;
    let mut out = vec![0u8; height * width * channels];
    for row in 0..src_height {
        for col in 0..src_width {
            let dst = ((off_row + row) * width + (off_col + col)) * channels;
            let src = (row * src_width + col) * channels;
            out[dst..dst + channels].copy_from_slice(&data[src..src + channels]);
        }
    }
    Ok(value::tensor_from_u8(
        value::shape_i64(&[height, width, channels]),
        out,
    ))
}

/// Cast an image to `dtype`, optionally mapping 8-bit values into a target
/// range. `None` skips normalization; `Some((low, high))` maps `[0, 255]` into
/// `[low, high]` (the conventional `Some((0.0, 1.0))` is the old `/255`).
pub fn finalize_dtype(
    tensor: &Tensor,
    dtype: &str,
    normalize: Option<(f64, f64)>,
) -> Result<Tensor, ApplyError> {
    let target = DType::from_name(dtype).ok_or_else(|| {
        ApplyError::new(format!("unsupported dtype {dtype:?} for an image output"))
    })?;
    if let Some((low, high)) = normalize {
        let (low, high) = (low as f32, high as f32);
        let scaled: Vec<f32> = value::to_f32_vec(tensor)
            .into_iter()
            .map(|value| low + (value / 255.0) * (high - low))
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
    use image::ImageEncoder;

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
    fn encoded_image_bytes_decode_to_rgb_tensor() {
        let mut encoded = Vec::new();
        image::codecs::png::PngEncoder::new(&mut encoded)
            .write_image(&[10, 20, 30], 1, 1, image::ColorType::Rgb8.into())
            .expect("encode png");

        let decoded = decode_image(&Value::Bytes(encoded), None).expect("decode");

        assert_eq!(decoded.shape(), &[1, 1, 3]);
        assert_eq!(decoded.to_contiguous_bytes().as_ref(), [10u8, 20, 30]);
    }

    #[test]
    fn encoded_grayscale_bytes_decode_to_single_channel() {
        // A grayscale-encoded image must decode to a 1-channel tensor, not be
        // silently expanded to 3 channels (which would feed a channels=1 model a
        // wrong-shaped input the resolver's declared-channel check cannot catch).
        let mut encoded = Vec::new();
        image::codecs::png::PngEncoder::new(&mut encoded)
            .write_image(&[42], 1, 1, image::ColorType::L8.into())
            .expect("encode gray png");

        let decoded = decode_image(&Value::Bytes(encoded), None).expect("decode");

        assert_eq!(decoded.shape(), &[1, 1, 1]);
        assert_eq!(decoded.to_contiguous_bytes().as_ref(), [42u8]);
    }

    #[test]
    fn encoded_rgba_bytes_preserve_the_alpha_channel() {
        // RGBA must decode to 4 channels, not drop alpha to 3.
        let mut encoded = Vec::new();
        image::codecs::png::PngEncoder::new(&mut encoded)
            .write_image(&[10, 20, 30, 128], 1, 1, image::ColorType::Rgba8.into())
            .expect("encode rgba png");

        let decoded = decode_image(&Value::Bytes(encoded), None).expect("decode");

        assert_eq!(decoded.shape(), &[1, 1, 4]);
        assert_eq!(decoded.to_contiguous_bytes().as_ref(), [10u8, 20, 30, 128]);
    }

    #[test]
    fn invalid_encoded_image_bytes_are_rejected() {
        let err = decode_image(&Value::Bytes(b"not an image".to_vec()), None).unwrap_err();
        assert!(err.message.contains("could not decode image bytes"));
    }

    #[test]
    fn resize_rejects_zero_dimensions() {
        let image = value::tensor_from_u8(vec![0, 2, 3], Vec::new());
        let error = resize_image(&image, 4, 4, "bilinear").expect_err("zero dim");
        assert!(error.to_string().contains("zero dimension"), "{error}");
    }

    #[test]
    fn crop_center_takes_the_middle() {
        // A 1x4 row cropped to 1x2 keeps the middle two pixels (offset 1).
        let image = value::tensor_from_u8(vec![1, 4, 1], vec![10, 20, 30, 40]);
        let cropped = crop_center(&image, 1, 2).expect("crop");
        assert_eq!(cropped.shape(), &[1, 2, 1]);
        assert_eq!(cropped.to_contiguous_bytes().as_ref(), [20u8, 30]);
    }

    #[test]
    fn pad_center_places_image_in_a_zero_field() {
        // A 1x1 pixel padded to 3x3 sits centered, black (zero) around it.
        let image = value::tensor_from_u8(vec![1, 1, 1], vec![255]);
        let padded = pad_center(&image, 3, 3).expect("pad");
        assert_eq!(padded.shape(), &[3, 3, 1]);
        assert_eq!(
            padded.to_contiguous_bytes().as_ref(),
            [0u8, 0, 0, 0, 255, 0, 0, 0, 0]
        );
    }

    #[test]
    fn fit_pad_letterboxes_into_a_square() {
        // 1x2 into 2x2 with pad: contain-scale is 1 (keeps 1x2), then a black
        // row is added below -> the image sits in the top row.
        let image = value::tensor_from_u8(vec![1, 2, 1], vec![100, 200]);
        let out = fit_resize(&image, 2, 2, "bilinear", FitMode::Pad).expect("fit");
        assert_eq!(out.shape(), &[2, 2, 1]);
        assert_eq!(out.to_contiguous_bytes().as_ref(), [100u8, 200, 0, 0]);
    }

    #[test]
    fn fit_crop_covers_then_center_crops_to_target_shape() {
        // 1x4 into 2x2 with crop: cover-scale 2 -> 2x8, center-crop -> 2x2.
        let image = value::tensor_from_u8(vec![1, 4, 1], vec![10, 20, 30, 40]);
        let out = fit_resize(&image, 2, 2, "bilinear", FitMode::Crop).expect("fit");
        assert_eq!(out.shape(), &[2, 2, 1]);
    }

    #[test]
    fn fit_stretch_matches_a_plain_resize() {
        let image = value::tensor_from_u8(vec![1, 2, 1], vec![100, 200]);
        let stretched = fit_resize(&image, 2, 4, "bilinear", FitMode::Stretch).expect("fit");
        let plain = resize_image(&image, 2, 4, "bilinear").expect("resize");
        assert_eq!(
            stretched.to_contiguous_bytes().as_ref(),
            plain.to_contiguous_bytes().as_ref()
        );
    }

    #[test]
    fn normalize_default_range_is_the_unit_interval() {
        let image = value::tensor_from_u8(vec![1, 2, 1], vec![0, 255]);
        let out = finalize_dtype(&image, "float32", Some((0.0, 1.0))).expect("normalize");
        assert_eq!(value::to_f32_vec(&out), vec![0.0, 1.0]);
    }

    #[test]
    fn normalize_maps_into_a_declared_signed_range() {
        // [0, 255] -> [-1, 1]: 0 -> -1, 255 -> 1, ~mid -> ~0.
        let image = value::tensor_from_u8(vec![1, 3, 1], vec![0, 128, 255]);
        let out = finalize_dtype(&image, "float32", Some((-1.0, 1.0))).expect("normalize");
        let floats = value::to_f32_vec(&out);
        assert!((floats[0] + 1.0).abs() < 1e-6, "got: {floats:?}");
        assert!((floats[2] - 1.0).abs() < 1e-6, "got: {floats:?}");
        assert!(floats[1].abs() < 0.01, "got: {floats:?}");
    }

    #[test]
    fn zero_fill_synthesizes_a_black_frame() {
        let plan = ImagePlan {
            placement: crate::path::NodePath::root().push_key("image"),
            source: crate::path::NodePath::root(),
            src_layout: ImageLayout::Hwc,
            dst_layout: ImageLayout::Hwc,
            flip: false,
            size: None,
            fit: FitMode::Stretch,
            resample: "bilinear_aa".to_owned(),
            dtype: "uint8".to_owned(),
            normalize: None,
            lead_dims: 0,
            src_range: None,
            stack: 1,
            zero_fill: Some((2, 2, 3)),
            absent_fill: 0,
        };
        let Value::Tensor(tensor) =
            apply_image(&plan, &std::collections::BTreeMap::new()).expect("zero-fill")
        else {
            panic!("expected a tensor")
        };
        assert_eq!(tensor.shape(), &[2, 2, 3]);
        assert!(
            tensor
                .to_contiguous_bytes()
                .as_ref()
                .iter()
                .all(|&b| b == 0)
        );
    }

    #[test]
    fn zero_fill_uses_the_absent_fill_level() {
        let plan = ImagePlan {
            placement: crate::path::NodePath::root().push_key("image"),
            source: crate::path::NodePath::root(),
            src_layout: ImageLayout::Hwc,
            dst_layout: ImageLayout::Hwc,
            flip: false,
            size: None,
            fit: FitMode::Stretch,
            resample: "bilinear_aa".to_owned(),
            dtype: "uint8".to_owned(),
            normalize: None,
            lead_dims: 0,
            src_range: None,
            stack: 1,
            zero_fill: Some((2, 2, 3)),
            absent_fill: 128,
        };
        let Value::Tensor(tensor) =
            apply_image(&plan, &std::collections::BTreeMap::new()).expect("zero-fill")
        else {
            panic!("expected a tensor")
        };
        assert!(
            tensor
                .to_contiguous_bytes()
                .as_ref()
                .iter()
                .all(|&b| b == 128)
        );
    }
}
