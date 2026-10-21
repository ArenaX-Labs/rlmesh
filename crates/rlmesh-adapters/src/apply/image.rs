//! Image operations used by resolved adapters.
//!
//! The pixel operations (resize kernels, aspect fits, crops, the 180°
//! rotation, BGR, normalisation) are the shared ones in
//! `rlmesh_session_core::spec::image`, which v2 bindings run too; this module
//! carries pixels in `rlmesh_spaces::Tensor` and adds what only v1 has
//! (decoding, the JPEG round trip, layouts and lead axes).

use std::collections::BTreeMap;

use rlmesh_session_core::spec::image::{self as shared, Image, normalise_table};
use rlmesh_session_core::spec::plan::Fit;
use rlmesh_session_core::spec::policy::Resample;
use rlmesh_spaces::{DType, Tensor, dtype_size};

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
    // Before the crop/resize: a training pipeline that stored JPEGs encoded the
    // camera frame it was handed, then cropped and resized the decoded result.
    if let Some(quality) = plan.jpeg_quality {
        image = jpeg_roundtrip(&image, quality)?;
    }
    // A `slice` crop is an integer center cut taken before the resize; a `zoom`
    // crop is the fractional box the resize itself samples through, so it rides
    // along into `fit_resize` instead of costing a second resampling pass.
    let mut zoom = None;
    if let Some(crop) = &plan.crop {
        if crop.slice {
            let (height, width, _) = image_dims(&image)?;
            image = crop_center(
                &image,
                crop_cut(height, crop.fraction),
                crop_cut(width, crop.fraction),
            )?;
        } else {
            zoom = Some(crop.fraction);
        }
    }
    let size = match (plan.size, zoom) {
        (Some(size), _) => Some(size),
        // A zoom with no declared target resamples the box back to the frame's
        // own size -- that magnification is the whole point of the crop.
        (None, Some(_)) => {
            let (height, width, _) = image_dims(&image)?;
            Some((height as u32, width as u32))
        }
        (None, None) => None,
    };
    if let Some((height, width)) = size {
        image = fit_resize(&image, height, width, &plan.resample, plan.fit, zoom)?;
    }
    finalize_image(image, plan)
}

/// The integer center-cut length a `slice` crop keeps from a `src`-long axis.
/// Shared by apply (which cuts the frame it is handed) and the resolver (which
/// precomputes the same box at the env's declared resolution for describe), so
/// the two can never print and cut different numbers.
pub(crate) fn crop_cut(src: usize, fraction: f64) -> usize {
    shared::crop_cut(src, fraction)
}

/// The shared tail both image paths (real and zero-fill) end with: map the HWC
/// uint8 frame into the model's dtype/range, transpose to the model's layout, and
/// prepend any leading axes. Kept in one place so a normalize/layout/lead policy
/// change cannot silently diverge between the two paths.
fn finalize_image(image: Tensor, plan: &ImagePlan) -> Result<Value, ApplyError> {
    let image = if plan.swap_rb {
        swap_rb(&image)?
    } else {
        image
    };
    if let Some(image) = finalize_camera_chw(&image, plan) {
        return Ok(Value::Tensor(add_lead_dims(image, plan.lead_dims)));
    }
    let image = finalize_dtype(&image, &plan.dtype, plan.normalize)?;
    let image = to_layout(&image, ImageLayout::Hwc, plan.dst_layout)?;
    Ok(Value::Tensor(add_lead_dims(image, plan.lead_dims)))
}

/// The camera fast path: a uint8 HWC frame going to a normalized float32 CHW
/// tensor, which is what every vision policy asks for. One pass reads each
/// source byte once and writes its f32 straight to the transposed offset, so
/// neither the HWC float staging tensor nor the transpose index vector exists.
/// Byte-identical to `finalize_dtype` + `to_layout` (same f32 expression, see
/// there); `None` hands any other plan to the generic tail.
fn finalize_camera_chw(image: &Tensor, plan: &ImagePlan) -> Option<Tensor> {
    let (low, high) = plan.normalize?;
    if image.dtype() != DType::Uint8
        || plan.dtype != "float32"
        || plan.dst_layout != ImageLayout::Chw
        || image.shape().len() != 3
    {
        return None;
    }
    let table = normalise_table(low, high);
    let shape = value::shape_usize(image);
    let (height, width, channels) = (shape[0], shape[1], shape[2]);
    let plane = height * width;
    let bytes = image.to_contiguous_bytes();
    let mut out = vec![0f32; bytes.len()];
    for (pixel, px) in bytes.chunks_exact(channels.max(1)).enumerate() {
        for (channel, &v) in px.iter().enumerate() {
            out[channel * plane + pixel] = table[v as usize];
        }
    }
    let shape = value::shape_i64(&[channels, height, width]);
    Some(value::tensor_from_f32(shape, &out))
}

/// Synthesize the model input for an optional image the env did not provide: a
/// frame filled with the spec's `fill` level (black by default), run
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
        vec![plan.fill; height * width * channels],
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
    // Fast path: an already-uint8, already-contiguous tensor is the pipeline's
    // canonical pixel buffer as-is. `to_u8_pixels` would copy every byte out and
    // `tensor_from_u8` re-wrap it into an identical tensor, so skip both.
    if tensor.dtype() == DType::Uint8 && tensor.is_contiguous() {
        return Ok(tensor);
    }
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
///
/// A true 180° rotation (rows AND columns reversed), intentional — NOT a
/// vertical flip. `upside_down` is a both-ends *declaration*: the env tag and
/// the model `Image` each state their stored orientation, and this fires only
/// when they differ (`flip = env.upside_down != model.upside_down`, see
/// `resolver/image.rs`). Don't "fix" this into a vertical flip — that would
/// silently mirror every frame left-right for everyone relying on rot180.
pub fn flip_180(tensor: &Tensor) -> Result<Tensor, ApplyError> {
    Ok(from_image(to_image(tensor)?.rotate180()))
}

/// An HWC uint8 tensor as the shared [`Image`].
fn to_image(tensor: &Tensor) -> Result<Image<u8>, ApplyError> {
    let (height, width, channels) = image_dims(tensor)?;
    Ok(Image {
        height,
        width,
        channels,
        data: value::u8_pixels(tensor)?,
    })
}

/// The shared [`Image`] as an HWC uint8 tensor.
fn from_image(image: Image<u8>) -> Tensor {
    value::tensor_from_u8(
        value::shape_i64(&[image.height, image.width, image.channels]),
        image.data,
    )
}

/// Swap the red and blue channels of an HWC image (`channel_order = "bgr"`).
///
/// A 3-channel op by definition: there is no meaningful red/blue pair in a
/// grayscale or RGBA frame, so a wrong-shaped source fails loudly rather than
/// reordering whatever happens to sit at those offsets.
pub fn swap_rb(tensor: &Tensor) -> Result<Tensor, ApplyError> {
    let (_, _, channels) = image_dims(tensor)?;
    if channels != 3 {
        return Err(ApplyError::new(format!(
            "channel_order \"bgr\" needs a 3-channel image, got {channels} channel(s)"
        )));
    }
    Ok(from_image(to_image(tensor)?.bgr()))
}

/// Encode an HWC 8-bit RGB frame as JPEG at `quality` and decode it back
/// (`jpeg_quality`), reproducing the codec artifacts a model trained on stored
/// JPEGs saw.
///
/// The profile is part of the v1 contract, because a different one is a
/// different picture: **baseline sequential, 4:2:0 chroma subsampling with
/// box-averaged (libjpeg `h2v2_downsample`) chroma, the standard IJG
/// quantization tables scaled by `quality`, and the standard IJG Huffman
/// tables** -- what TensorFlow's `encode_jpeg` and Pillow's `save(..., "JPEG")`
/// write by default. Encoding is `jpeg-encoder` (pinned exactly); decoding is
/// the `image` crate's JPEG reader, so only one codec is vendored.
///
/// A 3-channel op: JPEG's subsampling is defined on YCbCr, which a grayscale or
/// RGBA frame has no equivalent of, so a wrong-shaped frame fails loudly
/// (resolution rejects it first -- see `resolver::image`).
pub fn jpeg_roundtrip(tensor: &Tensor, quality: u8) -> Result<Tensor, ApplyError> {
    let encoded = jpeg_encode(tensor, quality)?;
    let decoded = image::load_from_memory_with_format(&encoded, image::ImageFormat::Jpeg)
        .map_err(|err| ApplyError::new(format!("could not decode the jpeg image: {err}")))?;
    Ok(value::tensor_from_u8(
        tensor.shape().to_vec(),
        decoded.to_rgb8().into_raw(),
    ))
}

/// The encode half of [`jpeg_roundtrip`] (split out so the profile itself can
/// be asserted against the encoded stream's headers).
fn jpeg_encode(tensor: &Tensor, quality: u8) -> Result<Vec<u8>, ApplyError> {
    let (height, width, channels) = image_dims(tensor)?;
    if channels != 3 {
        return Err(ApplyError::new(format!(
            "jpeg_quality needs a 3-channel image, got {channels} channel(s)"
        )));
    }
    // JPEG's frame header carries 16-bit dimensions; a larger frame has no
    // encoding, so say so rather than truncating the cast.
    let (Ok(encoded_width), Ok(encoded_height)) = (u16::try_from(width), u16::try_from(height))
    else {
        return Err(ApplyError::new(format!(
            "jpeg_quality cannot encode a {height}x{width} frame; JPEG dimensions are 16-bit"
        )));
    };
    let pixels = value::u8_pixels(tensor)?;
    let mut encoded: Vec<u8> = Vec::new();
    let mut encoder = jpeg_encoder::Encoder::new(&mut encoded, quality);
    // The crate picks 4:4:4 above quality 90; upstream (libjpeg, and so
    // TensorFlow and Pillow) subsamples at every quality, so pin it.
    encoder.set_sampling_factor(jpeg_encoder::SamplingFactor::R_4_2_0);
    encoder.set_chroma_subsampling_method(jpeg_encoder::ChromaSubsamplingMethod::Average);
    encoder
        .encode(
            &pixels,
            encoded_width,
            encoded_height,
            jpeg_encoder::ColorType::Rgb,
        )
        .map_err(|err| ApplyError::new(format!("could not jpeg-encode the image: {err}")))?;
    Ok(encoded)
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
    // A typed transpose: walk the destination in order and copy one
    // `itemsize`-byte element per step, no index vector.
    let size = dtype_size(tensor.dtype());
    let bytes = tensor.to_contiguous_bytes();
    let mut out = vec![0u8; bytes.len()];
    let shape = match target {
        ImageLayout::Chw => {
            let (height, width, channels) = image_dims(tensor)?;
            let plane = height * width;
            for channel in 0..channels {
                for pixel in 0..plane {
                    let src = (pixel * channels + channel) * size;
                    let dst = (channel * plane + pixel) * size;
                    out[dst..dst + size].copy_from_slice(&bytes[src..src + size]);
                }
            }
            vec![channels, height, width]
        }
        ImageLayout::Hwc => {
            let (channels, height, width) = image_dims(tensor)?;
            let plane = height * width;
            for pixel in 0..plane {
                for channel in 0..channels {
                    let src = (channel * plane + pixel) * size;
                    let dst = (pixel * channels + channel) * size;
                    out[dst..dst + size].copy_from_slice(&bytes[src..src + size]);
                }
            }
            vec![height, width, channels]
        }
    };
    Tensor::from_vec(out, value::shape_i64(&shape), tensor.dtype())
        .map_err(|err| ApplyError::new(err.to_string()))
}

/// The resample algorithms `resize_roi` accepts. Un-suffixed names have
/// cv2/torch semantics, `_aa` names PIL's; bare `bicubic`/`lanczos3` are
/// deliberately absent, so a spec naming one fails resolution rather than
/// silently getting the other library's kernel.
pub const RESAMPLES: [&str; 5] = [
    "bilinear",
    "bilinear_aa",
    "bicubic_aa",
    "lanczos3_aa",
    "area",
];

/// The kernel a resample name names, and the frame's dimensions, refusing a
/// zero dimension on either side.
fn prepare(
    tensor: &Tensor,
    (height, width): (u32, u32),
    resample: &str,
) -> Result<(Image<u8>, Resample), ApplyError> {
    let (src_height, src_width, _) = image_dims(tensor)?;
    if height == 0 || width == 0 || src_height == 0 || src_width == 0 {
        return Err(ApplyError::new(format!(
            "cannot resize an image with a zero dimension (source \
             {src_height}x{src_width}, target {height}x{width})"
        )));
    }
    let kernel = Resample::parse(resample)
        .ok_or_else(|| ApplyError::new(format!("unsupported resample {resample:?}")))?;
    Ok((to_image(tensor)?, kernel))
}

/// Resize an HWC uint8 image to `(height, width)`, sampling only the center
/// box `zoom` names (a side fraction of each axis) — `None` is the whole frame.
///
/// The box is resampled *straight* to the target by the declared kernel, so a
/// zoom crop costs no extra pass and no intermediate rounding; this is PIL's
/// `Image.resize(size, box=...)`, which is the anchor the vectors pin.
#[cfg(test)]
fn resize_roi(
    tensor: &Tensor,
    size: (u32, u32),
    zoom: Option<f64>,
    resample: &str,
) -> Result<Tensor, ApplyError> {
    let (image, kernel) = prepare(tensor, size, resample)?;
    let rows = shared::Roi::zoom(image.height, size.0 as usize, zoom);
    let cols = shared::Roi::zoom(image.width, size.1 as usize, zoom);
    Ok(from_image(image.resize_roi(rows, cols, kernel)))
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
    zoom: Option<f64>,
) -> Result<Tensor, ApplyError> {
    let (image, kernel) = prepare(tensor, (height, width), resample)?;
    let fit = match fit {
        FitMode::Stretch => Fit::Stretch,
        FitMode::Crop => Fit::Crop,
        FitMode::Pad => Fit::Pad,
    };
    Ok(from_image(shared::fit_resize(
        &image,
        (height as usize, width as usize),
        Some(fit),
        kernel,
        zoom,
    )))
}

/// Center-crop an HWC uint8 image to `(height, width)`; the source must be at
/// least that size in both axes.
fn crop_center(tensor: &Tensor, height: usize, width: usize) -> Result<Tensor, ApplyError> {
    Ok(from_image(to_image(tensor)?.crop_center(height, width)))
}

/// Center-pad an HWC uint8 image to `(height, width)` with zeros (black bars);
/// the source must fit within the target.
#[cfg(test)]
fn pad_center(tensor: &Tensor, height: usize, width: usize) -> Result<Tensor, ApplyError> {
    Ok(from_image(to_image(tensor)?.pad_center(height, width, 0)))
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
        if tensor.dtype() == DType::Uint8 && target == DType::Float32 {
            // The camera case (uint8 in, float32 out): one table lookup per
            // element, written straight as little-endian f32. The table holds
            // exactly `low + (v / 255) * (high - low)` in f32, which is what the
            // generic path computes before widening to f64 and narrowing back,
            // so the bytes are identical to it.
            let table = normalise_table(low, high).map(f32::to_le_bytes);
            let bytes = tensor.to_contiguous_bytes();
            let mut out = Vec::with_capacity(bytes.len() * 4);
            for &v in bytes.iter() {
                out.extend_from_slice(&table[v as usize]);
            }
            return Tensor::from_vec(out, tensor.shape().to_vec(), target)
                .map_err(|err| ApplyError::new(err.to_string()));
        }
        // Fuse normalize + cast: scale in f32 (the exact arithmetic of the old
        // Float32 staging tensor), widen to f64, and encode straight to the
        // target dtype — skipping the intermediate float32 tensor and its
        // decode/re-encode round-trip while preserving the bit-identical result.
        let (low, high) = (low as f32, high as f32);
        let scaled: Vec<f64> = value::to_f32_vec(tensor)
            .into_iter()
            .map(|value| f64::from(low + (value / 255.0) * (high - low)))
            .collect();
        return value::encode_f64_to(scaled, tensor.shape().to_vec(), target);
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

    /// The generic element-by-element implementations the typed kernels
    /// replaced, kept only to prove the kernels are byte-identical to them.
    mod generic {
        use super::*;

        pub fn flip_180(tensor: &Tensor) -> Tensor {
            let (height, width, channels) = image_dims(tensor).unwrap();
            let mut indices = Vec::with_capacity(tensor.numel());
            for row in 0..height {
                for col in 0..width {
                    let src = ((height - 1 - row) * width + (width - 1 - col)) * channels;
                    indices.extend(src..src + channels);
                }
            }
            value::gather(tensor, &indices, tensor.shape().to_vec())
        }

        pub fn to_layout(tensor: &Tensor, target: ImageLayout) -> Tensor {
            let mut indices = Vec::with_capacity(tensor.numel());
            let shape = match target {
                ImageLayout::Chw => {
                    let (height, width, channels) = image_dims(tensor).unwrap();
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
                    let (channels, height, width) = image_dims(tensor).unwrap();
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
            value::gather(tensor, &indices, value::shape_i64(&shape))
        }

        pub fn normalize_f32(tensor: &Tensor, (low, high): (f64, f64)) -> Tensor {
            let (low, high) = (low as f32, high as f32);
            let scaled: Vec<f64> = value::to_f32_vec(tensor)
                .into_iter()
                .map(|value| f64::from(low + (value / 255.0) * (high - low)))
                .collect();
            value::encode_f64_to(scaled, tensor.shape().to_vec(), DType::Float32).unwrap()
        }
    }

    /// Deterministic pseudo-random uint8 HWC frame (xorshift), so the
    /// equivalence checks cover arbitrary pixel values, not gradients.
    fn noise_frame(height: usize, width: usize, channels: usize, seed: u32) -> Tensor {
        let mut state = seed | 1;
        let data: Vec<u8> = (0..height * width * channels)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state & 0xff) as u8
            })
            .collect();
        value::tensor_from_u8(value::shape_i64(&[height, width, channels]), data)
    }

    #[test]
    fn typed_kernels_match_the_generic_implementations_byte_for_byte() {
        let mut seed = 7;
        for &(height, width) in &[(1, 1), (2, 3), (7, 5), (64, 64), (256, 256)] {
            for &channels in &[1usize, 3, 4] {
                seed += 1;
                let frame = noise_frame(height, width, channels, seed);
                assert_eq!(flip_180(&frame).unwrap(), generic::flip_180(&frame));
                for &range in &[(0.0, 1.0), (-1.0, 1.0), (0.0, 255.0)] {
                    let typed = finalize_dtype(&frame, "float32", Some(range)).unwrap();
                    assert_eq!(typed, generic::normalize_f32(&frame, range), "{range:?}");
                    // the float image is what the layout transpose sees in practice
                    let chw = to_layout(&typed, ImageLayout::Hwc, ImageLayout::Chw).unwrap();
                    assert_eq!(chw, generic::to_layout(&typed, ImageLayout::Chw));
                    assert_eq!(
                        to_layout(&chw, ImageLayout::Chw, ImageLayout::Hwc).unwrap(),
                        typed
                    );
                }
                let chw_u8 = to_layout(&frame, ImageLayout::Hwc, ImageLayout::Chw).unwrap();
                assert_eq!(chw_u8, generic::to_layout(&frame, ImageLayout::Chw));
            }
        }
    }

    /// `cargo test -p rlmesh-adapters camera_pipeline_timing -- --ignored --nocapture`:
    /// ms per 256x256x3 camera frame through flip + normalize + chw, typed
    /// kernels against the generic implementations they replaced.
    #[test]
    #[ignore = "timing, not correctness: run with --release --nocapture"]
    #[allow(clippy::print_stdout, reason = "the point of the test is the report")]
    fn camera_pipeline_timing() {
        let frame = noise_frame(256, 256, 3, 11);
        let reps = 20;
        let typed = std::time::Instant::now();
        for _ in 0..reps {
            let f = flip_180(&frame).unwrap();
            let n = finalize_dtype(&f, "float32", Some((0.0, 1.0))).unwrap();
            let _ = to_layout(&n, ImageLayout::Hwc, ImageLayout::Chw).unwrap();
        }
        let typed = typed.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let plan_like = |img: &Tensor| {
            let (low, high) = (0.0f32, 1.0f32);
            let table: [f32; 256] =
                std::array::from_fn(|v| low + (v as f32 / 255.0) * (high - low));
            let bytes = img.to_contiguous_bytes();
            let plane = 256 * 256;
            let mut out = vec![0f32; bytes.len()];
            for (pixel, px) in bytes.as_chunks::<3>().0.iter().enumerate() {
                for (channel, &v) in px.iter().enumerate() {
                    out[channel * plane + pixel] = table[v as usize];
                }
            }
            value::tensor_from_f32(vec![3, 256, 256], &out)
        };
        let fused = std::time::Instant::now();
        for _ in 0..reps {
            let f = flip_180(&frame).unwrap();
            let _ = plan_like(&f);
        }
        let fused = fused.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let generic = std::time::Instant::now();
        for _ in 0..reps {
            let f = generic::flip_180(&frame);
            let n = generic::normalize_f32(&f, (0.0, 1.0));
            let _ = generic::to_layout(&n, ImageLayout::Chw);
        }
        let generic = generic.elapsed().as_secs_f64() * 1e3 / reps as f64;
        println!(
            "camera frame flip+normalize+chw: fused {fused:.3} ms, typed {typed:.3} ms, generic {generic:.3} ms"
        );
    }

    /// The whole camera plan (flip + pad-to-same-size + normalize + chw) through
    /// `apply_image` — the fused fast path — against the generic kernels composed
    /// by hand, with and without the fast path's eligibility (hwc output, swap_rb).
    #[test]
    fn the_camera_plan_matches_the_generic_kernels_end_to_end() {
        let camera_plan = |dst_layout: ImageLayout, swap_rb: bool| ImagePlan {
            placement: crate::path::NodePath::root().push_key("image"),
            source: crate::path::NodePath::root().push_key("cam"),
            src_layout: ImageLayout::Hwc,
            dst_layout,
            flip: true,
            size: Some((256, 256)),
            fit: FitMode::Pad,
            resample: "bilinear_aa".to_owned(),
            dtype: "float32".to_owned(),
            normalize: Some((0.0, 1.0)),
            lead_dims: 0,
            src_range: None,
            stack: 1,
            zero_fill: None,
            fill: 0,
            crop: None,
            jpeg_quality: None,
            swap_rb,
            render: None,
            offsets: None,
            stack_pad: crate::spec::StackPad::First,
            frame_bytes: 0,
            role_rebound: None,
            part: None,
        };
        let frame = noise_frame(256, 256, 3, 23);
        let obs =
            std::collections::BTreeMap::from([("cam".to_owned(), Value::Tensor(frame.clone()))]);
        for (dst_layout, swap_rb) in [
            (ImageLayout::Chw, false),
            (ImageLayout::Chw, true),
            (ImageLayout::Hwc, false),
        ] {
            let plan = camera_plan(dst_layout, swap_rb);
            let Value::Tensor(fast) = apply_image(&plan, &obs).expect("camera plan applies") else {
                panic!("expected a tensor")
            };
            let mut expected = generic::flip_180(&frame);
            if swap_rb {
                expected = swap_rb_generic(&expected);
            }
            let expected = generic::normalize_f32(&expected, (0.0, 1.0));
            let expected = match dst_layout {
                ImageLayout::Chw => generic::to_layout(&expected, ImageLayout::Chw),
                ImageLayout::Hwc => expected,
            };
            assert_eq!(fast, expected, "{dst_layout:?} swap_rb={swap_rb}");
        }
    }

    fn swap_rb_generic(tensor: &Tensor) -> Tensor {
        let bytes = tensor.to_contiguous_bytes();
        let mut out = bytes.to_vec();
        for px in out.as_chunks_mut::<3>().0 {
            px.swap(0, 2);
        }
        value::tensor_from_u8(tensor.shape().to_vec(), out)
    }

    #[test]
    fn pad_to_the_same_size_is_the_frame_itself() {
        let frame = noise_frame(9, 11, 3, 3);
        assert_eq!(pad_center(&frame, 9, 11).unwrap(), frame);
        let padded = pad_center(&frame, 12, 12).unwrap();
        assert_eq!(padded.shape(), &[12, 12, 3]);
    }

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
    fn every_named_resample_resizes() {
        // RESAMPLES is the resolver's accept-set; a name in it that resize
        // rejects would pass resolution and then fail per frame at serve time.
        let image = value::tensor_from_u8(vec![4, 4, 1], (0u8..16).collect());
        for name in RESAMPLES {
            resize_roi(&image, (2, 3), None, name).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
        assert!(resize_roi(&image, (2, 3), None, "bicubic").is_err());
        assert!(resize_roi(&image, (2, 3), None, "lanczos3").is_err());
    }

    #[test]
    fn a_constant_image_resamples_to_the_same_constant() {
        // Partition-of-unity weights in, one rounded value out: no kernel may
        // ring, darken an edge, or drop a tap on a flat field.
        for name in RESAMPLES {
            for (src, dst) in [(8usize, 3usize), (3, 8), (16, 5), (5, 16)] {
                let side = src as i64;
                let image = value::tensor_from_u8(vec![side, side, 3], vec![137; src * src * 3]);
                let out = resize_roi(&image, (dst as u32, dst as u32), None, name).expect("resize");
                assert_eq!(out.shape(), &[dst as i64, dst as i64, 3]);
                assert!(
                    out.to_contiguous_bytes().as_ref().iter().all(|&b| b == 137),
                    "{name} {src}->{dst} did not stay constant"
                );
            }
        }
    }

    #[test]
    fn an_identity_resize_is_byte_identical() {
        let image = value::tensor_from_u8(vec![4, 5, 3], (0u8..60).collect());
        for name in RESAMPLES {
            let out = resize_roi(&image, (4, 5), None, name).expect("resize");
            assert_eq!(
                out.to_contiguous_bytes().as_ref(),
                image.to_contiguous_bytes().as_ref(),
                "{name} changed an identity resize"
            );
        }
    }

    #[test]
    fn area_downscale_averages_the_source_block() {
        // cv2 INTER_AREA at an integer ratio is a plain block mean: the 2x2
        // blocks of a 4x4 ramp average to 2.5, 4.5, 10.5, 12.5.
        let image = value::tensor_from_u8(vec![4, 4, 1], (0u8..16).collect());
        let out = resize_roi(&image, (2, 2), None, "area").expect("resize");
        // round-half-to-even on .5 ties: 2.5 -> 2, 4.5 -> 4, 10.5 -> 10, 12.5 -> 12.
        assert_eq!(out.to_contiguous_bytes().as_ref(), [2u8, 4, 10, 12]);
    }

    #[test]
    fn area_upscale_splits_by_footprint_coverage() {
        // cv2 INTER_AREA upscaling is not the widened filter the `_aa` kernels
        // use: each output pixel covers a sub-pixel span, so a 2x doubling is
        // pure replication, not interpolation.
        let image = value::tensor_from_u8(vec![1, 2, 1], vec![0, 200]);
        let out = resize_roi(&image, (1, 4), None, "area").expect("resize");
        assert_eq!(out.to_contiguous_bytes().as_ref(), [0u8, 0, 200, 200]);
    }

    #[test]
    fn widened_support_is_what_separates_aa_from_plain_bilinear() {
        // The naming rule, made falsifiable: on a 4x downscale the `_aa`
        // triangle's support widens to cover the whole source span, while
        // plain bilinear keeps sampling two taps per axis at 1.5 and 5.5 --
        // so a bright pixel at the edge survives in one and vanishes in the
        // other.
        let mut pixels = vec![0u8; 8];
        pixels[0] = 255;
        let image = value::tensor_from_u8(vec![1, 8, 1], pixels);
        let plain = resize_roi(&image, (1, 2), None, "bilinear").expect("resize");
        let aa = resize_roi(&image, (1, 2), None, "bilinear_aa").expect("resize");
        assert_eq!(plain.to_contiguous_bytes().as_ref(), [0u8, 0]);
        assert!(aa.to_contiguous_bytes()[0] > 0);
    }

    /// Bench gate for the kernels, run by hand: this crate has no criterion
    /// harness, so the cost is reported rather than asserted.
    ///
    ///     cargo test -p rlmesh-adapters --release resample_kernel_timing -- --ignored --nocapture
    #[test]
    #[ignore = "timing, not correctness: run with --release --nocapture"]
    #[allow(clippy::print_stdout, reason = "the point of the test is the report")]
    fn resample_kernel_timing() {
        // Two 256x256 cameras on each of 32 lanes, one step.
        let image = value::tensor_from_u8(
            vec![256, 256, 3],
            (0..256 * 256 * 3).map(|i| (i % 251) as u8).collect(),
        );
        for name in RESAMPLES {
            let started = std::time::Instant::now();
            for _ in 0..2 * 32 {
                resize_roi(&image, (224, 224), None, name).expect("resize");
            }
            println!("{name}: 64 x 256^2 -> 224^2 in {:?}", started.elapsed());
        }
    }

    #[test]
    fn resize_rejects_zero_dimensions() {
        let image = value::tensor_from_u8(vec![0, 2, 3], Vec::new());
        let error = resize_roi(&image, (4, 4), None, "bilinear").expect_err("zero dim");
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
        let out = fit_resize(&image, 2, 2, "bilinear", FitMode::Pad, None).expect("fit");
        assert_eq!(out.shape(), &[2, 2, 1]);
        assert_eq!(out.to_contiguous_bytes().as_ref(), [100u8, 200, 0, 0]);
    }

    #[test]
    fn fit_crop_covers_then_center_crops_to_target_shape() {
        // 1x4 into 2x2 with crop: cover-scale 2 -> 2x8, center-crop -> 2x2.
        let image = value::tensor_from_u8(vec![1, 4, 1], vec![10, 20, 30, 40]);
        let out = fit_resize(&image, 2, 2, "bilinear", FitMode::Crop, None).expect("fit");
        assert_eq!(out.shape(), &[2, 2, 1]);
    }

    #[test]
    fn fit_stretch_matches_a_plain_resize() {
        let image = value::tensor_from_u8(vec![1, 2, 1], vec![100, 200]);
        let stretched = fit_resize(&image, 2, 4, "bilinear", FitMode::Stretch, None).expect("fit");
        let plain = resize_roi(&image, (2, 4), None, "bilinear").expect("resize");
        assert_eq!(
            stretched.to_contiguous_bytes().as_ref(),
            plain.to_contiguous_bytes().as_ref()
        );
    }

    #[test]
    fn swap_rb_reverses_only_the_outer_channels() {
        let image = value::tensor_from_u8(vec![1, 2, 3], vec![1, 2, 3, 4, 5, 6]);
        let out = swap_rb(&image).expect("swap");
        assert_eq!(out.shape(), &[1, 2, 3]);
        assert_eq!(out.to_contiguous_bytes().as_ref(), [3u8, 2, 1, 6, 5, 4]);
        // Swapping twice is the identity -- the op is its own inverse.
        let back = swap_rb(&out).expect("swap");
        assert_eq!(back.to_contiguous_bytes().as_ref(), [1u8, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn swap_rb_rejects_a_non_rgb_frame() {
        let gray = value::tensor_from_u8(vec![1, 2, 1], vec![10, 20]);
        let err = swap_rb(&gray).expect_err("channels");
        assert!(err.message.contains("3-channel"), "got: {}", err.message);
    }

    #[test]
    fn crop_cut_rounds_and_never_empties_the_frame() {
        assert_eq!(crop_cut(480, 2.0 / 3.0), 320);
        assert_eq!(crop_cut(8, 0.9f64.sqrt()), 8); // 7.589 rounds up
        assert_eq!(crop_cut(3, 0.1), 1); // a tiny fraction still keeps a pixel
        assert_eq!(crop_cut(4, 1.0), 4);
    }

    #[test]
    fn a_zoom_crop_magnifies_the_center_of_the_frame() {
        // A horizontal ramp over 0..248. Resampling only its middle half back
        // to the same output width must land every sample inside the box's
        // value range instead of spanning the whole frame's -- that narrowing
        // IS the magnification, and it holds for every kernel.
        //
        // The box is a *resample* window, not a cut: like PIL's
        // `Image.resize(size, box=...)` the filter still reaches past the box
        // edge into the neighbouring pixels, so the bound below has margin for
        // that bleed rather than pinning the box exactly.
        let width = 32usize;
        let image = value::tensor_from_u8(
            value::shape_i64(&[1, width, 1]),
            (0..width).map(|col| (col * 8) as u8).collect(),
        );
        let spread = |tensor: &Tensor| -> (u8, u8) {
            let bytes = tensor.to_contiguous_bytes();
            (
                *bytes.iter().min().expect("non-empty"),
                *bytes.iter().max().expect("non-empty"),
            )
        };
        for name in RESAMPLES {
            let (low, high) = spread(&resize_roi(&image, (1, 8), Some(0.5), name).expect("zoom"));
            assert!(low >= 48 && high <= 208, "{name}: zoomed to {low}..{high}");
            let (low, high) = spread(&resize_roi(&image, (1, 8), None, name).expect("resize"));
            assert!(
                low <= 24 && high >= 224,
                "{name}: unzoomed to {low}..{high}"
            );
        }
    }

    #[test]
    fn a_full_frame_zoom_is_the_same_as_no_zoom() {
        // crop=1.0 names the whole frame: the box degenerates to the full axis,
        // so it must not perturb the resize by even a rounding step.
        let image = value::tensor_from_u8(
            vec![4, 4, 1],
            (0..16).map(|i| (i * 17 % 251) as u8).collect(),
        );
        for name in RESAMPLES {
            let zoomed = resize_roi(&image, (3, 3), Some(1.0), name).expect("zoom");
            let plain = resize_roi(&image, (3, 3), None, name).expect("resize");
            assert_eq!(
                zoomed.to_contiguous_bytes().as_ref(),
                plain.to_contiguous_bytes().as_ref(),
                "{name}"
            );
        }
    }

    /// A natural-looking frame: two smooth gradients plus a few hard edges.
    /// Deliberately not noise -- JPEG is tuned for photographic content, and
    /// white noise would put the PSNR floor somewhere that says nothing about
    /// what a camera frame survives.
    fn natural_frame(height: usize, width: usize) -> Tensor {
        let mut pixels = Vec::with_capacity(height * width * 3);
        for row in 0..height {
            for col in 0..width {
                let (y, x) = (row as f64 / height as f64, col as f64 / width as f64);
                // Luma-dominant, like a photograph: the channels vary together
                // with a mild constant tint, so the chroma planes stay smooth.
                let luma = 30.0 + 190.0 * (0.6 * x + 0.4 * y);
                let mut rgb = [luma * 1.02, luma * 0.96, luma * 0.88];
                // A bright block and a dark bar: the ringing cases a gradient
                // alone would never exercise.
                if row > height / 3 && row < height * 2 / 3 && col > width / 4 && col < width / 2 {
                    rgb = [236.0, 231.0, 214.0];
                }
                if col > width * 3 / 4 && col < width * 3 / 4 + width / 16 {
                    rgb = [26.0, 25.0, 22.0];
                }
                pixels.extend(rgb.map(|channel| channel.round().clamp(0.0, 255.0) as u8));
            }
        }
        value::tensor_from_u8(value::shape_i64(&[height, width, 3]), pixels)
    }

    /// Peak signal-to-noise ratio, in dB, between two equal-shaped 8-bit frames.
    fn psnr(left: &Tensor, right: &Tensor) -> f64 {
        let (left, right) = (
            value::u8_pixels(left).expect("pixels"),
            value::u8_pixels(right).expect("pixels"),
        );
        let squared: f64 = left
            .iter()
            .zip(&right)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
            .sum();
        let mse = squared / left.len() as f64;
        if mse == 0.0 {
            return f64::INFINITY;
        }
        10.0 * (255.0f64.powi(2) / mse).log10()
    }

    #[test]
    fn jpeg_encodes_a_baseline_4_2_0_frame() {
        // The profile IS the contract -- another engine reproducing the vectors
        // has to write the same stream shape -- so assert it on the bytes, not
        // on a crate default that could move under a patch bump. Walk the
        // markers to SOF0 (0xFFC0, baseline sequential; a progressive stream
        // would be 0xFFC2) and read its per-component sampling factors.
        let encoded = jpeg_encode(&natural_frame(32, 32), 95).expect("encode");
        assert_eq!(&encoded[..2], &[0xFF, 0xD8], "not a JPEG (missing SOI)");
        let mut at = 2;
        let sof = loop {
            assert_eq!(encoded[at], 0xFF, "desynced at byte {at}");
            let marker = encoded[at + 1];
            let length = usize::from(u16::from_be_bytes([encoded[at + 2], encoded[at + 3]]));
            assert_ne!(marker, 0xC2, "progressive SOF2, expected baseline SOF0");
            if marker == 0xC0 {
                break &encoded[at + 4..at + 2 + length];
            }
            at += 2 + length;
        };
        // SOF0 payload: precision, height, width, component count, then three
        // bytes per component (id, sampling factors packed 4:4, quant table).
        assert_eq!(sof[0], 8, "expected 8-bit samples");
        assert_eq!(sof[5], 3, "expected 3 components");
        let factors: Vec<(u8, u8)> = sof[6..]
            .chunks(3)
            .map(|component| (component[1] >> 4, component[1] & 0x0F))
            .collect();
        // 4:2:0 -- luma 2x2, both chroma planes 1x1 (half resolution on each
        // axis). This is the whole reason for the pinned encoder: 4:4:4 would
        // read (1, 1) for every component and could not match the upstream
        // anchor.
        assert_eq!(factors, vec![(2, 2), (1, 1), (1, 1)], "expected 4:2:0");
    }

    #[test]
    fn jpeg_roundtrip_at_q95_stays_above_the_psnr_floor() {
        // q95 is the value the training pipelines use; a round-trip that lost
        // more than this would be a different picture, not a codec artifact.
        let frame = natural_frame(64, 64);
        let out = jpeg_roundtrip(&frame, 95).expect("roundtrip");
        assert_eq!(out.shape(), frame.shape());
        let quality95 = psnr(&frame, &out);
        assert!(quality95 >= 35.0, "q95 PSNR {quality95} dB below the floor");
        // ... and the dial really is a dial: q10 is visibly worse.
        let coarse = psnr(&frame, &jpeg_roundtrip(&frame, 10).expect("roundtrip"));
        assert!(coarse < quality95, "q10 {coarse} dB not worse than q95");
    }

    #[test]
    fn jpeg_roundtrip_rejects_a_non_three_channel_frame() {
        // JPEG subsampling is defined on YCbCr; a grayscale or RGBA frame has
        // no such thing. Resolution rejects it first -- this is the backstop.
        let gray = value::tensor_from_u8(vec![2, 2, 1], vec![0, 64, 128, 255]);
        let error = jpeg_roundtrip(&gray, 95).expect_err("3-channel only");
        assert!(
            error.to_string().contains("needs a 3-channel image"),
            "got: {error}"
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
            fill: 0,
            crop: None,
            jpeg_quality: None,
            swap_rb: false,
            render: None,
            offsets: None,
            stack_pad: crate::spec::StackPad::First,
            frame_bytes: 0,
            role_rebound: None,
            part: None,
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
    fn zero_fill_uses_the_fill_level() {
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
            fill: 128,
            crop: None,
            jpeg_quality: None,
            swap_rb: false,
            render: None,
            offsets: None,
            stack_pad: crate::spec::StackPad::First,
            frame_bytes: 0,
            role_rebound: None,
            part: None,
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
