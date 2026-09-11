//! Image operations used by resolved adapters.
//!
//! Resizing implements the algorithms pinned by the v1 conformance vectors,
//! named by one rule: an un-suffixed kernel has cv2/torch semantics, an `_aa`
//! kernel has PIL semantics -- an antialiased filter whose support widens with
//! the downscale factor. So `bilinear` is the 4-tap half-pixel-center sampler,
//! `bilinear_aa`/`bicubic_aa`/`lanczos3_aa` are PIL's triangle, cubic
//! (a = -0.5) and Lanczos-3 filters, and `area` is cv2's `INTER_AREA` average
//! over each output pixel's source footprint. All compute in float64 with one
//! final round-half-to-even like the reference, and all but `bilinear` share
//! one weight builder. Pixels are carried in `rlmesh_spaces::Tensor`.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

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
    ((src as f64 * fraction).round() as usize).clamp(1, src.max(1))
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
    let image = finalize_dtype(&image, &plan.dtype, plan.normalize)?;
    let image = to_layout(&image, ImageLayout::Hwc, plan.dst_layout)?;
    Ok(Value::Tensor(add_lead_dims(image, plan.lead_dims)))
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

/// Swap the red and blue channels of an HWC image (`channel_order = "bgr"`).
///
/// A 3-channel op by definition: there is no meaningful red/blue pair in a
/// grayscale or RGBA frame, so a wrong-shaped source fails loudly rather than
/// reordering whatever happens to sit at those offsets.
pub fn swap_rb(tensor: &Tensor) -> Result<Tensor, ApplyError> {
    let (height, width, channels) = image_dims(tensor)?;
    if channels != 3 {
        return Err(ApplyError::new(format!(
            "channel_order \"bgr\" needs a 3-channel image, got {channels} channel(s)"
        )));
    }
    let mut indices = Vec::with_capacity(tensor.numel());
    for pixel in 0..height * width {
        indices.extend([pixel * 3 + 2, pixel * 3 + 1, pixel * 3]);
    }
    Ok(value::gather(tensor, &indices, tensor.shape().to_vec()))
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

/// 4-tap half-pixel-center bilinear resize (OpenCV/torch-compatible), sampling
/// each axis through its [`Roi`] (the whole axis unless a zoom crop narrowed it).
fn resize_bilinear(tensor: &Tensor, row_roi: Roi, col_roi: Roi) -> Result<Tensor, ApplyError> {
    let (_, src_width, channels) = image_dims(tensor)?;
    let (height, width) = (row_roi.dst, col_roi.dst);
    let data = value::u8_pixels(tensor)?;
    let coords = |roi: Roi| -> Vec<(usize, usize, f64)> {
        let scale = (roi.end - roi.start) / roi.dst as f64;
        (0..roi.dst)
            .map(|i| {
                let pos = roi.start + (i as f64 + 0.5) * scale - 0.5;
                let lo = (pos.floor() as i64).clamp(0, roi.src as i64 - 1) as usize;
                let hi = (lo + 1).min(roi.src - 1);
                (lo, hi, (pos - lo as f64).clamp(0.0, 1.0))
            })
            .collect()
    };
    let rows = coords(row_roi);
    let cols = coords(col_roi);
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

/// A resample filter shape. Naming rule for the wire names in [`RESAMPLES`]:
/// un-suffixed = cv2/torch semantics, `_aa` = PIL semantics (the filter's
/// support widens with the downscale factor, so decimation averages rather
/// than point-samples).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Kernel {
    /// PIL `BILINEAR`: triangle, support 1.
    Triangle,
    /// PIL `BICUBIC`: Keys cubic with a = -0.5, support 2.
    Cubic,
    /// PIL `LANCZOS`: windowed sinc, support 3.
    Lanczos3,
    /// cv2 `INTER_AREA`: box average over the output pixel's source footprint.
    Area,
}

impl Kernel {
    /// Filter half-width in its own (unwidened) units.
    fn support(self) -> f64 {
        match self {
            Kernel::Area => 0.5,
            Kernel::Triangle => 1.0,
            Kernel::Cubic => 2.0,
            Kernel::Lanczos3 => 3.0,
        }
    }

    /// How far the filter is stretched for a source-to-output `scale`. PIL's
    /// antialiased filters widen by the downscale factor and keep the unit
    /// filter when upscaling; `area` is always the output pixel's footprint,
    /// which is what makes cv2's `INTER_AREA` a plain average one way and a
    /// two-tap coverage split the other.
    fn filterscale(self, scale: f64) -> f64 {
        match self {
            Kernel::Area => scale,
            _ => scale.max(1.0),
        }
    }

    /// Weight of source pixel `tap` for an output pixel centered at `center`,
    /// with the filter widened by `filterscale`.
    fn weight(self, tap: usize, center: f64, filterscale: f64) -> f64 {
        let tap = tap as f64;
        let x = ((tap + 0.5 - center) / filterscale).abs();
        match self {
            // cv2 integrates the box over each source pixel instead of
            // sampling it at the pixel center, so a partially covered edge
            // pixel gets exactly its coverage as weight.
            Kernel::Area => {
                let (low, high) = (center - filterscale / 2.0, center + filterscale / 2.0);
                (high.min(tap + 1.0) - low.max(tap)).max(0.0)
            }
            Kernel::Triangle => (1.0 - x).max(0.0),
            Kernel::Cubic => {
                const A: f64 = -0.5;
                if x < 1.0 {
                    ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
                } else if x < 2.0 {
                    (((x - 5.0) * x + 8.0) * x - 4.0) * A
                } else {
                    0.0
                }
            }
            Kernel::Lanczos3 => {
                fn sinc(x: f64) -> f64 {
                    if x == 0.0 {
                        1.0
                    } else {
                        let x = x * std::f64::consts::PI;
                        x.sin() / x
                    }
                }
                if x < 3.0 {
                    sinc(x) * sinc(x / 3.0)
                } else {
                    0.0
                }
            }
        }
    }
}

/// One resample axis: `src` input pixels, the half-open source span
/// `[start, end)` sampled from them, and `dst` output pixels. [`Roi::full`]
/// samples the whole axis; a narrower span is the seam a crop resamples
/// through without a separate pass.
#[derive(Debug, Clone, Copy)]
struct Roi {
    src: usize,
    start: f64,
    end: f64,
    dst: usize,
}

impl Roi {
    fn full(src: usize, dst: usize) -> Self {
        Roi {
            src,
            start: 0.0,
            end: src as f64,
            dst,
        }
    }

    /// The whole axis, mapped 1:1 onto the output — nothing to resample.
    fn is_identity(self) -> bool {
        self.dst == self.src && self.start == 0.0 && self.end == self.src as f64
    }
}

/// Per-output-pixel normalized filter weights, PIL-style: for each output
/// pixel, the first source pixel it touches and the normalized weight of each
/// pixel from there on.
fn filter_weights(kernel: Kernel, support: f64, roi: Roi) -> Vec<(usize, Vec<f64>)> {
    let scale = (roi.end - roi.start) / roi.dst as f64;
    let filterscale = kernel.filterscale(scale);
    let reach = support * filterscale;
    (0..roi.dst)
        .map(|out| {
            let center = roi.start + (out as f64 + 0.5) * scale;
            // PIL snaps the tap window to the nearest pixel centers; `area`
            // needs every pixel the box touches, a partly covered one included.
            let (low, high) = match kernel {
                Kernel::Area => ((center - reach).floor(), (center + reach).ceil()),
                _ => (
                    (center - reach + 0.5).trunc(),
                    (center + reach + 0.5).trunc(),
                ),
            };
            let lo = (low as i64).clamp(0, roi.src as i64) as usize;
            let hi = (high as i64).clamp(lo as i64, roi.src as i64) as usize;
            let mut row: Vec<f64> = (lo..hi)
                .map(|tap| kernel.weight(tap, center, filterscale))
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

/// One axis' memoized weights.
type AxisWeights = Rc<Vec<(usize, Vec<f64>)>>;

/// What a memoized axis is keyed by: the kernel, the two sizes, and the
/// sampled span's bits (a zoom crop narrows the span; `Roi` is not `Hash`
/// because `f64` is not, hence the bits).
type WeightKey = (Kernel, usize, usize, u64, u64);

thread_local! {
    /// Weights depend only on the kernel and the two sizes, so a plan resizing
    /// the same camera every step builds them once instead of once per frame
    /// (a Lanczos-3 row is six taps of `sin` per output pixel). Thread-local
    /// rather than a field on the plan: no locking on the per-lane apply path,
    /// and the key covers the intermediate sizes `fit_resize` derives.
    static WEIGHT_CACHE: RefCell<HashMap<WeightKey, AxisWeights>> =
        RefCell::new(HashMap::new());
}

fn axis_weights(kernel: Kernel, roi: Roi) -> AxisWeights {
    // The span joins the key by its bits: a zoom crop is a fixed box per plan,
    // so its weights are as reusable as a full-axis resize's.
    let key = (
        kernel,
        roi.src,
        roi.dst,
        roi.start.to_bits(),
        roi.end.to_bits(),
    );
    WEIGHT_CACHE.with_borrow_mut(|cache| {
        Rc::clone(
            cache
                .entry(key)
                .or_insert_with(|| Rc::new(filter_weights(kernel, kernel.support(), roi))),
        )
    })
}

/// Separable filtered resize: a horizontal pass then a vertical one, both in
/// float64, with the per-axis weights taken from the cache.
fn resize_filter(
    tensor: &Tensor,
    row_roi: Roi,
    col_roi: Roi,
    kernel: Kernel,
) -> Result<Tensor, ApplyError> {
    let (src_height, src_width, channels) = image_dims(tensor)?;
    let (height, width) = (row_roi.dst, col_roi.dst);
    let data = value::u8_pixels(tensor)?;
    let col_weights = axis_weights(kernel, col_roi);
    let row_weights = axis_weights(kernel, row_roi);

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
                // PIL's intermediate is an 8-bit image, so a filter with
                // negative lobes (cubic, Lanczos) has its overshoot clipped
                // here, not only at the end. Clipping the float64 intermediate
                // to the same range reproduces that without also taking on
                // PIL's intermediate rounding; non-negative filters
                // (triangle, box) never reach the clamp.
                horizontal[(row * width + out_col) * channels + channel] = acc.clamp(0.0, 255.0);
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

/// Resize an HWC uint8 image to `(height, width)`, sampling only the center
/// box `zoom` names (a side fraction of each axis) — `None` is the whole frame.
///
/// The box is resampled *straight* to the target by the declared kernel, so a
/// zoom crop costs no extra pass and no intermediate rounding; this is PIL's
/// `Image.resize(size, box=...)`, which is the anchor the vectors pin.
fn resize_roi(
    tensor: &Tensor,
    (height, width): (u32, u32),
    zoom: Option<f64>,
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
    let row_roi = zoom_roi(src_height, height, zoom);
    let col_roi = zoom_roi(src_width, width, zoom);
    if row_roi.is_identity() && col_roi.is_identity() {
        return Ok(tensor.clone());
    }
    match resample {
        "bilinear" => resize_bilinear(tensor, row_roi, col_roi),
        "bilinear_aa" => resize_filter(tensor, row_roi, col_roi, Kernel::Triangle),
        "bicubic_aa" => resize_filter(tensor, row_roi, col_roi, Kernel::Cubic),
        "lanczos3_aa" => resize_filter(tensor, row_roi, col_roi, Kernel::Lanczos3),
        "area" => resize_filter(tensor, row_roi, col_roi, Kernel::Area),
        other => Err(ApplyError::new(format!("unsupported resample {other:?}"))),
    }
}

/// The centered span of a `src`-long axis a zoom crop samples, as the [`Roi`]
/// the weight builders take. `None` is the whole axis.
fn zoom_roi(src: usize, dst: usize, zoom: Option<f64>) -> Roi {
    let Some(fraction) = zoom else {
        return Roi::full(src, dst);
    };
    let span = src as f64 * fraction;
    let start = (src as f64 - span) / 2.0;
    Roi {
        src,
        start,
        end: start + span,
        dst,
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
    zoom: Option<f64>,
) -> Result<Tensor, ApplyError> {
    let (src_height, src_width, _) = image_dims(tensor)?;
    // A zoom crop keeps the same fraction of both axes, so the box has the
    // source's aspect ratio and the fit math only needs its (fractional) size.
    let fraction = zoom.unwrap_or(1.0);
    let (box_height, box_width) = (src_height as f64 * fraction, src_width as f64 * fraction);
    let (target_height, target_width) = (height as usize, width as usize);
    match fit {
        FitMode::Stretch => resize_roi(tensor, (height, width), zoom, resample),
        FitMode::Crop => {
            // Cover: scale uniformly so both axes reach the target, then crop.
            let scale = (target_height as f64 / box_height).max(target_width as f64 / box_width);
            let scaled_height = ((box_height * scale).round() as usize).max(target_height);
            let scaled_width = ((box_width * scale).round() as usize).max(target_width);
            let scaled = resize_roi(
                tensor,
                (scaled_height as u32, scaled_width as u32),
                zoom,
                resample,
            )?;
            crop_center(&scaled, target_height, target_width)
        }
        FitMode::Pad => {
            // Contain: scale uniformly so both axes fit within the target, then pad.
            let scale = (target_height as f64 / box_height).min(target_width as f64 / box_width);
            let scaled_height = ((box_height * scale).round() as usize).clamp(1, target_height);
            let scaled_width = ((box_width * scale).round() as usize).clamp(1, target_width);
            let scaled = resize_roi(
                tensor,
                (scaled_height as u32, scaled_width as u32),
                zoom,
                resample,
            )?;
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
        // Fuse normalize + cast: scale in f32 (the exact arithmetic of the old
        // Float32 staging tensor), widen to f64, and encode straight to the
        // target dtype — skipping the intermediate float32 tensor and its
        // decode/re-encode round-trip while preserving the bit-identical result.
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

    /// Every kernel in the public vocabulary, paired with its filter shape
    /// (`bilinear` has none -- it is the 4-tap sampler, not a filter).
    const FILTER_KERNELS: [(&str, Kernel); 4] = [
        ("bilinear_aa", Kernel::Triangle),
        ("bicubic_aa", Kernel::Cubic),
        ("lanczos3_aa", Kernel::Lanczos3),
        ("area", Kernel::Area),
    ];

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
    fn filter_weights_sum_to_one() {
        // Weights are a normalized partition of unity, so a resize can never
        // change an image's overall level -- the property the constant-image
        // test below observes end to end.
        for (name, kernel) in FILTER_KERNELS {
            for (src, dst) in [(8usize, 3usize), (3, 8), (7, 7), (256, 224), (1, 5), (5, 1)] {
                for (out, (lo, row)) in
                    filter_weights(kernel, kernel.support(), Roi::full(src, dst))
                        .iter()
                        .enumerate()
                {
                    let total: f64 = row.iter().sum();
                    assert!(
                        (total - 1.0).abs() < 1e-12,
                        "{name} {src}->{dst} out {out}: weights sum to {total}"
                    );
                    assert!(
                        lo + row.len() <= src,
                        "{name} {src}->{dst}: tap past the edge"
                    );
                }
            }
        }
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

    #[test]
    fn the_weight_cache_returns_the_same_weights_it_built() {
        // Memoization must be keyed tightly enough that a second axis size
        // cannot pick up the first one's weights.
        let first = axis_weights(Kernel::Lanczos3, Roi::full(256, 224));
        let again = axis_weights(Kernel::Lanczos3, Roi::full(256, 224));
        assert_eq!(first.as_ref(), again.as_ref());
        assert!(Rc::ptr_eq(&first, &again), "the second call rebuilt them");
        let other = axis_weights(Kernel::Lanczos3, Roi::full(256, 112));
        assert_ne!(first.len(), other.len());
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
            role_rebound: None,
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
            role_rebound: None,
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
