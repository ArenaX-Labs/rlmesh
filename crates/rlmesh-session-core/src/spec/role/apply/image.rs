//! One env camera frame to a model image input.
//!
//! The pixel operations (resize kernels, aspect fits, crops, the 180°
//! rotation, BGR, normalisation) are the shared ones in
//! [`spec::image`](crate::spec::image), which v2 bindings run too; this
//! module adds the 8-bit decode of a non-`u8` frame, the JPEG round trip,
//! layouts and lead axes.

use crate::spec::apply::{ApplyError, Tensor, decode};
use crate::spec::image::{Image, crop_cut, fit_resize, normalise_table};
use crate::spec::policy::ImageLayout;
use crate::spec::role::apply::{Array, cast, err, f32_tensor};
use crate::spec::role::plan::ImagePlan;
use crate::wire::part::DType;

/// The host's JPEG codec, for `jpeg_quality`: encode an HWC 8-bit RGB frame
/// at `quality` and decode it back. The profile is part of the v1 contract
/// (baseline sequential, 4:2:0 chroma subsampling with box-averaged chroma,
/// the standard IJG quantization and Huffman tables scaled by `quality`):
/// what TensorFlow's `encode_jpeg` and Pillow's `save(..., "JPEG")` write.
pub trait Codec {
    /// The frame after a JPEG round trip.
    fn jpeg(&self, image: &Image<u8>, quality: u8) -> Result<Image<u8>, ApplyError>;
}

impl ImagePlan {
    /// One frame through the plan: `frame` is the env camera's value (`None`
    /// for a zero-filled input, which has no source).
    pub fn apply(&self, frame: Option<Array<'_>>, codec: &dyn Codec) -> Result<Tensor, ApplyError> {
        if let Some((height, width, channels)) = self.zero_fill {
            let (height, width, channels) = (height as usize, width as usize, channels as usize);
            return self.finalize(Image {
                height,
                width,
                channels,
                data: vec![self.fill; height * width * channels],
            });
        }
        let frame = frame.ok_or_else(|| err(format!("image {}: no frame", self.placement)))?;
        let mut image = decode_image(frame, self.src_range, self.src_layout)?;
        if self.flip {
            image = image.rotate180();
        }
        // Before the crop/resize: a training pipeline that stored JPEGs encoded the
        // camera frame it was handed, then cropped and resized the decoded result.
        if let Some(quality) = self.jpeg_quality {
            if image.channels != 3 {
                return Err(err(format!(
                    "jpeg_quality needs a 3-channel image, got {} channel(s)",
                    image.channels
                )));
            }
            image = codec.jpeg(&image, quality)?;
        }
        // A `slice` crop is an integer center cut taken before the resize; a `zoom`
        // crop is the fractional box the resize itself samples through, so it rides
        // along into `fit_resize` instead of costing a second resampling pass.
        let mut zoom = None;
        if let Some(crop) = &self.crop {
            if crop.slice {
                image = image.crop_center(
                    crop_cut(image.height, crop.fraction),
                    crop_cut(image.width, crop.fraction),
                );
            } else {
                zoom = Some(crop.fraction);
            }
        }
        let size = match (self.size, zoom) {
            (Some(size), _) => Some(size),
            // A zoom with no declared target resamples the box back to the frame's
            // own size -- that magnification is the whole point of the crop.
            (None, Some(_)) => Some((image.height as u32, image.width as u32)),
            (None, None) => None,
        };
        if let Some((height, width)) = size {
            if height == 0 || width == 0 || image.height == 0 || image.width == 0 {
                return Err(err(format!(
                    "cannot resize an image with a zero dimension (source \
                     {}x{}, target {height}x{width})",
                    image.height, image.width
                )));
            }
            image = fit_resize(
                &image,
                (height as usize, width as usize),
                Some(self.fit),
                self.resample,
                zoom,
            );
        }
        self.finalize(image)
    }

    /// The tail a real frame and a zero fill share: map the HWC uint8 frame
    /// into the model's dtype/range, transpose to the model's layout, and
    /// prepend any leading axes.
    fn finalize(&self, image: Image<u8>) -> Result<Tensor, ApplyError> {
        let image = if self.swap_rb {
            if image.channels != 3 {
                return Err(err(format!(
                    "channel_order \"bgr\" needs a 3-channel image, got {} channel(s)",
                    image.channels
                )));
            }
            image.bgr()
        } else {
            image
        };
        let hwc = [image.height, image.width, image.channels]
            .map(|d| d as u32)
            .to_vec();
        let mut tensor = match self.normalize {
            // The camera case: one table lookup per element. The table holds
            // exactly `low + (v / 255) * (high - low)` in f32, which is what
            // the generic path below computes before widening to f64 and
            // narrowing back, so the bytes are identical to it.
            Some((low, high)) if self.dtype == DType::F32 => {
                let table = normalise_table(low, high);
                let values: Vec<f32> = image.data.iter().map(|v| table[usize::from(*v)]).collect();
                f32_tensor(hwc, &values)
            }
            Some((low, high)) => {
                let (low, high) = (low as f32, high as f32);
                let scaled: Vec<f64> = image
                    .data
                    .iter()
                    .map(|v| f64::from(low + (f32::from(*v) / 255.0) * (high - low)))
                    .collect();
                cast(&scaled, hwc, self.dtype)?
            }
            None if self.dtype == DType::U8 => Tensor {
                dtype: DType::U8,
                shape: hwc,
                data: image.data,
            },
            None => {
                let values: Vec<f64> = image.data.iter().map(|v| f64::from(*v)).collect();
                cast(&values, hwc, self.dtype)?
            }
        };
        if self.dst_layout == ImageLayout::Chw {
            tensor = to_chw(&tensor);
        }
        let mut shape = vec![1; self.lead_dims as usize];
        shape.extend_from_slice(&tensor.shape);
        tensor.shape = shape;
        Ok(tensor)
    }
}

/// An env frame as an HWC uint8 image. A float frame is mapped from its
/// declared `range` into `[0, 255]` (a `[0, 1]` image is scaled, a
/// `[0, 255]` one passes through), assumed `[0, 1]` when unbounded; integer
/// frames carry raw pixel magnitudes and are only rounded and clamped.
fn decode_image(
    frame: Array<'_>,
    range: Option<(f64, f64)>,
    layout: ImageLayout,
) -> Result<Image<u8>, ApplyError> {
    let [a, b, c] = *frame.shape else {
        return Err(err(format!(
            "expected an HWC image with 3 axes, got shape {:?}",
            frame.shape
        )));
    };
    let data = if frame.dtype == DType::U8 {
        frame.data.to_vec()
    } else {
        let float = frame.dtype.is_float();
        decode(frame.dtype, frame.data)
            .into_iter()
            .map(|value| {
                let value = if float {
                    match range {
                        Some((low, high)) if high > low => (value - low) / (high - low) * 255.0,
                        _ => value * 255.0,
                    }
                } else {
                    value
                };
                value.round_ties_even().clamp(0.0, 255.0) as u8
            })
            .collect()
    };
    Ok(match layout {
        ImageLayout::Hwc => Image {
            height: a,
            width: b,
            channels: c,
            data,
        },
        ImageLayout::Chw => {
            let (channels, height, width) = (a, b, c);
            let plane = height * width;
            let mut out = vec![0_u8; data.len()];
            for pixel in 0..plane {
                for channel in 0..channels {
                    out[pixel * channels + channel] = data[channel * plane + pixel];
                }
            }
            Image {
                height,
                width,
                channels,
                data: out,
            }
        }
    })
}

/// An HWC tensor transposed to CHW (any dtype).
fn to_chw(tensor: &Tensor) -> Tensor {
    let size = tensor.dtype.size() as usize;
    let (height, width, channels) = (
        tensor.shape[0] as usize,
        tensor.shape[1] as usize,
        tensor.shape[2] as usize,
    );
    let plane = height * width;
    let mut out = vec![0_u8; tensor.data.len()];
    for channel in 0..channels {
        for pixel in 0..plane {
            let src = (pixel * channels + channel) * size;
            let dst = (channel * plane + pixel) * size;
            out[dst..dst + size].copy_from_slice(&tensor.data[src..src + size]);
        }
    }
    Tensor {
        dtype: tensor.dtype,
        shape: vec![channels as u32, height as u32, width as u32],
        data: out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::plan::Fit;
    use crate::spec::policy::Resample;
    use crate::spec::role::input::StackPad;
    use crate::spec::role::path::NodePath;

    /// No codec: these plans never round-trip a JPEG.
    struct NoJpeg;

    impl Codec for NoJpeg {
        fn jpeg(&self, _: &Image<u8>, _: u8) -> Result<Image<u8>, ApplyError> {
            unreachable!("no plan here asks for a JPEG round trip")
        }
    }

    /// A pass-through plan of `dtype`, normalised into `normalize`.
    fn plan(dtype: DType, normalize: Option<(f64, f64)>) -> ImagePlan {
        ImagePlan {
            placement: NodePath::root().push_key("image"),
            source: NodePath::root().push_key("cam"),
            src_layout: ImageLayout::Hwc,
            dst_layout: ImageLayout::Hwc,
            flip: false,
            size: None,
            fit: Fit::Stretch,
            resample: Resample::Bilinear,
            dtype,
            normalize,
            lead_dims: 0,
            src_range: None,
            stack: 1,
            zero_fill: None,
            fill: 0,
            crop: None,
            jpeg_quality: None,
            swap_rb: false,
            render: None,
            offsets: None,
            stack_pad: StackPad::First,
            frame_bytes: 0,
            role_rebound: None,
            part: None,
        }
    }

    fn apply(plan: &ImagePlan, dtype: DType, shape: &[usize], data: &[u8]) -> Tensor {
        let frame = Array { dtype, shape, data };
        plan.apply(Some(frame), &NoJpeg).expect("apply")
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn a_float_frame_decodes_over_its_range_and_unit_when_unbounded() {
        // A normalized [0, 1] float image scales into 8-bit rather than
        // flooring to black; a [0, 255] one passes through; an unbounded one
        // is taken as [0, 1]; u8 is untouched.
        for (range, values, want) in [
            (
                Some((0.0, 1.0)),
                vec![0.0, 0.4, 0.6, 1.0],
                [0_u8, 102, 153, 255],
            ),
            (
                Some((0.0, 255.0)),
                vec![0.0, 64.0, 200.0, 255.0],
                [0, 64, 200, 255],
            ),
            (None, vec![0.0, 1.0, 0.5, 0.25], [0, 255, 128, 64]),
        ] {
            let mut plan = plan(DType::U8, None);
            plan.src_range = range;
            let out = apply(&plan, DType::F32, &[2, 2, 1], &f32_bytes(&values));
            assert_eq!(out.data, want, "{range:?}");
        }
        let out = apply(&plan(DType::U8, None), DType::U8, &[1, 1, 3], &[10, 20, 30]);
        assert_eq!(out.data, [10, 20, 30]);
    }

    #[test]
    fn normalising_maps_bytes_into_the_declared_range() {
        let out = apply(
            &plan(DType::F32, Some((-1.0, 1.0))),
            DType::U8,
            &[1, 3, 1],
            &[0, 128, 255],
        );
        let floats: Vec<f32> = out
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        assert!((floats[0] + 1.0).abs() < 1e-6, "{floats:?}");
        assert!(floats[1].abs() < 0.01, "{floats:?}");
        assert!((floats[2] - 1.0).abs() < 1e-6, "{floats:?}");
    }

    #[test]
    fn a_zero_fill_is_a_frame_of_the_fill_level_through_the_plan() {
        let mut fill = plan(DType::U8, None);
        fill.zero_fill = Some((2, 2, 3));
        fill.fill = 128;
        let out = fill.apply(None, &NoJpeg).expect("zero fill");
        assert_eq!(out.shape, [2, 2, 3]);
        assert!(out.data.iter().all(|&b| b == 128));
    }

    #[test]
    fn bgr_and_a_zero_resize_are_refused_on_the_frame() {
        let mut bgr = plan(DType::U8, None);
        bgr.swap_rb = true;
        let frame = Array {
            dtype: DType::U8,
            shape: &[1, 1, 1],
            data: &[7],
        };
        let error = bgr.apply(Some(frame), &NoJpeg).expect_err("one channel");
        assert!(error.0.contains("3-channel"), "{error}");
        let mut resize = plan(DType::U8, None);
        resize.size = Some((0, 4));
        let frame = Array {
            dtype: DType::U8,
            shape: &[2, 2, 3],
            data: &[0; 12],
        };
        let error = resize.apply(Some(frame), &NoJpeg).expect_err("zero target");
        assert!(error.0.contains("zero dimension"), "{error}");
    }
}
