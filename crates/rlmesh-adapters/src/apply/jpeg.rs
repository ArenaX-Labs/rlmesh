//! The JPEG round trip a model trained on stored JPEGs needs
//! (`jpeg_quality`), the binder's injected [`Codec`].

use rlmesh_session_core::spec::apply::ApplyError;
use rlmesh_session_core::spec::image::Image;
use rlmesh_session_core::spec::role::apply::Codec;

/// The JPEG codec: `jpeg-encoder` (pinned exactly) to encode, the `image`
/// crate's reader to decode, so only one codec is vendored.
pub struct Jpeg;

impl Codec for Jpeg {
    fn jpeg(&self, image: &Image<u8>, quality: u8) -> Result<Image<u8>, ApplyError> {
        let encoded = encode(image, quality)?;
        let decoded = image::load_from_memory_with_format(&encoded, image::ImageFormat::Jpeg)
            .map_err(|err| ApplyError(format!("could not decode the jpeg image: {err}")))?;
        Ok(Image {
            data: decoded.to_rgb8().into_raw(),
            ..*image
        })
    }
}

/// Encode an HWC 8-bit RGB frame at `quality`.
///
/// The profile is part of the v1 contract, because a different one is a
/// different picture: **baseline sequential, 4:2:0 chroma subsampling with
/// box-averaged (libjpeg `h2v2_downsample`) chroma, the standard IJG
/// quantization tables scaled by `quality`, and the standard IJG Huffman
/// tables** -- what TensorFlow's `encode_jpeg` and Pillow's `save(..., "JPEG")`
/// write by default.
fn encode(image: &Image<u8>, quality: u8) -> Result<Vec<u8>, ApplyError> {
    if image.channels != 3 {
        return Err(ApplyError(format!(
            "jpeg_quality needs a 3-channel image, got {} channel(s)",
            image.channels
        )));
    }
    // JPEG's frame header carries 16-bit dimensions; a larger frame has no
    // encoding, so say so rather than truncating the cast.
    let (Ok(width), Ok(height)) = (u16::try_from(image.width), u16::try_from(image.height)) else {
        return Err(ApplyError(format!(
            "jpeg_quality cannot encode a {}x{} frame; JPEG dimensions are 16-bit",
            image.height, image.width
        )));
    };
    let mut encoded: Vec<u8> = Vec::new();
    let mut encoder = jpeg_encoder::Encoder::new(&mut encoded, quality);
    // The crate picks 4:4:4 above quality 90; upstream (libjpeg, and so
    // TensorFlow and Pillow) subsamples at every quality, so pin it.
    encoder.set_sampling_factor(jpeg_encoder::SamplingFactor::R_4_2_0);
    encoder.set_chroma_subsampling_method(jpeg_encoder::ChromaSubsamplingMethod::Average);
    encoder
        .encode(&image.data, width, height, jpeg_encoder::ColorType::Rgb)
        .map_err(|err| ApplyError(format!("could not jpeg-encode the image: {err}")))?;
    Ok(encoded)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A natural-looking frame: two smooth gradients plus a few hard edges.
    /// Deliberately not noise -- JPEG is tuned for photographic content, and
    /// white noise would put the PSNR floor somewhere that says nothing about
    /// what a camera frame survives.
    fn natural_frame(height: usize, width: usize) -> Image<u8> {
        let mut data = Vec::with_capacity(height * width * 3);
        for row in 0..height {
            for col in 0..width {
                let (y, x) = (row as f64 / height as f64, col as f64 / width as f64);
                let luma = 30.0 + 190.0 * (0.6 * x + 0.4 * y);
                let mut rgb = [luma * 1.02, luma * 0.96, luma * 0.88];
                if row > height / 3 && row < height * 2 / 3 && col > width / 4 && col < width / 2 {
                    rgb = [236.0, 231.0, 214.0];
                }
                if col > width * 3 / 4 && col < width * 3 / 4 + width / 16 {
                    rgb = [26.0, 25.0, 22.0];
                }
                data.extend(rgb.map(|channel| channel.round().clamp(0.0, 255.0) as u8));
            }
        }
        Image {
            height,
            width,
            channels: 3,
            data,
        }
    }

    /// Peak signal-to-noise ratio, in dB, between two equal-shaped frames.
    fn psnr(left: &Image<u8>, right: &Image<u8>) -> f64 {
        let squared: f64 = left
            .data
            .iter()
            .zip(&right.data)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
            .sum();
        let mse = squared / left.data.len() as f64;
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
        let encoded = encode(&natural_frame(32, 32), 95).expect("encode");
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
        assert_eq!(sof[0], 8, "expected 8-bit samples");
        assert_eq!(sof[5], 3, "expected 3 components");
        let factors: Vec<(u8, u8)> = sof[6..]
            .chunks(3)
            .map(|component| (component[1] >> 4, component[1] & 0x0F))
            .collect();
        // 4:2:0 -- luma 2x2, both chroma planes 1x1.
        assert_eq!(factors, vec![(2, 2), (1, 1), (1, 1)], "expected 4:2:0");
    }

    #[test]
    fn a_q95_round_trip_stays_above_the_psnr_floor_and_q10_is_worse() {
        let frame = natural_frame(64, 64);
        let out = Jpeg.jpeg(&frame, 95).expect("roundtrip");
        assert_eq!((out.height, out.width, out.channels), (64, 64, 3));
        let quality95 = psnr(&frame, &out);
        assert!(quality95 >= 35.0, "q95 PSNR {quality95} dB below the floor");
        let coarse = psnr(&frame, &Jpeg.jpeg(&frame, 10).expect("roundtrip"));
        assert!(coarse < quality95, "q10 {coarse} dB not worse than q95");
        let gray = Image {
            height: 2,
            width: 2,
            channels: 1,
            data: vec![0, 64, 128, 255],
        };
        let error = Jpeg.jpeg(&gray, 95).expect_err("3-channel only");
        assert!(error.0.contains("needs a 3-channel image"), "{error}");
    }
}
