//! Frame encoding for the viewer.
//!
//! The Session feeds an already-decoded HWC uint8 array (from `session.read()`),
//! so the viewer only normalizes channels to RGB and encodes to the wire format.
//! JPEG is the default (lighter/faster over a remote link); PNG is lossless for
//! pixel-exact inspection.

use std::io::Cursor;

use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::PngEncoder;
use image::{ExtendedColorType, ImageEncoder, RgbImage};

/// Wire encoding for a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameFormat {
    /// Lossy, quality `1..=100`. The default — much smaller over a remote link.
    Jpeg(u8),
    /// Lossless. Use when an artifact must not be confused with the model's input.
    Png,
}

impl FrameFormat {
    pub fn content_type(self) -> &'static str {
        match self {
            FrameFormat::Jpeg(_) => "image/jpeg",
            FrameFormat::Png => "image/png",
        }
    }
}

/// Build an RGB frame from a contiguous HWC uint8 buffer (`channels` = 1/3/4).
/// Grayscale expands to RGB; alpha is dropped. Returns `None` if the buffer is
/// too short or `channels` is unsupported.
pub fn rgb_from_hwc(buf: &[u8], width: u32, height: u32, channels: u32) -> Option<RgbImage> {
    let (w, h, c) = (width as usize, height as usize, channels as usize);
    let needed = w.checked_mul(h)?.checked_mul(c)?;
    if c == 0 || buf.len() < needed {
        return None;
    }
    if c == 3 {
        return RgbImage::from_raw(width, height, buf[..needed].to_vec());
    }
    let mut rgb = Vec::with_capacity(w * h * 3);
    for px in 0..w * h {
        let base = px * c;
        let (r, g, b) = if c == 1 {
            (buf[base], buf[base], buf[base])
        } else {
            (buf[base], buf[base + 1], buf[base + 2]) // c == 4: drop alpha
        };
        rgb.extend_from_slice(&[r, g, b]);
    }
    RgbImage::from_raw(width, height, rgb)
}

/// Encode an RGB frame in the chosen wire format.
pub fn encode(img: &RgbImage, format: FrameFormat) -> Option<Vec<u8>> {
    let mut out = Cursor::new(Vec::new());
    let (buf, w, h) = (img.as_raw(), img.width(), img.height());
    match format {
        FrameFormat::Jpeg(quality) => JpegEncoder::new_with_quality(&mut out, quality)
            .write_image(buf, w, h, ExtendedColorType::Rgb8)
            .ok()?,
        FrameFormat::Png => PngEncoder::new(&mut out)
            .write_image(buf, w, h, ExtendedColorType::Rgb8)
            .ok()?,
    }
    Some(out.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_roundtrip_png_is_exact() {
        // 2x2 RGB: red, green, blue, white.
        let buf = [255u8, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];
        let img = rgb_from_hwc(&buf, 2, 2, 3).expect("rgb");
        let png = encode(&img, FrameFormat::Png).expect("png");
        assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
        let back = image::load_from_memory(&png).expect("reload").to_rgb8();
        assert_eq!(back.dimensions(), (2, 2));
        assert_eq!(back.get_pixel(0, 0).0, [255, 0, 0]);
        assert_eq!(back.get_pixel(1, 1).0, [255, 255, 255]);
    }

    #[test]
    fn grayscale_expands_and_jpeg_encodes() {
        let buf = [10u8, 20, 30, 40]; // 2x2x1
        let img = rgb_from_hwc(&buf, 2, 2, 1).expect("gray");
        assert_eq!(img.get_pixel(0, 0).0, [10, 10, 10]);
        let jpg = encode(&img, FrameFormat::Jpeg(80)).expect("jpeg");
        assert_eq!(&jpg[..2], &[0xff, 0xd8]); // JPEG SOI marker
        assert!(image::load_from_memory(&jpg).is_ok());
    }

    #[test]
    fn rejects_short_buffer() {
        assert!(rgb_from_hwc(&[0, 1, 2], 4, 4, 3).is_none());
    }
}
