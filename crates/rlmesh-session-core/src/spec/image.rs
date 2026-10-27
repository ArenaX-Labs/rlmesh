//! The image operations of bound camera and depth inputs (§7.2) and of v1
//! adapters, one implementation with v1's pinned semantics: an un-suffixed kernel has cv2/torch semantics, an
//! `_aa` kernel PIL's (an antialiased filter whose support widens with the
//! downscale factor). `bilinear` is the 4-tap half-pixel-center sampler;
//! `bilinear_aa`, `bicubic_aa` and `lanczos3_aa` are PIL's triangle, cubic
//! (a = -0.5) and Lanczos-3 filters; `area` is cv2's `INTER_AREA`. All
//! compute in float64 with one final round-half-to-even. Depth images are
//! resampled nearest-neighbour and keep NaN.
//!
//! An aspect fit is v1's `fit`: `crop` scales uniformly to cover the target
//! then center-crops; `pad` scales uniformly to fit inside it then
//! letterboxes (zeros; NaN for depth); `stretch` scales each axis. A zoom
//! (v1's fractional `crop`) resamples a centered box straight to the
//! target, PIL's `resize(size, box=...)`.

#[allow(unused_imports)]
use crate::prelude::*;

use alloc::rc::Rc;
#[cfg(feature = "std")]
use core::cell::RefCell;
#[cfg(feature = "std")]
use std::collections::HashMap;

use crate::spec::plan::Fit;
use crate::spec::policy::Resample;

/// An image: `height × width × channels`, row-major, one `T` per element.
#[derive(Clone, Debug, PartialEq)]
pub struct Image<T> {
    /// Rows.
    pub height: usize,
    /// Columns.
    pub width: usize,
    /// Channels (1 for depth).
    pub channels: usize,
    /// `height * width * channels` elements.
    pub data: Vec<T>,
}

impl<T: Copy> Image<T> {
    /// A 180° rotation: rows and columns reversed (not a vertical flip).
    pub fn rotate180(&self) -> Self {
        let mut data = Vec::with_capacity(self.data.len());
        for px in self.data.chunks_exact(self.channels.max(1)).rev() {
            data.extend_from_slice(px);
        }
        Self { data, ..*self }
    }

    /// The centered `height × width` window (each at most the source's).
    pub fn crop_center(&self, height: usize, width: usize) -> Self {
        let (top, left) = ((self.height - height) / 2, (self.width - width) / 2);
        let c = self.channels;
        let mut data = Vec::with_capacity(height * width * c);
        for row in top..top + height {
            let start = (row * self.width + left) * c;
            data.extend_from_slice(&self.data[start..start + width * c]);
        }
        Self {
            height,
            width,
            channels: c,
            data,
        }
    }

    /// Centered in a `height × width` frame of `fill` (each at least the
    /// source's).
    pub fn pad_center(&self, height: usize, width: usize, fill: T) -> Self {
        let (top, left) = ((height - self.height) / 2, (width - self.width) / 2);
        let c = self.channels;
        let mut data = vec![fill; height * width * c];
        for row in 0..self.height {
            let dst = ((top + row) * width + left) * c;
            let src = row * self.width * c;
            data[dst..dst + self.width * c].copy_from_slice(&self.data[src..src + self.width * c]);
        }
        Self {
            height,
            width,
            channels: c,
            data,
        }
    }

    /// Nearest-neighbour resample (half-pixel centres).
    pub fn resize_nearest(&self, height: usize, width: usize) -> Self {
        let pick = |dst: usize, src: usize, i: usize| {
            (((i as f64 + 0.5) * src as f64 / dst as f64) as usize).min(src - 1)
        };
        let c = self.channels;
        let mut data = Vec::with_capacity(height * width * c);
        for row in 0..height {
            let sr = pick(height, self.height, row);
            for col in 0..width {
                let sc = pick(width, self.width, col);
                let start = (sr * self.width + sc) * c;
                data.extend_from_slice(&self.data[start..start + c]);
            }
        }
        Self {
            height,
            width,
            channels: c,
            data,
        }
    }
}

/// The integer center-cut a training-time crop keeps of a `src`-long axis.
pub fn crop_cut(src: usize, fraction: f64) -> usize {
    ((src as f64 * fraction).round() as usize).clamp(1, src.max(1))
}

/// One resample axis: `src` input pixels, the half-open source span
/// `[start, end)` sampled from them, and `dst` output pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Roi {
    /// Source pixels.
    pub src: usize,
    /// First sampled source coordinate.
    pub start: f64,
    /// End of the sampled span.
    pub end: f64,
    /// Output pixels.
    pub dst: usize,
}

impl Roi {
    /// The whole axis.
    pub fn full(src: usize, dst: usize) -> Self {
        Self {
            src,
            start: 0.0,
            end: src as f64,
            dst,
        }
    }

    /// The centered `fraction` of the axis (`None`: the whole axis).
    pub fn zoom(src: usize, dst: usize, fraction: Option<f64>) -> Self {
        let Some(fraction) = fraction else {
            return Self::full(src, dst);
        };
        let span = src as f64 * fraction;
        let start = (src as f64 - span) / 2.0;
        Self {
            src,
            start,
            end: start + span,
            dst,
        }
    }

    /// The whole axis mapped 1:1: nothing to resample.
    fn is_identity(self) -> bool {
        self.dst == self.src && self.start == 0.0 && self.end == self.src as f64
    }
}

/// A resample filter (the `_aa` kernels and `area`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kernel {
    Triangle,
    Cubic,
    Lanczos3,
    Area,
}

impl Kernel {
    fn support(self) -> f64 {
        match self {
            Self::Area => 0.5,
            Self::Triangle => 1.0,
            Self::Cubic => 2.0,
            Self::Lanczos3 => 3.0,
        }
    }

    /// PIL's filters widen by the downscale factor and keep the unit filter
    /// when upscaling; `area` is always the output pixel's footprint.
    fn filterscale(self, scale: f64) -> f64 {
        match self {
            Self::Area => scale,
            _ => scale.max(1.0),
        }
    }

    fn weight(self, tap: usize, center: f64, filterscale: f64) -> f64 {
        let tap = tap as f64;
        let x = ((tap + 0.5 - center) / filterscale).abs();
        match self {
            // cv2 integrates the box over each source pixel, so a partly
            // covered edge pixel weighs exactly its coverage.
            Self::Area => {
                let (low, high) = (center - filterscale / 2.0, center + filterscale / 2.0);
                (high.min(tap + 1.0) - low.max(tap)).max(0.0)
            }
            Self::Triangle => (1.0 - x).max(0.0),
            Self::Cubic => {
                const A: f64 = -0.5;
                if x < 1.0 {
                    ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
                } else if x < 2.0 {
                    (((x - 5.0) * x + 8.0) * x - 4.0) * A
                } else {
                    0.0
                }
            }
            Self::Lanczos3 => {
                let sinc = |x: f64| {
                    if x == 0.0 {
                        1.0
                    } else {
                        let x = x * core::f64::consts::PI;
                        crate::math::sin(x) / x
                    }
                };
                if x < 3.0 {
                    sinc(x) * sinc(x / 3.0)
                } else {
                    0.0
                }
            }
        }
    }

    /// Per output pixel: the first source pixel it reads and the
    /// normalised weights from there.
    fn weights(self, roi: Roi) -> Vec<(usize, Vec<f64>)> {
        let scale = (roi.end - roi.start) / roi.dst as f64;
        let filterscale = self.filterscale(scale);
        let reach = self.support() * filterscale;
        (0..roi.dst)
            .map(|out| {
                let center = roi.start + (out as f64 + 0.5) * scale;
                // PIL snaps the tap window to the nearest pixel centers;
                // `area` needs every pixel the box touches.
                let (low, high) = match self {
                    Self::Area => ((center - reach).floor(), (center + reach).ceil()),
                    _ => (
                        (center - reach + 0.5).trunc(),
                        (center + reach + 0.5).trunc(),
                    ),
                };
                let lo = (low as i64).clamp(0, roi.src as i64) as usize;
                let hi = (high as i64).clamp(lo as i64, roi.src as i64) as usize;
                let mut row: Vec<f64> = (lo..hi)
                    .map(|tap| self.weight(tap, center, filterscale))
                    .collect();
                let total: f64 = row.iter().sum();
                if total > 0.0 {
                    for w in &mut row {
                        *w /= total;
                    }
                }
                (lo, row)
            })
            .collect()
    }
}

/// One axis' weights.
type AxisWeights = Rc<Vec<(usize, Vec<f64>)>>;

#[cfg(feature = "std")]
/// What an axis' weights are keyed by: the kernel, both sizes and the span's
/// bits.
type WeightKey = (Kernel, usize, usize, u64, u64);

#[cfg(feature = "std")]
std::thread_local! {
    /// Weights depend only on the kernel and the axis, so a camera resized
    /// every step builds them once (a Lanczos-3 row is six `sin` per output
    /// pixel).
    static WEIGHTS: RefCell<HashMap<WeightKey, AxisWeights>> =
        RefCell::new(HashMap::new());
}

#[cfg(not(feature = "std"))]
fn axis_weights(kernel: Kernel, roi: Roi) -> AxisWeights {
    // No thread-locals without `std`: build the weights each call.
    Rc::new(kernel.weights(roi))
}

#[cfg(feature = "std")]
fn axis_weights(kernel: Kernel, roi: Roi) -> AxisWeights {
    let key = (
        kernel,
        roi.src,
        roi.dst,
        roi.start.to_bits(),
        roi.end.to_bits(),
    );
    WEIGHTS.with_borrow_mut(|cache| {
        Rc::clone(
            cache
                .entry(key)
                .or_insert_with(|| Rc::new(kernel.weights(roi))),
        )
    })
}

fn finish(values: Vec<f64>) -> Vec<u8> {
    values
        .into_iter()
        .map(|v| v.round_ties_even().clamp(0.0, 255.0) as u8)
        .collect()
}

impl Image<u8> {
    /// The 4-tap half-pixel-center bilinear resample (cv2/torch).
    fn resize_bilinear(&self, rows: Roi, cols: Roi) -> Self {
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
        let (row_taps, col_taps) = (coords(rows), coords(cols));
        let c = self.channels;
        let px =
            |r: usize, col: usize, ch: usize| f64::from(self.data[(r * self.width + col) * c + ch]);
        let mut out = Vec::with_capacity(rows.dst * cols.dst * c);
        for &(r0, r1, rw) in &row_taps {
            for &(c0, c1, cw) in &col_taps {
                for ch in 0..c {
                    let top = px(r0, c0, ch) * (1.0 - cw) + px(r0, c1, ch) * cw;
                    let bottom = px(r1, c0, ch) * (1.0 - cw) + px(r1, c1, ch) * cw;
                    out.push(top * (1.0 - rw) + bottom * rw);
                }
            }
        }
        Self {
            height: rows.dst,
            width: cols.dst,
            channels: c,
            data: finish(out),
        }
    }

    /// A separable filtered resample: horizontal then vertical, in float64.
    fn resize_filter(&self, rows: Roi, cols: Roi, kernel: Kernel) -> Self {
        let c = self.channels;
        let (height, width) = (rows.dst, cols.dst);
        let col_weights = axis_weights(kernel, cols);
        let row_weights = axis_weights(kernel, rows);
        let mut horizontal = vec![0.0_f64; self.height * width * c];
        for row in 0..self.height {
            for (out_col, (lo, weights)) in col_weights.iter().enumerate() {
                for ch in 0..c {
                    let mut acc = 0.0;
                    for (k, w) in weights.iter().enumerate() {
                        acc += w * f64::from(self.data[(row * self.width + lo + k) * c + ch]);
                    }
                    // PIL's 8-bit intermediate clips negative-lobe overshoot.
                    horizontal[(row * width + out_col) * c + ch] = acc.clamp(0.0, 255.0);
                }
            }
        }
        let mut out = vec![0.0_f64; height * width * c];
        for (out_row, (lo, weights)) in row_weights.iter().enumerate() {
            for col in 0..width {
                for ch in 0..c {
                    let mut acc = 0.0;
                    for (k, w) in weights.iter().enumerate() {
                        acc += w * horizontal[((lo + k) * width + col) * c + ch];
                    }
                    out[(out_row * width + col) * c + ch] = acc;
                }
            }
        }
        Self {
            height,
            width,
            channels: c,
            data: finish(out),
        }
    }

    /// A resample of the `rows` × `cols` spans with `kernel`.
    pub fn resize_roi(&self, rows: Roi, cols: Roi, kernel: Resample) -> Self {
        if rows.is_identity() && cols.is_identity() {
            return self.clone();
        }
        match kernel {
            Resample::Bilinear => self.resize_bilinear(rows, cols),
            Resample::BilinearAa => self.resize_filter(rows, cols, Kernel::Triangle),
            Resample::BicubicAa => self.resize_filter(rows, cols, Kernel::Cubic),
            Resample::Lanczos3Aa => self.resize_filter(rows, cols, Kernel::Lanczos3),
            Resample::Area => self.resize_filter(rows, cols, Kernel::Area),
            Resample::Nearest => self.resize_nearest(rows.dst, cols.dst),
        }
    }

    /// A plain resample with `kernel`.
    pub fn resize(&self, height: usize, width: usize, kernel: Resample) -> Self {
        self.resize_roi(
            Roi::full(self.height, height),
            Roi::full(self.width, width),
            kernel,
        )
    }

    /// Swaps the first and third channels.
    pub fn bgr(&self) -> Self {
        let mut data = self.data.clone();
        for px in data.chunks_exact_mut(self.channels.max(1)) {
            if px.len() >= 3 {
                px.swap(0, 2);
            }
        }
        Self { data, ..*self }
    }
}

/// `low + (v / 255) * (high - low)` for every byte, in `f32`: the
/// normalisation v1 pins for a `u8` image into `[low, high]`.
pub fn normalise_table(low: f64, high: f64) -> [f32; 256] {
    let (low, high) = (low as f32, high as f32);
    core::array::from_fn(|v| low + (v as f32 / 255.0) * (high - low))
}

/// The intermediate size a fit resamples a `height × width` box to before
/// its crop or pad.
fn fitted(height: f64, width: f64, target: (usize, usize), fit: Fit) -> (usize, usize) {
    let (th, tw) = target;
    let (sy, sx) = (th as f64 / height, tw as f64 / width);
    match fit {
        Fit::Stretch => (th, tw),
        Fit::Crop => {
            let scale = sy.max(sx);
            (
                ((height * scale).round() as usize).max(th),
                ((width * scale).round() as usize).max(tw),
            )
        }
        Fit::Pad => {
            let scale = sy.min(sx);
            (
                ((height * scale).round() as usize).clamp(1, th),
                ((width * scale).round() as usize).clamp(1, tw),
            )
        }
    }
}

/// Resamples a pixel image to `target` (height, width) through `fit`,
/// sampling only the centered `zoom` fraction of each axis (`None`: the
/// whole frame).
pub fn fit_resize(
    image: &Image<u8>,
    target: (usize, usize),
    fit: Option<Fit>,
    kernel: Resample,
    zoom: Option<f64>,
) -> Image<u8> {
    let fit = fit.unwrap_or(Fit::Stretch);
    let fraction = zoom.unwrap_or(1.0);
    let (h, w) = fitted(
        image.height as f64 * fraction,
        image.width as f64 * fraction,
        target,
        fit,
    );
    let scaled = image.resize_roi(
        Roi::zoom(image.height, h, zoom),
        Roi::zoom(image.width, w, zoom),
        kernel,
    );
    match fit {
        Fit::Stretch => scaled,
        Fit::Crop => scaled.crop_center(target.0, target.1),
        Fit::Pad => scaled.pad_center(target.0, target.1, 0),
    }
}

/// Resamples a depth image nearest-neighbour through `fit`, padding with
/// `fill` (NaN for metres, 0 for raw millimetres).
pub fn fit_resize_depth<T: Copy>(
    image: &Image<T>,
    target: (usize, usize),
    fit: Option<Fit>,
    fill: T,
) -> Image<T> {
    match fit {
        None | Some(Fit::Stretch) => image.resize_nearest(target.0, target.1),
        Some(fit) => {
            let (h, w) = fitted(image.height as f64, image.width as f64, target, fit);
            let scaled = image.resize_nearest(h, w);
            if fit == Fit::Crop {
                scaled.crop_center(target.0, target.1)
            } else {
                scaled.pad_center(target.0, target.1, fill)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(height: usize, width: usize) -> Image<u8> {
        Image {
            height,
            width,
            channels: 3,
            data: (0..height * width * 3).map(|i| (i % 251) as u8).collect(),
        }
    }

    /// Pinned bytes: the host (`std`) and a device (no `std`) both take
    /// `sin` from libm, so they resample to the same pixels. The platform
    /// `sin` gives 127 for the middle pixel here.
    #[test]
    fn lanczos_resamples_to_pinned_bytes() {
        let stripes = Image {
            height: 1,
            width: 14,
            channels: 1,
            data: [255_u8, 0].repeat(7),
        };
        assert_eq!(
            stripes.resize(1, 3, Resample::Lanczos3Aa).data,
            [136, 128, 119]
        );
    }

    #[test]
    fn identity_resizes_and_downscales_keep_the_mean() {
        let img = gradient(8, 8);
        for kernel in [
            Resample::Bilinear,
            Resample::BilinearAa,
            Resample::BicubicAa,
            Resample::Lanczos3Aa,
            Resample::Area,
        ] {
            assert_eq!(img.resize(8, 8, kernel), img, "{kernel:?}");
        }
        let flat = Image {
            height: 6,
            width: 6,
            channels: 1,
            data: vec![100_u8; 36],
        };
        for kernel in [
            Resample::Bilinear,
            Resample::BilinearAa,
            Resample::BicubicAa,
            Resample::Area,
        ] {
            assert!(
                flat.resize(3, 2, kernel).data.iter().all(|v| *v == 100),
                "{kernel:?}"
            );
        }
        // area of a 2x2 block is its average.
        let block = Image {
            height: 2,
            width: 2,
            channels: 1,
            data: vec![0, 10, 20, 30],
        };
        assert_eq!(block.resize(1, 1, Resample::Area).data, [15]);
    }

    #[test]
    fn fits_cover_or_letterbox_like_v1() {
        let img = gradient(480, 640);
        let padded = fit_resize(&img, (224, 224), Some(Fit::Pad), Resample::Bilinear, None);
        assert_eq!((padded.height, padded.width), (224, 224));
        // 640x480 into 224x224: 224x168, letterboxed 28 rows top and bottom.
        assert!(padded.data[..28 * 224 * 3].iter().all(|v| *v == 0));
        assert!(padded.data[(224 - 28) * 224 * 3..].iter().all(|v| *v == 0));
        let cropped = fit_resize(&img, (96, 96), Some(Fit::Crop), Resample::Bilinear, None);
        assert_eq!((cropped.height, cropped.width), (96, 96));
        assert_eq!(fitted(480.0, 640.0, (96, 96), Fit::Crop), (96, 128));
        assert_eq!(crop_cut(480, 0.95), 456);
        let rotated = img.rotate180();
        assert_eq!(rotated.data[..3], img.data[img.data.len() - 3..]);
        let depth = Image {
            height: 2,
            width: 4,
            channels: 1,
            data: vec![1.0_f32; 8],
        };
        let boxed = fit_resize_depth(&depth, (4, 4), Some(Fit::Pad), f32::NAN);
        assert!(boxed.data[0].is_nan() && boxed.data[4] == 1.0);
    }

    #[test]
    fn filter_weights_partition_unity_and_are_built_once() {
        for kernel in [
            Kernel::Triangle,
            Kernel::Cubic,
            Kernel::Lanczos3,
            Kernel::Area,
        ] {
            for (src, dst) in [(8, 3), (3, 8), (7, 7), (256, 224), (1, 5), (5, 1)] {
                for (lo, row) in kernel.weights(Roi::full(src, dst)) {
                    let total: f64 = row.iter().sum();
                    assert!(
                        (total - 1.0).abs() < 1e-12,
                        "{kernel:?} {src}->{dst}: {total}"
                    );
                    assert!(
                        lo + row.len() <= src,
                        "{kernel:?} {src}->{dst}: past the edge"
                    );
                }
            }
        }
        let first = axis_weights(Kernel::Lanczos3, Roi::full(256, 224));
        let again = axis_weights(Kernel::Lanczos3, Roi::full(256, 224));
        assert!(Rc::ptr_eq(&first, &again), "the second call rebuilt them");
        assert_ne!(
            first.len(),
            axis_weights(Kernel::Lanczos3, Roi::full(256, 112)).len()
        );
    }
}
