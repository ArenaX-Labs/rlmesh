//! Image parts: JPEG encoding, a bitrate budget and newest-wins skipping.
//!
//! An [`ImagePublisher`] wraps a [`SamplePublisher`] for a part whose single
//! field is a uint8 `[H, W, 3]` image. Each [`ImagePublisher::offer`] of a
//! captured frame:
//!
//! 1. is skipped outright when the stream's publication plan (or its `send`
//!    rate ceiling) would coalesce it, or while the bitrate budget (`send:
//!    latest(max_mbps=…)`, §8.5) is spent, before any encoding work. The
//!    bucket refills at the budget; its capacity is the larger of 100 ms of
//!    budget and one maximum-size frame, a frame is admitted while the
//!    bucket is positive, and its actual bytes are then subtracted (debt);
//! 2. is encoded as the part declares (`jpeg` at its quality, or `raw`);
//! 3. lands in a one-frame mailbox that a sender thread drains. When the
//!    link is congested the sender is still blocked on the previous frame,
//!    and the new frame replaces the unsent one: the newest frame wins and
//!    no queue of stale frames builds up behind the link.
//!
//! Subscribers decode with [`decode_image`].

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use bytes::Bytes;
use rlmesh_wire::part::MAX_ENCODED_OVERHEAD;
use rlmesh_wire::{Encoding, Sample, StreamSpec};
use tracing::warn;

use crate::error::SessionError;
use crate::sample::SamplePublisher;

/// The token bucket holds this much of the budget.
const BUCKET_NS: i64 = 100_000_000;

/// What happened to one offered frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Offer {
    /// Encoded and handed to the sender.
    Queued {
        /// The encoded size.
        bytes: usize,
    },
    /// Over the bitrate budget: not encoded, not sent.
    OverBudget,
    /// Coalesced by the publication plan or the rate ceiling: not encoded,
    /// not sent.
    Coalesced,
}

/// Counts for one image publisher.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MediaStats {
    /// Frames offered.
    pub offered: u64,
    /// Skipped before encoding: over the bitrate budget.
    pub skipped_over_budget: u64,
    /// Skipped before encoding: the publication plan coalesced it.
    pub coalesced: u64,
    /// Frames encoded.
    pub encoded: u64,
    /// Encoded, then replaced by a newer frame before the sender got to it.
    pub superseded: u64,
    /// Frames sent.
    pub sent: u64,
    /// Encoded bytes sent.
    pub bytes_sent: u64,
    /// Sends that failed.
    pub send_errors: u64,
    /// Total encode time.
    pub encode_ns_total: u64,
    /// Longest encode.
    pub encode_ns_max: u64,
}

#[derive(Debug)]
struct Bucket {
    /// Bytes per second; `None`: no cap.
    rate: Option<f64>,
    tokens: f64,
    capacity: f64,
    last_ns: i64,
}

impl Bucket {
    /// A bucket for `max_bitrate_bps`, holding at least one `max_frame`.
    fn new(max_bitrate_bps: Option<u64>, max_frame: usize, now_ns: i64) -> Self {
        let rate = max_bitrate_bps.map(|bps| bps as f64 / 8.0);
        let capacity = rate.map_or(0.0, |r| (r * BUCKET_NS as f64 / 1e9).max(max_frame as f64));
        Self {
            rate,
            tokens: capacity,
            capacity,
            last_ns: now_ns,
        }
    }

    fn refill(&mut self, now_ns: i64) {
        if let Some(rate) = self.rate {
            let elapsed = (now_ns - self.last_ns).max(0) as f64 / 1e9;
            self.tokens = (self.tokens + elapsed * rate).min(self.capacity);
        }
        self.last_ns = now_ns;
    }

    fn has_budget(&mut self, now_ns: i64) -> bool {
        self.refill(now_ns);
        self.rate.is_none() || self.tokens >= 0.0
    }

    fn spend(&mut self, bytes: usize) {
        if self.rate.is_some() {
            self.tokens -= bytes as f64;
        }
    }

    fn refund(&mut self, bytes: usize) {
        if self.rate.is_some() {
            self.tokens = (self.tokens + bytes as f64).min(self.capacity);
        }
    }
}

struct Frame {
    capture_ns: i64,
    bytes: Bytes,
}

struct Inner {
    publisher: SamplePublisher,
    width: u16,
    height: u16,
    epoch: Instant,
    bucket: Mutex<Bucket>,
    slot: Mutex<Option<Frame>>,
    ready: Condvar,
    stop: AtomicBool,
    stats: Mutex<MediaStats>,
}

impl Inner {
    fn now_ns(&self) -> i64 {
        i64::try_from(self.epoch.elapsed().as_nanos()).unwrap_or(i64::MAX)
    }

    fn bump(&self, f: impl FnOnce(&mut MediaStats)) {
        if let Ok(mut stats) = self.stats.lock() {
            f(&mut stats);
        }
    }

    fn send_loop(&self) {
        loop {
            let frame = {
                let Ok(mut slot) = self.slot.lock() else {
                    return;
                };
                loop {
                    if self.stop.load(Ordering::Relaxed) {
                        return;
                    }
                    if let Some(frame) = slot.take() {
                        break frame;
                    }
                    slot = match self.ready.wait(slot) {
                        Ok(slot) => slot,
                        Err(_) => return,
                    };
                }
            };
            let len = frame.bytes.len() as u64;
            // A BLOCK channel waits here while the link is congested; newer
            // frames meanwhile replace each other in the slot.
            match self.publisher.publish(frame.capture_ns, frame.bytes) {
                Ok(Some(_)) => self.bump(|s| {
                    s.sent += 1;
                    s.bytes_sent += len;
                }),
                Ok(None) => self.bump(|s| s.coalesced += 1),
                Err(err) => {
                    warn!(channel = %self.publisher.spec().channel, %err, "frame send failed");
                    self.bump(|s| s.send_errors += 1);
                }
            }
        }
    }
}

/// Publishes frames of one image channel. Dropping it stops its sender.
pub struct ImagePublisher {
    inner: Arc<Inner>,
    sender: Option<JoinHandle<()>>,
}

impl ImagePublisher {
    /// Wraps the publisher of a uint8 `[H, W, 3]` image part, with its
    /// bitrate budget (`None`: no budget).
    pub fn new(
        publisher: SamplePublisher,
        max_bitrate_bps: Option<u64>,
    ) -> Result<Self, SessionError> {
        let spec = publisher.spec();
        let (height, width) = image_dims(spec)?;
        let max_frame = usize::from(height) * usize::from(width) * 3 + MAX_ENCODED_OVERHEAD;
        let inner = Arc::new(Inner {
            width,
            height,
            epoch: Instant::now(),
            bucket: Mutex::new(Bucket::new(max_bitrate_bps, max_frame, 0)),
            slot: Mutex::new(None),
            ready: Condvar::new(),
            stop: AtomicBool::new(false),
            stats: Mutex::new(MediaStats::default()),
            publisher,
        });
        let sender = std::thread::Builder::new()
            .name(format!("rlmesh-media-{}", inner.publisher.spec().channel))
            .spawn({
                let inner = inner.clone();
                move || inner.send_loop()
            })
            .map_err(|err| SessionError::Media(format!("sender thread: {err}")))?;
        Ok(Self {
            inner,
            sender: Some(sender),
        })
    }

    /// The stream.
    pub fn spec(&self) -> &StreamSpec {
        self.inner.publisher.spec()
    }

    /// The counters so far.
    pub fn stats(&self) -> MediaStats {
        self.inner.stats.lock().map(|s| *s).unwrap_or_default()
    }

    /// Offers one frame captured at `capture_ns` (this participant's clock),
    /// as `H * W * 3` row-major RGB bytes. Never blocks on the link.
    pub fn offer(&self, capture_ns: i64, rgb: &[u8]) -> Result<Offer, SessionError> {
        let inner = &self.inner;
        let expected = usize::from(inner.width) * usize::from(inner.height) * 3;
        if rgb.len() != expected {
            return Err(SessionError::InvalidValue {
                channel: self.spec().channel,
                reason: format!("frame is {} bytes, not {expected}", rgb.len()),
            });
        }
        inner.bump(|s| s.offered += 1);
        let now = inner.now_ns();
        if !inner.publisher.due(now) {
            inner.bump(|s| s.coalesced += 1);
            return Ok(Offer::Coalesced);
        }
        let budget = inner
            .bucket
            .lock()
            .map(|mut b| b.has_budget(now))
            .unwrap_or(true);
        if !budget {
            inner.bump(|s| s.skipped_over_budget += 1);
            return Ok(Offer::OverBudget);
        }
        let started = Instant::now();
        let bytes = match self.spec().encoding {
            Encoding::Raw => Bytes::copy_from_slice(rgb),
            Encoding::Jpeg { quality } => {
                Bytes::from(encode_jpeg(rgb, inner.width, inner.height, quality)?)
            }
        };
        let encode_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let len = bytes.len();
        if let Ok(mut bucket) = inner.bucket.lock() {
            bucket.spend(len);
        }
        let replaced = inner
            .slot
            .lock()
            .map(|mut slot| slot.replace(Frame { capture_ns, bytes }))
            .ok()
            .flatten();
        inner.ready.notify_one();
        if let Some(old) = &replaced
            && let Ok(mut bucket) = inner.bucket.lock()
        {
            bucket.refund(old.bytes.len());
        }
        inner.bump(|s| {
            s.encoded += 1;
            s.encode_ns_total += encode_ns;
            s.encode_ns_max = s.encode_ns_max.max(encode_ns);
            if replaced.is_some() {
                s.superseded += 1;
            }
        });
        Ok(Offer::Queued { bytes: len })
    }
}

impl Drop for ImagePublisher {
    fn drop(&mut self) {
        self.inner.stop.store(true, Ordering::Relaxed);
        self.inner.ready.notify_all();
        if let Some(sender) = self.sender.take() {
            // A sender blocked on a congested link returns once Zenoh does.
            let _ = sender.join();
        }
    }
}

fn image_dims(spec: &StreamSpec) -> Result<(u16, u16), SessionError> {
    let dims = match spec.fields.as_slice() {
        [field] => field
            .rgb8_dims()
            .and_then(|(h, w)| Some((u16::try_from(h).ok()?, u16::try_from(w).ok()?))),
        _ => None,
    };
    dims.ok_or_else(|| SessionError::WrongChannel {
        channel: spec.channel,
        reason: "not one uint8 [H, W, 3] image field with 16-bit dimensions".into(),
    })
}

/// A baseline 4:2:0 JPEG of `rgb` (the subsampling every mainstream encoder
/// uses at every quality).
pub fn encode_jpeg(
    rgb: &[u8],
    width: u16,
    height: u16,
    quality: u8,
) -> Result<Vec<u8>, SessionError> {
    let mut encoded = Vec::with_capacity(rgb.len() / 8);
    let mut encoder = jpeg_encoder::Encoder::new(&mut encoded, quality);
    encoder.set_sampling_factor(jpeg_encoder::SamplingFactor::R_4_2_0);
    encoder
        .encode(rgb, width, height, jpeg_encoder::ColorType::Rgb)
        .map_err(|err| SessionError::Media(format!("jpeg encode: {err}")))?;
    Ok(encoded)
}

/// A received image sample as `H * W * 3` RGB bytes: decoded from JPEG, or
/// the raw payload as is. Checks the declared dimensions (a mismatch is the
/// spec's `decode_mismatch`).
pub fn decode_image(spec: &StreamSpec, sample: &Sample) -> Result<Vec<u8>, SessionError> {
    decode_payload(spec, &sample.payload)
}

/// [`decode_image`] on a payload.
pub fn decode_payload(spec: &StreamSpec, leaf: &[u8]) -> Result<Vec<u8>, SessionError> {
    let (height, width) = image_dims(spec)?;
    let pixels = match spec.encoding {
        Encoding::Raw => leaf.to_vec(),
        Encoding::Jpeg { .. } => {
            let decoded = image::load_from_memory_with_format(leaf, image::ImageFormat::Jpeg)
                .map_err(|err| SessionError::Media(format!("{} jpeg: {err}", spec.channel)))?;
            if (decoded.width(), decoded.height()) != (u32::from(width), u32::from(height)) {
                return Err(SessionError::InvalidValue {
                    channel: spec.channel,
                    reason: format!(
                        "jpeg is {}x{}, declared {width}x{height}",
                        decoded.width(),
                        decoded.height()
                    ),
                });
            }
            decoded.to_rgb8().into_raw()
        }
    };
    let expected = usize::from(width) * usize::from(height) * 3;
    if pixels.len() != expected {
        return Err(SessionError::InvalidValue {
            channel: spec.channel,
            reason: format!("image is {} bytes, not {expected}", pixels.len()),
        });
    }
    Ok(pixels)
}

/// A synthetic camera frame with the statistics of a real one (smooth
/// gradients, edges that move with `t`, and sensor noise), so encoded sizes
/// are realistic. For tests, demos and the benchmark.
pub fn synthetic_frame(width: usize, height: usize, t: u32, out: &mut Vec<u8>) {
    out.clear();
    out.reserve(width * height * 3);
    let mut noise = 0x9E37_79B9_u32 ^ t.wrapping_mul(0x85EB_CA6B);
    let shift = (t as usize).wrapping_mul(7);
    for y in 0..height {
        for x in 0..width {
            noise ^= noise << 13;
            noise ^= noise >> 17;
            noise ^= noise << 5;
            let n = (noise & 0x1F) as i32 - 16;
            let band = (((x + shift) / 40 + y / 40) % 2) as i32 * 60;
            let r = (x * 255 / width.max(1)) as i32 + band + n;
            let g = (y * 255 / height.max(1)) as i32 + n;
            let b = (((x + y + shift) % 256) as i32 / 2) + band / 2 + n;
            out.extend([r, g, b].map(|v| v.clamp(0, 255) as u8));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bucket_skips_frames_over_budget_and_refills_at_the_cap() {
        // 8 Mbit/s = 1 MB/s; the bucket holds 100 kB.
        let mut bucket = Bucket::new(Some(8_000_000), 50_000, 0);
        let mut sent = 0_usize;
        // 50 kB frames offered at 100 Hz for one second: twice the budget.
        for i in 0..100_i64 {
            if bucket.has_budget(i * 10_000_000) {
                bucket.spend(50_000);
                sent += 1;
            }
        }
        let rate = sent * 50_000;
        assert!((1_000_000..=1_150_000).contains(&rate), "{sent} frames");
        let mut free = Bucket::new(None, 0, 0);
        assert!(free.has_budget(0));
        // A raw VGA frame (921 600 B) is larger than 100 ms at 8 Mbit/s: the
        // bucket holds one anyway, so frames still flow at about 1 Hz.
        let mut raw = Bucket::new(Some(8_000_000), 921_600, 0);
        let sent = (0..300_i64)
            .filter(|i| {
                let ok = raw.has_budget(i * 10_000_000);
                if ok {
                    raw.spend(921_600);
                }
                ok
            })
            .count();
        assert!((3..=5).contains(&sent), "{sent} raw frames in 3 s");
    }

    #[test]
    fn jpeg_round_trips_within_a_lossy_bound() {
        let (w, h) = (64_usize, 48_usize);
        let mut frame = Vec::new();
        synthetic_frame(w, h, 3, &mut frame);
        let jpeg = encode_jpeg(&frame, w as u16, h as u16, 90).expect("encode");
        assert!(jpeg.len() < frame.len(), "{} bytes", jpeg.len());
        let decoded = image::load_from_memory_with_format(&jpeg, image::ImageFormat::Jpeg)
            .expect("decode")
            .to_rgb8()
            .into_raw();
        let mean_err = frame
            .iter()
            .zip(&decoded)
            .map(|(a, b)| f64::from(a.abs_diff(*b)))
            .sum::<f64>()
            / frame.len() as f64;
        assert!(mean_err < 12.0, "mean error {mean_err}");
    }
}
