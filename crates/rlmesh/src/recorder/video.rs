//! Recording: encode HWC frames to an `.mp4`, AV1 in process by default or
//! H.264 through the user's ffmpeg (`h264`).
//!
//! `VideoWriter` is the recording sink dual of the display sinks: it takes the
//! same RGB frames the viewer already normalizes (via `rgb_from_hwc`),
//! encodes them to AV1 with `rav1e` (pure Rust, royalty-free), and muxes the
//! bitstream into a plain MP4 (`av01` sample entry) as each packet comes out --
//! streaming, so peak memory is one frame plus the encoder state, never the whole
//! episode. The MP4 muxer is a purpose-built minimum for our one shape: a single
//! 8-bit 4:2:0 video track, one frame per sample, `pts == dts` (no reordering).
//!
//! The codec-configuration box (`av1C`) is taken verbatim from
//! `rav1e`'s `container_sequence_header()`, so it always matches the bitstream.

use std::fs::File;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use rav1e::config::SpeedSettings;
use rav1e::prelude::*;

use image::RgbImage;

use super::VideoCodec;

mod h264;

/// Fixed MP4 offset where sample data begins: `ftyp` (32) + `mdat` header (16).
const MDAT_DATA_OFFSET: u64 = 32 + 16;

/// What [`VideoWriter::finish`] reports for the media manifest row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    /// Frames written.
    pub frame_count: u32,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

/// One encoded frame's location in the `mdat`, gathered for the sample tables.
struct Sample {
    size: u32,
    sync: bool,
}

/// How a recording is encoded: the recorder's `codec`, `quality` and
/// `ffmpeg` options, checked once at construction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Encoding {
    /// In-process rav1e; `quality` is 1..=100 (higher is better/larger).
    Av1 { quality: u8 },
    /// libx264 in this ffmpeg; `quality` does not apply.
    H264 { ffmpeg: PathBuf },
}

impl Default for Encoding {
    fn default() -> Self {
        Self::Av1 { quality: 60 }
    }
}

impl Encoding {
    /// `ffmpeg` (default `ffmpeg` on `PATH`) is checked for libx264 only for
    /// [`VideoCodec::H264`]: a missing or encoder-less ffmpeg is an error
    /// naming what to install, never a fallback to AV1.
    pub fn resolve(codec: VideoCodec, quality: u8, ffmpeg: Option<&Path>) -> io::Result<Self> {
        match codec {
            VideoCodec::Av1 => Ok(Self::Av1 { quality }),
            VideoCodec::H264 => {
                let ffmpeg = ffmpeg.map_or_else(|| PathBuf::from("ffmpeg"), Path::to_path_buf);
                h264::require(&ffmpeg)?;
                Ok(Self::H264 { ffmpeg })
            }
        }
    }
}

/// Streaming MP4 recorder for a single camera's frames.
pub struct VideoWriter {
    encoder: Encoder,
    width: u32,
    height: u32,
}

enum Encoder {
    Av1(Box<Av1Writer>),
    H264(h264::H264Writer),
}

impl VideoWriter {
    /// Open `path` for `width` x `height` frames at `fps`.
    ///
    /// AV1 4:2:0 crops an odd side by one pixel; H.264 pads it by one, as the
    /// managed runner does. Every [`push`](Self::push) must then hand the same
    /// original `width`/`height`.
    pub fn create(
        path: &Path,
        width: u32,
        height: u32,
        fps: u32,
        encoding: &Encoding,
    ) -> io::Result<Self> {
        let encoder = match encoding {
            Encoding::Av1 { quality } => Encoder::Av1(Box::new(Av1Writer::create(
                path, width, height, fps, *quality,
            )?)),
            Encoding::H264 { ffmpeg } => {
                if width == 0 || height == 0 {
                    return Err(io::Error::other(format!(
                        "frame too small to record: {width}x{height}"
                    )));
                }
                Encoder::H264(h264::H264Writer::create(
                    ffmpeg,
                    path,
                    width,
                    height,
                    fps.max(1),
                )?)
            }
        };
        Ok(Self {
            encoder,
            width,
            height,
        })
    }

    /// Encode one HWC uint8 frame (`channels` = 1/3/4).
    ///
    /// Rejects a frame whose dimensions differ from the ones [`create`](Self::create)
    /// fixed -- a camera must not change resolution mid-episode.
    pub fn push(&mut self, buf: &[u8], width: u32, height: u32, channels: u32) -> io::Result<()> {
        if width != self.width || height != self.height {
            return Err(io::Error::other(format!(
                "frame size changed mid-recording: expected {}x{}, got {width}x{height}",
                self.width, self.height
            )));
        }
        match &mut self.encoder {
            Encoder::Av1(writer) => writer.push(buf, width, height, channels),
            Encoder::H264(writer) => {
                let not_hwc = || io::Error::other("frame is not a 1/3/4-channel HWC image");
                match channels {
                    3 => {
                        let needed = (width as usize)
                            .checked_mul(height as usize)
                            .and_then(|n| n.checked_mul(3))
                            .ok_or_else(not_hwc)?;
                        writer.push(buf.get(..needed).ok_or_else(not_hwc)?)
                    }
                    1 | 4 => writer.push(
                        rgb_from_hwc(buf, width, height, channels)
                            .ok_or_else(not_hwc)?
                            .as_raw(),
                    ),
                    _ => Err(not_hwc()),
                }
            }
        }
    }

    /// Finalize the file.
    pub fn finish(self) -> io::Result<Stats> {
        match self.encoder {
            Encoder::Av1(writer) => writer.finish(),
            Encoder::H264(writer) => writer.finish(),
        }
    }
}

/// Streaming AV1-to-MP4 encoder.
///
/// The `y`/`u`/`v` fields are per-frame conversion scratch, allocated once at
/// [`create`](Self::create) and overwritten each [`push`](Self::push) so the
/// hot path performs no per-frame heap allocation.
struct Av1Writer {
    ctx: Context<u8>,
    out: BufWriter<File>,
    width: u32,
    height: u32,
    fps: u32,
    mdat_len: u64,
    samples: Vec<Sample>,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

impl Av1Writer {
    /// Open `path` and prepare to record even-cropped `width` x `height` frames.
    ///
    /// AV1 4:2:0 needs even dimensions, so odd inputs are cropped by one pixel;
    /// every [`push`](Self::push) must then hand the same original `width`/`height`.
    /// `quality` is 1..=100 (higher is better/larger), mapped to the AV1 quantizer.
    fn create(path: &Path, width: u32, height: u32, fps: u32, quality: u8) -> io::Result<Self> {
        let (ew, eh) = (width & !1, height & !1);
        if ew == 0 || eh == 0 {
            return Err(io::Error::other(format!(
                "frame too small to record: {width}x{height}"
            )));
        }
        if ew > 0xFFFF || eh > 0xFFFF {
            return Err(io::Error::other(format!(
                "frame too large to record: {width}x{height} (max 65535 per side)"
            )));
        }
        let fps = fps.max(1);
        let enc = EncoderConfig {
            width: ew as usize,
            height: eh as usize,
            speed_settings: SpeedSettings::from_preset(10),
            time_base: Rational::new(1, fps as u64),
            chroma_sampling: ChromaSampling::Cs420,
            low_latency: true,
            min_key_frame_interval: 0,
            max_key_frame_interval: 150,
            quantizer: quantizer_for(quality),
            ..Default::default()
        };
        let cfg = Config::new().with_encoder_config(enc).with_threads(0);
        let ctx: Context<u8> = cfg
            .new_context()
            .map_err(|e| io::Error::other(format!("rav1e context: {e:?}")))?;

        let mut out = BufWriter::new(File::create(path)?);
        write_ftyp(&mut out)?;
        write_mdat_header(&mut out)?;

        Ok(Self {
            ctx,
            out,
            width,
            height,
            fps,
            mdat_len: 0,
            samples: Vec::new(),
            y: vec![0u8; ew as usize * eh as usize],
            u: vec![0u8; (ew as usize / 2) * (eh as usize / 2)],
            v: vec![0u8; (ew as usize / 2) * (eh as usize / 2)],
        })
    }

    /// Encode one HWC uint8 frame (`channels` = 1/3/4) and mux any ready packets.
    fn push(&mut self, buf: &[u8], width: u32, height: u32, channels: u32) -> io::Result<()> {
        let (w, h, c) = (width as usize, height as usize, channels as usize);
        let needed = w
            .checked_mul(h)
            .and_then(|n| n.checked_mul(c))
            .filter(|&n| matches!(c, 1 | 3 | 4) && buf.len() >= n)
            .ok_or_else(|| io::Error::other("frame is not a 1/3/4-channel HWC image"))?;
        let ew = self.width & !1;
        if c == 3 {
            rgb_to_i420(
                &buf[..needed],
                w,
                h,
                (&mut self.y, &mut self.u, &mut self.v),
            );
        } else {
            let rgb = rgb_from_hwc(buf, width, height, channels)
                .ok_or_else(|| io::Error::other("frame is not a 1/3/4-channel HWC image"))?;
            rgb_to_i420(rgb.as_raw(), w, h, (&mut self.y, &mut self.u, &mut self.v));
        }

        let mut frame = self.ctx.new_frame();
        frame.planes[0].copy_from_raw_u8(&self.y, ew as usize, 1);
        frame.planes[1].copy_from_raw_u8(&self.u, (ew / 2) as usize, 1);
        frame.planes[2].copy_from_raw_u8(&self.v, (ew / 2) as usize, 1);
        self.ctx
            .send_frame(frame)
            .map_err(|e| io::Error::other(format!("rav1e send_frame: {e:?}")))?;
        self.drain(false)
    }

    /// Mux every packet the encoder has ready.
    ///
    /// When `flushing`, polls through `NeedMoreData`/`Encoded` until
    /// `LimitReached`, bounded so an encoder that never reports completion
    /// (rav1e documents no post-flush guarantee) errors out instead of
    /// spinning forever.
    fn drain(&mut self, flushing: bool) -> io::Result<()> {
        let mut spins = 0u32;
        loop {
            match self.ctx.receive_packet() {
                Ok(pkt) => {
                    spins = 0;
                    self.write_packet(&pkt.data, pkt.frame_type == FrameType::KEY)?;
                }
                Err(EncoderStatus::LimitReached) => return Ok(()),
                Err(EncoderStatus::NeedMoreData) | Err(EncoderStatus::Encoded) => {
                    if !flushing {
                        return Ok(());
                    }
                    spins += 1;
                    if spins > 100_000 {
                        return Err(io::Error::other(
                            "rav1e did not report completion after flush",
                        ));
                    }
                }
                Err(e) => return Err(io::Error::other(format!("rav1e receive: {e:?}"))),
            }
        }
    }

    /// Flush the encoder, write the `moov`, and finalize the file.
    fn finish(mut self) -> io::Result<Stats> {
        self.ctx.flush();
        self.drain(true)?;
        if self.samples.is_empty() {
            return Err(io::Error::other("no frames were recorded"));
        }

        let (ew, eh) = (self.width & !1, self.height & !1);
        let av1c = self.ctx.container_sequence_header();
        let moov = build_moov(ew, eh, self.fps, &self.samples, &av1c);
        self.out.write_all(&moov)?;

        let mdat_box_size = 16 + self.mdat_len;
        self.out.seek(SeekFrom::Start(40))?;
        self.out.write_all(&mdat_box_size.to_be_bytes())?;
        self.out.flush()?;

        Ok(Stats {
            frame_count: self.samples.len() as u32,
            width: ew,
            height: eh,
        })
    }

    fn write_packet(&mut self, data: &[u8], sync: bool) -> io::Result<()> {
        self.out.write_all(data)?;
        self.mdat_len += data.len() as u64;
        self.samples.push(Sample {
            size: data.len() as u32,
            sync,
        });
        Ok(())
    }
}

/// One HWC uint8 frame queued for the encoder thread.
struct FrameMsg {
    buf: Vec<u8>,
    width: u32,
    height: u32,
    channels: u32,
}

/// A `VideoWriter` running on its own encoder thread.
///
/// `push` hands the frame to a bounded queue and returns immediately, so the
/// caller (the eval loop) overlaps env stepping and model inference with
/// encoding instead of stalling on it; the queue bound keeps at most a few
/// frames in flight. An encode error surfaces on the next `push` (or at
/// `finish`), which is also when a dimension change is rejected.
pub struct ThreadedVideoWriter {
    sender: Option<std::sync::mpsc::SyncSender<FrameMsg>>,
    worker: Option<std::thread::JoinHandle<io::Result<Stats>>>,
}

impl ThreadedVideoWriter {
    /// Open the file and start the encoder thread (see [`VideoWriter::create`]).
    pub fn create(
        path: &Path,
        width: u32,
        height: u32,
        fps: u32,
        encoding: &Encoding,
    ) -> io::Result<Self> {
        let mut writer = VideoWriter::create(path, width, height, fps, encoding)?;
        let (sender, receiver) = std::sync::mpsc::sync_channel::<FrameMsg>(3);
        let worker = std::thread::Builder::new()
            .name("rlmesh-video-encode".to_string())
            .spawn(move || {
                let mut failed: Option<io::Error> = None;
                while let Ok(msg) = receiver.recv() {
                    if failed.is_none()
                        && let Err(err) = writer.push(&msg.buf, msg.width, msg.height, msg.channels)
                    {
                        failed = Some(err);
                    }
                }
                match failed {
                    Some(err) => Err(err),
                    None => writer.finish(),
                }
            })?;
        Ok(Self {
            sender: Some(sender),
            worker: Some(worker),
        })
    }

    /// Queue one frame for encoding, blocking only when the queue is full.
    pub fn push(&mut self, buf: Vec<u8>, width: u32, height: u32, channels: u32) -> io::Result<()> {
        let Some(sender) = self.sender.as_ref() else {
            return Err(io::Error::other("video writer already finished"));
        };
        let msg = FrameMsg {
            buf,
            width,
            height,
            channels,
        };
        if sender.send(msg).is_ok() {
            return Ok(());
        }
        self.sender = None;
        match self.join() {
            Err(err) => Err(err),
            Ok(_) => Err(io::Error::other("encoder thread exited unexpectedly")),
        }
    }

    /// Close the queue, wait for the encoder to drain, and finalize the file.
    pub fn finish(&mut self) -> io::Result<Stats> {
        self.sender = None;
        self.join()
    }

    fn join(&mut self) -> io::Result<Stats> {
        let Some(worker) = self.worker.take() else {
            return Err(io::Error::other("video writer already finished"));
        };
        worker
            .join()
            .map_err(|_| io::Error::other("encoder thread panicked"))?
    }
}

/// Map a 1..=100 quality (higher is better) to a rav1e quantizer (0 best, 255 worst).
fn quantizer_for(quality: u8) -> usize {
    let q = quality.clamp(1, 100) as usize;
    (100 - q) * 255 / 100
}

/// BT.601 limited-range RGB -> I420 planar, cropping to even dimensions.
///
/// Writes into caller-provided plane buffers (sized for the even-cropped
/// dimensions) so the per-frame hot path allocates nothing.
fn rgb_to_i420(rgb: &[u8], w: usize, h: usize, out: (&mut [u8], &mut [u8], &mut [u8])) {
    let (y, u, v) = out;
    let (ew, eh) = (w & !1, h & !1);
    for row in 0..eh {
        for col in 0..ew {
            let i = (row * w + col) * 3;
            let (r, g, b) = (rgb[i] as i32, rgb[i + 1] as i32, rgb[i + 2] as i32);
            y[row * ew + col] = (((66 * r + 129 * g + 25 * b + 128) >> 8) + 16) as u8;
        }
    }
    for row in (0..eh).step_by(2) {
        for col in (0..ew).step_by(2) {
            let (mut sr, mut sg, mut sb) = (0i32, 0i32, 0i32);
            for dy in 0..2 {
                for dx in 0..2 {
                    let i = ((row + dy) * w + col + dx) * 3;
                    sr += rgb[i] as i32;
                    sg += rgb[i + 1] as i32;
                    sb += rgb[i + 2] as i32;
                }
            }
            let (r, g, b) = (sr / 4, sg / 4, sb / 4);
            let ci = (row / 2) * (ew / 2) + col / 2;
            u[ci] = (((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128) as u8;
            v[ci] = (((112 * r - 94 * g - 18 * b + 128) >> 8) + 128) as u8;
        }
    }
}

fn write_ftyp(out: &mut impl Write) -> io::Result<()> {
    let mut b = Vec::new();
    open_box(&mut b, b"ftyp");
    b.extend_from_slice(b"isom");
    b.extend_from_slice(&0x200u32.to_be_bytes());
    for brand in [b"isom", b"iso2", b"av01", b"mp41"] {
        b.extend_from_slice(brand);
    }
    close_box(&mut b, 0);
    out.write_all(&b)
}

/// `mdat` header with a 64-bit largesize (patched in [`VideoWriter::finish`]).
fn write_mdat_header(out: &mut impl Write) -> io::Result<()> {
    out.write_all(&1u32.to_be_bytes())?;
    out.write_all(b"mdat")?;
    out.write_all(&0u64.to_be_bytes())
}

fn build_moov(ew: u32, eh: u32, fps: u32, samples: &[Sample], av1c: &[u8]) -> Vec<u8> {
    let n = samples.len() as u32;
    let mut b = Vec::new();
    let moov = open_box(&mut b, b"moov");

    full_box(&mut b, b"mvhd", 0, 0, |b| {
        b.extend_from_slice(&[0u8; 8]);
        b.extend_from_slice(&fps.to_be_bytes());
        b.extend_from_slice(&n.to_be_bytes());
        b.extend_from_slice(&0x0001_0000u32.to_be_bytes());
        b.extend_from_slice(&0x0100u16.to_be_bytes());
        b.extend_from_slice(&[0u8; 10]);
        matrix(b);
        b.extend_from_slice(&[0u8; 24]);
        b.extend_from_slice(&2u32.to_be_bytes());
    });

    let trak = open_box(&mut b, b"trak");
    full_box(&mut b, b"tkhd", 0, 0x7, |b| {
        b.extend_from_slice(&[0u8; 8]);
        b.extend_from_slice(&1u32.to_be_bytes());
        b.extend_from_slice(&0u32.to_be_bytes());
        b.extend_from_slice(&n.to_be_bytes());
        b.extend_from_slice(&[0u8; 16]);
        matrix(b);
        b.extend_from_slice(&(ew << 16).to_be_bytes());
        b.extend_from_slice(&(eh << 16).to_be_bytes());
    });

    let mdia = open_box(&mut b, b"mdia");
    full_box(&mut b, b"mdhd", 0, 0, |b| {
        b.extend_from_slice(&[0u8; 8]);
        b.extend_from_slice(&fps.to_be_bytes());
        b.extend_from_slice(&n.to_be_bytes());
        b.extend_from_slice(&0x55c4u16.to_be_bytes());
        b.extend_from_slice(&0u16.to_be_bytes());
    });
    full_box(&mut b, b"hdlr", 0, 0, |b| {
        b.extend_from_slice(&0u32.to_be_bytes());
        b.extend_from_slice(b"vide");
        b.extend_from_slice(&[0u8; 12]);
        b.extend_from_slice(b"VideoHandler\0");
    });

    let minf = open_box(&mut b, b"minf");
    full_box(&mut b, b"vmhd", 0, 1, |b| {
        b.extend_from_slice(&0u16.to_be_bytes());
        b.extend_from_slice(&[0u8; 6]);
    });
    let dinf = open_box(&mut b, b"dinf");
    full_box(&mut b, b"dref", 0, 0, |b| {
        b.extend_from_slice(&1u32.to_be_bytes());
        full_box(b, b"url ", 0, 1, |_| {});
    });
    close_box(&mut b, dinf);

    let stbl = open_box(&mut b, b"stbl");
    full_box(&mut b, b"stsd", 0, 0, |b| {
        b.extend_from_slice(&1u32.to_be_bytes());
        let av01 = open_box(b, b"av01");
        b.extend_from_slice(&[0u8; 6]);
        b.extend_from_slice(&1u16.to_be_bytes());
        b.extend_from_slice(&[0u8; 16]);
        b.extend_from_slice(&(ew as u16).to_be_bytes());
        b.extend_from_slice(&(eh as u16).to_be_bytes());
        b.extend_from_slice(&0x0048_0000u32.to_be_bytes());
        b.extend_from_slice(&0x0048_0000u32.to_be_bytes());
        b.extend_from_slice(&0u32.to_be_bytes());
        b.extend_from_slice(&1u16.to_be_bytes());
        b.extend_from_slice(&[0u8; 32]);
        b.extend_from_slice(&0x0018u16.to_be_bytes());
        b.extend_from_slice(&0xffffu16.to_be_bytes());
        let av1c_box = open_box(b, b"av1C");
        b.extend_from_slice(av1c);
        close_box(b, av1c_box);
        write_colr_bt601(b);
        close_box(b, av01);
    });
    full_box(&mut b, b"stts", 0, 0, |b| {
        b.extend_from_slice(&1u32.to_be_bytes());
        b.extend_from_slice(&n.to_be_bytes());
        b.extend_from_slice(&1u32.to_be_bytes());
    });
    full_box(&mut b, b"stsc", 0, 0, |b| {
        b.extend_from_slice(&1u32.to_be_bytes());
        b.extend_from_slice(&1u32.to_be_bytes());
        b.extend_from_slice(&n.to_be_bytes());
        b.extend_from_slice(&1u32.to_be_bytes());
    });
    full_box(&mut b, b"stsz", 0, 0, |b| {
        b.extend_from_slice(&0u32.to_be_bytes());
        b.extend_from_slice(&n.to_be_bytes());
        for s in samples {
            b.extend_from_slice(&s.size.to_be_bytes());
        }
    });
    full_box(&mut b, b"co64", 0, 0, |b| {
        b.extend_from_slice(&1u32.to_be_bytes());
        b.extend_from_slice(&MDAT_DATA_OFFSET.to_be_bytes());
    });
    let sync: Vec<u32> = samples
        .iter()
        .enumerate()
        .filter(|(_, s)| s.sync)
        .map(|(i, _)| i as u32 + 1)
        .collect();
    full_box(&mut b, b"stss", 0, 0, |b| {
        b.extend_from_slice(&(sync.len() as u32).to_be_bytes());
        for idx in &sync {
            b.extend_from_slice(&idx.to_be_bytes());
        }
    });
    close_box(&mut b, stbl);
    close_box(&mut b, minf);
    close_box(&mut b, mdia);
    close_box(&mut b, trak);
    close_box(&mut b, moov);
    b
}

fn matrix(b: &mut Vec<u8>) {
    for v in [0x0001_0000u32, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000] {
        b.extend_from_slice(&v.to_be_bytes());
    }
}

/// `colr` (nclx) box declaring BT.601 limited-range, matching [`rgb_to_i420`].
///
/// Without it a player guesses the matrix from resolution (BT.709 for >=720p),
/// which would shift the colors of a recorded HD camera away from what was drawn.
fn write_colr_bt601(b: &mut Vec<u8>) {
    let colr = open_box(b, b"colr");
    b.extend_from_slice(b"nclx");
    b.extend_from_slice(&6u16.to_be_bytes());
    b.extend_from_slice(&6u16.to_be_bytes());
    b.extend_from_slice(&6u16.to_be_bytes());
    b.push(0x00);
    close_box(b, colr);
}

/// Open a box, returning the index of its size field for [`close_box`].
fn open_box(b: &mut Vec<u8>, box_type: &[u8; 4]) -> usize {
    let pos = b.len();
    b.extend_from_slice(&[0u8; 4]);
    b.extend_from_slice(box_type);
    pos
}

/// Patch a box's size field to span from `pos` to the current end.
fn close_box(b: &mut [u8], pos: usize) {
    let size = (b.len() - pos) as u32;
    b[pos..pos + 4].copy_from_slice(&size.to_be_bytes());
}

/// Write a full box (version + flags header) with `body` appended, then close it.
fn full_box(
    b: &mut Vec<u8>,
    box_type: &[u8; 4],
    version: u8,
    flags: u32,
    body: impl FnOnce(&mut Vec<u8>),
) {
    let pos = open_box(b, box_type);
    b.push(version);
    b.extend_from_slice(&flags.to_be_bytes()[1..]);
    body(b);
    close_box(b, pos);
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

#[cfg(test)]
mod tests {
    use super::*;

    const AV1: Encoding = Encoding::Av1 { quality: 60 };

    /// One synthetic RGB gradient frame, HWC.
    fn frame(w: u32, h: u32, t: usize) -> Vec<u8> {
        let mut rgb = vec![0u8; (w * h * 3) as usize];
        for yy in 0..h as usize {
            for xx in 0..w as usize {
                let i = (yy * w as usize + xx) * 3;
                rgb[i] = ((xx + t * 4) % 256) as u8;
                rgb[i + 1] = ((yy + t * 2) % 256) as u8;
                rgb[i + 2] = (t * 4 % 256) as u8;
            }
        }
        rgb
    }

    fn record(path: &Path, w: u32, h: u32, n: usize) -> Stats {
        let mut vw = VideoWriter::create(path, w, h, 30, &AV1).expect("create");
        for t in 0..n {
            vw.push(&frame(w, h, t), w, h, 3).expect("push");
        }
        vw.finish().expect("finish")
    }

    /// ffprobe's codec, decoded frame count and size of the file's video
    /// stream; `None` when ffprobe is not installed.
    fn ffprobe(path: &Path) -> Option<(String, u32, u32, u32)> {
        let output = std::process::Command::new("ffprobe")
            .args(["-v", "error", "-count_frames", "-select_streams", "v:0"])
            .args([
                "-show_entries",
                "stream=codec_name,nb_read_frames,width,height",
            ])
            .args(["-of", "default=noprint_wrappers=1"])
            .arg(path)
            .output()
            .ok()?;
        assert!(output.status.success(), "ffprobe {}", path.display());
        let text = String::from_utf8(output.stdout).expect("utf-8");
        let field = |key: &str| {
            text.lines()
                .find_map(|line| line.strip_prefix(key)?.strip_prefix('='))
                .unwrap_or_else(|| panic!("ffprobe reported no {key}: {text}"))
                .to_string()
        };
        let number = |key: &str| field(key).parse::<u32>().expect("number");
        Some((
            field("codec_name"),
            number("nb_read_frames"),
            number("width"),
            number("height"),
        ))
    }

    #[test]
    fn each_codec_writes_an_mp4_ffprobe_decodes() {
        // An odd width: AV1 crops it, H.264 pads it as the managed runner does.
        for (codec, expected) in [
            (VideoCodec::Av1, ("av1", 64)),
            (VideoCodec::H264, ("h264", 66)),
        ] {
            let encoding = match Encoding::resolve(codec, 60, None) {
                Ok(encoding) => encoding,
                Err(err) => {
                    eprintln!("skipping {codec}: {err}");
                    continue;
                }
            };
            let dir = tempfile::tempdir().expect("tempdir");
            let path = dir.path().join("camera.mp4");
            let mut vw = ThreadedVideoWriter::create(&path, 65, 48, 30, &encoding).expect("create");
            for t in 0..5 {
                vw.push(frame(65, 48, t), 65, 48, 3).expect("push");
            }
            let stats = vw.finish().expect("finish");
            assert_eq!((stats.frame_count, stats.width), (5, expected.1), "{codec}");
            let Some(probed) = ffprobe(&path) else {
                eprintln!("skipping {codec} playback: ffprobe is not installed");
                continue;
            };
            assert_eq!(probed, (expected.0.into(), 5, expected.1, 48), "{codec}");
        }
    }

    #[test]
    fn threaded_writer_matches_sync_stats() {
        let dir = std::env::temp_dir().join(format!("rlmesh-threaded-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let path = dir.join("threaded.mp4");
        let mut vw = ThreadedVideoWriter::create(&path, 64, 48, 30, &AV1).expect("create");
        for t in 0..12 {
            vw.push(frame(64, 48, t), 64, 48, 3).expect("push");
        }
        let stats = vw.finish().expect("finish");
        assert_eq!(stats.frame_count, 12);
        assert_eq!((stats.width, stats.height), (64, 48));
        assert!(
            vw.finish().is_err(),
            "second finish reports already-finished"
        );
        let bytes = std::fs::read(&path).expect("read");
        assert_eq!(&bytes[4..8], b"ftyp");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn threaded_writer_surfaces_encode_error_on_push_or_finish() {
        let dir = std::env::temp_dir().join(format!("rlmesh-threaded-err-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let path = dir.join("bad.mp4");
        let mut vw = ThreadedVideoWriter::create(&path, 64, 48, 30, &AV1).expect("create");
        vw.push(frame(64, 48, 0), 64, 48, 3).expect("push");
        vw.push(frame(32, 32, 1), 32, 32, 3)
            .expect("queued; error surfaces later");
        let err = vw
            .finish()
            .expect_err("size change must fail the recording");
        assert!(err.to_string().contains("size changed"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn odd_dims_crop_and_roundtrip_through_demux() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("out.mp4");
        let stats = record(&path, 129, 97, 40);
        assert_eq!((stats.width, stats.height), (128, 96));
        assert_eq!(stats.frame_count, 40);

        let bytes = std::fs::read(&path).expect("read");
        assert_eq!(&bytes[4..8], b"ftyp");
        assert!(
            bytes.windows(4).any(|w| w == b"nclx"),
            "must carry a colr color box"
        );
        let demux = re_mp4::Mp4::read_bytes(&bytes).expect("demux");
        let (_, track) = demux.tracks().iter().next().expect("track");
        assert!(track.codec_string(&demux).unwrap().starts_with("av01"));
        assert_eq!((track.width, track.height), (128, 96));
        assert_eq!(track.samples.len(), 40);
        assert!(track.samples[0].is_sync, "first sample must be a keyframe");
    }

    #[test]
    fn single_frame_file_is_valid() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("one.mp4");
        let stats = record(&path, 64, 64, 1);
        assert_eq!(stats.frame_count, 1);
        let bytes = std::fs::read(&path).expect("read");
        let demux = re_mp4::Mp4::read_bytes(&bytes).expect("demux");
        let (_, track) = demux.tracks().iter().next().expect("track");
        assert_eq!(track.samples.len(), 1);
    }

    #[test]
    fn finish_with_no_frames_errors() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vw =
            VideoWriter::create(&dir.path().join("empty.mp4"), 64, 64, 30, &AV1).expect("create");
        assert!(vw.finish().is_err());
    }

    #[test]
    fn size_change_mid_recording_is_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut vw =
            VideoWriter::create(&dir.path().join("x.mp4"), 64, 64, 30, &AV1).expect("create");
        vw.push(&frame(64, 64, 0), 64, 64, 3).expect("push");
        assert!(vw.push(&frame(48, 48, 1), 48, 48, 3).is_err());
    }

    #[test]
    fn rejects_out_of_range_dimensions() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(VideoWriter::create(&dir.path().join("z.mp4"), 1, 64, 30, &AV1).is_err());
        assert!(VideoWriter::create(&dir.path().join("z.mp4"), 70000, 64, 30, &AV1).is_err());
    }
}
