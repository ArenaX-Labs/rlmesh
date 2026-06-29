//! Image → terminal-bytes encoders: the native inline-image protocols — Kitty
//! (Kitty, Ghostty) and iTerm2 (iTerm2, WezTerm) — with a truecolor ANSI
//! half-block fallback, plus env-based protocol detection.
//!
//! Detection is env-only (no terminal query), so it never competes with the
//! terminal backend's key thread for stdin; Kitty also transmits with `q=2`, so
//! the terminal sends nothing back either.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};

use image::RgbImage;
use image::imageops::FilterType;

use crate::frame::{self, FrameFormat};

/// Upper half block (▀): one cell renders two vertically-stacked pixels.
const UPPER_HALF: char = '\u{2580}';

/// Ping-pong counter for the two Kitty image ids. A process-global is fine: only
/// one terminal viewer (one alt-screen takeover) can exist at a time.
static KITTY_FRAME: AtomicU64 = AtomicU64::new(0);

/// Cap the longest side fed to the PNG encoder on the graphics path, to bound
/// per-frame encode + transmission cost; the terminal scales it into the cells.
const MAX_GRAPHICS_PX: u32 = 1024;

/// Which inline-image protocol the terminal speaks (detected from env).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Graphics {
    /// No graphics protocol — render with half-blocks.
    None,
    /// Kitty graphics protocol (Kitty, Ghostty).
    Kitty,
    /// iTerm2 inline-images protocol (iTerm2, WezTerm).
    Iterm,
}

/// Detect the terminal's inline-image protocol from env only — no terminal query.
///
/// Ghostty/Kitty speak the Kitty protocol; iTerm2/WezTerm speak the iTerm2 one.
/// Multiplexers (tmux/screen) strip graphics escapes but leak the outer terminal's
/// env into their panes, so they fall back to half-blocks; so does anything not
/// recognized.
pub(crate) fn detect() -> Graphics {
    if std::env::var_os("TMUX").is_some() || std::env::var_os("STY").is_some() {
        return Graphics::None;
    }
    let term = std::env::var("TERM").unwrap_or_default();
    if term.starts_with("screen") || term.starts_with("tmux") {
        return Graphics::None;
    }
    let prog = std::env::var("TERM_PROGRAM").unwrap_or_default();
    if term.contains("kitty")
        || term.contains("ghostty")
        || prog == "ghostty"
        || std::env::var_os("KITTY_WINDOW_ID").is_some()
    {
        Graphics::Kitty
    } else if prog == "iTerm.app" || prog == "WezTerm" || prog == "mintty" {
        Graphics::Iterm
    } else {
        Graphics::None
    }
}

/// Append one frame to `out` for protocol `g`, sized to `cols`×`img_rows` cells
/// (the caller reserves the rows below for its footer).
///
/// The half-block path packs two vertical pixels per cell; the graphics paths
/// PNG-encode (the source long side capped to `MAX_GRAPHICS_PX`) and size the cell
/// box to the image's aspect — cells are ~2× taller than wide, so the vertical
/// bound is doubled — so Kitty's exact `c`×`r` fill does not stretch.
pub(crate) fn render_image(
    g: Graphics,
    img: &RgbImage,
    cols: u16,
    img_rows: u16,
    out: &mut String,
) {
    match g {
        Graphics::None => {
            let (nw, nh) = fit(
                img.width(),
                img.height(),
                u32::from(cols.max(1)),
                u32::from(img_rows) * 2,
            );
            let small = image::imageops::resize(img, nw, (nh.max(2)) & !1, FilterType::Triangle);
            out.reserve(small.len() * 8);
            half_block_cells(&small, out);
            out.push_str("\x1b[J");
        }
        Graphics::Kitty | Graphics::Iterm => {
            let resized;
            let src = if img.width().max(img.height()) > MAX_GRAPHICS_PX {
                let (nw, nh) = fit(img.width(), img.height(), MAX_GRAPHICS_PX, MAX_GRAPHICS_PX);
                resized = image::imageops::resize(img, nw.max(1), nh.max(1), FilterType::Triangle);
                &resized
            } else {
                img
            };
            let Some(png) = frame::encode(src, FrameFormat::Png) else {
                return;
            };
            let b64 = base64_encode(&png);
            if matches!(g, Graphics::Kitty) {
                let (cw, ch2) = fit(
                    img.width(),
                    img.height(),
                    u32::from(cols.max(1)),
                    u32::from(img_rows) * 2,
                );
                let cells_w = u16::try_from(cw.max(1)).unwrap_or(cols);
                let cells_h = u16::try_from(ch2.div_ceil(2).max(1)).unwrap_or(img_rows);
                kitty_emit(&b64, cells_w, cells_h, out);
            } else {
                iterm_emit(&b64, cols, img_rows, out);
            }
        }
    }
}

/// Fit `w x h` pixels into `max_w x max_h`, preserving aspect.
fn fit(w: u32, h: u32, max_w: u32, max_h: u32) -> (u32, u32) {
    let w = w.max(1);
    let h = h.max(1);
    let nw = w.min(max_w).max(1);
    let nh = ((u64::from(nw) * u64::from(h)) / u64::from(w)) as u32;
    if nh > max_h {
        let nh = max_h.max(1);
        let nw = ((u64::from(nh) * u64::from(w)) / u64::from(h)) as u32;
        (nw.max(1), nh)
    } else {
        (nw, nh.max(1))
    }
}

/// Encode an (even-height) RGB image as ANSI half-blocks, diffing color runs.
fn half_block_cells(img: &RgbImage, out: &mut String) {
    let (w, h) = (img.width(), img.height());
    for ry in 0..h / 2 {
        let (top_y, bot_y) = (ry * 2, ry * 2 + 1);
        let mut prev: Option<([u8; 3], [u8; 3])> = None;
        for x in 0..w {
            let t = img.get_pixel(x, top_y).0;
            let b = img.get_pixel(x, bot_y).0;
            if prev != Some((t, b)) {
                let _ = write!(
                    out,
                    "\x1b[38;2;{};{};{};48;2;{};{};{}m",
                    t[0], t[1], t[2], b[0], b[1], b[2]
                );
                prev = Some((t, b));
            }
            out.push(UPPER_HALF);
        }
        out.push_str("\x1b[0m\r\n");
    }
}

/// Emit a PNG frame via the Kitty graphics protocol, scaled into `cols`×`rows`
/// cells. Double-buffers across two image ids: the new frame is drawn (with a
/// strictly increasing `z` so it sits above the previous one — kitty breaks equal-z
/// ties by image id, which would otherwise put the lower-id frame underneath) and
/// only *then* is the previous image freed. No delete-then-redraw gap, so live
/// updates don't flicker, even if a redraw lands mid-stream over SSH. `C=1` keeps the
/// cursor put; `q=2` means the terminal never replies on stdin.
fn kitty_emit(b64: &str, cols: u16, rows: u16, out: &mut String) {
    let n = KITTY_FRAME.fetch_add(1, Ordering::Relaxed);
    let id = (n % 2) + 1;
    let prev = (id % 2) + 1;
    let z = (n & 0x7fff_ffff) as i32;
    let total = b64.len();
    let mut start = 0;
    let mut first = true;
    while start < total {
        let end = (start + 4096).min(total);
        let more = u8::from(end < total);
        if first {
            let _ = write!(
                out,
                "\x1b_Gf=100,a=T,i={id},q=2,c={cols},r={rows},z={z},C=1,m={more};{}\x1b\\",
                &b64[start..end]
            );
            first = false;
        } else {
            let _ = write!(out, "\x1b_Gm={more};{}\x1b\\", &b64[start..end]);
        }
        start = end;
    }
    let _ = write!(out, "\x1b_Ga=d,d=I,i={prev},q=2\x1b\\");
}

/// Emit a PNG frame via the iTerm2 inline-images protocol, scaled into `cols`×`rows`
/// cells. Fire-and-forget — no response to read.
fn iterm_emit(b64: &str, cols: u16, rows: u16, out: &mut String) {
    let _ = write!(
        out,
        "\x1b]1337;File=inline=1;width={cols};height={rows};preserveAspectRatio=1:{b64}\x07"
    );
}

/// Standard base64 (RFC 4648); the Kitty/iTerm2 transmitters base64 their PNG payload.
fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b1 = u32::from(*chunk.get(1).unwrap_or(&0));
        let b2 = u32::from(*chunk.get(2).unwrap_or(&0));
        let n = (u32::from(chunk[0]) << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[(n >> 18 & 63) as usize] as char);
        out.push(ALPHABET[(n >> 12 & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6 & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_rfc4648_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn fit_preserves_aspect_within_bounds() {
        assert_eq!(fit(100, 50, 40, 100), (40, 20));
        assert_eq!(fit(50, 100, 100, 20), (10, 20));
    }

    #[test]
    fn half_block_cells_one_cell_per_column_then_reset() {
        let img = RgbImage::from_raw(2, 2, vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]).expect("img");
        let mut out = String::new();
        half_block_cells(&img, &mut out);
        assert_eq!(out.matches(UPPER_HALF).count(), 2);
        assert!(out.ends_with("\x1b[0m\r\n"));
    }
}
