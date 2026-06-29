//! Terminal backend: draws frames in place via ANSI half-blocks, with a key
//! thread for live source switching.
//!
//! Zero image-codec deps — `crossterm` handles the alt-screen takeover, raw-mode
//! key input, and terminal size; the half-block encoder is hand-rolled (one cell
//! = two vertical pixels, fg = top, bg = bottom, truecolor). The alt-screen makes
//! the takeover reversible: on drop the prior scrollback returns intact.

use std::fmt::Write as _;
use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossterm::terminal::{EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::{cursor, event, execute, terminal};
use image::RgbImage;
use image::imageops::FilterType;

const UPPER_HALF: char = '\u{2580}'; // ▀

/// Owns the terminal takeover and the key thread; restores everything on drop.
pub struct Terminal {
    running: Arc<AtomicBool>,
    keys: Option<JoinHandle<()>>,
}

impl Terminal {
    /// Take over the terminal (alt-screen, raw mode, hidden cursor) and start the
    /// key thread that cycles the selected source. `None` if stdout is not a TTY.
    pub fn new(shared: Arc<super::http::HttpShared>) -> Option<Self> {
        terminal::enable_raw_mode().ok()?;
        if execute!(io::stdout(), EnterAlternateScreen, cursor::Hide).is_err() {
            let _ = terminal::disable_raw_mode();
            return None;
        }
        let running = Arc::new(AtomicBool::new(true));
        let keys = spawn_keys(Arc::clone(&running), shared);
        Some(Self {
            running,
            keys: Some(keys),
        })
    }

    /// Draw one frame in place, with a source-selector + HUD footer.
    pub fn draw(&self, img: &RgbImage, shared: &super::http::HttpShared) {
        let (cols, rows) = terminal::size().unwrap_or((80, 24));
        let max_w = u32::from(cols.max(1));
        // Reserve two rows for the footer; each char row is two vertical pixels.
        let max_h = u32::from(rows.saturating_sub(2).max(1)) * 2;
        let (nw, nh) = fit(img.width(), img.height(), max_w, max_h);
        let small = image::imageops::resize(img, nw, (nh.max(2)) & !1, FilterType::Triangle);

        let mut out = String::with_capacity(small.len() * 8);
        out.push_str("\x1b[H"); // cursor home
        half_blocks(&small, &mut out);
        out.push_str("\x1b[J");
        footer(shared, rows, &mut out);

        let mut stdout = io::stdout().lock();
        let _ = stdout.write_all(out.as_bytes());
        let _ = stdout.flush();
    }
}

impl Drop for Terminal {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.keys.take() {
            let _ = handle.join();
        }
        let _ = execute!(io::stdout(), cursor::Show, LeaveAlternateScreen);
        let _ = terminal::disable_raw_mode();
    }
}

/// Poll keys on a dedicated thread (only `crossterm::event`, never a second stdin
/// reader) and cycle the shared `selected` index. Polls so it notices `running`
/// flipping without needing a keypress.
fn spawn_keys(running: Arc<AtomicBool>, shared: Arc<super::http::HttpShared>) -> JoinHandle<()> {
    thread::Builder::new()
        .name("rlmesh-viewer-keys".to_string())
        .spawn(move || {
            use event::{Event, KeyCode};
            while running.load(Ordering::Relaxed) {
                if !matches!(event::poll(Duration::from_millis(100)), Ok(true)) {
                    continue;
                }
                let Ok(Event::Key(key)) = event::read() else {
                    continue;
                };
                if is_quit(&key) {
                    shared.quit.store(true, Ordering::Relaxed);
                    continue;
                }
                let n = super::http::lock(&shared.sources).len();
                if n == 0 {
                    continue;
                }
                let cur = shared.selected.load(Ordering::Relaxed).min(n - 1);
                let next = match key.code {
                    KeyCode::Right | KeyCode::Tab | KeyCode::Char(' ') => (cur + 1) % n,
                    KeyCode::Left => (cur + n - 1) % n,
                    KeyCode::Char(c @ '1'..='9') => {
                        let i = (c as usize) - ('1' as usize);
                        if i < n { i } else { cur }
                    }
                    _ => cur,
                };
                shared.selected.store(next, Ordering::Relaxed);
            }
        })
        .expect("spawn rlmesh-viewer key thread")
}

/// Quit keys: `q` / Esc, or Ctrl-C / Ctrl-D — the raw-mode-safe replacements for
/// the SIGINT the kernel no longer delivers.
fn is_quit(key: &event::KeyEvent) -> bool {
    use event::{KeyCode, KeyModifiers};
    match key.code {
        KeyCode::Esc | KeyCode::Char('q' | 'Q') => true,
        KeyCode::Char('c' | 'd') => key.modifiers.contains(KeyModifiers::CONTROL),
        _ => false,
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
fn half_blocks(img: &RgbImage, out: &mut String) {
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

/// The bottom two rows: the source selector and a HUD line.
fn footer(shared: &super::http::HttpShared, rows: u16, out: &mut String) {
    let sources = super::http::lock(&shared.sources);
    let sel = shared.selected.load(Ordering::Relaxed);
    let hud = super::http::lock(&shared.hud).clone();

    let _ = write!(out, "\x1b[{};1H\x1b[2K", rows.saturating_sub(1).max(1));
    for (i, source) in sources.iter().enumerate() {
        if i == sel {
            let _ = write!(out, "\x1b[1m[{source}]\x1b[0m ");
        } else {
            let _ = write!(out, " {source}  ");
        }
    }
    let outcome = if hud.outcome.is_empty() {
        String::new()
    } else {
        format!("   {}", hud.outcome)
    };
    let _ = write!(
        out,
        "\x1b[{};1H\x1b[2K\x1b[2mstep {}   R {:+.2}{}\x1b[0m",
        rows.max(1),
        hud.step,
        hud.reward,
        outcome
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_preserves_aspect_within_bounds() {
        // Width-bound: 100x50 into 40x100 -> 40x20.
        assert_eq!(fit(100, 50, 40, 100), (40, 20));
        // Height-bound: 50x100 into 100x20 -> 10x20.
        assert_eq!(fit(50, 100, 100, 20), (10, 20));
    }

    #[test]
    fn half_blocks_one_cell_per_column_then_reset() {
        // 2x2 -> one char row, two cells, terminated by reset + CRLF.
        let img = RgbImage::from_raw(2, 2, vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]).expect("img");
        let mut out = String::new();
        half_blocks(&img, &mut out);
        assert_eq!(out.matches(UPPER_HALF).count(), 2);
        assert!(out.ends_with("\x1b[0m\r\n"));
    }
}
