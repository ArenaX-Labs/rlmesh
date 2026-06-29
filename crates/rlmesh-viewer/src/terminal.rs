//! Terminal backend: takes over the terminal (alt-screen, raw mode), runs the key
//! thread for live source switching + quitting, and orchestrates per-frame drawing.
//!
//! Frame rendering itself — the native inline-image protocols and the truecolor
//! half-block fallback — lives in [`crate::graphics`]. This module owns the terminal
//! lifecycle, key input, the HUD footer, and resize reflow. `crossterm` handles the
//! alt-screen takeover (reversible: scrollback returns on drop), raw-mode key input,
//! and terminal size.

use std::fmt::Write as _;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossterm::terminal::{EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::{cursor, event, execute, terminal};
use image::RgbImage;

use crate::graphics::{self, Graphics};

/// Owns the terminal takeover and the key thread; restores everything on drop.
pub struct Terminal {
    running: Arc<AtomicBool>,
    keys: Option<JoinHandle<()>>,
    graphics: Graphics,
    /// Last frame drawn, cached so the key thread can reflow it on a terminal resize
    /// — the draw loop is fed by the eval, so a paused eval wouldn't reflow otherwise.
    last: Arc<Mutex<Option<RgbImage>>>,
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
        let graphics = graphics::detect();
        let last: Arc<Mutex<Option<RgbImage>>> = Arc::new(Mutex::new(None));
        let keys = spawn_keys(Arc::clone(&running), shared, graphics, Arc::clone(&last));
        Some(Self {
            running,
            keys: Some(keys),
            graphics,
            last,
        })
    }

    /// Draw one frame in place. Caches it so a resize can reflow it from the key
    /// thread even while the eval is paused.
    pub fn draw(&self, img: &RgbImage, shared: &super::http::HttpShared) {
        *super::http::lock(&self.last) = Some(img.clone());
        render(self.graphics, img, shared);
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

/// Render one frame at the current terminal size: the image (via [`crate::graphics`])
/// plus the source-selector / HUD footer, in a single flush. Shared by the draw path
/// and the key thread's resize reflow; each call writes a whole frame, so concurrent
/// calls never interleave mid-escape.
fn render(g: Graphics, img: &RgbImage, shared: &super::http::HttpShared) {
    let (cols, rows) = terminal::size().unwrap_or((80, 24));
    let img_rows = rows.saturating_sub(2).max(1);

    let mut out = String::new();
    out.push_str("\x1b[H");
    graphics::render_image(g, img, cols, img_rows, &mut out);
    footer(shared, rows, &mut out);

    let mut stdout = io::stdout().lock();
    let _ = stdout.write_all(out.as_bytes());
    let _ = stdout.flush();
}

/// Poll keys on a dedicated thread (only `crossterm::event`, never a second stdin
/// reader): cycle the selected source, set the quit flag, and reflow the cached frame
/// on resize. Polls so it notices `running` flipping without needing a keypress.
fn spawn_keys(
    running: Arc<AtomicBool>,
    shared: Arc<super::http::HttpShared>,
    graphics: Graphics,
    last: Arc<Mutex<Option<RgbImage>>>,
) -> JoinHandle<()> {
    thread::Builder::new()
        .name("rlmesh-viewer-keys".to_string())
        .spawn(move || {
            use event::{Event, KeyCode};
            while running.load(Ordering::Relaxed) {
                if !matches!(event::poll(Duration::from_millis(100)), Ok(true)) {
                    continue;
                }
                let key = match event::read() {
                    Ok(Event::Key(key)) => key,
                    Ok(Event::Resize(_, _)) => {
                        if let Some(img) = super::http::lock(&last).clone() {
                            render(graphics, &img, &shared);
                        }
                        continue;
                    }
                    _ => continue,
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

/// Quit keys: `q` / Esc, or Ctrl-C / Ctrl-D — the raw-mode-safe replacements for the
/// SIGINT the kernel no longer delivers.
fn is_quit(key: &event::KeyEvent) -> bool {
    use event::{KeyCode, KeyModifiers};
    match key.code {
        KeyCode::Esc | KeyCode::Char('q' | 'Q') => true,
        KeyCode::Char('c' | 'd') => key.modifiers.contains(KeyModifiers::CONTROL),
        _ => false,
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
