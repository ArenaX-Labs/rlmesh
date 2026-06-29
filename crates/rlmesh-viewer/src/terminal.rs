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
    footer(shared, cols, rows, &mut out);

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
                if matches!(key.code, KeyCode::Char('n' | 'N')) {
                    // End the current episode early (soft, non-failure) and advance.
                    shared.skip.store(true, Ordering::Relaxed);
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

/// The bottom two rows: the source selector and the compact HUD line.
fn footer(shared: &super::http::HttpShared, cols: u16, rows: u16, out: &mut String) {
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
    // Key hints (dim): source cycling is via ←/→; `n` ends the episode, `q` quits.
    let _ = write!(out, "\x1b[2m   ·  n skip ep  ·  q quit\x1b[0m");
    let line = hud_line(&hud, usize::from(cols));
    let _ = write!(out, "\x1b[{};1H\x1b[2K\x1b[2m{line}\x1b[0m", rows.max(1));
}

/// Build the single, width-adaptive HUD line as labeled groups joined by ` | `.
/// When the terminal is too narrow, the lowest-priority groups are dropped first
/// (frame info, then chunk, then timing) while `step` and the reward/outcome always
/// stay — so a narrow window still shows the essentials instead of wrapping.
fn hud_line(hud: &super::http::Hud, cols: usize) -> String {
    // Group A — progress (mandatory; carries `step`): "ep i/N  step S  M:SS  seed N".
    let mut a = String::new();
    if hud.episodes > 0 {
        let _ = write!(a, "ep {}/{}  ", hud.episode, hud.episodes);
    }
    let _ = write!(a, "step {}", hud.step);
    if hud.elapsed_s > 0.0 {
        let _ = write!(a, "  {}", fmt_elapsed(hud.elapsed_s));
    }
    if hud.seed >= 0 {
        let _ = write!(a, "  seed {}", hud.seed);
    }

    // Group B — timing (the point of the HUD; dropped only just before the core).
    let b = format!(
        "model {}  env {}  {:.1}sps",
        fmt_ms(hud.model_ms),
        fmt_ms(hud.env_ms),
        hud.sps
    );

    // Group D — reward + env-reported outcome (mandatory).
    let mut d = format!("R {:+.2}", hud.reward);
    if !hud.outcome.is_empty() {
        let _ = write!(d, "  {}", hud.outcome);
    }

    // (text, priority): lower = kept longer; 0 = never dropped. Built in display order.
    let groups: Vec<(String, u8)> = [
        Some((a, 0u8)),
        Some((b, 1)),
        (hud.chunk_len > 1).then(|| (format!("chunk {}/{}", hud.chunk_pos, hud.chunk_len), 3)),
        Some((d, 0)),
        (hud.width > 0).then(|| {
            (
                format!("{}x{}  {:.0}fps", hud.width, hud.height, hud.fps),
                4,
            )
        }),
    ]
    .into_iter()
    .flatten()
    .collect();

    // Richest detail level whose width fits `cols`; thresholds are the distinct
    // priority breakpoints (4 = all, then drop frame, then chunk, then timing).
    for threshold in [4u8, 3, 1, 0] {
        let line = join_groups(&groups, threshold);
        if line.chars().count() <= cols {
            return line;
        }
    }
    join_groups(&groups, 0)
}

/// Join the groups at or below `threshold` priority with ` | `, in display order.
fn join_groups(groups: &[(String, u8)], threshold: u8) -> String {
    groups
        .iter()
        .filter(|(_, p)| *p <= threshold)
        .map(|(s, _)| s.as_str())
        .collect::<Vec<_>>()
        .join("  |  ")
}

/// Wall-clock seconds as `M:SS` (or `H:MM:SS` past an hour).
fn fmt_elapsed(s: f64) -> String {
    let total = s.max(0.0) as u64;
    let (h, m, sec) = (total / 3600, (total % 3600) / 60, total % 60);
    if h > 0 {
        format!("{h}:{m:02}:{sec:02}")
    } else {
        format!("{m}:{sec:02}")
    }
}

/// A millisecond timing: one decimal under 10ms, whole numbers above (78ms, 3.4ms).
fn fmt_ms(v: f64) -> String {
    if v < 10.0 {
        format!("{v:.1}ms")
    } else {
        format!("{v:.0}ms")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::Hud;

    fn sample() -> Hud {
        Hud {
            step: 42,
            reward: 1.5,
            outcome: "success".to_string(),
            fps: 30.0,
            model_ms: 78.0,
            env_ms: 23.0,
            sps: 6.1,
            elapsed_s: 12.0,
            episode: 2,
            episodes: 10,
            seed: 7,
            width: 320,
            height: 240,
            chunk_pos: 3,
            chunk_len: 8,
        }
    }

    #[test]
    fn wide_line_shows_every_group() {
        let line = hud_line(&sample(), 200);
        for needle in [
            "ep 2/10",
            "step 42",
            "0:12",
            "seed 7",
            "model 78ms",
            "env 23ms",
            "6.1sps",
            "chunk 3/8",
            "R +1.50",
            "success",
            "320x240",
            "30fps",
        ] {
            assert!(line.contains(needle), "missing {needle:?} in {line:?}");
        }
    }

    #[test]
    fn medium_line_drops_only_the_frame_group() {
        // Wide enough for everything but the lowest-priority frame group.
        let line = hud_line(&sample(), 100);
        assert!(line.chars().count() <= 100, "too wide: {line:?}");
        assert!(line.contains("chunk 3/8"), "chunk should survive: {line:?}");
        assert!(
            line.contains("model 78ms"),
            "timing should survive: {line:?}"
        );
        assert!(
            !line.contains("320x240"),
            "frame group should drop first: {line:?}"
        );
    }

    #[test]
    fn narrow_line_keeps_only_step_and_outcome() {
        let line = hud_line(&sample(), 60);
        assert!(line.chars().count() <= 60, "too wide: {line:?}");
        assert!(line.contains("step 42"), "step is mandatory: {line:?}");
        assert!(
            line.contains("R +1.50"),
            "reward/outcome is mandatory: {line:?}"
        );
        assert!(!line.contains("model 78ms"), "timing should drop: {line:?}");
        assert!(!line.contains("chunk 3/8"), "chunk should drop: {line:?}");
    }

    #[test]
    fn omits_unknown_sentinels() {
        // A hand-driven step with no episode/seed/chunk/frame info: those groups vanish.
        let hud = Hud {
            step: 5,
            reward: 0.0,
            ..Default::default()
        };
        let line = hud_line(&hud, 200);
        assert!(line.contains("step 5"), "{line:?}");
        // `ep i/N` and `chunk k/H` are the only groups with a slash; their absence
        // confirms both are omitted (a substring like "ep " collides with "step ").
        assert!(!line.contains('/'), "no episode/chunk labels: {line:?}");
        assert!(!line.contains("seed"), "no seed label: {line:?}");
        assert!(!line.contains('x'), "no frame resolution: {line:?}");
    }

    #[test]
    fn formatters() {
        assert_eq!(fmt_elapsed(12.0), "0:12");
        assert_eq!(fmt_elapsed(75.0), "1:15");
        assert_eq!(fmt_elapsed(3661.0), "1:01:01");
        assert_eq!(fmt_ms(78.0), "78ms");
        assert_eq!(fmt_ms(3.4), "3.4ms");
    }
}
