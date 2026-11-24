//! Terminal backend: takes over the terminal (alt-screen, raw mode), runs the key
//! thread for live source switching + quitting, and orchestrates per-frame drawing.

use std::fmt::Write as _;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossterm::terminal::{EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::{cursor, event, execute, terminal};
use image::RgbImage;

use crate::viewer::graphics::{self, Graphics};

/// Owns the terminal takeover and the key thread; restores everything on drop.
pub struct Terminal {
    running: Arc<AtomicBool>,
    keys: Option<JoinHandle<()>>,
    graphics: Graphics,
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

fn is_quit(key: &event::KeyEvent) -> bool {
    use event::{KeyCode, KeyModifiers};
    match key.code {
        KeyCode::Esc | KeyCode::Char('q' | 'Q') => true,
        KeyCode::Char('c' | 'd') => key.modifiers.contains(KeyModifiers::CONTROL),
        _ => false,
    }
}

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
    let _ = write!(out, "\x1b[2m   ·  n skip ep  ·  q quit\x1b[0m");
    let line = hud_line(&hud, usize::from(cols));
    let _ = write!(out, "\x1b[{};1H\x1b[2K\x1b[2m{line}\x1b[0m", rows.max(1));
}

fn hud_line(hud: &super::Hud, cols: usize) -> String {
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

    let b = format!(
        "model {}  env {}  {:.1}sps",
        fmt_ms(hud.model_ms),
        fmt_ms(hud.env_ms),
        hud.sps
    );

    let mut d = format!("R {:+.2}", hud.reward);
    if !hud.outcome.is_empty() {
        let _ = write!(d, "  {}", hud.outcome);
    }

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

    for threshold in [4u8, 3, 1, 0] {
        let line = join_groups(&groups, threshold);
        if line.chars().count() <= cols {
            return line;
        }
    }
    join_groups(&groups, 0)
}

fn join_groups(groups: &[(String, u8)], threshold: u8) -> String {
    groups
        .iter()
        .filter(|(_, p)| *p <= threshold)
        .map(|(s, _)| s.as_str())
        .collect::<Vec<_>>()
        .join("  |  ")
}

fn fmt_elapsed(s: f64) -> String {
    let total = s.max(0.0) as u64;
    let (h, m, sec) = (total / 3600, (total % 3600) / 60, total % 60);
    if h > 0 {
        format!("{h}:{m:02}:{sec:02}")
    } else {
        format!("{m}:{sec:02}")
    }
}

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
    use crate::viewer::Hud;

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
        let hud = Hud {
            step: 5,
            reward: 0.0,
            ..Default::default()
        };
        let line = hud_line(&hud, 200);
        assert!(line.contains("step 5"), "{line:?}");
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
