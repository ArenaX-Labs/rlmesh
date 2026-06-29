//! The viewer handle the Python `Session` loop feeds directly.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tiny_http::Server;

use crate::frame::{self, FrameFormat};
use crate::http::{self, HttpShared, Hud};
use crate::terminal::Terminal;

/// Where the viewer draws. Terminal is the primary backend (works over bare SSH);
/// HTTP is the zero-terminal-impact one (watch in a browser). Enable either or
/// both — they share one selected-source state, so switching stays in sync.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Terminal,
    Http(u16),
}

/// A synchronous, directly-fed viewer.
///
/// The Session loop is the caller's own loop, so feeding inline is fine: each
/// [`feed_frame`](Viewer::feed_frame) normalizes to RGB and draws to every active
/// backend (throttled to `fps`). The caller reads
/// [`selected_source`](Viewer::selected_source) to know which camera to feed.
pub struct Viewer {
    shared: Arc<HttpShared>,
    format: FrameFormat,
    frame_interval: Duration,
    last_render: Mutex<Option<Instant>>,
    terminal: Option<Terminal>,
    server: Option<Arc<Server>>,
    warnings: Vec<String>,
}

impl Viewer {
    /// Build a viewer over the given backends, encoding HTTP frames as `format`
    /// at up to `fps`.
    pub fn new(backends: &[Backend], fps: u32, format: FrameFormat) -> Self {
        let shared = Arc::new(HttpShared::new(format.content_type()));
        let mut terminal = None;
        let mut server = None;
        let mut warnings = Vec::new();
        for backend in backends {
            match backend {
                Backend::Terminal => {
                    terminal = Terminal::new(Arc::clone(&shared));
                    if terminal.is_none() {
                        warnings
                            .push("terminal backend unavailable (stdout not a TTY?)".to_string());
                    }
                }
                Backend::Http(port) => match http::spawn(*port, Arc::clone(&shared)) {
                    Ok(handle) => server = Some(handle),
                    Err(err) => {
                        warnings.push(format!("HTTP backend failed to bind port {port}: {err}"));
                    }
                },
            }
        }
        Self {
            shared,
            format,
            frame_interval: Duration::from_secs_f64(1.0 / f64::from(fps.max(1))),
            last_render: Mutex::new(None),
            terminal,
            server,
            warnings,
        }
    }

    /// Setup warnings collected while bringing up backends (a backend that failed
    /// to start); the caller surfaces these once after construction.
    pub fn warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }

    /// Whether the user asked to quit via the terminal key thread. Raw mode swallows
    /// Ctrl-C, so the eval polls this each step and stops the loop itself.
    pub fn should_quit(&self) -> bool {
        self.shared.quit.load(Ordering::Relaxed)
    }

    /// Declare the selectable sources and the initially-selected index.
    pub fn set_sources(&self, sources: Vec<String>, default: usize) {
        let clamped = default.min(sources.len().saturating_sub(1));
        *http::lock(&self.shared.sources) = sources;
        self.shared.selected.store(clamped, Ordering::Relaxed);
    }

    /// The source currently selected (key thread or browser) — what to feed next.
    pub fn selected_source(&self) -> Option<String> {
        let sources = http::lock(&self.shared.sources);
        sources
            .get(self.shared.selected.load(Ordering::Relaxed))
            .cloned()
    }

    /// Whether the fps throttle would actually draw a frame right now. Non-mutating
    /// — lets the caller skip an expensive frame fetch it would only have dropped.
    pub fn wants_frame(&self) -> bool {
        http::lock(&self.last_render).is_none_or(|t| t.elapsed() >= self.frame_interval)
    }

    /// Feed one HWC uint8 frame for the selected source. Throttled to `fps`:
    /// normalize + draw (and dropping when ahead of cadence) happens here.
    pub fn feed_frame(&self, buf: &[u8], width: u32, height: u32, channels: u32) {
        {
            let mut last = http::lock(&self.last_render);
            if last.is_some_and(|t| t.elapsed() < self.frame_interval) {
                return;
            }
            *last = Some(Instant::now());
        }
        let Some(img) = frame::rgb_from_hwc(buf, width, height, channels) else {
            return;
        };
        if let Some(terminal) = &self.terminal {
            terminal.draw(&img, &self.shared);
        }
        if self.server.is_some()
            && let Some(bytes) = frame::encode(&img, self.format)
        {
            *http::lock(&self.shared.latest_frame) = bytes;
        }
    }

    /// Update the HUD (step / cumulative reward / outcome label). The outcome is
    /// computed by the caller from the env-reported task result, not inferred from
    /// `terminated` here (a terminal state is not necessarily a success).
    pub fn feed_hud(&self, step: i64, reward: f64, outcome: &str) {
        *http::lock(&self.shared.hud) = Hud {
            step,
            reward,
            outcome: outcome.to_string(),
        };
    }
}

impl Drop for Viewer {
    fn drop(&mut self) {
        if let Some(server) = &self.server {
            server.unblock();
        }
        // The terminal backend restores the screen via its own Drop.
    }
}
