//! Live viewer configuration and session driver.
use crate::env::factory::{py_float, py_str};
use crate::{Error, Result, RunContext, Value};
use std::{
    fmt,
    str::FromStr,
    time::{Duration, Instant},
};

/// Where to display live frames.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ViewBackend {
    /// Draw in the terminal.
    #[default]
    Terminal,
    /// Serve a local HTTP page.
    Http,
    /// Draw in the terminal and serve HTTP.
    Both,
}
/// HTTP image encoding.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ViewFormat {
    /// JPEG frames.
    #[default]
    Jpeg,
    /// Lossless PNG frames.
    Png,
}
impl ViewBackend {
    /// Python's backend spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Terminal => "terminal",
            Self::Http => "http",
            Self::Both => "both",
        }
    }
}
impl fmt::Display for ViewBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
impl ViewFormat {
    /// Python's format spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Jpeg => "jpeg",
            Self::Png => "png",
        }
    }
}
impl fmt::Display for ViewFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
/// How long a successful session retains its final frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ViewHold {
    /// Close immediately.
    #[default]
    Off,
    /// Wait for an interactive user to quit.
    UntilQuit,
    /// Wait at most this long.
    For(Duration),
}
impl From<bool> for ViewHold {
    fn from(value: bool) -> Self {
        if value { Self::UntilQuit } else { Self::Off }
    }
}
impl From<Duration> for ViewHold {
    fn from(value: Duration) -> Self {
        Self::For(value)
    }
}

/// Live display options, with Python's `View` defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct View {
    /// Terminal, HTTP, or both.
    pub backend: ViewBackend,
    /// HTTP port (1..65535).
    pub port: u16,
    /// Maximum drawing rate (at least 1).
    pub fps: u64,
    /// Initially selected frame source; otherwise the first image role.
    pub source: Option<String>,
    /// HTTP image format.
    pub format: ViewFormat,
    /// JPEG quality (1..100).
    pub quality: u8,
    /// Retain the final frame on a successful close.
    pub hold: ViewHold,
    /// Optional positive finite env stepping rate.
    pub step_hz: Option<f64>,
}
impl Default for View {
    fn default() -> Self {
        Self {
            backend: ViewBackend::Terminal,
            port: 8008,
            fps: 30,
            source: None,
            format: ViewFormat::Jpeg,
            quality: 75,
            hold: ViewHold::Off,
            step_hz: None,
        }
    }
}
impl View {
    /// Validate before consuming this plain value.
    pub fn validate(&self) -> Result<()> {
        if self.port == 0 {
            return Err(Error::invalid_argument(
                "View.port must be in 1..65535; got 0",
            ));
        }
        if self.fps == 0 {
            return Err(Error::invalid_argument("View.fps must be >= 1; got 0"));
        }
        if !(1..=100).contains(&self.quality) {
            return Err(Error::invalid_argument(format!(
                "View.quality must be in 1..100; got {}",
                self.quality
            )));
        }
        if let Some(hz) = self.step_hz
            && (!hz.is_finite() || hz <= 0.0)
        {
            return Err(Error::invalid_argument(format!(
                "View.step_hz must be a finite rate > 0 or None; got {}",
                py_float(hz)
            )));
        }
        Ok(())
    }
    /// Parse Python's case-insensitive shorthand, ignoring outer whitespace.
    pub fn parse(text: &str) -> Result<Self> {
        let spec = text.trim().to_ascii_lowercase();
        let mut view = Self::default();
        match spec.as_str() {
            "" | "terminal" | "term" | "tty" => {}
            "both" => view.backend = ViewBackend::Both,
            "http" | "http:" => view.backend = ViewBackend::Http,
            _ if spec.starts_with("http:") => {
                let tail = &spec[5..];
                if !tail.bytes().all(|b| b.is_ascii_digit()) {
                    return Err(Error::invalid_argument(format!(
                        "invalid view port {} in view={}; use 'http:PORT' with an integer port",
                        py_str(tail),
                        py_str(text)
                    )));
                }
                view.backend = ViewBackend::Http;
                view.port = tail.parse().map_err(|_| {
                    Error::invalid_argument(format!("View.port must be in 1..65535; got {tail}"))
                })?;
            }
            _ => {
                return Err(Error::invalid_argument(format!(
                    "unrecognized view={}; use 'terminal', 'http', 'http:PORT', 'both', or a View(...)",
                    py_str(text)
                )));
            }
        }
        view.validate()?;
        Ok(view)
    }
}
impl FromStr for View {
    type Err = Error;
    fn from_str(text: &str) -> Result<Self> {
        Self::parse(text)
    }
}
impl fmt::Display for View {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let backend = match self.backend {
            ViewBackend::Terminal => "terminal",
            ViewBackend::Http => "http",
            ViewBackend::Both => "both",
        };
        let format = match self.format {
            ViewFormat::Jpeg => "jpeg",
            ViewFormat::Png => "png",
        };
        let hold = match self.hold {
            ViewHold::Off => "False".into(),
            ViewHold::UntilQuit => "True".into(),
            ViewHold::For(d) => py_float(d.as_secs_f64()),
        };
        write!(
            f,
            "View(backend={}, port={}, fps={}, source={}, format={}, quality={}, hold={}, step_hz={})",
            py_str(backend),
            self.port,
            self.fps,
            self.source.as_deref().map_or_else(|| "None".into(), py_str),
            py_str(format),
            self.quality,
            hold,
            self.step_hz.map_or_else(|| "None".into(), py_float)
        )
    }
}

/// A normalized uint8 HWC frame, shared by viewing and recording.
pub(crate) struct Frame {
    pub(crate) data: Vec<u8>,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) channels: u32,
}
pub(crate) fn frame(value: &Value) -> Option<Frame> {
    // Metadata's tensor conversion handles strided tensors and nested numeric lists.
    fn shape_values(
        v: &Value,
        shape: &mut Vec<usize>,
        out: &mut Vec<f64>,
        depth: usize,
    ) -> Option<()> {
        match v {
            Value::List(items) | Value::Tuple(items) => {
                if shape.len() == depth {
                    shape.push(items.len());
                } else if shape[depth] != items.len() {
                    return None;
                }
                for item in items {
                    shape_values(item, shape, out, depth + 1)?;
                }
            }
            Value::Int(x) if depth == 3 => out.push(*x as f64),
            Value::Float(x) if depth == 3 => out.push(*x),
            Value::Bool(x) if depth == 3 => out.push(f64::from(*x)),
            _ => return None,
        }
        Some(())
    }
    let (shape, values, uint8) = match value {
        Value::Tensor(t) => {
            let shape = t.shape().to_vec();
            let scalars = rlmesh_spaces::decode_scalars(&t.tobytes(), t.dtype().to_core()).ok()?;
            (
                shape,
                scalars
                    .into_iter()
                    .map(|x| x.to_f64(t.dtype().to_core()))
                    .collect::<Vec<_>>(),
                t.dtype() == crate::DType::Uint8,
            )
        }
        _ => {
            let (mut shape, mut values) = (Vec::new(), Vec::new());
            shape_values(value, &mut shape, &mut values, 0)?;
            (shape, values, false)
        }
    };
    if shape.len() != 3 || !matches!(shape[2], 1 | 3 | 4) || shape[0] == 0 || shape[1] == 0 {
        return None;
    }
    let (lo, hi) = if values.iter().any(|value| value.is_nan()) {
        (f64::NAN, f64::NAN)
    } else {
        (
            values.iter().copied().fold(f64::INFINITY, f64::min),
            values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )
    };
    let data = values
        .into_iter()
        .map(|x| {
            let x = if uint8 {
                x
            } else if lo >= 0.0 && hi <= 1.0 + 1e-6 {
                x * 255.0
            } else if lo >= -1.0 - 1e-6 && hi <= 1.0 + 1e-6 {
                (x + 1.0) * 127.5
            } else if !(lo >= 0.0 && hi <= 255.0 + 1e-6) {
                if hi - lo > 1e-12 {
                    (x - lo) * 255.0 / (hi - lo)
                } else {
                    x - lo
                }
            } else {
                x
            };
            x.clamp(0.0, 255.0) as u8
        })
        .collect();
    Some(Frame {
        data,
        width: shape[1].try_into().ok()?,
        height: shape[0].try_into().ok()?,
        channels: shape[2] as u32,
    })
}
pub(crate) fn read_frame(read: &crate::ReadFn, camera: &str) -> Option<Frame> {
    let mut image = crate::adapters::Image::new(camera);
    image.layout = crate::adapters::ImageLayout::Hwc;
    frame(&read(&image.into()).ok()?)
}

pub(crate) struct ViewerDriver {
    options: View,
    viewer: Option<rlmesh_viewer::Viewer>,
    disabled: bool,
    pub(crate) quit: bool,
    next_step: Option<Instant>,
    last: Option<Value>,
    hud: rlmesh_viewer::Hud,
}
impl ViewerDriver {
    pub(crate) fn new(options: View) -> Self {
        Self {
            options,
            viewer: None,
            disabled: false,
            quit: false,
            next_step: None,
            last: None,
            hud: Default::default(),
        }
    }
    pub(crate) fn feed(
        &mut self,
        ctx: &dyn RunContext,
        obs: Value,
        hud: rlmesh_viewer::Hud,
    ) -> Result<bool> {
        if self.disabled {
            return Ok(false);
        }
        let mut sources = ctx.frame_sources()?;
        if self.viewer.is_none() {
            if sources.labels().is_empty() {
                self.disabled = true;
                crate::warnings::warn(
                    crate::WarningCategory::UserWarning,
                    "rlmesh view: env declares no image roles and has no rgb render mode; viewer disabled.",
                )?;
                return Ok(false);
            }
            use rlmesh_viewer::{Backend, FrameFormat, Viewer};
            let backends = match self.options.backend {
                ViewBackend::Terminal => vec![Backend::Terminal],
                ViewBackend::Http => vec![Backend::Http(self.options.port)],
                ViewBackend::Both => vec![Backend::Terminal, Backend::Http(self.options.port)],
            };
            let format = match self.options.format {
                ViewFormat::Jpeg => FrameFormat::Jpeg(self.options.quality),
                ViewFormat::Png => FrameFormat::Png,
            };
            let viewer = Viewer::new(
                &backends,
                self.options.fps.min(u32::MAX as u64) as u32,
                format,
            );
            for message in viewer.warnings() {
                crate::warnings::warn(
                    crate::WarningCategory::UserWarning,
                    format!("rlmesh view: {message}"),
                )?;
            }
            let labels = sources.labels();
            let selected = self
                .options
                .source
                .as_ref()
                .filter(|source| labels.contains(source))
                .or(sources.default_label().as_ref())
                .and_then(|s| labels.iter().position(|l| l == s))
                .unwrap_or(0);
            viewer.set_sources(labels, selected);
            self.viewer = Some(viewer);
        }
        let viewer = self.viewer.as_ref().expect("viewer was initialized above");
        if viewer.wants_frame() {
            let selected = viewer.selected_source();
            let value = if selected == sources.render_label
                && let Some(render) = sources.render.as_mut()
            {
                render.call().ok()
            } else {
                selected.and_then(|s| {
                    let mut image = crate::adapters::Image::new(s);
                    image.layout = crate::adapters::ImageLayout::Hwc;
                    ctx.read(&obs, &image.into()).ok()
                })
            };
            if let Some(frame) = value.as_ref().and_then(frame) {
                viewer.feed_frame(&frame.data, frame.width, frame.height, frame.channels);
                self.hud.width = frame.width;
                self.hud.height = frame.height;
            }
        }
        self.last = Some(obs);
        self.hud = rlmesh_viewer::Hud {
            width: self.hud.width,
            height: self.hud.height,
            ..hud
        };
        viewer.feed_hud(self.hud.clone());
        self.quit |= viewer.should_quit();
        Ok(viewer.take_skip() || self.quit)
    }
    pub(crate) fn delay(&mut self) -> Duration {
        let Some(hz) = self.options.step_hz else {
            return Duration::ZERO;
        };
        if self.quit {
            return Duration::ZERO;
        }
        let period = Duration::try_from_secs_f64(1.0 / hz).unwrap_or(Duration::MAX);
        let now = Instant::now();
        let due = self.next_step.filter(|due| *due >= now);
        self.next_step = due.unwrap_or(now).checked_add(period);
        due.map_or(Duration::ZERO, |due| due - now)
    }
    pub(crate) fn disable(&mut self) {
        self.disabled = true;
        self.viewer = None;
    }
    pub(crate) fn hold(&self) -> ViewHold {
        if self.disabled || self.quit || self.last.is_none() {
            ViewHold::Off
        } else {
            self.options.hold
        }
    }
    pub(crate) fn quittable(&self) -> bool {
        use std::io::IsTerminal;
        std::io::stdin().is_terminal()
            && (self.options.backend != ViewBackend::Terminal || std::io::stdout().is_terminal())
    }
    pub(crate) fn held(&mut self, ctx: &dyn RunContext) -> Result<()> {
        if let Some(obs) = self.last.clone() {
            let mut hud = self.hud.clone();
            if !hud.outcome.ends_with("[held]") {
                hud.outcome.push_str(" [held]");
            }
            self.feed(ctx, obs, hud)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
