//! PyO3 wrapper over the `rlmesh-viewer` engine, fed from the Python Session loop.

use std::sync::{Mutex, MutexGuard};

use pyo3::prelude::*;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rlmesh_viewer::{Backend, FrameFormat, Viewer};

/// Native debug viewer, built by the Python `Session` when `view=` is set and fed
/// one decoded HWC uint8 frame (the selected camera) per step.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(module = "rlmesh._rlmesh", name = "PyViewer")]
pub struct PyViewer {
    inner: Mutex<Option<Viewer>>,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[pymethods]
impl PyViewer {
    #[new]
    #[pyo3(signature = (terminal=true, http_port=None, fps=30, format=None, quality=75))]
    fn new(
        terminal: bool,
        http_port: Option<u16>,
        fps: u32,
        format: Option<String>,
        quality: u8,
    ) -> Self {
        let mut backends = Vec::new();
        if terminal {
            backends.push(Backend::Terminal);
        }
        if let Some(port) = http_port {
            backends.push(Backend::Http(port));
        }
        let format = match format.as_deref() {
            Some(f) if f.eq_ignore_ascii_case("png") => FrameFormat::Png,
            _ => FrameFormat::Jpeg(quality),
        };
        Self {
            inner: Mutex::new(Some(Viewer::new(&backends, fps, format))),
        }
    }

    /// Declare the selectable camera labels and the initial selection index.
    fn set_sources(&self, sources: Vec<String>, default: usize) {
        if let Some(viewer) = self.lock().as_ref() {
            viewer.set_sources(sources, default);
        }
    }

    /// The currently-selected source label (key thread or browser), or `None`.
    fn selected_source(&self) -> Option<String> {
        self.lock().as_ref().and_then(Viewer::selected_source)
    }

    /// Whether the throttle would draw now — gate an expensive frame fetch on this.
    fn wants_frame(&self) -> bool {
        self.lock().as_ref().is_some_and(Viewer::wants_frame)
    }

    /// Feed one contiguous HWC uint8 frame for the selected source.
    fn feed_frame(&self, buf: Vec<u8>, width: u32, height: u32, channels: u32) {
        if let Some(viewer) = self.lock().as_ref() {
            viewer.feed_frame(&buf, width, height, channels);
        }
    }

    /// Update the HUD (step / cumulative reward / outcome label computed by the caller).
    fn feed_hud(&self, step: i64, reward: f64, outcome: &str) {
        if let Some(viewer) = self.lock().as_ref() {
            viewer.feed_hud(step, reward, outcome);
        }
    }

    /// Whether the user asked to quit via the terminal (q / Esc / Ctrl-C). The eval
    /// polls this each step and stops, since raw mode swallows the real SIGINT.
    fn should_quit(&self) -> bool {
        self.lock().as_ref().is_some_and(Viewer::should_quit)
    }

    /// Setup warnings (a backend that failed to come up), surfaced once by Python.
    fn warnings(&self) -> Vec<String> {
        self.lock()
            .as_ref()
            .map(Viewer::warnings)
            .unwrap_or_default()
    }

    /// Tear down the viewer (restores the terminal / stops the HTTP server).
    fn close(&self) {
        *self.lock() = None;
    }
}

impl PyViewer {
    fn lock(&self) -> MutexGuard<'_, Option<Viewer>> {
        self.inner.lock().unwrap_or_else(|err| err.into_inner())
    }
}
