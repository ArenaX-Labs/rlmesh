//! Opt-in H.264 recording through the user's own ffmpeg.
//!
//! rlmesh never bundles or links an H.264 encoder: `codec="h264"` runs the
//! ffmpeg the user installed (on `PATH`, or an explicit path) as a subprocess
//! and pipes raw RGB frames to its stdin. The arguments are the managed
//! runner's (libx264 `ultrafast`/`zerolatency`, yuv420p, odd sides padded to
//! even, `+faststart`), so its videos and these play the same in a browser.
//!
//! Writes block: a slow encoder backpressures the caller through the pipe,
//! where the managed runner drops frames instead. Every frame is kept.

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::thread::JoinHandle;

/// The ffmpeg encoder H.264 recording needs.
const ENCODER: &str = "libx264";

/// Bytes of ffmpeg stderr kept for an error message.
const STDERR_LIMIT: usize = 64 * 1024;

/// Checks that `program` runs and lists libx264, once per program per
/// process; otherwise an error naming what to install.
pub(super) fn require(program: &Path) -> io::Result<()> {
    static CHECKED: OnceLock<Mutex<HashMap<PathBuf, Result<(), String>>>> = OnceLock::new();
    let mut checked = CHECKED
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    checked
        .entry(program.to_path_buf())
        .or_insert_with(|| probe(program))
        .clone()
        .map_err(io::Error::other)
}

fn probe(program: &Path) -> Result<(), String> {
    let install = "install ffmpeg with libx264 (e.g. `apt install ffmpeg`, `brew install ffmpeg`) \
                   or pass ffmpeg= its path; codec=\"av1\" needs nothing installed";
    let output = Command::new(program)
        .args(["-hide_banner", "-encoders"])
        .stdin(Stdio::null())
        .output()
        .map_err(|err| {
            format!(
                "codec=\"h264\" needs ffmpeg, but `{}` could not run ({err}); {install}",
                program.display()
            )
        })?;
    let listed = String::from_utf8_lossy(&output.stdout)
        .lines()
        .any(|line| line.split_whitespace().nth(1) == Some(ENCODER));
    if !output.status.success() || !listed {
        return Err(format!(
            "codec=\"h264\" needs ffmpeg's {ENCODER} encoder, which `{}` does not list; {install}",
            program.display()
        ));
    }
    Ok(())
}

/// One ffmpeg process encoding a single camera's frames.
pub(super) struct H264Writer {
    child: Child,
    stdin: Option<ChildStdin>,
    stderr: Option<JoinHandle<Vec<u8>>>,
    width: u32,
    height: u32,
    frames: u32,
}

impl H264Writer {
    /// Start ffmpeg writing `path`; `program` has passed [`require`].
    pub(super) fn create(
        program: &Path,
        path: &Path,
        width: u32,
        height: u32,
        fps: u32,
    ) -> io::Result<Self> {
        let mut child = Command::new(program)
            .args(["-y", "-loglevel", "error"])
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24"])
            .args(["-s", &format!("{width}x{height}"), "-r", &fps.to_string()])
            .args(["-i", "pipe:0", "-an"])
            .args(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
            .args([
                "-c:v",
                ENCODER,
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
            ])
            .args(["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            .arg(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|err| io::Error::other(format!("start {}: {err}", program.display())))?;
        let stdin = child.stdin.take();
        // Drained so a chatty ffmpeg never blocks on a full stderr pipe.
        let stderr = child.stderr.take().map(|mut pipe| {
            std::thread::spawn(move || {
                let mut buf = Vec::new();
                let _ = pipe.read_to_end(&mut buf);
                buf.drain(..buf.len().saturating_sub(STDERR_LIMIT));
                buf
            })
        });
        Ok(Self {
            child,
            stdin,
            stderr,
            width,
            height,
            frames: 0,
        })
    }

    /// Pipe one packed RGB frame; blocks while ffmpeg is behind.
    pub(super) fn push(&mut self, rgb: &[u8]) -> io::Result<()> {
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| io::Error::other("ffmpeg writer is closed"))?;
        if let Err(err) = stdin.write_all(rgb) {
            // ffmpeg exited: its stderr names why, not the broken pipe.
            return Err(self.exit().err().unwrap_or(err));
        }
        self.frames += 1;
        Ok(())
    }

    /// Close ffmpeg's input and wait for it to finalize the file.
    pub(super) fn finish(mut self) -> io::Result<super::Stats> {
        if self.frames == 0 {
            return Err(io::Error::other("no frames were recorded"));
        }
        self.exit()?;
        Ok(super::Stats {
            frame_count: self.frames,
            width: (self.width + 1) & !1,
            height: (self.height + 1) & !1,
        })
    }

    fn exit(&mut self) -> io::Result<()> {
        drop(self.stdin.take());
        let status = self.child.wait()?;
        let stderr = self
            .stderr
            .take()
            .and_then(|handle| handle.join().ok())
            .unwrap_or_default();
        if !status.success() {
            return Err(io::Error::other(format!(
                "ffmpeg failed with {status}: {}",
                String::from_utf8_lossy(&stderr).trim()
            )));
        }
        Ok(())
    }
}

impl Drop for H264Writer {
    /// An abandoned writer stops its encoder instead of leaving it running.
    fn drop(&mut self) {
        if self.stdin.take().is_some() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}
