//! `rlmesh viewtest` — smoke-test the native terminal/HTTP renderer with synthetic
//! frames (a box bouncing on a shifting gradient) plus a live HUD.
//!
//! Drives [`rlmesh_viewer::Viewer`] directly — the SAME renderer the Python `view=`
//! path uses — with no env, model, or container. If THIS shows a picture, your
//! terminal + the renderer are fine and a blank real run is a frame-*source* problem
//! (env camera / role); if THIS is also blank, the problem is the renderer / TTY /
//! terminal compatibility. Quit with `q` / `Esc` / `Ctrl-C`.

use std::io::Write;
use std::time::Duration;

use anyhow::Result;
use rlmesh_viewer::{Backend, FrameFormat, Viewer};

use crate::cli::ViewtestArgs;

/// Synthetic frame size; the renderer scales it to the terminal / browser.
const W: u32 = 240;
const H: u32 = 180;

/// A bouncing box: position + velocity in pixels, fixed size.
struct BBox {
    x: f32,
    y: f32,
    w: u32,
    h: u32,
    vx: f32,
    vy: f32,
}

/// Fill `buf` with an HxWx3 uint8 frame: a moving diagonal RGB gradient plus a
/// bouncing white box with a dark border.
fn make_frame(t: u32, b: &BBox, buf: &mut Vec<u8>) {
    buf.clear();
    let phase = t as f32 * 0.05;
    for y in 0..H {
        let yn = y as f32 / (H - 1) as f32;
        for x in 0..W {
            let xn = x as f32 / (W - 1) as f32;
            let r = 0.5 + 0.5 * (6.0 * xn + phase).sin();
            let g = 0.5 + 0.5 * (6.0 * yn + phase * 1.3 + 2.0).sin();
            let bl = 0.5 + 0.5 * (6.0 * (xn + yn) + phase * 0.7 + 4.0).sin();
            buf.push((r * 255.0) as u8);
            buf.push((g * 255.0) as u8);
            buf.push((bl * 255.0) as u8);
        }
    }
    let (x0, y0) = (b.x as u32, b.y as u32);
    for yy in y0..(y0 + b.h).min(H) {
        for xx in x0..(x0 + b.w).min(W) {
            let border = xx < x0 + 2 || xx + 2 >= x0 + b.w || yy < y0 + 2 || yy + 2 >= y0 + b.h;
            let px: [u8; 3] = if border {
                [10, 10, 10]
            } else {
                [255, 255, 255]
            };
            let i = ((yy * W + xx) * 3) as usize;
            buf[i..i + 3].copy_from_slice(&px);
        }
    }
}

/// Run the diagnostic loop until `--frames` elapse or the user quits. Diagnostic
/// messages go to `stderr`; the rendered frames take over the real terminal stdout.
pub fn run(args: &ViewtestArgs, stderr: &mut impl Write) -> Result<i32> {
    let terminal = args.both || args.http.is_none();
    let mut backends = Vec::new();
    if terminal {
        backends.push(Backend::Terminal);
    }
    if let Some(port) = args.http {
        backends.push(Backend::Http(port));
    }

    let viewer = Viewer::new(&backends, args.fps, FrameFormat::Jpeg(80));

    // Surface setup warnings before the alt-screen can hide them.
    let warns = viewer.warnings();
    if !warns.is_empty() {
        let _ = std::fs::write("viewtest.warnings", warns.join("\n") + "\n");
        for w in &warns {
            writeln!(stderr, "[viewtest] WARNING: {w}")?;
        }
    }

    viewer.set_sources(vec!["demo".to_string()], 0);
    if let Some(port) = args.http {
        writeln!(
            stderr,
            "[viewtest] http viewer on http://localhost:{port}  \
             (over SSH: ssh -L {port}:localhost:{port} …)"
        )?;
    }

    let mut b = BBox {
        x: 20.0,
        y: 20.0,
        w: 48,
        h: 48,
        vx: 2.3,
        vy: 1.7,
    };
    let mut reward = 0.0_f64;
    let period = Duration::from_secs_f64(1.0 / f64::from(args.fps.max(1)));
    let mut buf: Vec<u8> = Vec::with_capacity((W * H * 3) as usize);

    for t in 0..args.frames {
        b.x += b.vx;
        b.y += b.vy;
        if b.x <= 0.0 || b.x + b.w as f32 >= W as f32 {
            b.vx = -b.vx;
            b.x = b.x.clamp(0.0, (W - b.w) as f32);
        }
        if b.y <= 0.0 || b.y + b.h as f32 >= H as f32 {
            b.vy = -b.vy;
            b.y = b.y.clamp(0.0, (H - b.h) as f32);
        }

        if !args.no_frames && viewer.wants_frame() {
            make_frame(t, &b, &mut buf);
            viewer.feed_frame(&buf, W, H, 3);
        }
        reward += 0.1 * (f64::from(t) * 0.1).sin();
        viewer.feed_hud(i64::from(t), reward, "running");

        if viewer.should_quit() {
            break;
        }
        std::thread::sleep(period);
    }

    // viewer drops here → terminal restored / HTTP server stopped.
    writeln!(stderr, "[viewtest] done.")?;
    Ok(0)
}
