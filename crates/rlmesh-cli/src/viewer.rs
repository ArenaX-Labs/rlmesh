use std::io::{self, Read};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Duration;

use anyhow::{Result, anyhow};
use eframe::egui;
use egui::{ColorImage, TextureHandle, TextureOptions};
use image::ImageFormat;

use crate::cli::ViewerArgs;

pub fn run(args: &ViewerArgs) -> Result<i32> {
    run_render_viewer(RenderViewerConfig {
        title: args.title.clone(),
    })
    .map_err(|err| anyhow!("failed to launch render viewer: {err}"))?;
    Ok(0)
}

#[derive(Debug, Clone)]
struct RenderViewerConfig {
    title: String,
}

struct RenderViewerApp {
    title: String,
    rx: Receiver<ViewerEvent>,
    frame: Option<ImageFrame>,
    texture: TextureSlot,
    status: String,
    should_close: bool,
}

impl RenderViewerApp {
    fn new(config: RenderViewerConfig) -> Self {
        Self::with_receiver(config, spawn_stdin_reader())
    }

    fn with_receiver(config: RenderViewerConfig, rx: Receiver<ViewerEvent>) -> Self {
        Self {
            title: config.title,
            rx,
            frame: None,
            texture: TextureSlot::default(),
            status: "Waiting for render frames...".to_string(),
            should_close: false,
        }
    }

    fn poll_events(&mut self) {
        while let Ok(event) = self.rx.try_recv() {
            match event {
                ViewerEvent::Frame(frame) => {
                    self.status = format!("{} x {}", frame.width, frame.height);
                    self.frame = Some(frame);
                }
                ViewerEvent::Clear => {
                    self.frame = None;
                    self.status = "No frame available".to_string();
                }
                ViewerEvent::Exit => {
                    self.should_close = true;
                }
                ViewerEvent::Error(message) => {
                    self.frame = None;
                    self.status = message;
                }
            }
        }
    }
}

impl eframe::App for RenderViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_events();
        ctx.request_repaint_after(Duration::from_millis(33));

        if self.should_close {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }

        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading(&self.title);
            ui.label(&self.status);
            ui.separator();

            if let Some(frame) = &self.frame {
                update_texture(&ctx, &mut self.texture, frame);
                if let Some(texture) = &self.texture.texture {
                    let available = ui.available_size();
                    let size = fit_to_bounds(frame.width as f32, frame.height as f32, available);
                    ui.image((texture.id(), size));
                }
            } else {
                ui.label("Open the viewer from Python and call reset(), step(), or render().");
            }
        });
    }
}

#[derive(Default)]
struct TextureSlot {
    texture: Option<TextureHandle>,
}

#[derive(Debug, Clone)]
struct ImageFrame {
    label: String,
    width: usize,
    height: usize,
    rgba: Vec<u8>,
}

enum ViewerEvent {
    Frame(ImageFrame),
    Clear,
    Exit,
    Error(String),
}

fn run_render_viewer(config: RenderViewerConfig) -> eframe::Result<()> {
    let title = config.title.clone();
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title(title.clone())
            .with_inner_size([960.0, 720.0])
            .with_min_inner_size([480.0, 360.0]),
        ..Default::default()
    };

    eframe::run_native(
        &title,
        native_options,
        Box::new(move |_cc| Ok(Box::new(RenderViewerApp::new(config.clone())))),
    )
}

fn spawn_stdin_reader() -> Receiver<ViewerEvent> {
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let stdin = io::stdin();
        let mut reader = stdin.lock();
        pump_wire_frames(&mut reader, &tx);
    });

    rx
}

/// Read framed viewer events from `reader` and forward them on `tx` until the
/// stream ends, errors, or the receiver is dropped.
fn pump_wire_frames(reader: &mut impl Read, tx: &mpsc::Sender<ViewerEvent>) {
    loop {
        match read_wire_frame(reader) {
            Ok(Some(event)) => {
                if tx.send(event).is_err() {
                    break;
                }
            }
            Ok(None) => {
                let _ = tx.send(ViewerEvent::Exit);
                break;
            }
            Err(err) => {
                // Surface the error and keep the window open so the user can
                // read it. Do NOT follow with Exit: the UI drains all pending
                // events in one frame, so an immediate Exit would close the
                // window before the error status is ever rendered.
                let _ = tx.send(ViewerEvent::Error(err));
                break;
            }
        }
    }
}

fn read_wire_frame(reader: &mut impl Read) -> Result<Option<ViewerEvent>, String> {
    let mut header = [0_u8; 5];
    match reader.read_exact(&mut header) {
        Ok(()) => {}
        Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(err) => return Err(format!("failed to read viewer header: {err}")),
    }

    let kind = header[0];
    let len =
        u32::from_le_bytes(header[1..5].try_into().expect("header[1..5] is 4 bytes")) as usize;

    if kind == 0 {
        return Ok(Some(ViewerEvent::Clear));
    }
    if kind != 1 {
        return Err(format!("unsupported viewer event kind: {kind}"));
    }

    let mut raw = vec![0_u8; len];
    reader
        .read_exact(&mut raw)
        .map_err(|err| format!("failed to read viewer frame payload: {err}"))?;

    let frame = decode_wire_frame(raw)?;
    Ok(Some(ViewerEvent::Frame(frame)))
}

fn decode_wire_frame(raw: Vec<u8>) -> Result<ImageFrame, String> {
    let image = image::load_from_memory_with_format(&raw, ImageFormat::Png)
        .map_err(|err| format!("failed to decode render frame: {err}"))?;
    let rgba = image.to_rgba8();
    let (width, height) = rgba.dimensions();
    Ok(ImageFrame {
        label: "render".to_string(),
        width: width as usize,
        height: height as usize,
        rgba: rgba.into_raw(),
    })
}

fn update_texture(ctx: &egui::Context, slot: &mut TextureSlot, frame: &ImageFrame) {
    let image = ColorImage::from_rgba_unmultiplied([frame.width, frame.height], &frame.rgba);
    if let Some(texture) = &mut slot.texture {
        texture.set(image, TextureOptions::LINEAR);
    } else {
        slot.texture = Some(ctx.load_texture(frame.label.clone(), image, TextureOptions::LINEAR));
    }
}

fn fit_to_bounds(width: f32, height: f32, bounds: egui::Vec2) -> egui::Vec2 {
    let scale = (bounds.x / width).min(bounds.y / height).max(0.1);
    egui::vec2(width * scale, height * scale)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::sync::mpsc;

    use image::{DynamicImage, ImageBuffer, ImageFormat, Luma};

    use super::*;

    #[test]
    fn decode_wire_frame_handles_png() {
        let raw = encode_image(
            DynamicImage::ImageLuma8(ImageBuffer::from_pixel(1, 1, Luma([255_u8]))),
            ImageFormat::Png,
        );
        let frame = decode_wire_frame(raw).unwrap();
        assert_eq!(frame.width, 1);
        assert_eq!(frame.height, 1);
    }

    #[test]
    fn decode_wire_frame_rejects_invalid_payloads() {
        let err = decode_wire_frame(vec![0; 3]).unwrap_err();
        assert!(!err.is_empty());
    }

    fn encode_image(image: DynamicImage, format: ImageFormat) -> Vec<u8> {
        let mut raw = Vec::new();
        image.write_to(&mut Cursor::new(&mut raw), format).unwrap();
        raw
    }

    // Finding #88: a decode error must emit Error and NOT an immediate Exit, so
    // the UI (which drains all pending events in one frame) keeps the window open
    // to display the error instead of closing before it renders.
    #[test]
    fn decode_error_emits_error_without_exit() {
        // Frame header kind=1 (Frame) with a 3-byte payload that is not valid PNG.
        let mut stream = Vec::new();
        stream.extend_from_slice(&[1, 3, 0, 0, 0]); // kind=1, len=3
        stream.extend_from_slice(&[0, 0, 0]); // bogus payload -> decode error
        let mut reader = Cursor::new(stream);

        let (tx, rx) = mpsc::channel();
        pump_wire_frames(&mut reader, &tx);
        drop(tx);

        let events: Vec<ViewerEvent> = rx.into_iter().collect();
        assert_eq!(events.len(), 1, "expected exactly one event (the error)");
        assert!(
            matches!(&events[0], ViewerEvent::Error(_)),
            "expected an Error event"
        );

        // Driving the UI with the error must keep the window open.
        let (tx2, rx2) = mpsc::channel();
        for event in events {
            tx2.send(event).unwrap();
        }
        drop(tx2);
        let mut app = RenderViewerApp::with_receiver(
            RenderViewerConfig {
                title: "test".to_string(),
            },
            rx2,
        );
        app.poll_events();
        assert!(!app.should_close, "decode error must not close the window");
    }
}
