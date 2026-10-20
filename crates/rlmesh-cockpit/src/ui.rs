//! The egui layer: renders one [`View`] and returns the verbs the operator
//! asked for. It holds no session state beyond what the widgets need (the
//! selected tab, the arm confirmation, uploaded thumbnails), so the same
//! code renders a live session, a test fixture or a headless snapshot.

use std::collections::BTreeMap;

use egui::{
    Align, Button, Color32, CornerRadius, Grid, Layout, Modal, RichText, ScrollArea, Sense, Stroke,
    TextureHandle, TextureOptions, Vec2, pos2, vec2,
};
use rlmesh_wire::{ChannelId, EdgeMode, Mode};

use crate::model::Verb;
use crate::timeline::Tone;
use crate::values::Value;
use crate::view::{View, clocks};

/// The central panel's tabs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tab {
    /// Declared vs measured (§8.10).
    #[default]
    Measured,
    /// The lease and mode timeline.
    Timeline,
    /// The binding tables (§6.8).
    Bindings,
}

/// What the widgets keep between frames.
#[derive(Default)]
pub struct UiState {
    /// The selected tab.
    pub tab: Tab,
    /// The arm confirmation is open.
    pub confirm_arm: bool,
    textures: BTreeMap<ChannelId, (u64, TextureHandle)>,
}

impl std::fmt::Debug for UiState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UiState")
            .field("tab", &self.tab)
            .field("confirm_arm", &self.confirm_arm)
            .finish_non_exhaustive()
    }
}

/// The colour of a tone.
pub fn tone_color(tone: Tone) -> Color32 {
    match tone {
        Tone::Off => Color32::from_gray(110),
        Tone::Neutral => Color32::from_rgb(70, 130, 200),
        Tone::Good => Color32::from_rgb(50, 160, 80),
        Tone::Warn => Color32::from_rgb(220, 160, 30),
        Tone::Bad => Color32::from_rgb(210, 50, 50),
    }
}

/// The e-stop button's fill.
pub const ESTOP_RED: Color32 = Color32::from_rgb(200, 16, 16);

fn mode_color(mode: Option<Mode>) -> Color32 {
    match mode {
        None => Color32::from_gray(90),
        Some(mode) => tone_color(crate::timeline::mode_tone(mode)),
    }
}

fn hz(v: Option<f64>) -> String {
    v.map_or_else(|| "—".into(), |v| format!("{v:.1}"))
}

fn ms(v: Option<f64>) -> String {
    v.map_or_else(|| "—".into(), |v| format!("{v:.1}"))
}

/// Renders `view` into `ui`; returns the verbs to send.
pub fn show(ui: &mut egui::Ui, view: &View, state: &mut UiState) -> Vec<Verb> {
    let mut verbs = Vec::new();
    egui::Panel::top("cockpit_header").show(ui, |ui| header(ui, view, state, &mut verbs));
    egui::Panel::left("cockpit_channels")
        .default_size(440.0)
        .show(ui, |ui| channels(ui, view, state));
    egui::CentralPanel::default_margins().show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut state.tab, Tab::Measured, "Declared vs measured");
            ui.selectable_value(&mut state.tab, Tab::Timeline, "Lease and mode timeline");
            ui.selectable_value(&mut state.tab, Tab::Bindings, "Bindings");
        });
        ui.separator();
        ScrollArea::both()
            .auto_shrink(false)
            .show(ui, |ui| match state.tab {
                Tab::Measured => measured(ui, view),
                Tab::Timeline => timeline(ui, view),
                Tab::Bindings => bindings(ui, view),
            });
    });
    if state.confirm_arm {
        let modal = Modal::new(egui::Id::unique("cockpit_confirm_arm")).show(ui.ctx(), |ui| {
            ui.set_width(380.0);
            ui.heading("Arm the session?");
            ui.label(
                "Arming starts a fresh epoch. Controllers may then take leases and move \
                 the robot.",
            );
            ui.add_space(8.0);
            let mut done = false;
            ui.horizontal(|ui| {
                if ui
                    .add(Button::new(RichText::new("Arm").strong()).fill(tone_color(Tone::Good)))
                    .clicked()
                {
                    verbs.push(Verb::Arm);
                    done = true;
                }
                if ui.button("Cancel").clicked() {
                    done = true;
                }
            });
            done
        });
        if modal.inner || modal.should_close() {
            state.confirm_arm = false;
        }
    }
    verbs
}

fn header(ui: &mut egui::Ui, view: &View, state: &mut UiState, verbs: &mut Vec<Verb>) {
    let h = &view.header;
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        let mode = h
            .mode
            .map_or_else(|| "NOT CONNECTED".into(), |m| m.to_string());
        egui::Frame::new()
            .fill(mode_color(h.mode))
            .corner_radius(CornerRadius::same(4))
            .inner_margin(egui::Margin::symmetric(10, 4))
            .show(ui, |ui| {
                ui.label(
                    RichText::new(mode)
                        .size(20.0)
                        .strong()
                        .color(Color32::WHITE),
                );
            });
        ui.vertical(|ui| {
            ui.label(RichText::new(format!(
                "epoch {} · revision {} · session {}",
                h.epoch, h.revision, h.session
            )));
            let host = if h.host_present {
                RichText::new("host present")
            } else {
                RichText::new("host not visible").color(tone_color(Tone::Bad))
            };
            ui.horizontal(|ui| {
                ui.label(host);
                ui.label(RichText::new(&h.clock).weak());
            });
        });
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            let estop = ui
                .add(
                    Button::new(
                        RichText::new("E-STOP")
                            .size(22.0)
                            .strong()
                            .color(Color32::WHITE),
                    )
                    .fill(ESTOP_RED)
                    .min_size(vec2(150.0, 44.0)),
                )
                .on_hover_text("Emergency stop: one click, latches until the robot clears it");
            if estop.clicked() {
                verbs.push(Verb::Estop);
            }
            ui.add_space(12.0);
            if ui
                .add_enabled(
                    h.can_disarm,
                    Button::new("Disarm").min_size(vec2(80.0, 32.0)),
                )
                .on_hover_text("Back to DISARMED, revoking every lease")
                .clicked()
            {
                verbs.push(Verb::Disarm);
            }
            if ui
                .add_enabled(h.can_arm, Button::new("Arm…").min_size(vec2(80.0, 32.0)))
                .on_hover_text("DISARMED or SAFE to ARMED, in a fresh epoch (asks first)")
                .clicked()
            {
                state.confirm_arm = true;
            }
        });
    });
    if let Some(note) = &h.estop_note {
        ui.label(RichText::new(note).strong().color(tone_color(Tone::Bad)));
    }
    if let Some((verb, result, ago)) = &h.last_outcome {
        let text = match result {
            Ok(()) => RichText::new(format!("{} accepted ({ago:.0} s ago)", verb.name())),
            Err(why) => RichText::new(format!("{} refused: {why} ({ago:.0} s ago)", verb.name()))
                .color(tone_color(Tone::Warn)),
        };
        ui.label(text);
    }
    if h.overrun > 0 {
        ui.label(
            RichText::new(format!(
                "the cockpit dropped {} inputs it could not keep up with (not network loss)",
                h.overrun
            ))
            .color(tone_color(Tone::Warn)),
        );
    }
    ui.add_space(4.0);
}

fn channels(ui: &mut egui::Ui, view: &View, state: &mut UiState) {
    ui.heading("Live channels");
    ui.label(RichText::new("age = now − capture, cockpit clock").weak());
    ui.separator();
    ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
        for row in &view.tree {
            ui.horizontal(|ui| {
                ui.add_space(row.depth as f32 * 14.0);
                let Some(kind) = &row.kind else {
                    let label = RichText::new(&row.label).strong();
                    ui.label(if row.depth == 0 {
                        label.size(16.0)
                    } else {
                        label
                    });
                    if !row.detail.is_empty() {
                        ui.label(RichText::new(&row.detail).weak().small());
                    }
                    return;
                };
                ui.label(RichText::new(&row.label).strong());
                ui.label(RichText::new(kind).small().color(tone_color(Tone::Neutral)));
                let age = match row.age_ms {
                    Some(age) => format!("{age:.0} ms"),
                    None => "no sample".into(),
                };
                let color = if row.stale {
                    tone_color(Tone::Warn)
                } else {
                    ui.visuals().weak_text_color()
                };
                ui.label(RichText::new(age).small().color(color));
                if row.lost > 0 {
                    ui.label(
                        RichText::new(format!("{} lost", row.lost))
                            .small()
                            .color(tone_color(Tone::Warn)),
                    );
                }
            })
            .response
            .on_hover_text(&row.detail);
            let Some(value) = &row.value else {
                continue;
            };
            ui.horizontal(|ui| {
                ui.add_space(row.depth as f32 * 14.0 + 14.0);
                match value {
                    Value::Image(thumb) => {
                        let channel = row.channel.unwrap_or(ChannelId(0));
                        let entry = state.textures.get(&channel);
                        let image =
                            egui::ColorImage::from_rgb([thumb.width, thumb.height], &thumb.rgb);
                        let texture = match entry {
                            Some((seq, handle)) if *seq == thumb.seq => handle.clone(),
                            Some((_, handle)) => {
                                let mut handle = handle.clone();
                                handle.set(image, TextureOptions::LINEAR);
                                state.textures.insert(channel, (thumb.seq, handle.clone()));
                                handle
                            }
                            None => {
                                let handle = ui.ctx().load_texture(
                                    format!("thumb-{}", channel.0),
                                    image,
                                    TextureOptions::LINEAR,
                                );
                                state.textures.insert(channel, (thumb.seq, handle.clone()));
                                handle
                            }
                        };
                        ui.image((texture.id(), vec2(thumb.width as f32, thumb.height as f32)));
                        ui.label(RichText::new(value.brief()).small().weak());
                    }
                    _ => {
                        if !row.plot.is_empty() {
                            sparkline(ui, &row.plot, view.lane_window_s.min(10.0));
                        }
                        ui.add(
                            egui::Label::new(RichText::new(value.brief()).monospace().small())
                                .wrap(),
                        );
                    }
                }
            });
        }
    });
}

const SERIES: [Color32; 3] = [
    Color32::from_rgb(70, 130, 200),
    Color32::from_rgb(220, 120, 40),
    Color32::from_rgb(60, 170, 110),
];

fn sparkline(ui: &mut egui::Ui, series: &[Vec<[f64; 2]>], window_s: f64) {
    let (response, painter) = ui.allocate_painter(vec2(120.0, 28.0), Sense::hover());
    let rect = response.rect;
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    let (lo, hi) = series
        .iter()
        .flatten()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
            (lo.min(p[1]), hi.max(p[1]))
        });
    if !lo.is_finite() || !hi.is_finite() {
        return;
    }
    let span = (hi - lo).max(1e-9);
    for (i, points) in series.iter().enumerate() {
        let line: Vec<egui::Pos2> = points
            .iter()
            .map(|p| {
                let x = rect.right() + (p[0] / window_s.max(1e-3)) as f32 * rect.width();
                let y = rect.bottom() - ((p[1] - lo) / span) as f32 * (rect.height() - 4.0) - 2.0;
                pos2(x.max(rect.left()), y)
            })
            .collect();
        painter.add(egui::Shape::line(
            line,
            Stroke::new(1.2, SERIES[i % SERIES.len()]),
        ));
    }
}

fn section(ui: &mut egui::Ui, title: &str, note: &str) {
    ui.add_space(6.0);
    ui.label(RichText::new(title).heading());
    ui.label(RichText::new(note).weak());
}

fn measured(ui: &mut egui::Ui, view: &View) {
    section(
        ui,
        "Streams",
        &format!(
            "Hz, each on its own clock: declared ({}), produced and published ({}), \
             admitted delivery (contract), unique received ({}), per capture second ({}), \
             and each subscriber's seq advance (its heartbeats).",
            clocks::DECLARED,
            clocks::PUBLISHER,
            clocks::COCKPIT,
            clocks::CAPTURE
        ),
    );
    Grid::new("streams").striped(true).show(ui, |ui| {
        for h in [
            "stream",
            "kind",
            "declared",
            "plan",
            "admitted",
            "produced",
            "published",
            "received",
            "per capture s",
            "subscribers",
            "age ms",
            "bound ms",
            "lost",
            "",
        ] {
            ui.label(RichText::new(h).strong());
        }
        ui.end_row();
        for r in &view.streams {
            ui.label(&r.label);
            ui.label(&r.kind);
            ui.label(
                r.declared_hz
                    .map_or_else(|| "event".into(), |v| format!("{v}")),
            );
            ui.label(&r.plan)
                .on_hover_text(format!("send ceiling: {}", r.ceiling));
            let admitted: Vec<String> = r
                .admitted
                .iter()
                .map(|(who, c)| format!("{who} {c}"))
                .collect();
            ui.label(if admitted.is_empty() {
                "—".into()
            } else {
                admitted.join(", ")
            });
            ui.label(hz(r.produced_hz));
            ui.label(hz(r.published_hz));
            ui.label(hz(r.received_hz));
            ui.label(hz(r.capture_hz));
            let heard: Vec<String> = r
                .heard
                .iter()
                .map(|(who, rate)| format!("{who} {}", hz(*rate)))
                .collect();
            ui.label(if heard.is_empty() {
                "—".into()
            } else {
                heard.join(", ")
            });
            let age = RichText::new(ms(r.age_ms));
            ui.label(if r.stale {
                age.color(tone_color(Tone::Warn))
            } else {
                age
            });
            ui.label(
                r.bound
                    .map_or_else(|| "—".into(), |(b, from)| format!("{b:.0} ({from})")),
            );
            ui.label(r.lost.to_string());
            match &r.drift {
                Some(drift) => ui.label(RichText::new(drift).color(tone_color(Tone::Warn))),
                None => ui.label(""),
            };
            ui.end_row();
        }
    });

    section(
        ui,
        "Controllers",
        &format!(
            "starts, completions and busy skips per second ({}); chunks per second and \
             chunk age at receipt ({}); valid coverage remaining is the cockpit's estimate \
             in session time.",
            clocks::CONTROLLER,
            clocks::COCKPIT
        ),
    );
    Grid::new("controllers").striped(true).show(ui, |ui| {
        for h in [
            "controller",
            "class",
            "binding",
            "replan",
            "starts",
            "completions",
            "busy skips",
            "other skips",
            "pending ms",
            "p50/p99 ms",
            "chunks",
            "chunk age ms",
            "coverage ms",
            "leases",
        ] {
            ui.label(RichText::new(h).strong());
        }
        ui.end_row();
        for r in &view.controllers {
            ui.label(&r.id);
            ui.label(&r.class);
            ui.label(r.binding.as_deref().unwrap_or("—"));
            ui.label(hz(r.replan_hz));
            if r.manual {
                ui.label(RichText::new("by hand").weak());
                ui.label(RichText::new("by hand").weak());
            } else {
                ui.label(hz(r.starts_hz));
                let completions = RichText::new(hz(r.completions_hz));
                ui.label(if r.drifting {
                    completions.color(tone_color(Tone::Warn))
                } else {
                    completions
                })
                .on_hover_text(if r.drifting { "rate_drift" } else { "" });
            }
            ui.label(format!("{} ({}/s)", r.busy_skips, hz(r.busy_hz)));
            let skips: Vec<String> = r.skips.iter().map(|(k, n)| format!("{k} {n}")).collect();
            ui.label(if skips.is_empty() {
                "—".into()
            } else {
                skips.join(", ")
            });
            let pending = RichText::new(ms(r.pending_ms));
            ui.label(match r.pending_ms {
                Some(p) if p > 1000.0 => pending.color(tone_color(Tone::Bad)),
                _ => pending,
            });
            ui.label(
                r.latency_ms
                    .map_or_else(|| "—".into(), |(a, b)| format!("{a:.1}/{b:.1}")),
            );
            ui.label(hz(r.chunks_hz));
            ui.label(
                r.chunk_age_ms
                    .map_or_else(|| "—".into(), |(a, b)| format!("{a:.1} (max {b:.1})")),
            );
            ui.label(ms(r.coverage_ms));
            ui.label(if r.leases.is_empty() {
                "—".into()
            } else {
                r.leases.join(", ")
            });
            ui.end_row();
        }
    });

    section(
        ui,
        "Edges",
        &format!(
            "control rate ({}); a deadline miss is a tick more than two control periods \
             after the previous one (§9.1), seq gaps excluded.",
            clocks::EDGE
        ),
    );
    Grid::new("edges").striped(true).show(ui, |ui| {
        for h in [
            "part",
            "command",
            "declared",
            "ticks",
            "misses",
            "recent",
            "longest gap ms",
            "mode",
            "dispatched",
            "lease",
            "coverage ms",
            "fenced",
            "queue",
            "hold / safe",
        ] {
            ui.label(RichText::new(h).strong());
        }
        ui.end_row();
        for r in &view.edges {
            ui.label(&r.label);
            ui.label(&r.command);
            ui.label(format!("{}", r.control_hz));
            ui.label(hz(r.tick_hz));
            ui.label(r.misses.to_string());
            ui.label(r.recent_misses.to_string());
            ui.label(ms(r.longest_gap_ms));
            match r.mode {
                Some((mode, since)) => ui.label(
                    RichText::new(format!("{mode} ({since:.1} s)"))
                        .strong()
                        .color(tone_color(edge_tone(mode))),
                ),
                None => ui.label("—"),
            };
            ui.label(format!("{}/{}", r.present.0, r.present.1));
            ui.label(r.holder.as_deref().unwrap_or("free"))
                .on_hover_text(r.last_chunk.as_deref().map_or_else(
                    || "no chunk under this lease yet".to_string(),
                    |c| format!("newest chunk: {c}"),
                ));
            ui.label(ms(r.coverage_ms));
            ui.label(r.fenced.to_string());
            ui.label(r.queue_depth.map_or_else(|| "—".into(), |q| q.to_string()));
            ui.label(format!("{:.0} ms / {}", r.hold.0, r.hold.1));
            ui.end_row();
        }
    });

    section(
        ui,
        "Participants",
        "clock mapping (session = participant + offset, ± error) and heartbeats",
    );
    Grid::new("participants").striped(true).show(ui, |ui| {
        for h in [
            "participant",
            "roles",
            "class",
            "offset ms",
            "± ms",
            "rtt ms",
            "health",
            "heartbeat ms ago",
        ] {
            ui.label(RichText::new(h).strong());
        }
        ui.end_row();
        for r in &view.participants {
            ui.label(&r.id);
            ui.label(&r.roles);
            ui.label(&r.class);
            ui.label(ms(r.offset_ms));
            ui.label(r.error_ms.map_or_else(|| "—".into(), |v| format!("{v:.3}")));
            ui.label(r.rtt_ms.map_or_else(|| "—".into(), |v| format!("{v:.3}")));
            ui.label(r.health.as_deref().unwrap_or("—"));
            ui.label(ms(r.heartbeat_age_ms));
            ui.end_row();
        }
    });
}

fn edge_tone(mode: EdgeMode) -> Tone {
    crate::model::edge_tone(mode)
}

fn timeline(ui: &mut egui::Ui, view: &View) {
    section(
        ui,
        "Lanes",
        &format!(
            "the last {:.0} s, now at the right edge",
            view.lane_window_s
        ),
    );
    let label_width = 220.0;
    let width = (ui.available_width() - label_width - 16.0).max(200.0);
    for lane in &view.lanes {
        ui.horizontal(|ui| {
            ui.allocate_ui_with_layout(
                vec2(label_width, 20.0),
                Layout::left_to_right(Align::Center),
                |ui| {
                    ui.set_min_width(label_width);
                    ui.add(egui::Label::new(&lane.label).truncate());
                },
            );
            let (response, painter) = ui.allocate_painter(vec2(width, 20.0), Sense::hover());
            let rect = response.rect;
            painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
            let x =
                |t: f64| rect.right() + (t / view.lane_window_s.max(1e-3)) as f32 * rect.width();
            for (start, end, label, tone) in &lane.segments {
                let seg = egui::Rect::from_min_max(
                    pos2(x(*start), rect.top() + 2.0),
                    pos2(x(*end).max(x(*start) + 1.0), rect.bottom() - 2.0),
                );
                painter.rect_filled(seg, 2.0, tone_color(*tone));
                if seg.width() > 40.0 {
                    painter.text(
                        seg.left_center() + Vec2::new(4.0, 0.0),
                        egui::Align2::LEFT_CENTER,
                        label,
                        egui::FontId::proportional(11.0),
                        Color32::WHITE,
                    );
                }
            }
        });
    }
    section(
        ui,
        "Events",
        "newest first; t = seconds before now; session time from the snapshot",
    );
    Grid::new("timeline").striped(true).show(ui, |ui| {
        for h in ["t", "session s", "epoch", "kind", "event"] {
            ui.label(RichText::new(h).strong());
        }
        ui.end_row();
        for row in view.timeline.iter().take(300) {
            ui.label(format!("−{:.1}", row.ago_s));
            ui.label(
                row.session_s
                    .map_or_else(|| "—".into(), |s| format!("{s:.3}")),
            );
            ui.label(row.epoch.to_string());
            ui.label(RichText::new(row.kind).color(tone_color(row.tone)));
            ui.label(&row.text);
            ui.end_row();
        }
    });
}

fn bindings(ui: &mut egui::Ui, view: &View) {
    if view.bindings.is_empty() {
        ui.label("No controller has declared a binding.");
    }
    for (i, b) in view.bindings.iter().enumerate() {
        ui.add_space(6.0);
        ui.label(RichText::new(&b.title).monospace().strong());
        ui.label(RichText::new(&b.subtitle).weak());
        if let Some(raw) = &b.raw {
            ui.label(RichText::new(raw).monospace().small());
            continue;
        }
        table(
            ui,
            ("entries", i),
            &["entry", "dir", "part", "status", "plan", "delivers"],
            b.entries.iter().map(|r| r.as_slice()),
        );
        table(
            ui,
            ("inputs", i),
            &["input", "contract", "source", "delivery", "freshness"],
            b.inputs.iter().map(|r| r.as_slice()),
        );
        table(
            ui,
            ("actions", i),
            &["action", "grid", "chunk", "valid", "edge", "age cap"],
            b.actions.iter().map(|r| r.as_slice()),
        );
        ui.separator();
    }
}

fn table<'a>(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash + std::fmt::Debug,
    head: &[&str],
    rows: impl Iterator<Item = &'a [String]>,
) {
    let mut rows = rows.peekable();
    if rows.peek().is_none() {
        return;
    }
    ui.add_space(4.0);
    Grid::new(id).striped(true).show(ui, |ui| {
        for h in head {
            ui.label(RichText::new(*h).strong().monospace());
        }
        ui.end_row();
        for row in rows {
            for cell in row {
                ui.label(RichText::new(cell).monospace());
            }
            ui.end_row();
        }
    });
}
