//! The eframe app: each frame brings the model up to date (a live feed
//! drained, or a replay advanced), renders the model's view and sends the
//! verbs the operator asked for (live only).

use std::time::{Duration, Instant};

use crate::feed::Feed;
use crate::model::Model;
use crate::replay::Replay;
use crate::ui::{self, UiState};

/// How often the window repaints without new authority state.
pub const REFRESH: Duration = Duration::from_millis(100);

enum Source {
    Live { feed: Feed, model: Model },
    Replay { replay: Replay, last: Instant },
}

/// The cockpit window.
pub struct CockpitApp {
    source: Source,
    ui: UiState,
}

impl CockpitApp {
    /// A window over a live `feed`.
    pub fn new(feed: Feed, model: Model) -> Self {
        Self {
            source: Source::Live { feed, model },
            ui: UiState::default(),
        }
    }

    /// A window over a recorded session.
    pub fn replay(replay: Replay) -> Self {
        let mut ui = UiState::default();
        ui.replay = true;
        Self {
            source: Source::Replay {
                replay,
                last: Instant::now(),
            },
            ui,
        }
    }
}

impl eframe::App for CockpitApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        match &mut self.source {
            Source::Live { feed, model } => {
                feed.drain(model);
                let view = model.view(feed.now_ns());
                for verb in ui::show(ui, &view, &mut self.ui) {
                    feed.send(verb);
                }
                ui.ctx().request_repaint_after(REFRESH);
            }
            Source::Replay { replay, last } => {
                let now = Instant::now();
                replay.tick(now - *last);
                *last = now;
                egui::Panel::top("cockpit_replay").show(ui, |ui| ui::replay_bar(ui, replay));
                let view = replay.model().view(replay.at_ns());
                ui::show(ui, &view, &mut self.ui);
                if replay.playing {
                    ui.ctx().request_repaint();
                }
            }
        }
    }
}
