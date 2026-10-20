//! The eframe app: each frame drains the feed into the model, renders the
//! model's view and sends the verbs the operator asked for.

use std::time::Duration;

use crate::feed::Feed;
use crate::model::Model;
use crate::ui::{self, UiState};

/// How often the window repaints without new authority state.
pub const REFRESH: Duration = Duration::from_millis(100);

/// The cockpit window.
pub struct CockpitApp {
    feed: Feed,
    model: Model,
    ui: UiState,
}

impl CockpitApp {
    /// A window over `feed`.
    pub fn new(feed: Feed, model: Model) -> Self {
        Self {
            feed,
            model,
            ui: UiState::default(),
        }
    }
}

impl eframe::App for CockpitApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.feed.drain(&mut self.model);
        let view = self.model.view(self.feed.now_ns());
        for verb in ui::show(ui, &view, &mut self.ui) {
            self.feed.send(verb);
        }
        ui.ctx().request_repaint_after(REFRESH);
    }
}
