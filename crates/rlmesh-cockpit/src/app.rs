//! The eframe app: each frame brings the model up to date (a live feed
//! drained, or a replay advanced), renders the model's view and sends the
//! verbs the operator asked for (live only).
//!
//! A live session can be paused: the cockpit replays its recent history
//! ([`History`]) while the feed keeps draining underneath, and only the
//! e-stop stays on. Keep asks the host's recorder, through this machine's
//! recordings directory, to keep the last minutes as an incident.

use std::time::{Duration, Instant};

use rlmesh_session::record::{self, Rolling};
use rlmesh_session::wire::SessionId;

use crate::feed::Feed;
use crate::model::{Model, ModelConfig};
use crate::replay::{History, Replay};
use crate::ui::{self, LiveAction, UiState};

/// How often the window repaints without new authority state.
pub const REFRESH: Duration = Duration::from_millis(100);

enum Source {
    Live(Box<Live>),
    Replay { replay: Box<Replay>, last: Instant },
}

struct Live {
    feed: Feed,
    model: Model,
    history: History,
    /// Paused on the history.
    paused: Option<(Replay, Instant)>,
    /// The host's recordings root, when it is on this machine.
    recordings: Option<Rolling>,
    /// The last Keep's answer.
    kept: Option<Result<String, String>>,
}

impl Live {
    /// The session's rolling recording is on this machine.
    fn session(&self) -> Option<(SessionId, &Rolling)> {
        let session = self.model.snapshot()?.session;
        let rolling = self.recordings.as_ref()?;
        rolling
            .session_dir(session)
            .is_dir()
            .then_some((session, rolling))
    }
}

/// The cockpit window.
pub struct CockpitApp {
    source: Source,
    ui: UiState,
}

impl CockpitApp {
    /// A window over a live `feed`; `recordings` is the recordings root
    /// the host records into, when it is on this machine (Keep).
    pub fn new(feed: Feed, model: Model, recordings: Option<Rolling>) -> Self {
        Self {
            source: Source::Live(Box::new(Live {
                feed,
                model,
                history: History::new(History::WINDOW, History::MAX_BYTES),
                paused: None,
                recordings,
                kept: None,
            })),
            ui: UiState::default(),
        }
    }

    /// A window over a recorded session.
    pub fn replay(replay: Replay) -> Self {
        let mut ui = UiState::default();
        ui.replay = true;
        Self {
            source: Source::Replay {
                replay: Box::new(replay),
                last: Instant::now(),
            },
            ui,
        }
    }
}

/// Advances `replay` by the wall time since `last`.
fn tick(replay: &mut Replay, last: &mut Instant) {
    let now = Instant::now();
    replay.tick(now - *last);
    *last = now;
}

impl eframe::App for CockpitApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        match &mut self.source {
            Source::Live(live) => {
                live.feed.drain(&mut live.model, Some(&mut live.history));
                let can_keep = live.session().is_some();
                let action = egui::Panel::top("cockpit_live")
                    .show(ui, |ui| {
                        let action =
                            ui::live_bar(ui, live.paused.is_some(), can_keep, live.kept.as_ref());
                        if let Some((replay, _)) = &mut live.paused {
                            ui::replay_bar(ui, replay, "LAST MINUTE");
                        }
                        action
                    })
                    .inner;
                match action {
                    Some(LiveAction::Pause) => {
                        let replay = live
                            .history
                            .replay(live.model.me().clone(), ModelConfig::default());
                        live.paused = Some((replay, Instant::now()));
                    }
                    Some(LiveAction::Resume) => live.paused = None,
                    Some(LiveAction::Keep) => {
                        live.kept = live.session().map(|(session, rolling)| {
                            record::keep(rolling, session, "keep")
                                .map(|dir| dir.display().to_string())
                                .map_err(|err| err.to_string())
                        });
                    }
                    None => {}
                }
                self.ui.paused = live.paused.is_some();
                let view = match &mut live.paused {
                    Some((replay, last)) => {
                        tick(replay, last);
                        if replay.playing {
                            ui.ctx().request_repaint();
                        }
                        replay.model().view(replay.at_ns())
                    }
                    None => live.model.view(live.feed.now_ns()),
                };
                for verb in ui::show(ui, &view, &mut self.ui) {
                    live.feed.send(verb);
                }
                ui.ctx().request_repaint_after(REFRESH);
            }
            Source::Replay { replay, last } => {
                tick(replay, last);
                egui::Panel::top("cockpit_replay")
                    .show(ui, |ui| ui::replay_bar(ui, replay, "REPLAY"));
                let view = replay.model().view(replay.at_ns());
                ui::show(ui, &view, &mut self.ui);
                if replay.playing {
                    ui.ctx().request_repaint();
                }
            }
        }
    }
}
