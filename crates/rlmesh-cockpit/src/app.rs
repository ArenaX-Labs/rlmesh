//! The eframe app: each frame brings the model up to date (a live feed
//! drained, or a replay advanced), renders the model's view and sends the
//! verbs the operator asked for (live only).
//!
//! A live session can be paused: the cockpit replays its recent history
//! ([`History`]) while the feed keeps draining underneath, and only the
//! e-stop stays on. Keep and the episode controls work on the host's
//! recording, through this machine's recordings directory: Keep keeps the
//! last minutes as an incident, an episode is collected into
//! `collected/{dataset}/` when the operator marks it a success or a
//! failure.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use rlmesh_session::record::{self, Episode, Rolling};
use rlmesh_session::wire::SessionId;

use crate::feed::Feed;
use crate::model::{Model, ModelConfig};
use crate::replay::{History, Replay};
use crate::ui::{self, LiveAction, LiveBar, UiState};

/// How often the window repaints without new authority state.
pub const REFRESH: Duration = Duration::from_millis(100);

enum Source {
    Live(Box<Live>),
    Replay {
        replay: Box<Replay>,
        last: Instant,
        /// The recording's file or directory.
        path: PathBuf,
        /// Why Rerun did not open.
        rerun_failed: Option<String>,
    },
}

struct Live {
    feed: Feed,
    model: Model,
    history: History,
    /// Paused on the history.
    paused: Option<(Replay, Instant)>,
    /// The host's recordings root, when it is on this machine.
    recordings: Option<Rolling>,
    /// Where episodes are collected: `collected/{dataset}/`.
    dataset: String,
    bar: LiveBar,
    /// The running episode: its start, wall clock and session time.
    episode: Option<(SystemTime, i64)>,
    /// What the bar asked for, done next frame.
    action: Option<LiveAction>,
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
    /// the host records into, when it is on this machine (Keep and
    /// episodes, collected into `dataset`).
    pub fn new(feed: Feed, model: Model, recordings: Option<Rolling>, dataset: String) -> Self {
        Self {
            source: Source::Live(Box::new(Live {
                feed,
                model,
                history: History::new(History::WINDOW, History::MAX_BYTES),
                paused: None,
                recordings,
                dataset,
                bar: LiveBar::default(),
                episode: None,
                action: None,
            })),
            ui: UiState::default(),
        }
    }

    /// A window over the session recorded at `path`.
    pub fn replay(replay: Replay, path: PathBuf) -> Self {
        let mut ui = UiState::default();
        ui.replay = true;
        Self {
            source: Source::Replay {
                replay: Box::new(replay),
                last: Instant::now(),
                path,
                rerun_failed: None,
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
                let now_ns = live.feed.now_ns();
                // What the bar asked for last frame (it is drawn inside the
                // header, after the view is chosen).
                match live.action.take() {
                    Some(LiveAction::Pause) => {
                        let replay = live
                            .history
                            .replay(live.model.me().clone(), ModelConfig::default());
                        live.paused = Some((replay, Instant::now()));
                    }
                    Some(LiveAction::Resume) => live.paused = None,
                    Some(LiveAction::Keep) => {
                        live.bar.note = live.session().map(|(session, rolling)| {
                            record::keep(rolling, session, "keep")
                                .map(|dir| format!("kept in {}", dir.display()))
                                .map_err(|err| format!("not kept: {err}"))
                        });
                    }
                    Some(LiveAction::Start) => {
                        live.episode = Some((SystemTime::now(), live.model.session_ns(now_ns)));
                        live.bar.note = None;
                    }
                    Some(LiveAction::Finish { success }) => {
                        let started = live.episode.take();
                        live.bar.note = live.session().zip(started).map(
                            |((session, rolling), (wall, start_ns))| {
                                let episode = Episode {
                                    task: live.bar.task.clone(),
                                    start_ns,
                                    end_ns: live.model.session_ns(now_ns),
                                    success: Some(success),
                                };
                                record::collect(rolling, session, &live.dataset, &episode, wall)
                                    .map(|dir| format!("episode in {}", dir.display()))
                                    .map_err(|err| format!("episode not collected: {err}"))
                            },
                        );
                    }
                    Some(LiveAction::Discard) => {
                        live.episode = None;
                        live.bar.note = Some(Ok("episode discarded".into()));
                    }
                    None => {}
                }
                live.bar.paused = live.paused.is_some();
                live.bar.local = live.session().is_some();
                live.bar.episode_s = live
                    .episode
                    .map(|(_, start)| (live.model.session_ns(now_ns) - start) as f64 * 1e-9);
                self.ui.paused = live.paused.is_some();
                let view = match &mut live.paused {
                    Some((replay, last)) => {
                        tick(replay, last);
                        if replay.playing {
                            ui.ctx().request_repaint();
                        }
                        replay.model().view(replay.at_ns())
                    }
                    None => live.model.view(now_ns),
                };
                let bar = |ui: &mut egui::Ui| {
                    live.action = ui::live_bar(ui, &mut live.bar);
                    if let Some((replay, _)) = &mut live.paused {
                        ui::replay_bar(ui, replay, "LAST MINUTE", false);
                    }
                };
                for verb in ui::show(ui, &view, &mut self.ui, bar) {
                    live.feed.send(verb);
                }
                if live.action.is_some() {
                    ui.ctx().request_repaint();
                }
                ui.ctx().request_repaint_after(REFRESH);
            }
            Source::Replay {
                replay,
                last,
                path,
                rerun_failed,
            } => {
                tick(replay, last);
                let view = replay.model().view(replay.at_ns());
                let bar = |ui: &mut egui::Ui| {
                    if ui::replay_bar(ui, replay, "REPLAY", true) {
                        *rerun_failed = open_in_rerun(path).err();
                    }
                    if let Some(why) = rerun_failed {
                        ui.label(why.as_str());
                    }
                };
                ui::show(ui, &view, &mut self.ui, bar);
                // The bar acts after the view is drawn: show its effect.
                if replay.playing {
                    ui.ctx().request_repaint();
                } else {
                    ui.ctx().request_repaint_after(REFRESH);
                }
            }
        }
    }
}

/// Opens the MCAP files at `path` in the Rerun viewer, which decodes their
/// protobuf envelopes itself.
fn open_in_rerun(path: &Path) -> Result<(), String> {
    let files = record::mcap_files(path).map_err(|err| err.to_string())?;
    std::process::Command::new("rerun")
        .args(files)
        .spawn()
        .map(drop)
        .map_err(|err| format!("could not start rerun: {err}"))
}
