//! The RLMesh cockpit (protocol v2, wave 6): a native single-session viewer.
//!
//! It joins a session as an **observer** (never a controller), so it adds
//! nothing to the hot loop, can attach late and works against a remote
//! robot. It shows four things:
//!
//! 1. **Live channels**: the robot path tree, each part's kind, latest value
//!    (numbers with a short plot, or a camera thumbnail), age and seq gaps.
//! 2. **Declared vs measured** (§8.10), each counter on its own clock:
//!    per stream the declared source rate, the produced and published rates
//!    (publisher reports), the admitted delivery rate (contracts), the
//!    unique received rate here and each subscriber's seq advance
//!    (heartbeats); per controller starts, completions, busy skips,
//!    inference pending, chunk age and valid coverage remaining; per edge
//!    the control rate, deadline misses and the edge mode (`_applied`).
//! 3. **The lease and mode timeline**: authority changes derived from
//!    consecutive snapshots (leases granted, preempted, rebound, expired,
//!    revoked; epochs; e-stop), edge mode transitions, and lanes of mode,
//!    lease holder and edge mode over the last minute; and the binding
//!    table of every controller (§6.8).
//! 4. **Arm and e-stop** through the host's verbs: e-stop in one click, arm
//!    behind a confirmation, disarm. A latched e-stop is cleared at the
//!    robot (§11.1), which the cockpit says rather than offers.
//!
//! The layers: [`feed`] (the network side: an observer participant and its
//! subscriptions) produces [`model::Input`]s; [`model::Model`] is sans-IO
//! state with an injected `now_ns`; [`view`] derives plain data from it;
//! [`ui`] renders a [`view::View`] with egui and returns
//! [`model::Verb`]s. Tests drive the model and render the UI headless.
//!
//! [`replay`] feeds a recorded session (MCAP) through the same model, with
//! a play, pause, scrub and step bar instead of the verbs.
//!
//! The counters only the measuring side knows (a publisher's produced
//! count, a controller's starts and skips) come from observer reports
//! (`rlmesh_session::report`); everything else is measured here.

pub mod app;
pub mod feed;
pub mod meter;
pub mod model;
pub mod replay;
pub mod timeline;
pub mod ui;
pub mod values;
pub mod view;

pub use app::CockpitApp;
pub use feed::Feed;
pub use model::{Input, Model, ModelConfig, Verb};
pub use replay::Replay;
pub use ui::{Tab, UiState};
pub use view::View;
