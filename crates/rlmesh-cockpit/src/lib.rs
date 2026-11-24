//! The RLMesh cockpit: a native single-session viewer.

pub mod app;
pub mod cli;
pub mod feed;
pub mod meter;
pub mod model;
pub mod replay;
pub mod timeline;
pub mod ui;
pub mod values;
pub mod view;

pub use app::CockpitApp;
pub use cli::CockpitArgs;
pub use feed::Feed;
pub use model::{Input, Model, ModelConfig, Verb};
pub use replay::Replay;
pub use ui::{Tab, UiState};
pub use view::View;
