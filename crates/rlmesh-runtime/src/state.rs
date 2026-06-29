//! Per-route mutable state: lane slots, episode lifecycle, and request building.

mod route;
#[cfg(test)]
mod tests;
mod types;

pub(crate) use route::{RequestPhase, RouteState};
pub(crate) use types::{EpisodeState, RouteSnapshot, SlotState, StartedEpisode};
