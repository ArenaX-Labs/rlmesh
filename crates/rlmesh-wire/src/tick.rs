//! The virtual-mode tick protocol (§11.3): what the host publishes on
//! `host/tick`. Embodiments answer with [`Command::Ready`], due controllers
//! with [`Command::Decided`].
//!
//! [`Command::Ready`]: crate::authority::Command::Ready
//! [`Command::Decided`]: crate::authority::Command::Decided

use crate::action::ChunkRef;
use crate::authority::Watermark;
use crate::ids::{Epoch, ParticipantId};

/// Phase 2, observe: every embodiment published tick `tick`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TickState {
    /// The epoch.
    pub epoch: Epoch,
    /// The tick.
    pub tick: u64,
    /// The last seq per stream, from every embodiment's `Ready`.
    pub watermarks: Vec<Watermark>,
    /// The controllers whose policy period falls on this tick.
    pub due: Vec<ParticipantId>,
}

/// Phase 4, step: the chunks scheduled for delivery at `tick`. Every
/// decision scheduled at or before it has arrived.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step {
    /// The epoch.
    pub epoch: Epoch,
    /// The tick.
    pub tick: u64,
    /// The chunks the embodiments' edges must judge before applying.
    pub chunks: Vec<ChunkRef>,
}

/// One message on `host/tick`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VirtualTick {
    /// Observe: due controllers snapshot and decide.
    Observe(TickState),
    /// Step: embodiments apply and advance physics.
    Step(Step),
}

impl VirtualTick {
    /// The epoch.
    pub fn epoch(&self) -> Epoch {
        match self {
            Self::Observe(state) => state.epoch,
            Self::Step(step) => step.epoch,
        }
    }

    /// The tick.
    pub fn tick(&self) -> u64 {
        match self {
            Self::Observe(state) => state.tick,
            Self::Step(step) => step.tick,
        }
    }
}
