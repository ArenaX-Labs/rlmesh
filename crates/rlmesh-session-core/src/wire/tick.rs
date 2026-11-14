//! The virtual-mode tick protocol: what the host publishes on
//! `host/tick`. Embodiments answer with [`Command::Ready`], due controllers
//! with [`Command::Decided`].
//!
//! [`Command::Ready`]: crate::wire::authority::Command::Ready
//! [`Command::Decided`]: crate::wire::authority::Command::Decided

#[allow(unused_imports)]
use crate::prelude::*;

use crate::wire::action::ChunkRef;
use crate::wire::authority::Watermark;
use crate::wire::ids::{Epoch, ParticipantId};

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

    /// Where it falls in the host's order: (epoch, tick, phase), observe
    /// before step. A receiver drops a tick at or below the last it took.
    pub fn order(&self) -> (Epoch, u64, u8) {
        let phase = match self {
            Self::Observe(_) => 0,
            Self::Step(_) => 1,
        };
        (self.epoch(), self.tick(), phase)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_order_by_epoch_then_tick_with_observe_before_step() {
        let observe = |epoch, tick| {
            VirtualTick::Observe(TickState {
                epoch: Epoch(epoch),
                tick,
                watermarks: Vec::new(),
                due: Vec::new(),
            })
        };
        let step = |epoch, tick| {
            VirtualTick::Step(Step {
                epoch: Epoch(epoch),
                tick,
                chunks: Vec::new(),
            })
        };
        let order = [
            observe(1, 7),
            step(1, 7),
            observe(1, 8),
            step(1, 8),
            observe(2, 1),
        ]
        .map(|t| t.order());
        assert!(order.windows(2).all(|w| w[0] < w[1]), "{order:?}");
        assert_eq!(order[1], (Epoch(1), 7, 1));
    }
}
