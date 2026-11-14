//! The virtual clock's tick barrier: who it waits for, and the tick verbs.

use super::Authority;
use crate::prelude::*;
use crate::virtual_clock::Participants;
use crate::wire::{CtlError, CtlStatus, Epoch, ParticipantId};

impl Authority {
    pub(super) fn require_tick_verb(
        &self,
        from: &ParticipantId,
        epoch: Option<Epoch>,
    ) -> Result<(), CtlError> {
        self.require_attached(from)?;
        if self.clock.is_none() {
            return Err(CtlError::new(
                CtlStatus::Unsupported,
                "tick verbs need a virtual-clock session",
            ));
        }
        if epoch != Some(self.epoch) {
            return Err(CtlError::new(
                CtlStatus::StaleEpoch,
                format!(
                    "tick verb for epoch {}, current {}",
                    epoch.unwrap_or_default(),
                    self.epoch
                ),
            ));
        }
        Ok(())
    }

    /// Who the virtual barrier waits for.
    pub(super) fn barrier_participants(&self) -> Participants {
        let tick_ns = self.config.virtual_clock.map_or(1, |c| c.tick_ns.max(1));
        Participants {
            embodiments: self
                .parts
                .values()
                .map(|p| &p.owner)
                .filter(|owner| self.is_attached(owner))
                .cloned()
                .collect(),
            controllers: self
                .bindings
                .values()
                .filter(|b| self.is_attached(&b.participant) && b.replan_hz > 0.0)
                .map(|b| {
                    let period = (1e9 / (b.replan_hz * tick_ns as f64)).round().max(1.0);
                    (b.participant.clone(), period as u64)
                })
                .collect(),
        }
    }

    pub(super) fn barrier_gone(&mut self, id: &ParticipantId, now_ns: i64) {
        let who = self.barrier_participants();
        if let Some(clock) = &mut self.clock {
            let ticks = clock.on_gone(id, &who, now_ns);
            self.ticks.extend(ticks);
        }
    }
}
