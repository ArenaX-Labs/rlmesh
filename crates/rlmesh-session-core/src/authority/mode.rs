//! The session mode: arming, the e-stop latch, SAFE, robots stopped on their own, and the
//! reason recorded for each change.

use core::fmt;

use super::{Authority, AuthorityEvent};
use crate::prelude::*;
use crate::wire::{ChannelId, LastStop, Mode, ParticipantId, Reason, ReasonCode};

impl Authority {
    pub(super) fn latch_estop(
        &mut self,
        code: ReasonCode,
        by: &ParticipantId,
        from: &ParticipantId,
        reason: String,
        events: &mut Vec<AuthorityEvent>,
    ) {
        if self.mode == Mode::Estop && self.estop_latched {
            return;
        }
        let why = Reason {
            verified: by == from,
            ..self.reason(code, by, &reason)
        };
        self.revoke_all(ReasonCode::VoidedByEstop, &why, events);
        self.estop_latched = true;
        self.estop_by = Some(by.clone());
        events.push(AuthorityEvent::EstopLatched {
            by: by.clone(),
            reason,
        });
        if self.mode == Mode::Estop {
            self.mode_reason = why;
        } else {
            self.set_mode(Mode::Estop, why, events);
        }
    }

    pub(super) fn reason(&self, code: ReasonCode, by: impl fmt::Display, detail: &str) -> Reason {
        Reason {
            code,
            by: by.to_string(),
            detail: detail.to_string(),
            at_ns: self.now_ns,
            revision: self.revision + 1,
            verified: false,
        }
    }

    pub(super) fn set_mode(&mut self, to: Mode, reason: Reason, events: &mut Vec<AuthorityEvent>) {
        if self.mode == to {
            return;
        }
        if matches!(to, Mode::Safe | Mode::Disarmed | Mode::Estop) {
            self.last_stop = LastStop {
                epoch: self.epoch,
                revision: self.revision + 1,
            };
        }
        events.push(AuthorityEvent::ModeChanged {
            from: self.mode,
            to,
            epoch: self.epoch,
            reason: reason.clone(),
        });
        self.mode = to;
        self.mode_reason = reason;
    }

    pub(super) fn stop_robot(
        &mut self,
        robot: &ParticipantId,
        code: ReasonCode,
        cause: &Reason,
        events: &mut Vec<AuthorityEvent>,
    ) {
        let channels: Vec<ChannelId> = self
            .parts
            .values()
            .filter(|p| &p.owner == robot)
            .map(|p| p.channel)
            .collect();
        for channel in channels {
            if let Some(lease) = self.leases.remove(&channel) {
                let why = Reason {
                    code,
                    ..cause.clone()
                };
                self.end(lease, why, events);
            }
        }
    }

    pub(super) fn hold(
        &mut self,
        robot: &ParticipantId,
        why: Reason,
        events: &mut Vec<AuthorityEvent>,
    ) {
        if !self.config.rearm_robots && self.commands_any(robot) {
            self.held.insert(robot.clone(), why.clone());
            events.push(AuthorityEvent::RobotHeld {
                robot: robot.clone(),
                reason: why,
            });
        }
    }

    pub(super) fn commands_any(&self, robot: &ParticipantId) -> bool {
        self.parts
            .values()
            .any(|p| &p.owner == robot && p.command.is_some())
    }

    pub(super) fn settle_after_release(
        &mut self,
        by: &ParticipantId,
        detail: &str,
        events: &mut Vec<AuthorityEvent>,
    ) {
        if self.mode == Mode::Active && self.leases.is_empty() {
            let why = self.reason(ReasonCode::LastRelease, by, detail);
            self.set_mode(Mode::Armed, why, events);
        }
    }

    pub(super) fn enter_safe(&mut self, reason: Reason, events: &mut Vec<AuthorityEvent>) {
        if self.mode == Mode::Active {
            self.revoke_all(ReasonCode::VoidedBySafe, &reason, events);
            self.set_mode(Mode::Safe, reason, events);
        }
    }
}
