//! The session mode: arming, the e-stop latch, SAFE, robots stopped on
//! their own, and the reason recorded for each change.

use core::fmt;

use super::{Authority, AuthorityEvent};
use crate::prelude::*;
use crate::wire::{ChannelId, LastStop, Mode, ParticipantId, Reason, ReasonCode};

impl Authority {
    /// Latches ESTOP for `by`'s stop, which `from` sent (`verified` when
    /// they are one: the host authenticated the stop's sender itself).
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
            // Cleared but not yet disarmed: latched again.
            self.mode_reason = why;
        } else {
            self.set_mode(Mode::Estop, why, events);
        }
    }

    /// A reason for a change landing in this call's revision.
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

    /// Changes the mode and records why; an entry into SAFE, DISARMED or
    /// ESTOP moves `last_stop` (never lowered: epochs and revisions only
    /// grow).
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

    /// Ends every lease on `robot`'s parts (`code`, carrying who and why
    /// from `cause`).
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

    /// `robot` takes no lease until it is re-armed by name, unless robots
    /// re-arm on their own. A participant with nothing commandable (one that
    /// only publishes) is never held: there is nothing to re-arm.
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

    /// Whether `robot` has declared a commandable part.
    pub(super) fn commands_any(&self, robot: &ParticipantId) -> bool {
        self.parts
            .values()
            .any(|p| &p.owner == robot && p.command.is_some())
    }

    /// ACTIVE with no lease left drops back to ARMED.
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

    /// Takes an ACTIVE session to SAFE and revokes every lease: re-arming (a
    /// fresh epoch) is the way back. Only a stalled barrier or a scene losing
    /// a member does; a lost holder stops its own parts alone.
    pub(super) fn enter_safe(&mut self, reason: Reason, events: &mut Vec<AuthorityEvent>) {
        if self.mode == Mode::Active {
            self.revoke_all(ReasonCode::VoidedBySafe, &reason, events);
            self.set_mode(Mode::Safe, reason, events);
        }
    }
}
