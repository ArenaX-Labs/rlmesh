//! The control verbs: [`Authority::apply_from`] checks and applies each one.

use super::{Applied, Authority, AuthorityEvent, DetachReason, Grant, Identity, Seen, Verb};
use crate::prelude::*;
use crate::wire::version::{PROTOCOL_MINOR, missing_features};
use crate::wire::{
    BindingRecord, Command, CtlError, CtlStatus, LeaseToken, Mode, ParticipantId, ParticipantInfo,
    PriorityClass, Reason, ReasonCode, Request, Role,
};

impl Authority {
    /// Applies a verb as device `local`, generation 0: sans-IO tests and simulations.
    pub fn apply(&mut self, request: Request, now_ns: i64) -> Result<Applied, CtlError> {
        let local = Identity {
            device: crate::auth::LOCAL_DEVICE.into(),
            generation: 0,
        };
        self.apply_from(request, &local, now_ns)
    }

    /// Applies a verb from `sender`.
    pub fn apply_from(
        &mut self,
        request: Request,
        sender: &Identity,
        now_ns: i64,
    ) -> Result<Applied, CtlError> {
        self.now_ns = now_ns;
        let mut applied = Applied::default();
        self.scene_refuses(&request.command)?;
        let from = request.from;
        match request.command {
            Command::Attach {
                name,
                roles,
                priority,
                protocol_minor,
                supports,
                requires,
                workflow_edition,
                workflow_editions,
                estop,
            } => self.on_attach(
                from,
                name,
                roles,
                priority,
                protocol_minor,
                supports,
                requires,
                workflow_edition,
                workflow_editions,
                estop,
                sender,
                now_ns,
                &mut applied,
            )?,
            Command::Detach => {
                self.require_attached(&from)?;
                self.detach(
                    &from,
                    ReasonCode::HolderDetached,
                    DetachReason::Requested,
                    &mut applied.events,
                );
            }
            Command::Arm => self.on_arm(from, sender, now_ns, &mut applied)?,
            Command::Rearm { robot } => self.on_rearm(from, robot, sender, &mut applied)?,
            Command::Disarm => self.on_disarm(from, sender, &mut applied)?,
            Command::DeclareParts {
                spec_version,
                robot_json,
                seal,
                board_id: _,
            } => {
                self.declare_parts(&from, spec_version, &robot_json, &seal, &mut applied.events)?
            }
            Command::DeclareBinding {
                fingerprint,
                revision,
                table_json,
                channels,
                replan_hz,
                contracts,
                grids,
            } => self.declare_binding(
                BindingRecord {
                    participant: from,
                    fingerprint,
                    revision,
                    table_json,
                    channels,
                    replan_hz,
                    contracts,
                    grids,
                },
                &mut applied.events,
            )?,
            Command::DemandAck { channel, revision } => {
                self.demand_ack(&from, channel, revision, &mut applied.events)?;
            }
            Command::GrantLease {
                channels,
                release,
                binding,
                ttl_ns,
            } => {
                applied.leases = self.grant(
                    &from,
                    &sender.device,
                    request.epoch,
                    &Grant {
                        channels,
                        release,
                        binding,
                        ttl_ns,
                    },
                    now_ns,
                    &mut applied.events,
                )?;
            }
            Command::ReleaseLease {
                channels,
                token,
                to,
            } => {
                self.release(&from, &channels, token, to.as_ref(), &mut applied.events)?;
            }
            Command::Estop {
                reason,
                code,
                origin,
                at_ns: _,
            } => self.on_estop(from, reason, code, origin, &mut applied)?,
            Command::Clear => self.on_clear(from, sender, &mut applied)?,
            Command::EnterSafe { reason, code } => {
                self.on_enter_safe(from, reason, code, &mut applied)?
            }
            Command::Ready { tick, watermarks } => {
                self.require_tick_verb(&from, request.epoch)?;
                let who = self.barrier_participants();
                if let Some(clock) = &mut self.clock {
                    let ticks = clock.on_ready(&from, tick, &watermarks, &who, now_ns)?;
                    self.ticks.extend(ticks);
                }
            }
            Command::Decided { tick, chunks } => {
                self.require_tick_verb(&from, request.epoch)?;
                let who = self.barrier_participants();
                if let Some(clock) = &mut self.clock {
                    let ticks = clock.on_decided(&from, tick, &chunks, &who, now_ns)?;
                    self.ticks.extend(ticks);
                }
            }
            Command::KeysAck(_) | Command::GetKeys => {
                self.require_attached(&from)?;
            }
            Command::Keep { .. } => {
                self.require_attached(&from)?;
                self.config.access.check_verb(&sender.device, Verb::Arm)?;
            }
            Command::LobbyWait(_)
            | Command::RunScene { .. }
            | Command::SceneStatus { .. }
            | Command::StopScene { .. }
            | Command::HostStarted(_)
            | Command::ArmScene { .. } => {
                return Err(CtlError::new(
                    CtlStatus::Unsupported,
                    "a lobby verb: this session runs no lobby",
                ));
            }
        }
        if !applied.events.is_empty() {
            self.revision += 1;
        }
        Ok(applied)
    }

    /// [`Self::apply_from`], minting `token` (from [`Self::reserve_token`]) if the verb mints
    /// one.
    pub fn apply_minting(
        &mut self,
        request: Request,
        sender: &Identity,
        now_ns: i64,
        token: LeaseToken,
    ) -> Result<Applied, CtlError> {
        self.minting = Some(token);
        let applied = self.apply_from(request, sender, now_ns);
        self.minting = None;
        applied
    }

    #[allow(clippy::too_many_arguments)]
    fn on_attach(
        &mut self,
        from: ParticipantId,
        name: String,
        roles: Vec<Role>,
        priority: PriorityClass,
        protocol_minor: u32,
        supports: Vec<String>,
        requires: Vec<String>,
        workflow_edition: String,
        workflow_editions: Vec<String>,
        estop: Option<Reason>,
        sender: &Identity,
        now_ns: i64,
        applied: &mut Applied,
    ) -> Result<(), CtlError> {
        if roles.is_empty() {
            return Err(CtlError::new(
                CtlStatus::InvalidRequest,
                "attach declares no role",
            ));
        }
        let missing = missing_features(&requires, &self.config.supports);
        if !missing.is_empty() {
            return Err(CtlError::new(
                CtlStatus::Incompatible,
                format!("this session (protocol minor {PROTOCOL_MINOR}) lacks {missing:?}"),
            ));
        }
        self.config
            .access
            .check_attach(&sender.device, &roles, priority)?;
        self.scene_admit(&from, sender, &roles, priority)?;
        if self
            .roster
            .get(&from)
            .is_some_and(|p| p.generation != sender.generation)
        {
            if self.scene_member(&from) {
                let before = applied.events.len();
                self.lose(&from, &mut applied.events);
                let mut at = 0;
                applied.events.retain(|event| {
                    at += 1;
                    at <= before
                        || !matches!(event, AuthorityEvent::Detached { participant, .. } if participant == &from)
                });
            }
            self.replace_generation(&from, &mut applied.events);
        }
        let left = self
            .roster
            .get(&from)
            .and_then(|p| p.left.clone())
            .filter(|_| self.scene_rejoining(&from));
        if left.is_none() {
            self.rejoined(&from);
        }
        self.roster.insert(
            from.clone(),
            ParticipantInfo {
                id: from.clone(),
                name,
                roles: roles.clone(),
                priority,
                attached_ns: now_ns,
                protocol_minor,
                supports,
                requires,
                workflow_edition,
                workflow_editions,
                present: true,
                left,
                device: sender.device.clone(),
                generation: sender.generation,
            },
        );
        self.seen.insert(
            from.clone(),
            Seen {
                last_ns: now_ns,
                hint_ns: None,
            },
        );
        if let Some(latch) = estop.filter(|_| roles.contains(&Role::Embodiment)) {
            let code = match latch.code {
                ReasonCode::EstopSignal => ReasonCode::EstopSignal,
                _ => ReasonCode::EstopLocal,
            };
            let by = ParticipantId::new(&latch.by).unwrap_or_else(|_| from.clone());
            self.latch_estop(code, &by, &from, latch.detail, &mut applied.events);
        }
        applied.events.push(AuthorityEvent::Attached {
            participant: from,
            roles,
            priority,
        });
        self.recompute_demands(&mut applied.events);
        Ok(())
    }

    fn on_arm(
        &mut self,
        from: ParticipantId,
        sender: &Identity,
        now_ns: i64,
        applied: &mut Applied,
    ) -> Result<(), CtlError> {
        self.require_attached(&from)?;
        self.config.access.check_verb(&sender.device, Verb::Arm)?;
        self.scene_arm_ready()?;
        match self.mode {
            Mode::Disarmed | Mode::Safe => {
                self.epoch = self.epoch.next();
                if let Some(clock) = &mut self.clock {
                    clock.reset(self.epoch, now_ns);
                }
                let why = self.reason(ReasonCode::Armed, &from, "");
                self.set_mode(Mode::Armed, why, &mut applied.events);
            }
            Mode::Estop if self.estop_latched => {
                return Err(CtlError::new(
                    CtlStatus::EstopLatched,
                    "e-stop is latched: clear it locally, then disarm and arm",
                ));
            }
            Mode::Estop => {
                return Err(CtlError::new(
                    CtlStatus::InvalidTransition,
                    "leaving ESTOP needs a disarm before arm",
                ));
            }
            Mode::Armed | Mode::Active => {}
        }
        Ok(())
    }

    fn on_rearm(
        &mut self,
        from: ParticipantId,
        robot: ParticipantId,
        sender: &Identity,
        applied: &mut Applied,
    ) -> Result<(), CtlError> {
        self.require_attached(&from)?;
        self.config.access.check_verb(&sender.device, Verb::Arm)?;
        self.scene_arm_ready()?;
        match self.mode {
            Mode::Armed | Mode::Active => {}
            Mode::Estop if self.estop_latched => {
                return Err(CtlError::new(
                    CtlStatus::EstopLatched,
                    format!("e-stop is latched: clear it, disarm and arm before re-arming {robot}"),
                ));
            }
            mode => {
                return Err(CtlError::new(
                    CtlStatus::InvalidTransition,
                    format!("session is {mode}; arm it first, then re-arm {robot}"),
                ));
            }
        }
        let Some(info) = self
            .roster
            .get(&robot)
            .filter(|p| p.has_role(Role::Embodiment))
        else {
            return Err(CtlError::new(
                CtlStatus::InvalidRequest,
                format!("{robot} is not a robot in this session"),
            ));
        };
        if !self.held.contains_key(&robot) {
            return Ok(());
        }
        if !info.present {
            return Err(CtlError::new(
                CtlStatus::NotReady,
                format!("{robot} is not present: re-arm it once it is back"),
            ));
        }
        self.held.remove(&robot);
        applied
            .events
            .push(AuthorityEvent::RobotRearmed { robot, by: from });
        Ok(())
    }

    fn on_disarm(
        &mut self,
        from: ParticipantId,
        sender: &Identity,
        applied: &mut Applied,
    ) -> Result<(), CtlError> {
        self.require_attached(&from)?;
        self.config
            .access
            .check_verb(&sender.device, Verb::Disarm)?;
        match self.mode {
            Mode::Disarmed => {}
            Mode::Estop if self.estop_latched => {
                return Err(CtlError::new(
                    CtlStatus::EstopLatched,
                    "e-stop is latched: clear it locally first",
                ));
            }
            _ => {
                let why = self.reason(ReasonCode::Disarmed, &from, "");
                self.revoke_all(ReasonCode::VoidedByDisarm, &why, &mut applied.events);
                self.set_mode(Mode::Disarmed, why, &mut applied.events);
            }
        }
        Ok(())
    }

    fn on_clear(
        &mut self,
        from: ParticipantId,
        sender: &Identity,
        applied: &mut Applied,
    ) -> Result<(), CtlError> {
        let info = self.require_attached(&from)?;
        if !info.has_role(Role::Embodiment) {
            return Err(CtlError::new(
                CtlStatus::Forbidden,
                "only an embodiment can clear an e-stop",
            ));
        }
        if !(self.mode == Mode::Estop && self.estop_latched) {
            return Err(CtlError::new(
                CtlStatus::InvalidTransition,
                "no latched e-stop to clear",
            ));
        }
        if sender.device != crate::auth::LOCAL_DEVICE && self.estop_by.as_ref() != Some(&from) {
            let by = self
                .estop_by
                .as_ref()
                .map_or_else(|| "someone else".into(), ToString::to_string);
            return Err(CtlError::new(
                CtlStatus::Forbidden,
                format!(
                    "{by} latched this e-stop: only {by}, or a robot process on the \
                             host's own machine, clears it"
                ),
            ));
        }
        self.estop_latched = false;
        self.estop_by = None;
        self.mode_reason = self.reason(ReasonCode::EstopCleared, &from, "");
        applied
            .events
            .push(AuthorityEvent::EstopCleared { by: from });
        Ok(())
    }

    fn on_estop(
        &mut self,
        from: ParticipantId,
        reason: String,
        code: ReasonCode,
        origin: Option<ParticipantId>,
        applied: &mut Applied,
    ) -> Result<(), CtlError> {
        let code = match code {
            ReasonCode::EstopLocal | ReasonCode::EstopSignal => code,
            _ => ReasonCode::EstopRemote,
        };
        let by = origin
            .filter(|_| code == ReasonCode::EstopSignal)
            .unwrap_or_else(|| from.clone());
        self.latch_estop(code, &by, &from, reason, &mut applied.events);
        Ok(())
    }

    fn on_enter_safe(
        &mut self,
        from: ParticipantId,
        reason: String,
        code: ReasonCode,
        applied: &mut Applied,
    ) -> Result<(), CtlError> {
        let info = self.require_attached(&from)?;
        if !info.has_role(Role::Embodiment) {
            return Err(CtlError::new(
                CtlStatus::Forbidden,
                "only an embodiment's edge reports a fallback",
            ));
        }
        let why = self.reason(code, &from, &reason);
        match self.mode {
            Mode::Active | Mode::Armed | Mode::Disarmed => {
                self.stop_robot(&from, ReasonCode::VoidedBySafe, &why, &mut applied.events);
                self.hold(&from, why, &mut applied.events);
                self.settle_after_release(&from, "its robot went safe", &mut applied.events);
            }
            Mode::Safe | Mode::Estop => {
                self.mode_reason = why.clone();
                applied.events.push(AuthorityEvent::ModeReason {
                    mode: self.mode,
                    reason: why,
                });
            }
        }
        Ok(())
    }
}
