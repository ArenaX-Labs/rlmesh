//! `ctl` verbs, and what the participant's heartbeats carry.

use std::sync::atomic::Ordering;
use std::time::Duration;

use crate::spec::{BindError, Binding, Policy, RunOptions, Target};
use crate::wire::version::SPEC_VERSION;
use crate::wire::{
    AuthoritySnapshot, ChannelId, Command, CtlReply, EdgeMode, EstopSignal, Fingerprint, Health,
    Heartbeat, HeartbeatEcho, Lease, LeaseToken, PartSpec, ParticipantId, ReasonCode,
};
use tokio::sync::mpsc;
use tracing::warn;

use crate::error::SessionError;
use crate::participant::tasks::health_code;
use crate::participant::{Participant, lock};
use crate::report::RunSource;
use crate::sample::{SampleSubscriber, SubscribeOptions};
use crate::transport::{CONTROL_CONGESTION, CONTROL_PRIORITY};

impl Participant {
    /// Relays another clock's heartbeats (a gateway's browser): this
    /// participant's own heartbeats stop, [`Self::send_heartbeat`] seals and
    /// sends each relayed one, and the host's echoes of them, opened, go to
    /// `echoes`.
    pub fn relay_heartbeats(&self, echoes: mpsc::UnboundedSender<HeartbeatEcho>) {
        *lock(&self.shared.relay) = Some(echoes);
    }

    /// Seals and sends one heartbeat under this participant's keys (a
    /// relayed one: see [`Self::relay_heartbeats`]).
    pub async fn send_heartbeat(&self, heartbeat: &Heartbeat) -> Result<(), SessionError> {
        self.shared.send_heartbeat(heartbeat.clone(), false).await
    }

    /// DISARMED or SAFE to ARMED, in a fresh epoch.
    pub async fn arm(&self) -> Result<AuthoritySnapshot, SessionError> {
        Ok(self.shared.ctl(Command::Arm).await?.0)
    }

    /// Back to DISARMED.
    pub async fn disarm(&self) -> Result<AuthoritySnapshot, SessionError> {
        Ok(self.shared.ctl(Command::Disarm).await?.0)
    }

    /// Emergency stop: publishes this member's e-stop signal,
    /// which every robot edge latches directly (`ESTOP_SIGNAL`), and tells
    /// the host (`Estop`) in the background. Returns once the signal is
    /// out; a missing host reply is only a warning. The confirmation is
    /// each robot's `_applied` mode: [`Self::wait_for_estop`].
    ///
    /// # Errors
    /// [`SessionError::Key`] once this process is no longer attached (its
    /// key ended): only the robot's own stop is left.
    pub async fn estop(&self, reason: impl Into<String>) -> Result<(), SessionError> {
        let shared = &self.shared;
        let reason = reason.into();
        let key = shared.keys.estop(&shared.id);
        let signal = EstopSignal {
            reason: reason.clone(),
            sent_ns: shared.clock.now_ns(),
        };
        let link = shared.link.get().ok_or(SessionError::Closed)?;
        let sealed = lock(link).estop_signal(&key, &signal).map_err(|err| {
            SessionError::Key(format!(
                "this process is no longer attached ({err}): use the robot's stop"
            ))
        })?;
        shared
            .transport
            .control()
            .put(&key, sealed)
            .priority(CONTROL_PRIORITY)
            .congestion_control(CONTROL_CONGESTION)
            .express(true)
            .await?;
        let shared = shared.clone();
        tokio::spawn(async move {
            let estop = Command::Estop {
                reason,
                code: ReasonCode::EstopRemote,
                origin: None,
                at_ns: 0,
            };
            if let Err(err) = shared.ctl(estop).await {
                warn!(participant = %shared.id, %err, "the host did not answer the e-stop; the signal stands");
            }
        });
        Ok(())
    }

    /// Waits until every robot's edge applies ESTOP (its first commandable
    /// part's `_applied` mode): what [`Self::estop`] confirms by.
    ///
    /// # Errors
    /// [`SessionError::Timeout`]; a subscription's error.
    pub async fn wait_for_estop(&self, timeout: Duration) -> Result<(), SessionError> {
        let view = self.authority().ok_or(SessionError::Closed)?;
        let mut robots: Vec<&ParticipantId> = Vec::new();
        let mut applied = Vec::new();
        for part in view.parts.iter().filter(|p| p.command.is_some()) {
            if !robots.contains(&&part.owner) {
                robots.push(&part.owner);
                let options = SubscribeOptions::default();
                let path = part.path.as_str();
                applied.push(self.subscribe_applied(&part.owner, path, options).await?);
            }
        }
        let stopped = |s: &SampleSubscriber| {
            s.latest()
                .and_then(|f| f.received.sample.applied)
                .is_some_and(|a| a.mode == EdgeMode::Estop)
        };
        let started = tokio::time::Instant::now();
        while !applied.iter().all(stopped) {
            if started.elapsed() >= timeout {
                return Err(SessionError::Timeout(timeout, "every robot's ESTOP".into()));
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        Ok(())
    }

    /// Releases a latched e-stop: an embodiment's local clear.
    pub async fn clear(&self) -> Result<AuthoritySnapshot, SessionError> {
        Ok(self.shared.ctl(Command::Clear).await?.0)
    }

    /// Asks the host to keep its recent recording of this session as an
    /// incident named `label`, out of the rolling purge's reach (the
    /// cockpit's Keep). Whoever may arm may keep, at most every 5 s; a host
    /// that records nothing refuses it (`UNSUPPORTED`).
    pub async fn keep(&self, label: &str) -> Result<(), SessionError> {
        self.shared
            .ctl(Command::Keep {
                label: label.to_string(),
            })
            .await
            .map(|_| ())
    }

    /// Declares this embodiment's robot map (the authored JSON): the
    /// host normalises it and assigns channel ids. Re-declaring identically
    /// is a no-op; the parts are part of the authority snapshot.
    ///
    /// This runtime can seal every part, so the session's choice of level
    /// stands. Returns once the host's keys for every part reached this
    /// process (or after `ctl_timeout`, with a warning): samples published
    /// before then are dropped unsent.
    pub async fn declare_parts(&self, robot_json: &str) -> Result<AuthoritySnapshot, SessionError> {
        let seal = crate::spec::normalise(&self.shared.id, SPEC_VERSION, robot_json)
            .map(|parts| parts.iter().map(|p| p.path.to_string()).collect())
            .unwrap_or_default();
        let snapshot = self
            .shared
            .ctl(Command::DeclareParts {
                spec_version: SPEC_VERSION,
                robot_json: robot_json.to_string(),
                seal,
                board_id: None,
            })
            .await?
            .0;
        let channels: Vec<ChannelId> = snapshot
            .parts
            .iter()
            .filter(|p| p.owner == self.shared.id)
            .map(|p| p.channel)
            .collect();
        let Some(link) = self.shared.link.get() else {
            return Ok(snapshot);
        };
        let ready = || {
            let link = lock(link);
            channels.iter().all(|c| link.keyring().sends_stream(*c))
        };
        let mut changed = self.shared.keys_changed.subscribe();
        let waited = tokio::time::timeout(self.shared.ctl_timeout, async {
            while !ready() {
                if changed.changed().await.is_err() {
                    break;
                }
            }
        })
        .await;
        if waited.is_err() {
            warn!(participant = %self.shared.id, "the host's stream keys have not arrived yet");
        }
        Ok(snapshot)
    }

    /// Declares an identity binding for this controller: the parts it will
    /// command, a table and a fingerprint, without delivery contracts or
    /// action grids (a controller that commands paths directly). A new
    /// fingerprint re-issues this controller's leases under a fresh token.
    /// [`Self::declare_bound`] declares what the binder produced.
    pub async fn declare_binding(
        &self,
        fingerprint: Fingerprint,
        table_json: &str,
        channels: Vec<ChannelId>,
        replan_hz: f64,
    ) -> Result<AuthoritySnapshot, SessionError> {
        let revision = self.shared.binding_revision.fetch_add(1, Ordering::Relaxed) + 1;
        Ok(self
            .shared
            .ctl(Command::DeclareBinding {
                fingerprint,
                revision,
                table_json: table_json.to_string(),
                channels,
                replan_hz,
                contracts: Vec::new(),
                grids: Vec::new(),
            })
            .await?
            .0)
    }

    /// Binds `policy` against every part in this participant's view,
    /// or its scene's view when it carries places and `options` names no
    /// robot: the binding table, or the refusal listing
    /// every problem.
    pub fn bind(&self, policy: &Policy, options: &RunOptions) -> Result<Binding, BindError> {
        let view = self.shared.view.borrow();
        match view.as_ref() {
            Some(state) => {
                // A policy's own parts (its annotations) are never bound to
                // it: they neither make a session several robots nor stand
                // in for the one it drives.
                let me = &self.shared.id;
                if state.parts.iter().any(|p| &p.owner == me) {
                    let others: Vec<PartSpec> = state
                        .parts
                        .iter()
                        .filter(|p| &p.owner != me)
                        .cloned()
                        .collect();
                    let target = Target {
                        parts: &others,
                        places: &state.places,
                    };
                    crate::spec::bind_in(policy, &target, options)
                } else {
                    crate::spec::bind_in(policy, &Target::of(state), options)
                }
            }
            None => crate::spec::bind(policy, &[], options),
        }
    }

    /// Declares a binding the binder produced: its fingerprint, its table,
    /// the parts it commands, its delivery contracts and its action grids. A
    /// lease under it is granted once every stream it reads has installed a
    /// publication plan that includes its contract (`NOT_READY` until
    /// then).
    pub async fn declare_bound(
        &self,
        binding: &Binding,
    ) -> Result<AuthoritySnapshot, SessionError> {
        let revision = self.shared.binding_revision.fetch_add(1, Ordering::Relaxed) + 1;
        Ok(self
            .shared
            .ctl(Command::DeclareBinding {
                fingerprint: binding.fingerprint,
                revision,
                table_json: binding.table_text(),
                channels: binding.channels(),
                replan_hz: binding.replan_hz,
                contracts: binding.contracts.clone(),
                grids: binding.grids(),
            })
            .await?
            .0)
    }

    /// Requests (or renews) leases on commandable parts, all or none, under
    /// this controller's current binding, releasing `release` in the same
    /// revision (a switch within a command group). `ttl` defaults to the
    /// session's lease grace.
    pub async fn grant_leases(
        &self,
        channels: Vec<ChannelId>,
        release: Vec<ChannelId>,
        ttl: Option<Duration>,
    ) -> Result<Vec<Lease>, SessionError> {
        let binding = self.binding_fingerprint();
        let ttl_ns = ttl.map(|ttl| u64::try_from(ttl.as_nanos()).unwrap_or(u64::MAX));
        let (_, leases) = self
            .shared
            .ctl(Command::GrantLease {
                channels,
                release,
                binding,
                ttl_ns,
            })
            .await?;
        Ok(leases)
    }

    /// The fingerprint of this participant's declared binding (zero: none).
    pub fn binding_fingerprint(&self) -> Fingerprint {
        self.authority()
            .and_then(|view| view.binding(&self.shared.id).map(|b| b.fingerprint))
            .unwrap_or_default()
    }

    /// Sends one verb and returns the host's reply, a refusal included (as
    /// its `result`); the reply's snapshot updates the view either way.
    pub async fn command(&self, command: Command) -> Result<CtlReply, SessionError> {
        self.shared.ctl_reply(command).await
    }

    /// [`Self::grant_leases`] for one part.
    pub async fn grant_lease(
        &self,
        channel: ChannelId,
        ttl: Option<Duration>,
    ) -> Result<Lease, SessionError> {
        self.grant_leases(vec![channel], Vec::new(), ttl)
            .await?
            .pop()
            .ok_or_else(|| {
                SessionError::Wire(crate::wire::WireError::Malformed {
                    message: "ctl reply",
                    reason: "a granted lease is missing".into(),
                })
            })
    }

    /// Gives a lease back.
    pub async fn release_lease(&self, lease: &Lease) -> Result<AuthoritySnapshot, SessionError> {
        self.release_leases(vec![lease.channel], lease.token).await
    }

    /// Gives leases held under one token back.
    pub async fn release_leases(
        &self,
        channels: Vec<ChannelId>,
        token: LeaseToken,
    ) -> Result<AuthoritySnapshot, SessionError> {
        Ok(self
            .shared
            .ctl(Command::ReleaseLease { channels, token })
            .await?
            .0)
    }

    /// Reports a bound controller's run counters to observers ([`crate::report`]); the
    /// [`Controller`](crate::Controller) installs
    /// its own. `None` stops them.
    pub fn set_run_source(&self, source: Option<RunSource>) {
        if let Ok(mut run) = self.shared.run.lock() {
            *run = source;
        }
    }

    /// The health this participant reports in its heartbeats.
    pub fn set_health(&self, health: Health) {
        self.shared
            .health
            .store(health_code(health), Ordering::Relaxed);
    }

    /// Fault injection for tests and demos: stop (or resume) heartbeats while
    /// staying present, as a hung controller would.
    pub fn set_heartbeats_enabled(&self, enabled: bool) {
        self.shared
            .heartbeats_enabled
            .store(enabled, Ordering::Relaxed);
    }
}
