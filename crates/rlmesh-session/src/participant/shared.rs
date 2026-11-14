//! What the participant's tasks share: `ctl` calls and their replies, the
//! authority snapshot it keeps, and the key task's work.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::wire::{
    AuthoritySnapshot, ChannelId, ClockMode, Command, CtlReply, Epoch, Heartbeat, KeysAck, Lease,
    Mode, Reason, ReasonCode, ViewReply,
};
use rlmesh_session_core::auth::{Link, ReplyOutcome};
use tokio::sync::watch;
use tracing::{debug, warn};
use zenoh::query::ConsolidationMode;

use crate::error::SessionError;
use crate::participant::attach::ask;
use crate::participant::edge::Report;
use crate::participant::sealer::Sealer;
use crate::participant::{Shared, lock};
use crate::transport::{CONTROL_CONGESTION, CONTROL_PRIORITY};

/// Work for the key task: acknowledge a delivery, recover missing keys,
/// or keep retrying a request whose caller timed out (its reply may carry
/// keys: a grant's).
#[derive(Debug)]
pub(super) enum KeyWork {
    Ack(KeysAck),
    Missing,
    Unanswered(u64, Command, watch::Sender<Option<CtlReply>>),
}

/// A request the host did not answer before its call timed out: its
/// outcome is unknown. The participant keeps sending it under its id, which
/// the host answers from its reply cache without applying the verb twice,
/// until the answer comes or this generation ends.
#[derive(Clone, Debug)]
pub struct Pending {
    id: u64,
    reply: watch::Receiver<Option<CtlReply>>,
}

impl Pending {
    /// The request's id.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The host's answer (a refusal stays in it, keys are installed), once
    /// it comes; [`SessionError::Closed`] when this generation ended first.
    pub async fn reply(mut self) -> Result<CtlReply, SessionError> {
        let reply = self.reply.wait_for(Option::is_some).await;
        reply
            .ok()
            .and_then(|reply| reply.clone())
            .ok_or(SessionError::Closed)
    }
}

/// The link, once the handshake completed.
pub(super) type SharedLink = Arc<std::sync::OnceLock<Mutex<Link>>>;

impl Shared {
    /// Keeps the newest snapshot.
    pub(super) fn observe(&self, snapshot: AuthoritySnapshot) {
        if let Some(link) = self.link.get() {
            lock(link).observe(&snapshot, self.clock.now_ns());
        }
        let id = &self.id;
        self.view.send_if_modified(|view| {
            if view
                .as_ref()
                .is_some_and(|old| !snapshot.is_newer_than(old))
            {
                return false;
            }
            if let Some(old) = view.as_ref()
                && old.mode != snapshot.mode
            {
                debug!(participant = %id, "mode {} -> {}", old.mode, snapshot.mode);
            }
            self.epoch.store(snapshot.epoch.0, Ordering::Relaxed);
            self.view_local_ns
                .store(self.clock.now_ns(), Ordering::Relaxed);
            *view = Some(snapshot);
            true
        });
    }

    pub(super) fn seen(&self, channel: ChannelId) -> Arc<AtomicU64> {
        self.seen
            .lock()
            .map(|mut seen| seen.entry(channel).or_default().clone())
            .unwrap_or_default()
    }

    /// Latches this robot process's e-stop for `reason`, and queues its
    /// report to the host.
    pub(super) fn latch(&self, reason: Reason) {
        let seq = self.estop.latch(reason.clone());
        let _ = self.reports.send(Report::estop(&reason, seq));
    }

    /// A member's e-stop signal on `key`: latched at once (`ESTOP_SIGNAL`,
    /// `by` its sender), when its `k_estop` verifies it.
    pub(super) fn on_estop_signal(&self, key: &str, payload: &zenoh::bytes::ZBytes) {
        let Some(link) = self.link.get() else {
            return;
        };
        let opened = lock(link).open_estop_signal(key, payload.to_bytes().into_owned());
        match opened {
            Ok((by, signal)) => {
                warn!(participant = %self.id, %by, reason = %signal.reason, "e-stop signal latched");
                self.latch(Reason {
                    code: ReasonCode::EstopSignal,
                    by: by.to_string(),
                    detail: signal.reason,
                    at_ns: self.clock.now_ns(),
                    ..Reason::default()
                });
            }
            Err(err) => debug!(participant = %self.id, %err, "e-stop signal dropped"),
        }
    }

    /// Whether this participant's generation ended: the view lists a newer
    /// one under its id, or none at all.
    pub(super) fn generation_ended(&self) -> bool {
        let Some(link) = self.link.get() else {
            return true;
        };
        let mine = lock(link).accept().generation;
        self.view.borrow().as_ref().is_some_and(|view| {
            view.participant(&self.id)
                .is_none_or(|p| p.generation != mine)
        })
    }

    pub(super) fn epoch_and_mode(&self) -> (u64, Mode) {
        self.view
            .borrow()
            .as_ref()
            .map_or((0, Mode::Disarmed), |s| (s.epoch.0, s.mode))
    }

    pub(super) fn clock_mode(&self) -> ClockMode {
        self.view
            .borrow()
            .as_ref()
            .map_or(ClockMode::Wall, |s| s.clock_mode)
    }

    /// Sends one verb and waits for the host's reply. Every reply carries the
    /// host's snapshot, which updates this participant's view even when the
    /// verb was refused.
    pub(super) async fn ctl(
        &self,
        command: Command,
    ) -> Result<(AuthoritySnapshot, Vec<Lease>), SessionError> {
        let CtlReply {
            result,
            state,
            leases,
            ..
        } = self.ctl_reply(command).await?;
        result?;
        Ok((state, leases))
    }

    /// [`Self::ctl`], with a refusal left in the reply. A request left
    /// unanswered is retried in the background under its id until the
    /// host's answer (its cached reply, keys included) arrives or this
    /// generation ends; a caller that timed out gets
    /// [`SessionError::Pending`], whose handle resolves with that answer.
    pub(super) async fn ctl_reply(&self, command: Command) -> Result<CtlReply, SessionError> {
        let (id, reply) = self.call(None, &command).await;
        let Some(id) = id.filter(|&id| {
            reply.is_err()
                && self
                    .link
                    .get()
                    .is_some_and(|link| lock(link).unanswered(id))
        }) else {
            return reply;
        };
        let (answer, pending) = watch::channel(None);
        let _ = self.key_work.send(KeyWork::Unanswered(id, command, answer));
        match reply {
            Err(SessionError::NoReply(timeout)) => Err(SessionError::Pending {
                timeout,
                pending: Pending { id, reply: pending },
            }),
            other => other,
        }
    }

    /// Sends `command` as a new request (`None`) or again as request `id`
    /// (a retry, which the host answers from its reply cache, never
    /// applying the verb twice), and takes the first reply that
    /// answers it. An unsealed refusal is only a hint (anyone can answer
    /// `ctl`): it is returned, as [`SessionError::Unverified`], when no
    /// sealed reply came. Returns the id.
    pub(super) async fn call(
        &self,
        request_id: Option<u64>,
        command: &Command,
    ) -> (Option<u64>, Result<CtlReply, SessionError>) {
        let Some(link) = self.link.get() else {
            return (request_id, Err(SessionError::Closed));
        };
        let (epoch, _) = self.epoch_and_mode();
        let epoch = (epoch > 0).then_some(Epoch(epoch));
        let key = self.keys.ctl();
        let sealed = {
            let mut link = lock(link);
            match request_id {
                Some(id) => link
                    .resend(&key, id, epoch, command)
                    .map(|sealed| (id, sealed)),
                None => link.request(&key, epoch, command),
            }
        };
        match sealed {
            Ok((id, sealed)) => (Some(id), self.exchange(link, &key, id, sealed).await),
            Err(err) => (
                request_id,
                Err(SessionError::Config(format!("ctl request: {err}"))),
            ),
        }
    }

    async fn exchange(
        &self,
        link: &Mutex<Link>,
        key: &str,
        request_id: u64,
        sealed: Vec<u8>,
    ) -> Result<CtlReply, SessionError> {
        let timeout = self.ctl_timeout;
        let replies = self
            .transport
            .control()
            .get(key)
            .payload(sealed)
            .consolidation(ConsolidationMode::None)
            .priority(CONTROL_PRIORITY)
            .congestion_control(CONTROL_CONGESTION)
            .express(true)
            .timeout(timeout)
            .await?;
        let mut hint = None;
        while let Ok(reply) = replies.recv_async().await {
            let Ok(sample) = reply.result() else {
                continue;
            };
            let mut bytes = sample.payload().to_bytes().into_owned();
            if self.takes_view {
                let outcome =
                    lock(link).reply_view(key, &mut bytes, request_id, self.clock.now_ns());
                match outcome {
                    Ok(ReplyOutcome::Answer(reply)) => {
                        // The reply carries the view, not the state: read
                        // the state the verb left.
                        let state = self.fetch_state().await?;
                        let ViewReply {
                            result,
                            leases,
                            request_id,
                            keys,
                            assignment,
                            run,
                            ..
                        } = *reply;
                        return Ok(CtlReply {
                            result,
                            state,
                            leases,
                            request_id,
                            keys,
                            assignment,
                            run,
                        });
                    }
                    Ok(ReplyOutcome::Hint(refusal)) => hint = Some(refusal),
                    Ok(ReplyOutcome::Other(_)) => {}
                    Err(err) => debug!(participant = %self.id, %err, "ctl reply dropped"),
                }
                continue;
            }
            let outcome = lock(link).reply(key, &mut bytes, request_id, self.clock.now_ns());
            match outcome {
                Ok(ReplyOutcome::Answer(reply)) => {
                    self.observe(reply.state.clone());
                    return Ok(*reply);
                }
                Ok(ReplyOutcome::Hint(refusal)) => hint = Some(refusal),
                Ok(ReplyOutcome::Other(_)) => {}
                Err(err) => debug!(participant = %self.id, %err, "ctl reply dropped"),
            }
        }
        Err(hint.map_or(SessionError::NoReply(timeout), SessionError::Unverified))
    }

    pub(super) fn sealer(&self) -> Sealer {
        Sealer {
            link: self.link.clone(),
            work: self.key_work.clone(),
        }
    }

    /// The host's current snapshot, queried and verified (a newer
    /// publication that overtook it stands).
    async fn fetch_state(&self) -> Result<AuthoritySnapshot, SessionError> {
        let key = self.keys.host_state();
        let signed = ask(self.transport.control(), &key, None, self.ctl_timeout).await?;
        if let Some(snapshot) = self.verify_state(&key, &signed) {
            self.observe(snapshot);
        }
        self.view
            .borrow()
            .clone()
            .ok_or(SessionError::NoReply(self.ctl_timeout))
    }

    /// Verifies a `host/state` broadcast, received on `key`.
    pub(super) fn verify_state(&self, key: &str, msg: &[u8]) -> Option<AuthoritySnapshot> {
        lock(self.link.get()?)
            .state(key, msg)
            .map_err(|err| debug!(participant = %self.id, %err, "authority update dropped"))
            .ok()
    }
}

impl Shared {
    /// Seals `heartbeat` under `k_up` and publishes it.
    /// Seals and sends `heartbeat`. With `stamp`, its `t1` is taken
    /// under the link's lock, so time spent waiting for the lock (another
    /// task sealing or opening) never counts as network time in the clock
    /// mapping.
    pub(super) async fn send_heartbeat(
        &self,
        mut heartbeat: Heartbeat,
        stamp: bool,
    ) -> Result<(), SessionError> {
        let (Some(link), Some(publisher)) = (self.link.get(), self.heartbeat.get()) else {
            return Err(SessionError::Closed);
        };
        let sealed = {
            let mut link = lock(link);
            if stamp {
                heartbeat.t1_ns = self.clock.now_ns();
            }
            link.heartbeat(publisher.key_expr().as_str(), &heartbeat)
        }
        .map_err(|err| SessionError::Config(format!("heartbeat: {err}")))?;
        Ok(publisher.put(sealed).await?)
    }
}
