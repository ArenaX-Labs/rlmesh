//! A board as a robot in a session, sans-IO: the board's transport
//! (zenoh-pico) carries out the [`Action`]s it asks for and hands back what
//! arrives, so the protocol is session-core's and the transport only moves
//! bytes.
//!
//! The order is the participant's (design §8.3): subscribe to the host's
//! state and this robot's heartbeat echo, read the host instance from the
//! state, handshake on `hs`, then `Attach` and `DeclareParts` sealed on
//! `ctl`, then heartbeat. Any failure starts over from the handshake after
//! [`RETRY_MS`].

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use rlmesh_session_core::auth::{self, Credential, Link, ReplyOutcome};
use rlmesh_session_core::crypto::Rng;
use rlmesh_session_core::crypto::handshake::IkInitiator;
use rlmesh_session_core::crypto::signed;
use rlmesh_session_core::wire::codec;
use rlmesh_session_core::wire::heartbeat::{Health, Heartbeat, TimingSample};
use rlmesh_session_core::wire::ids::{ParticipantId, SessionId, TenantId};
use rlmesh_session_core::wire::keys::SessionKeys;
use rlmesh_session_core::wire::version::{PROTOCOL_MINOR, SPEC_VERSION};
use rlmesh_session_core::wire::{
    AuthoritySnapshot, Command, CtlStatus, Epoch, Mode, PriorityClass, Role,
};

/// How long a query waits for its reply before the transport gives up.
pub const QUERY_MS: u32 = 2000;
/// The pause before starting over after a failure.
pub const RETRY_MS: u32 = 1000;
/// With no heartbeat echo for this long, the host counts as lost.
pub const HOST_LOST_MS: u32 = 3000;
/// `ctl` requests are resent (same id) this many times before starting
/// over.
const CTL_TRIES: u8 = 3;

/// What a robot needs to join.
#[derive(Debug, Clone)]
pub struct Config {
    /// The session's tenant (`local` for an embedded host).
    pub tenant: TenantId,
    /// The session.
    pub session: SessionId,
    /// This robot's participant id and display name.
    pub name: ParticipantId,
    /// The device key and the host's static key, from the address.
    pub credential: Credential,
    /// The robot's map, as authored JSON.
    pub robot_json: String,
    /// The heartbeat period.
    pub heartbeat_ms: u32,
}

/// The subscriptions a robot asks for, named in [`Robot::sample`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sub {
    /// `host/state`: signed authority snapshots.
    State = 0,
    /// `host/tick`: signed virtual ticks.
    Tick = 1,
    /// `p/{id}/hb/echo`: sealed heartbeat echoes.
    Echo = 2,
}

impl Sub {
    /// From its C value.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::State),
            1 => Some(Self::Tick),
            2 => Some(Self::Echo),
            _ => None,
        }
    }
}

/// What the transport is asked to do. Queries carry an id; their reply (or
/// none) comes back through [`Robot::reply`] and [`Robot::query_done`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Subscribe to `key`, passing its samples to [`Robot::sample`] as
    /// `sub`.
    Subscribe {
        /// What [`Robot::sample`] calls it.
        sub: Sub,
        /// The key expression.
        key: String,
    },
    /// Declare a liveliness token on `key`, held while the robot runs.
    Token {
        /// The key expression.
        key: String,
    },
    /// Query `key` (with `payload`, if any), for at most [`QUERY_MS`].
    Query {
        /// Names the replies.
        id: u32,
        /// The key expression.
        key: String,
        /// The query's payload.
        payload: Option<Vec<u8>>,
    },
    /// Publish `payload` on `key`.
    Publish {
        /// The key expression.
        key: String,
        /// The bytes.
        payload: Vec<u8>,
    },
}

/// Where the robot is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Reading the host instance from its state.
    Host,
    /// Waiting for the handshake's reply.
    Handshake,
    /// Waiting for `Attach`'s reply.
    Attach,
    /// Waiting for `DeclareParts`' reply.
    Declare,
    /// Attached and heartbeating.
    Running,
    /// Waiting to start over.
    Retry,
}

/// Why the robot last started over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fault {
    /// No reply from the host.
    NoReply,
    /// The host's state isn't signed.
    Unsigned,
    /// The handshake failed: the address's key isn't this host's.
    Handshake,
    /// The host refused a request.
    Refused(CtlStatus, String),
    /// No heartbeat echo for [`HOST_LOST_MS`].
    HostLost,
    /// A message failed to seal or decode (a bug, or the RNG failed).
    Internal,
}

/// The `ctl` request in flight.
struct Pending {
    request: u64,
    command: Command,
    tries: u8,
}

/// The robot.
pub struct Robot {
    config: Config,
    keys: SessionKeys,
    phase: Phase,
    out: Vec<Action>,
    started: bool,
    next_query: u32,
    /// The query the current phase waits on.
    waiting: Option<u32>,
    initiator: Option<IkInitiator>,
    link: Option<Link>,
    pending: Option<Pending>,
    /// Whether the current `ctl` query got a reply that wasn't its answer
    /// (an unsealed hint, or another request's).
    hint: Option<Fault>,
    snapshot: Option<AuthoritySnapshot>,
    fault: Option<Fault>,
    retry_at: u64,
    next_heartbeat: u64,
    last_echo: u64,
    echoes: u64,
    heartbeat_seq: u64,
    previous: Option<TimingSample>,
    generation: u32,
}

impl core::fmt::Debug for Robot {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Robot")
            .field("name", &self.config.name)
            .field("phase", &self.phase)
            .field("fault", &self.fault)
            .finish_non_exhaustive()
    }
}

fn ns(now_us: u64) -> i64 {
    i64::try_from(now_us.saturating_mul(1000)).unwrap_or(i64::MAX)
}

impl Robot {
    /// A robot that starts joining on the first [`Self::poll`].
    pub fn new(config: Config) -> Self {
        let keys = SessionKeys::new(&config.tenant, config.session);
        Self {
            config,
            keys,
            phase: Phase::Host,
            out: Vec::new(),
            started: false,
            next_query: 0,
            waiting: None,
            initiator: None,
            link: None,
            pending: None,
            hint: None,
            snapshot: None,
            fault: None,
            retry_at: 0,
            next_heartbeat: 0,
            last_echo: 0,
            echoes: 0,
            heartbeat_seq: 0,
            previous: None,
            generation: 0,
        }
    }

    /// Where it is.
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// Why it last started over, if it has.
    pub fn fault(&self) -> Option<&Fault> {
        self.fault.as_ref()
    }

    /// The last verified authority snapshot.
    pub fn snapshot(&self) -> Option<&AuthoritySnapshot> {
        self.snapshot.as_ref()
    }

    /// Heartbeat echoes received, over every attach.
    pub fn echoes(&self) -> u64 {
        self.echoes
    }

    /// This attach's generation (0 before the first).
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Runs the timers and returns what the transport should do next, in
    /// order.
    pub fn poll(&mut self, now_us: u64) -> Vec<Action> {
        if !self.started {
            self.started = true;
            self.out.extend([
                Action::Subscribe {
                    sub: Sub::State,
                    key: self.keys.host_state(),
                },
                Action::Subscribe {
                    sub: Sub::Tick,
                    key: self.keys.host_tick(),
                },
                Action::Subscribe {
                    sub: Sub::Echo,
                    key: self.keys.heartbeat_echo(&self.config.name),
                },
                // A multi-session host starts the session when it sees
                // this token.
                Action::Token {
                    key: self.keys.participant(&self.config.name),
                },
            ]);
            self.ask_host();
        }
        match self.phase {
            Phase::Retry if now_us >= self.retry_at => self.ask_host(),
            // No echo for HOST_LOST_MS: the host is gone, or restarted
            // under a new instance. Join again.
            Phase::Running if now_us >= self.last_echo + u64::from(HOST_LOST_MS) * 1000 => {
                self.fail(Fault::HostLost, now_us);
            }
            Phase::Running if now_us >= self.next_heartbeat => self.heartbeat(now_us),
            _ => {}
        }
        core::mem::take(&mut self.out)
    }

    /// A reply to query `id`. `now_us` and `rng` drive the next step.
    pub fn reply(&mut self, id: u32, payload: &[u8], now_us: u64, rng: &mut dyn Rng) {
        if self.waiting != Some(id) {
            return;
        }
        match self.phase {
            Phase::Host => {
                // Unverified: a wrong instance only fails the handshake.
                let Ok(peeked) = signed::peek(payload) else {
                    return self.fail(Fault::Unsigned, now_us);
                };
                self.start(peeked.header.host_instance, rng, now_us);
            }
            Phase::Handshake => {
                self.waiting = None;
                let Some(initiator) = self.initiator.take() else {
                    return self.fail(Fault::Internal, now_us);
                };
                let msg2 = match codec::decode_handshake_reply(payload) {
                    Ok(Ok(msg2)) => msg2,
                    Ok(Err(refusal)) => {
                        return self.fail(Fault::Refused(refusal.status, refusal.message), now_us);
                    }
                    Err(_) => return self.fail(Fault::Internal, now_us),
                };
                let Ok(attached) = auth::finish(initiator, &msg2) else {
                    return self.fail(Fault::Handshake, now_us);
                };
                self.generation = attached.accept.generation;
                self.link = Some(Link::new(self.config.name.clone(), attached));
                self.phase = Phase::Attach;
                let attach = Command::Attach {
                    name: self.config.name.as_str().to_string(),
                    roles: alloc::vec![Role::Embodiment],
                    priority: PriorityClass::Scripted,
                    protocol_minor: PROTOCOL_MINOR,
                    supports: Vec::new(),
                    requires: Vec::new(),
                    workflow_edition: String::new(),
                    workflow_editions: Vec::new(),
                };
                self.request(attach, now_us);
            }
            Phase::Attach | Phase::Declare => self.ctl_reply(payload, now_us),
            Phase::Running | Phase::Retry => {}
        }
    }

    /// Query `id` is over: every reply has arrived, or it timed out.
    pub fn query_done(&mut self, id: u32, now_us: u64) {
        if self.waiting != Some(id) {
            return;
        }
        self.waiting = None;
        match self.phase {
            Phase::Attach | Phase::Declare => {
                let Some(mut pending) = self.pending.take() else {
                    return self.fail(Fault::Internal, now_us);
                };
                if let Some(hint) = self.hint.take() {
                    return self.fail(hint, now_us);
                }
                pending.tries += 1;
                if pending.tries >= CTL_TRIES {
                    return self.fail(Fault::NoReply, now_us);
                }
                // The same request id: the host answers a retry from its
                // reply cache, never applying it twice.
                self.send_ctl(pending, now_us);
            }
            Phase::Host | Phase::Handshake => self.fail(Fault::NoReply, now_us),
            Phase::Running | Phase::Retry => {}
        }
    }

    /// A sample on a subscription.
    pub fn sample(&mut self, sub: Sub, payload: &mut [u8], now_us: u64) {
        let Some(link) = self.link.as_mut() else {
            return;
        };
        match sub {
            Sub::State => {
                if let Ok(snapshot) = link.state(&self.keys.host_state(), payload) {
                    self.snapshot = Some(snapshot);
                }
            }
            Sub::Tick => {
                // Verified to keep the host's sequence floor; lockstep
                // isn't run on a board yet.
                let _ = link.tick(&self.keys.host_tick(), payload);
            }
            Sub::Echo => {
                let key = self.keys.heartbeat_echo(&self.config.name);
                if let Ok(echo) = link.echo(&key, payload) {
                    self.previous = Some(echo.complete(ns(now_us)));
                    self.last_echo = now_us;
                    self.echoes += 1;
                }
            }
        }
    }

    fn query(&mut self, key: String, payload: Option<Vec<u8>>) {
        self.next_query = self.next_query.wrapping_add(1);
        let id = self.next_query;
        self.waiting = Some(id);
        self.out.push(Action::Query { id, key, payload });
    }

    fn ask_host(&mut self) {
        self.phase = Phase::Host;
        self.link = None;
        self.pending = None;
        self.hint = None;
        self.query(self.keys.host_state(), None);
    }

    fn start(&mut self, host_instance: u64, rng: &mut dyn Rng, now_us: u64) {
        let session = self.config.session.to_string();
        let started = self.config.credential.start(
            rng,
            self.config.tenant.as_str(),
            &session,
            host_instance,
            &self.config.name,
        );
        let Ok((initiator, hello)) = started else {
            return self.fail(Fault::Internal, now_us);
        };
        self.initiator = Some(initiator);
        self.phase = Phase::Handshake;
        let payload = codec::encode_handshake_request(&hello);
        self.query(self.keys.handshake(), Some(payload));
    }

    fn epoch(&self) -> Option<Epoch> {
        self.snapshot.as_ref().map(|s| s.epoch)
    }

    fn request(&mut self, command: Command, now_us: u64) {
        let epoch = self.epoch();
        let ctl = self.keys.ctl();
        let Some(link) = self.link.as_mut() else {
            return self.fail(Fault::Internal, now_us);
        };
        let Ok((request, sealed)) = link.request(&ctl, epoch, &command) else {
            return self.fail(Fault::Internal, now_us);
        };
        self.hint = None;
        self.query(ctl, Some(sealed));
        self.pending = Some(Pending {
            request,
            command,
            tries: 0,
        });
    }

    fn send_ctl(&mut self, pending: Pending, now_us: u64) {
        let epoch = self.epoch();
        let ctl = self.keys.ctl();
        let Some(link) = self.link.as_mut() else {
            return self.fail(Fault::Internal, now_us);
        };
        let Ok(sealed) = link.resend(&ctl, pending.request, epoch, &pending.command) else {
            return self.fail(Fault::Internal, now_us);
        };
        self.query(ctl, Some(sealed));
        self.pending = Some(pending);
    }

    fn ctl_reply(&mut self, payload: &[u8], now_us: u64) {
        let ctl = self.keys.ctl();
        let (Some(link), Some(pending)) = (self.link.as_mut(), self.pending.as_ref()) else {
            return;
        };
        let mut bytes = payload.to_vec();
        let reply = match link.reply(&ctl, &mut bytes, pending.request) {
            Ok(ReplyOutcome::Answer(reply)) => reply,
            // Anyone can answer `ctl` unsealed: a hint, kept in case no
            // sealed answer comes.
            Ok(ReplyOutcome::Hint(refusal)) => {
                self.hint = Some(Fault::Refused(refusal.status, refusal.message));
                return;
            }
            Ok(ReplyOutcome::Other(_)) | Err(_) => return,
        };
        self.waiting = None;
        self.pending = None;
        self.hint = None;
        if let Err(refusal) = reply.result {
            return self.fail(Fault::Refused(refusal.status, refusal.message), now_us);
        }
        self.snapshot = Some(reply.state);
        match self.phase {
            Phase::Attach => {
                self.phase = Phase::Declare;
                let declare = Command::DeclareParts {
                    spec_version: SPEC_VERSION,
                    robot_json: self.config.robot_json.clone(),
                };
                self.request(declare, now_us);
            }
            Phase::Declare => {
                self.phase = Phase::Running;
                self.fault = None;
                self.next_heartbeat = now_us;
                self.last_echo = now_us;
            }
            _ => {}
        }
    }

    fn heartbeat(&mut self, now_us: u64) {
        self.next_heartbeat = now_us + u64::from(self.config.heartbeat_ms) * 1000;
        self.heartbeat_seq += 1;
        let (epoch, mode) = self
            .snapshot
            .as_ref()
            .map_or((Epoch(0), Mode::Disarmed), |s| (s.epoch, s.mode));
        let heartbeat = Heartbeat {
            participant: self.config.name.clone(),
            epoch,
            mode,
            seq: self.heartbeat_seq,
            t1_ns: ns(now_us),
            previous: self.previous.take(),
            last_seq_seen: Vec::new(),
            last_chunk_accepted: 0,
            last_chunk_applied: 0,
            queue_depth: 0,
            health: Health::Ok,
        };
        let key = self.keys.heartbeat(&self.config.name);
        let Some(link) = self.link.as_mut() else {
            return;
        };
        if let Ok(sealed) = link.heartbeat(&key, &heartbeat) {
            self.out.push(Action::Publish {
                key,
                payload: sealed,
            });
        }
    }

    fn fail(&mut self, fault: Fault, now_us: u64) {
        self.fault = Some(fault);
        self.phase = Phase::Retry;
        self.waiting = None;
        self.initiator = None;
        self.link = None;
        self.pending = None;
        self.hint = None;
        self.retry_at = now_us + u64::from(RETRY_MS) * 1000;
    }
}
