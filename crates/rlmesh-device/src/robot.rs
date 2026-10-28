//! A board as a robot in a session, sans-IO: the board's transport
//! (zenoh-pico) carries out the [`Action`]s it asks for and hands back what
//! arrives. The protocol is session-core's: the handshake and sealing are
//! its [`Link`], the command edge its [`EdgeCore`], the map its
//! [`normalise`]; this only puts them in the participant's order.
//!
//! The order (design §8.3): subscribe to the host's state, ticks, this
//! robot's heartbeat echo and key deliveries; read the host instance from
//! the state; handshake on `hs`; `Attach` and `DeclareParts` sealed on
//! `ctl`; then build an edge per command group, subscribe to each part's
//! commands, and heartbeat. Any failure starts over from the handshake after
//! [`RETRY_MS`].
//!
//! Traffic rides the host's per-class sessions (one port each, from the
//! address's): control on the first, a part's commands and samples on its
//! priority's ([`Action`]'s `class`).

use alloc::collections::{BTreeMap, VecDeque};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use bytes::Bytes;
use rlmesh_session_core::auth::keyring::KeyError;
use rlmesh_session_core::auth::{self, Credential, Link, LinkError, ReplyOutcome};
use rlmesh_session_core::crypto::Rng;
use rlmesh_session_core::crypto::handshake::IkInitiator;
use rlmesh_session_core::crypto::signed;
use rlmesh_session_core::edge::{EdgeConfig, EdgeCore, EdgeTick};
use rlmesh_session_core::spec::normalise;
use rlmesh_session_core::wire::codec;
use rlmesh_session_core::wire::version::{PROTOCOL_MINOR, SPEC_VERSION};
use rlmesh_session_core::wire::{
    AuthoritySnapshot, ChannelId, ChannelPriority, Command, CtlStatus, Epoch, Health, Heartbeat,
    Mode, PartSpec, ParticipantId, PriorityClass, Reason, ReasonCode, Role, Sample, SessionId,
    SessionKeys, TenantId, TimingSample,
};

/// How long a query waits for its reply before the transport gives up.
pub const QUERY_MS: u32 = 2000;
/// The pause before starting over after a failure.
pub const RETRY_MS: u32 = 1000;
/// With no heartbeat echo and no valid signed state for this long, the host
/// counts as lost (scene design §3.5).
pub const HOST_LOST_MS: u32 = 5000;
/// `ctl` requests are resent (same id) this many times before they fail.
const CTL_TRIES: u8 = 3;
/// How often a robot missing keys asks for them again (`GetKeys`).
const GET_KEYS_MS: u64 = 250;

/// The control class: the host's first port.
pub const CLASS_CONTROL: u8 = 0;

/// Subscription ids, for [`Robot::sample`]: `host/state`.
pub const SUB_STATE: u8 = 0;
/// `host/tick`.
pub const SUB_TICK: u8 = 1;
/// `p/{id}/hb/echo`.
pub const SUB_ECHO: u8 = 2;
/// `p/{id}/keys`.
pub const SUB_KEYS: u8 = 3;
/// `estop/*`: every member's e-stop signal. Its samples need their key.
pub const SUB_ESTOP: u8 = 4;
/// The commands of part `i` (in the map's order) are `SUB_COMMANDS + i`.
pub const SUB_COMMANDS: u8 = 5;

/// What a robot needs to join.
#[derive(Debug, Clone)]
pub struct Config {
    /// The session's tenant (`local` for an embedded host).
    pub tenant: TenantId,
    /// The session.
    pub session: SessionId,
    /// This robot's participant id and display name.
    pub name: ParticipantId,
    /// Its device key and the host's static key: from the address's key
    /// (device `local`), or provisioned (tether spec §14).
    pub credential: Credential,
    /// The robot's map, as authored JSON.
    pub robot_json: String,
    /// The heartbeat period.
    pub heartbeat_ms: u32,
}

/// Why a [`Config`] can't make a robot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// The map doesn't normalise: the host would refuse it.
    Map(String),
    /// A zero heartbeat period.
    Heartbeat,
}

/// What the transport is asked to do, on traffic class `class` (0 control,
/// 1 realtime, 2 data, 3 bulk: the address's port plus `class`). Queries
/// carry an id; their reply (or none) comes back through [`Robot::reply`]
/// and [`Robot::query_done`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Subscribe to `key`, passing its samples to [`Robot::sample`] as
    /// `sub`.
    Subscribe {
        /// The class.
        class: u8,
        /// What [`Robot::sample`] calls it.
        sub: u8,
        /// The key expression.
        key: String,
    },
    /// Declare a liveliness token on `key`, held while the robot runs.
    Token {
        /// The class.
        class: u8,
        /// The key expression.
        key: String,
    },
    /// Query `key` (with `payload`, if any), for at most [`QUERY_MS`].
    Query {
        /// The class.
        class: u8,
        /// Names the replies.
        id: u32,
        /// The key expression.
        key: String,
        /// The query's payload.
        payload: Option<Vec<u8>>,
    },
    /// Publish `payload` on `key`.
    Publish {
        /// The class.
        class: u8,
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
    /// Attached, declared and heartbeating.
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
    /// No heartbeat echo and no signed state for [`HOST_LOST_MS`].
    HostLost,
    /// A message failed to seal or decode (a bug, or the RNG failed), or
    /// the host's parts don't match the map.
    Internal,
}

/// Why [`Robot::publish`] sent nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublishError {
    /// Not running yet, or the stream's key hasn't arrived.
    NotReady,
    /// No such part.
    UnknownPart,
    /// Not the part's dense layout.
    Invalid,
}

/// The `ctl` request in flight.
struct Pending {
    query: u32,
    request: u64,
    command: Command,
    tries: u8,
}

/// One command group's edge.
struct Edge {
    core: EdgeCore,
    /// The latch count it last saw: a latch cleared since still fences it.
    latches_seen: u64,
}

/// One control tick of a part's command group: session-core's
/// [`EdgeTick`], with the part it concerns as an index into the map.
#[derive(Debug, Clone)]
pub struct Tick {
    /// The edge's tick.
    pub tick: EdgeTick,
    /// The part the tick concerns (the active or last-driven member).
    pub part: Option<usize>,
}

/// The robot.
pub struct Robot {
    config: Config,
    keys: SessionKeys,
    /// The map's part paths, in its order. `DeclareParts.seal` names
    /// them all: this runtime seals any part.
    paths: Vec<String>,
    phase: Phase,
    out: Vec<Action>,
    started: bool,
    next_query: u32,
    /// The join step's query (host state, handshake).
    waiting: Option<u32>,
    initiator: Option<IkInitiator>,
    link: Option<Link>,
    pending: Option<Pending>,
    queue: VecDeque<Command>,
    /// An unsealed refusal of the `ctl` request in flight: a hint, kept in
    /// case no sealed answer comes.
    hint: Option<Fault>,
    snapshot: Option<AuthoritySnapshot>,
    /// Local time the snapshot was observed at.
    observed_ns: i64,
    /// The host's parts of this robot, in the map's order.
    parts: Vec<PartSpec>,
    edges: Vec<Edge>,
    edge_of: Vec<Option<usize>>,
    /// Command subscriptions made (they outlive a restart).
    subscribed: Vec<bool>,
    seqs: BTreeMap<ChannelId, u64>,
    fault: Option<Fault>,
    retry_at: u64,
    next_heartbeat: u64,
    last_heard: u64,
    echoes: u64,
    heartbeat_seq: u64,
    previous: Option<TimingSample>,
    generation: u32,
    last_get_keys: Option<u64>,
    accepted: u64,
    applied: u64,
    /// This robot's e-stop latch (DESIGN §6): its own stop (`ESTOP_LOCAL`)
    /// or a member's signal (`ESTOP_SIGNAL`). It outlives a session, and the
    /// next attach carries it.
    estop: Option<Reason>,
    latches: u64,
    /// The code it last latched with, which a cleared latch still fences
    /// an edge with.
    last_code: ReasonCode,
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

/// The class a part's traffic rides.
fn class_of(priority: ChannelPriority) -> u8 {
    match priority {
        ChannelPriority::Control => 0,
        ChannelPriority::Realtime => 1,
        ChannelPriority::Data => 2,
        ChannelPriority::Bulk => 3,
    }
}

impl Robot {
    /// A robot that starts joining on the first [`Self::poll`].
    ///
    /// # Errors
    /// [`ConfigError`]: a map the host would refuse, checked here first.
    pub fn new(config: Config) -> Result<Self, ConfigError> {
        if config.heartbeat_ms == 0 {
            return Err(ConfigError::Heartbeat);
        }
        let paths: Vec<String> = normalise(&config.name, SPEC_VERSION, &config.robot_json)
            .map_err(|err| ConfigError::Map(err.message))?
            .iter()
            .map(|p| p.path.to_string())
            .collect();
        let keys = SessionKeys::new(&config.tenant, config.session);
        Ok(Self {
            subscribed: alloc::vec![false; paths.len()],
            edge_of: alloc::vec![None; paths.len()],
            paths,
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
            queue: VecDeque::new(),
            hint: None,
            snapshot: None,
            observed_ns: 0,
            parts: Vec::new(),
            edges: Vec::new(),
            seqs: BTreeMap::new(),
            fault: None,
            retry_at: 0,
            next_heartbeat: 0,
            last_heard: 0,
            echoes: 0,
            heartbeat_seq: 0,
            previous: None,
            generation: 0,
            last_get_keys: None,
            accepted: 0,
            applied: 0,
            estop: None,
            latches: 0,
            last_code: ReasonCode::EstopLocal,
        })
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

    /// The map's part paths, in its order: the indices every call takes.
    pub fn paths(&self) -> &[String] {
        &self.paths
    }

    /// The index of `path` in the map.
    pub fn part(&self, path: &str) -> Option<usize> {
        self.paths.iter().position(|p| p == path)
    }

    /// The index in the map of the part with `channel`, once running.
    pub fn part_of(&self, channel: ChannelId) -> Option<usize> {
        self.parts.iter().position(|p| p.channel == channel)
    }

    /// Latches this robot's e-stop for its own stop (`ESTOP_LOCAL`): an
    /// e-stop pin. Every edge stops at once, with or without a host, and
    /// the host is told. Only [`Self::clear_estop`] clears it.
    pub fn estop(&mut self, reason: &str, now_us: u64) {
        self.latch(Reason {
            code: ReasonCode::EstopLocal,
            by: self.config.name.to_string(),
            detail: reason.to_string(),
            at_ns: ns(now_us),
            ..Reason::default()
        });
    }

    /// Clears the e-stop latch: the only way to, since no wire path clears
    /// it. The host is told after the `Estop` for every latch before, so it
    /// ends in ESTOP, unlatched; leaving that still needs disarm and arm.
    pub fn clear_estop(&mut self) {
        if self.estop.take().is_some() {
            self.queue.push_back(Command::Clear);
        }
    }

    /// Why the e-stop is latched, while it is.
    pub fn estop_reason(&self) -> Option<&Reason> {
        self.estop.as_ref()
    }

    /// Chunks the edges accepted, over every attach.
    pub fn chunks_accepted(&self) -> u64 {
        self.accepted
    }

    /// Runs the timers and returns what the transport should do next, in
    /// order.
    pub fn poll(&mut self, now_us: u64) -> Vec<Action> {
        if !self.started {
            self.started = true;
            let me = &self.config.name;
            self.out.extend([
                Action::Subscribe {
                    class: CLASS_CONTROL,
                    sub: SUB_STATE,
                    key: self.keys.host_state(),
                },
                Action::Subscribe {
                    class: CLASS_CONTROL,
                    sub: SUB_TICK,
                    key: self.keys.host_tick(),
                },
                Action::Subscribe {
                    class: CLASS_CONTROL,
                    sub: SUB_ECHO,
                    key: self.keys.heartbeat_echo(me),
                },
                Action::Subscribe {
                    class: CLASS_CONTROL,
                    sub: SUB_KEYS,
                    key: self.keys.participant_keys(me),
                },
                // A robot latches every member's e-stop signal at once,
                // with or without a host.
                Action::Subscribe {
                    class: CLASS_CONTROL,
                    sub: SUB_ESTOP,
                    key: self.keys.estops(),
                },
                // A multi-session host starts the session when it sees
                // this token.
                Action::Token {
                    class: CLASS_CONTROL,
                    key: self.keys.participant(me),
                },
            ]);
            self.ask_host();
        }
        match self.phase {
            Phase::Retry if now_us >= self.retry_at => self.ask_host(),
            // Nothing from the host for HOST_LOST_MS: it is gone, or
            // restarted under a new instance. Join again.
            Phase::Running if now_us >= self.last_heard + u64::from(HOST_LOST_MS) * 1000 => {
                self.fail(Fault::HostLost, now_us);
            }
            Phase::Running if now_us >= self.next_heartbeat => self.heartbeat(now_us),
            _ => {}
        }
        self.next_ctl(now_us);
        core::mem::take(&mut self.out)
    }

    /// A reply to query `id`. `now_us` and `rng` drive the next step.
    pub fn reply(&mut self, id: u32, payload: &[u8], now_us: u64, rng: &mut dyn Rng) {
        if self.pending.as_ref().is_some_and(|p| p.query == id) {
            return self.ctl_reply(payload, now_us);
        }
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
                self.queue.push_back(Command::Attach {
                    name: self.config.name.as_str().to_string(),
                    roles: alloc::vec![Role::Embodiment],
                    priority: PriorityClass::Scripted,
                    protocol_minor: PROTOCOL_MINOR,
                    supports: Vec::new(),
                    requires: Vec::new(),
                    workflow_edition: String::new(),
                    workflow_editions: Vec::new(),
                    // A latch outlives the session: the host learns it at
                    // once.
                    estop: self.estop.clone(),
                });
                self.next_ctl(now_us);
            }
            _ => {}
        }
    }

    /// Query `id` is over: every reply has arrived, or it timed out.
    pub fn query_done(&mut self, id: u32, now_us: u64) {
        if self.pending.as_ref().is_some_and(|p| p.query == id) {
            return self.ctl_unanswered(now_us);
        }
        if self.waiting != Some(id) {
            return;
        }
        self.waiting = None;
        if matches!(self.phase, Phase::Host | Phase::Handshake) {
            self.fail(Fault::NoReply, now_us);
        }
    }

    /// A sample on subscription `sub`, received on `key`.
    pub fn sample(&mut self, sub: u8, key: &str, payload: &mut [u8], now_us: u64) {
        let now = ns(now_us);
        let Some(link) = self.link.as_mut() else {
            return;
        };
        match sub {
            SUB_STATE => {
                if let Ok(snapshot) = link.state(&self.keys.host_state(), payload) {
                    self.last_heard = now_us;
                    link.observe(&snapshot, now);
                    self.observe(snapshot, now);
                }
            }
            SUB_TICK => {
                // Verified to keep the host's sequence floor; lockstep
                // isn't run on a board yet.
                let _ = link.tick(&self.keys.host_tick(), payload);
            }
            SUB_ECHO => {
                let key = self.keys.heartbeat_echo(&self.config.name);
                if let Ok(echo) = link.echo(&key, payload) {
                    self.previous = Some(echo.complete(now));
                    self.last_heard = now_us;
                    self.echoes += 1;
                }
            }
            SUB_ESTOP => {
                if let Ok((by, signal)) = link.open_estop_signal(key, payload.to_vec()) {
                    self.latch(Reason {
                        code: ReasonCode::EstopSignal,
                        by: by.to_string(),
                        detail: signal.reason,
                        at_ns: now,
                        ..Reason::default()
                    });
                }
            }
            SUB_KEYS => {
                let key = self.keys.participant_keys(&self.config.name);
                if let Ok(ack) = link.keys(&key, payload, now)
                    && !ack.is_empty()
                {
                    self.queue.push_back(Command::KeysAck(ack));
                }
            }
            _ => {
                if let Some(index) = sub.checked_sub(SUB_COMMANDS) {
                    self.chunk(usize::from(index), payload.to_vec(), now);
                }
            }
        }
    }

    /// One control tick of part `part`'s command group at `now_us`: what
    /// to dispatch, hold, or the safe behaviour to run. Publishes the
    /// group's applied samples, and tells the host when the edge went safe
    /// on its own account. `None` for a part that takes no commands, or
    /// before the robot runs.
    pub fn tick(&mut self, part: usize, now_us: u64) -> Option<Tick> {
        let now = ns(now_us);
        let index = (*self.edge_of.get(part)?)?;
        let latch = self.latch_now();
        let edge = self.edges.get_mut(index)?;
        apply_latch(edge, latch);
        let tick = edge
            .core
            .tick(now, self.snapshot.as_ref(), self.observed_ns);
        let epoch = self.snapshot.as_ref().map_or(Epoch(0), |s| s.epoch);
        for sample in &tick.applied {
            let Some(stream) = self
                .parts
                .iter()
                .find(|p| p.channel == sample.part)
                .and_then(|p| p.applied_stream())
                .and_then(Result::ok)
            else {
                continue;
            };
            let seq = self.seqs.entry(stream.channel).or_insert(0);
            *seq += 1;
            let sealed = Sample {
                channel: stream.channel,
                epoch,
                seq: *seq,
                capture_ns: now,
                publish_ns: now,
                tick: 0,
                payload: sample.value.clone(),
                applied: Some(sample.applied),
            };
            let key = self.keys.data(&self.config.name, &stream.path);
            if let Some(link) = self.link.as_mut()
                && let Ok(payload) = link.seal_sample(&key, &sealed, true)
            {
                self.out.push(Action::Publish {
                    class: class_of(stream.priority),
                    key,
                    payload,
                });
            }
        }
        if let Some(action) = &tick.action {
            self.applied = action.chunk.seq;
        }
        if let Some((_, cause)) = tick.report {
            let channel = tick.channel.map(|c| c.to_string()).unwrap_or_default();
            self.queue.push_back(Command::EnterSafe {
                reason: alloc::format!("{channel}: {cause}"),
                code: cause.code(),
            });
        }
        let part = tick.channel.and_then(|c| self.part_of(c));
        Some(Tick { tick, part })
    }

    /// The robot finished the handover of `part`'s group: it may dispatch
    /// again.
    pub fn handover_done(&mut self, part: usize) {
        if let Some(Some(index)) = self.edge_of.get(part)
            && let Some(edge) = self.edges.get_mut(*index)
        {
            edge.core.handover_done();
        }
    }

    /// Publishes a sample of part `part`: `value` in its dense layout,
    /// captured at `capture_us` (this board's clock).
    ///
    /// # Errors
    /// [`PublishError`]: nothing was sent.
    pub fn publish(
        &mut self,
        part: usize,
        capture_us: u64,
        value: &[u8],
        now_us: u64,
    ) -> Result<(), PublishError> {
        if part >= self.paths.len() {
            return Err(PublishError::UnknownPart);
        }
        let (Some(spec), Some(link)) = (self.parts.get(part), self.link.as_mut()) else {
            return Err(PublishError::NotReady);
        };
        let stream = spec.stream().map_err(|_| PublishError::Invalid)?;
        stream
            .check_payload(value)
            .map_err(|_| PublishError::Invalid)?;
        let seq = self.seqs.entry(stream.channel).or_insert(0);
        *seq += 1;
        let sample = Sample {
            channel: stream.channel,
            epoch: self.snapshot.as_ref().map_or(Epoch(0), |s| s.epoch),
            seq: *seq,
            capture_ns: ns(capture_us),
            publish_ns: ns(now_us),
            tick: 0,
            payload: Bytes::copy_from_slice(value),
            applied: None,
        };
        let key = self.keys.data(&self.config.name, &stream.path);
        match link.seal_sample(&key, &sample, false) {
            Ok(payload) => {
                self.out.push(Action::Publish {
                    class: class_of(stream.priority),
                    key,
                    payload,
                });
                Ok(())
            }
            Err(err) => {
                self.missing_key(&err, now_us);
                Err(PublishError::NotReady)
            }
        }
    }

    fn chunk(&mut self, index: usize, msg: Vec<u8>, now: i64) {
        let (Some(path), Some(link)) = (self.paths.get(index), self.link.as_mut()) else {
            return;
        };
        let key = self.keys.commands(&self.config.name, path);
        let chunk = match link.open_chunk(&key, msg) {
            Ok(chunk) => chunk,
            Err(err) => {
                let now_us = u64::try_from(now / 1000).unwrap_or(0);
                return self.missing_key(&err, now_us);
            }
        };
        let latch = self.latch_now();
        let Some(Some(edge)) = self
            .edge_of
            .get(index)
            .map(|e| e.and_then(|i| self.edges.get_mut(i)))
        else {
            return;
        };
        apply_latch(edge, latch);
        if edge
            .core
            .accept(chunk, now, self.snapshot.as_ref(), self.observed_ns)
            .is_ok()
        {
            self.accepted += 1;
        }
    }

    /// A message under a key this robot doesn't hold: ask for every key of
    /// its generation, at most every [`GET_KEYS_MS`].
    fn missing_key(&mut self, err: &LinkError, now_us: u64) {
        if !matches!(err, LinkError::Key(KeyError::Unknown(_))) {
            return;
        }
        if self
            .last_get_keys
            .is_some_and(|at| now_us < at + GET_KEYS_MS * 1000)
        {
            return;
        }
        self.last_get_keys = Some(now_us);
        self.queue.push_back(Command::GetKeys);
    }

    /// Latches the e-stop for `reason`, and queues its report to the host
    /// (in order: before any `Clear` queued after it).
    fn latch(&mut self, reason: Reason) {
        let origin = (reason.code == ReasonCode::EstopSignal)
            .then(|| ParticipantId::new(&reason.by).ok())
            .flatten();
        self.queue.push_back(Command::Estop {
            reason: reason.detail.clone(),
            code: reason.code,
            origin,
        });
        self.last_code = reason.code;
        self.estop = Some(reason);
        self.latches += 1;
    }

    /// The latch as the edges take it.
    fn latch_now(&self) -> Latch {
        Latch {
            latches: self.latches,
            code: self.last_code,
            latched: self.estop.is_some(),
        }
    }

    fn observe(&mut self, snapshot: AuthoritySnapshot, now: i64) {
        self.snapshot = Some(snapshot);
        self.observed_ns = now;
    }

    fn query(&mut self, class: u8, key: String, payload: Option<Vec<u8>>) -> u32 {
        self.next_query = self.next_query.wrapping_add(1);
        let id = self.next_query;
        self.out.push(Action::Query {
            class,
            id,
            key,
            payload,
        });
        id
    }

    fn ask_host(&mut self) {
        self.phase = Phase::Host;
        let id = self.query(CLASS_CONTROL, self.keys.host_state(), None);
        self.waiting = Some(id);
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
        let id = self.query(CLASS_CONTROL, self.keys.handshake(), Some(payload));
        self.waiting = Some(id);
    }

    fn epoch(&self) -> Option<Epoch> {
        self.snapshot.as_ref().map(|s| s.epoch)
    }

    /// Sends the next queued `ctl` request when none is in flight.
    fn next_ctl(&mut self, now_us: u64) {
        while self.pending.is_none() && self.link.is_some() {
            let Some(command) = self.queue.pop_front() else {
                return;
            };
            // A stop the host already shows needs no report.
            if matches!(command, Command::EnterSafe { .. })
                && self
                    .snapshot
                    .as_ref()
                    .is_some_and(|s| matches!(s.mode, Mode::Safe | Mode::Estop))
            {
                continue;
            }
            let epoch = self.epoch();
            let ctl = self.keys.ctl();
            let Some(link) = self.link.as_mut() else {
                return;
            };
            let Ok((request, sealed)) = link.request(&ctl, epoch, &command) else {
                return self.fail(Fault::Internal, now_us);
            };
            self.hint = None;
            let query = self.query(CLASS_CONTROL, ctl, Some(sealed));
            self.pending = Some(Pending {
                query,
                request,
                command,
                tries: 0,
            });
        }
    }

    /// Whether the request in flight is a join step: its failure starts
    /// over.
    fn joining(&self) -> bool {
        matches!(self.phase, Phase::Attach | Phase::Declare)
            && self.pending.as_ref().is_some_and(|p| {
                matches!(
                    p.command,
                    Command::Attach { .. } | Command::DeclareParts { .. }
                )
            })
    }

    fn ctl_unanswered(&mut self, now_us: u64) {
        let joining = self.joining();
        let Some(mut pending) = self.pending.take() else {
            return;
        };
        if let Some(hint) = self.hint.take() {
            if joining {
                return self.fail(hint, now_us);
            }
            return self.next_ctl(now_us);
        }
        pending.tries += 1;
        if pending.tries >= CTL_TRIES {
            if joining {
                return self.fail(Fault::NoReply, now_us);
            }
            // A stop report is retried until the host answers it.
            if is_stop(&pending.command) {
                self.queue.push_front(pending.command);
            }
            return self.next_ctl(now_us);
        }
        // The same request id: the host answers a retry from its reply
        // cache, never applying it twice.
        let epoch = self.epoch();
        let ctl = self.keys.ctl();
        let Some(link) = self.link.as_mut() else {
            return;
        };
        let Ok(sealed) = link.resend(&ctl, pending.request, epoch, &pending.command) else {
            return self.fail(Fault::Internal, now_us);
        };
        pending.query = self.query(CLASS_CONTROL, ctl, Some(sealed));
        self.pending = Some(pending);
    }

    fn ctl_reply(&mut self, payload: &[u8], now_us: u64) {
        let now = ns(now_us);
        let ctl = self.keys.ctl();
        let (Some(link), Some(pending)) = (self.link.as_mut(), self.pending.as_ref()) else {
            return;
        };
        let mut bytes = payload.to_vec();
        let reply = match link.reply(&ctl, &mut bytes, pending.request, now) {
            Ok(ReplyOutcome::Answer(reply)) => reply,
            Ok(ReplyOutcome::Hint(refusal)) => {
                self.hint = Some(Fault::Refused(refusal.status, refusal.message));
                return;
            }
            Ok(ReplyOutcome::Other(_)) | Err(_) => return,
        };
        let joining = self.joining();
        let Some(pending) = self.pending.take() else {
            return;
        };
        self.hint = None;
        let reply = *reply;
        self.observe(reply.state, now);
        match reply.result {
            Err(refusal) if joining => {
                return self.fail(Fault::Refused(refusal.status, refusal.message), now_us);
            }
            // Its first outcome is unknown: send it again, anew.
            Err(refusal) if refusal.status == CtlStatus::Expired && is_stop(&pending.command) => {
                self.queue.push_front(pending.command.clone());
            }
            _ => {}
        }
        match (self.phase, &pending.command) {
            (Phase::Attach, Command::Attach { .. }) => {
                self.phase = Phase::Declare;
                self.queue.push_front(Command::DeclareParts {
                    spec_version: SPEC_VERSION,
                    robot_json: self.config.robot_json.clone(),
                    seal: self.paths.clone(),
                });
            }
            (Phase::Declare, Command::DeclareParts { .. }) => {
                if !self.build_edges() {
                    return self.fail(Fault::Internal, now_us);
                }
                self.phase = Phase::Running;
                self.fault = None;
                self.next_heartbeat = now_us;
                self.last_heard = now_us;
            }
            _ => {}
        }
        self.next_ctl(now_us);
    }

    /// This robot's parts as the host declared them, an edge per command
    /// group, and a subscription to each commandable part's commands (as
    /// the host-side `Edge::attach` does).
    fn build_edges(&mut self) -> bool {
        let Some(view) = self.snapshot.as_ref() else {
            return false;
        };
        let me = &self.config.name;
        let mut parts = Vec::with_capacity(self.paths.len());
        for path in &self.paths {
            match view
                .parts
                .iter()
                .find(|p| &p.owner == me && p.path.as_str() == path)
            {
                Some(part) => parts.push(part.clone()),
                None => return false,
            }
        }
        let mut edges: Vec<Edge> = Vec::new();
        let mut edge_of = alloc::vec![None; parts.len()];
        for (i, part) in parts.iter().enumerate() {
            let Some(command) = &part.command else {
                continue;
            };
            if edge_of[i].is_some() {
                continue;
            }
            let members: Vec<usize> = match &command.group {
                Some(group) => parts
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| {
                        p.command
                            .as_ref()
                            .is_some_and(|c| c.group.as_deref() == Some(group))
                    })
                    .map(|(j, _)| j)
                    .collect(),
                None => alloc::vec![i],
            };
            let specs = members.iter().map(|&j| parts[j].clone()).collect();
            let Ok(core) = EdgeCore::new(specs, me.clone(), EdgeConfig::default()) else {
                return false;
            };
            for &j in &members {
                edge_of[j] = Some(edges.len());
            }
            edges.push(Edge {
                core,
                latches_seen: 0,
            });
        }
        for (i, part) in parts.iter().enumerate() {
            if part.command.is_none() || self.subscribed[i] {
                continue;
            }
            let Ok(sub) = u8::try_from(usize::from(SUB_COMMANDS) + i) else {
                return false;
            };
            self.subscribed[i] = true;
            self.out.push(Action::Subscribe {
                class: class_of(part.priority),
                sub,
                key: self.keys.commands(me, part.path.as_str()),
            });
        }
        self.parts = parts;
        self.edges = edges;
        self.edge_of = edge_of;
        self.seqs.clear();
        true
    }

    fn heartbeat(&mut self, now_us: u64) {
        self.next_heartbeat = now_us + u64::from(self.config.heartbeat_ms) * 1000;
        self.heartbeat_seq += 1;
        let (epoch, mode) = self
            .snapshot
            .as_ref()
            .map_or((Epoch(0), Mode::Disarmed), |s| (s.epoch, s.mode));
        let queue_depth = self
            .edges
            .iter()
            .map(|e| u32::try_from(e.core.queue_depth()).unwrap_or(u32::MAX))
            .sum();
        let heartbeat = Heartbeat {
            participant: self.config.name.clone(),
            epoch,
            mode,
            seq: self.heartbeat_seq,
            t1_ns: ns(now_us),
            previous: self.previous.take(),
            last_seq_seen: Vec::new(),
            last_chunk_accepted: self.accepted,
            last_chunk_applied: self.applied,
            queue_depth,
            health: Health::Ok,
        };
        let key = self.keys.heartbeat(&self.config.name);
        let Some(link) = self.link.as_mut() else {
            return;
        };
        if let Ok(sealed) = link.heartbeat(&key, &heartbeat) {
            self.out.push(Action::Publish {
                class: CLASS_CONTROL,
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
        self.queue.clear();
        self.hint = None;
        self.parts.clear();
        self.edges.clear();
        self.edge_of = alloc::vec![None; self.paths.len()];
        self.retry_at = now_us + u64::from(RETRY_MS) * 1000;
    }
}

/// Whether `command` is a stop report: sent until the host answers it.
fn is_stop(command: &Command) -> bool {
    matches!(
        command,
        Command::EnterSafe { .. } | Command::Estop { .. } | Command::Clear
    )
}

/// The robot's e-stop latch, as an edge takes it.
#[derive(Clone, Copy)]
struct Latch {
    /// How many times it latched.
    latches: u64,
    /// The code it last latched with.
    code: ReasonCode,
    latched: bool,
}

/// Gives `edge` the robot's latch: a latch it hasn't seen yet stops and
/// fences it even when cleared since (as the host-side edge does).
fn apply_latch(edge: &mut Edge, latch: Latch) {
    if edge.latches_seen != latch.latches {
        edge.latches_seen = latch.latches;
        edge.core.set_local_estop(Some(latch.code));
    }
    edge.core
        .set_local_estop(latch.latched.then_some(latch.code));
}
