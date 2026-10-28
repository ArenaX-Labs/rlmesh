//! The session side of the gateway: real `rlmesh-session` participants on
//! the real v2 keyspace.
//!
//! - The [`Gateway`] attaches one observer participant of its own, for the
//!   declarations (it resolves subscriptions against the host's parts), and
//!   relays the host's authority updates (`host/state`) as they were
//!   published, once their signature verifies (the browser gets the plain
//!   snapshot: the wasm holds no keys).
//! - Every participant attaches with the gateway's credential (the session
//!   address's key: device `local`), so each runs its own handshake and
//!   seals its own verbs and heartbeats.
//! - Every browser control connection is a [`BrowserLink`]: a participant of
//!   its own (a controller at the class the browser asked for, capped by
//!   `--max-priority`), with an identity binding over every commandable part
//!   so it can be granted leases. The host is the only authority: lease
//!   grants, releases and the session verbs go to it through the
//!   participant, and its replies come back as `CtlReply`s.
//! - **Heartbeats are the browser's own.** The participant relays them
//!   instead of its own: each browser heartbeat (with its four-timestamp
//!   exchange, in the browser's clock) is sealed under the participant's
//!   keys and forwarded, and the host's echoes come back opened. The host therefore renews leases only while
//!   the browser itself is alive, maps the browser's clock (not the
//!   gateway's), and lapses the lease when the browser stops.
//! - **Chunks are the browser's own**, forwarded as sent to the part's
//!   command key on its class; the edge fences them. The gateway refuses a
//!   chunk that names another source or a part that is not commandable, and
//!   reports (but still forwards) one the current snapshot would fence.
//! - Samples are forwarded as their publisher sent them: the gateway
//!   subscribes on the stream's class and prepends the subscription id.
//!
//! Dropping a link drops its participant: its heartbeats stop, so the host
//! ends its leases (the edge holds, then goes safe) and nothing is
//! resumed by a reconnect, which attaches a new participant.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::transport::{TrafficClass, payload_bytes, stream_qos};
use crate::wire::codec;
use crate::wire::{
    AuthoritySnapshot, ChannelId, ChannelPriority, Command, CtlError, CtlReply, CtlStatus,
    Fingerprint, PartSpec, ParticipantId, PriorityClass, Role, SessionId, SessionKeys, TenantId,
};
use crate::{MonotonicClock, Participant, ParticipantConfig, SessionError, Transport};
use bytes::Bytes;
use rlmesh_session_core::auth::{self, Credential};
use serde_json::json;
use tokio::sync::{Mutex, broadcast, mpsc, watch};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};
use zenoh::Wait;
use zenoh::bytes::ZBytes;

use crate::gateway::frame;

/// What goes out on a browser socket.
#[derive(Clone, Debug, PartialEq)]
pub enum Outbound {
    /// A binary frame (a tag and a v2 message).
    Binary(Bytes),
    /// A gateway JSON message.
    Text(String),
    /// Close the socket with this code and reason (the session ended).
    Close(u16, String),
}

/// How long the host may be gone before the gateway calls its session
/// ended (a blip, such as a host restart in place, is not an end).
pub const HOST_GONE_GRACE: Duration = Duration::from_secs(2);

/// Forwarding counters, by socket.
#[derive(Debug, Default)]
pub struct Counters {
    /// Frames queued for a browser.
    pub forwarded: AtomicU64,
    /// Frames dropped because a browser's queue was full.
    pub dropped: AtomicU64,
}

/// The gateway's session end: see the module docs.
pub struct Gateway {
    transport: Transport,
    keys: SessionKeys,
    tenant: TenantId,
    session: SessionId,
    credential: Credential,
    observer: Participant,
    state: broadcast::Sender<Bytes>,
    max_priority: PriorityClass,
    /// `state` and `bulk` socket counters.
    pub data: [Counters; 2],
    /// Control socket counters.
    pub control: Counters,
    _state: zenoh::pubsub::Subscriber<()>,
    /// Set once the session host has been gone for [`HOST_GONE_GRACE`].
    ended: watch::Receiver<bool>,
}

fn random_id(prefix: &str) -> Result<ParticipantId, SessionError> {
    let bits: u64 = rand::random();
    Ok(ParticipantId::new(format!("{prefix}-{:08x}", bits as u32))?)
}

/// Marks the session ended once the host's liveliness token has been gone
/// for [`HOST_GONE_GRACE`].
async fn watch_host(mut present: watch::Receiver<bool>, end: watch::Sender<bool>) {
    loop {
        if present.wait_for(|p| !*p).await.is_err() {
            break;
        }
        match tokio::time::timeout(HOST_GONE_GRACE, present.wait_for(|p| *p)).await {
            Ok(Ok(_)) => {}
            Ok(Err(_)) | Err(_) => break,
        }
    }
    warn!("the session host is gone: the session ended");
    let _ = end.send(true);
}

/// The class a data socket carries: `state` (control, realtime and data
/// priorities) or `bulk`.
pub fn socket_of(priority: ChannelPriority) -> &'static str {
    match priority {
        ChannelPriority::Bulk => "bulk",
        _ => "state",
    }
}

impl Gateway {
    /// Attaches the gateway's observer to the session on `transport`, with
    /// `credential` (the session address's key).
    pub async fn open(
        transport: Transport,
        tenant: TenantId,
        session: SessionId,
        credential: Credential,
        max_priority: PriorityClass,
    ) -> Result<Arc<Self>, SessionError> {
        let keys = SessionKeys::new(&tenant, session);
        let mut config = ParticipantConfig::new(
            tenant.clone(),
            session,
            random_id("gateway")?,
            vec![Role::Observer],
            PriorityClass::Scripted,
        );
        config.name = "rlmesh-gateway".into();
        config.credential = Some(credential.clone());
        let observer = Participant::attach(transport.clone(), config).await?;
        info!(gateway = %observer.id(), %session, "attached");
        let (state, _) = broadcast::channel(64);
        let verifier = std::sync::Mutex::new(observer.host_verifier().ok_or(SessionError::Closed)?);
        let raw_state = transport
            .control()
            .declare_subscriber(keys.host_state())
            .callback({
                let (state, key) = (state.clone(), keys.host_state());
                move |sample| {
                    let signed = payload_bytes(sample.payload());
                    let mut verifier = verifier
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    match auth::open_state(&mut verifier, &key, &signed) {
                        // No receivers is fine.
                        Ok(snapshot) => {
                            let _ = state.send(Bytes::from(codec::encode_state(&snapshot)));
                        }
                        Err(err) => debug!(%err, "authority update dropped"),
                    }
                }
            })
            .await?;
        let (end, ended) = watch::channel(false);
        tokio::spawn(watch_host(observer.host_present(), end));
        Ok(Arc::new(Self {
            transport,
            keys,
            tenant,
            session,
            credential,
            observer,
            state,
            max_priority,
            data: Default::default(),
            control: Counters::default(),
            _state: raw_state,
            ended,
        }))
    }

    /// Becomes true once the session ended: its host has been gone for
    /// [`HOST_GONE_GRACE`]. This gateway serves one session, so it is done.
    pub fn ended(&self) -> watch::Receiver<bool> {
        self.ended.clone()
    }

    /// The host's latest snapshot, as the gateway's observer holds it.
    pub fn view(&self) -> Option<AuthoritySnapshot> {
        self.observer.authority()
    }

    /// The session.
    pub fn session(&self) -> SessionId {
        self.session
    }

    /// The highest class a browser may attach at.
    pub fn max_priority(&self) -> PriorityClass {
        self.max_priority
    }

    /// Subscribes `sink` to the stream at `path` (`{robot}/{part}` or
    /// `{robot}/{part}/_applied`) for a `socket` (`state` or `bulk`)
    /// connection: every sample its publisher sends, as
    /// [`frame::SAMPLE`] frames for subscription `id`. A full sink drops the
    /// frame.
    pub async fn subscribe(
        self: &Arc<Self>,
        path: &str,
        socket: &'static str,
        id: u32,
        sink: mpsc::Sender<Bytes>,
    ) -> Result<(zenoh::pubsub::Subscriber<()>, ChannelId), String> {
        let view = self.view().ok_or("no authority snapshot yet")?;
        let (robot, part_path) = path
            .split_once('/')
            .ok_or_else(|| format!("{path}: not {{robot}}/{{part}}"))?;
        let robot = ParticipantId::new(robot).map_err(|e| e.to_string())?;
        let (part_path, applied) = match part_path.strip_suffix("/_applied") {
            Some(p) => (p, true),
            None => (part_path, false),
        };
        let part = view
            .part_at(&robot, part_path)
            .ok_or_else(|| format!("{path} is not declared"))?;
        let spec = if applied {
            part.applied_stream()
                .ok_or_else(|| format!("{path}: not a commandable part"))?
        } else {
            part.stream()
        }
        .map_err(|e| format!("{path}: {e}"))?;
        if socket_of(spec.priority) != socket {
            return Err(format!(
                "{path} rides the {} socket, not {socket}",
                socket_of(spec.priority)
            ));
        }
        let counters = usize::from(socket == "bulk");
        let gw = Arc::downgrade(self);
        let subscriber = self
            .transport
            .for_priority(spec.priority)
            .declare_subscriber(self.keys.data(&robot, &spec.path))
            .callback(move |sample| {
                // One copy, into the WebSocket message; the bytes are the
                // publisher's.
                let body = payload_bytes(sample.payload());
                let ok = sink.try_send(Bytes::from(frame::sample(id, &body))).is_ok();
                if let Some(gw) = gw.upgrade() {
                    let c = &gw.data[counters];
                    if ok { &c.forwarded } else { &c.dropped }.fetch_add(1, Ordering::Relaxed);
                }
            })
            .await
            .map_err(|e| e.to_string())?;
        Ok((subscriber, spec.channel))
    }

    /// Attaches a browser control connection as a participant of its own at
    /// `priority`; `out` carries its echoes.
    pub async fn attach(
        self: &Arc<Self>,
        priority: PriorityClass,
        out: mpsc::Sender<Outbound>,
    ) -> Result<BrowserLink, SessionError> {
        if priority > self.max_priority {
            return Err(SessionError::Config(format!(
                "{priority:?} is above this gateway's --max-priority {:?}",
                self.max_priority
            )));
        }
        let id = random_id("web")?;
        let mut config = ParticipantConfig::new(
            self.tenant.clone(),
            self.session,
            id.clone(),
            vec![Role::Controller],
            priority,
        );
        config.name = format!("browser via {}", self.observer.id());
        config.credential = Some(self.credential.clone());
        // The participant's own heartbeats would renew leases on the
        // gateway's account: it relays the browser's instead (one of its own
        // may leave before the switch below; it carries no timing sample).
        config.heartbeat_period = Duration::from_secs(3600);
        config.clock = MonotonicClock::new().shared();
        let participant = Participant::attach(self.transport.clone(), config).await?;
        let (echoes, mut opened) = mpsc::unbounded_channel();
        participant.relay_heartbeats(echoes);
        let relay = tokio::spawn(async move {
            while let Some(echo) = opened.recv().await {
                let frame = frame::tagged(frame::ECHO, &codec::encode_echo(&echo));
                let _ = out.try_send(Outbound::Binary(Bytes::from(frame)));
            }
        });
        let link = BrowserLink {
            gw: self.clone(),
            participant: Arc::new(participant),
            id,
            chunks: Mutex::new(BTreeMap::new()),
            bound: Arc::new(Mutex::new(Vec::new())),
            tasks: vec![relay],
        };
        Ok(link)
    }
}

/// One browser control connection's participant: see the module docs.
pub struct BrowserLink {
    gw: Arc<Gateway>,
    participant: Arc<Participant>,
    id: ParticipantId,
    chunks: Mutex<BTreeMap<ChannelId, zenoh::pubsub::Publisher<'static>>>,
    bound: Arc<Mutex<Vec<ChannelId>>>,
    tasks: Vec<JoinHandle<()>>,
}

/// Every commandable part, by id.
fn commandable(view: &AuthoritySnapshot) -> Vec<ChannelId> {
    let mut channels: Vec<ChannelId> = view
        .parts
        .iter()
        .filter(|p| p.command.is_some())
        .map(|p| p.channel)
        .collect();
    channels.sort();
    channels
}

/// Declares an identity binding over `channels` (a controller that commands
/// paths directly, §6.5), when they differ from `bound`.
async fn rebind(
    participant: &Participant,
    bound: &Mutex<Vec<ChannelId>>,
) -> Result<(), SessionError> {
    let Some(view) = participant.authority() else {
        return Ok(());
    };
    let channels = commandable(&view);
    let mut bound = bound.lock().await;
    if channels.is_empty() || *bound == channels {
        return Ok(());
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"rlmesh-gateway/identity/1\0");
    hasher.update(participant.id().as_str().as_bytes());
    for c in &channels {
        hasher.update(&c.0.to_le_bytes());
    }
    let digest = hasher.finalize();
    let mut fingerprint = [0_u8; 8];
    fingerprint.copy_from_slice(&digest.as_bytes()[..8]);
    let paths: Vec<String> = channels
        .iter()
        .filter_map(|c| view.part(*c))
        .map(|p| format!("{}/{}", p.owner, p.path))
        .collect();
    let table = json!({"binding": "identity", "by": "rlmesh-gateway", "parts": paths}).to_string();
    participant
        .declare_binding(Fingerprint(fingerprint), &table, channels.clone(), 0.0)
        .await?;
    debug!(participant = %participant.id(), ?paths, "bound");
    *bound = channels;
    Ok(())
}

fn refusal(err: &SessionError) -> CtlError {
    match err.refusal() {
        Some(refusal) => refusal.clone(),
        None => CtlError::new(CtlStatus::InvalidRequest, format!("gateway: {err}")),
    }
}

impl BrowserLink {
    /// This browser's participant id.
    pub fn id(&self) -> &ParticipantId {
        &self.id
    }

    /// The hello the gateway answers with (JSON).
    pub fn hello(&self, priority: PriorityClass) -> String {
        json!({
            "type": "hello",
            "participant": self.id.as_str(),
            "tenant": self.gw.tenant.as_str(),
            "session": self.gw.session.to_string(),
            "priority": format!("{priority:?}").to_lowercase(),
            "heartbeat_ms": 50,
            "gateway": env!("CARGO_PKG_VERSION"),
        })
        .to_string()
    }

    /// The current snapshot as a [`frame::STATE`] frame (encoded from the
    /// participant's view: the first one a browser sees; later ones are the
    /// host's own bytes).
    pub fn state_frame(&self) -> Option<Bytes> {
        let view = self.participant.authority()?;
        Some(Bytes::from(frame::tagged(
            frame::STATE,
            &codec::encode_state(&view),
        )))
    }

    /// Binds the browser's participant now and again whenever the
    /// commandable parts change, and relays the host's authority updates to
    /// `out`.
    pub async fn start(&mut self, out: mpsc::Sender<Outbound>) -> Result<(), SessionError> {
        rebind(&self.participant, &self.bound).await?;
        let (participant, bound) = (self.participant.clone(), self.bound.clone());
        let mut view = self.participant.watch();
        self.tasks.push(tokio::spawn(async move {
            while view.changed().await.is_ok() {
                if let Err(err) = rebind(&participant, &bound).await {
                    warn!(participant = %participant.id(), %err, "binding");
                }
            }
        }));
        let mut updates = self.gw.state.subscribe();
        let gw = self.gw.clone();
        self.tasks.push(tokio::spawn(async move {
            loop {
                match updates.recv().await {
                    Ok(state) => {
                        let frame =
                            Outbound::Binary(Bytes::from(frame::tagged(frame::STATE, &state)));
                        let c = &gw.control;
                        if out.try_send(frame).is_ok() {
                            &c.forwarded
                        } else {
                            &c.dropped
                        }
                        .fetch_add(1, Ordering::Relaxed);
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        gw.control.dropped.fetch_add(n, Ordering::Relaxed);
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        }));
        Ok(())
    }

    /// Handles one binary frame from the browser; returns what to send back
    /// (a reply, or a JSON report).
    pub async fn on_frame(&self, frame_bytes: Bytes) -> Option<Outbound> {
        let (&tag, _) = frame_bytes.split_first()?;
        match tag {
            frame::HEARTBEAT => self.on_heartbeat(frame_bytes.slice(1..)).await,
            frame::CHUNK => self.on_chunk(frame_bytes.slice(1..)).await,
            frame::REQUEST => {
                let id = frame::read_id(&frame_bytes)?;
                Some(self.on_request(id, &frame_bytes[5..]).await)
            }
            other => Some(report("error", format!("unexpected frame 0x{other:02x}"))),
        }
    }

    async fn on_heartbeat(&self, body: Bytes) -> Option<Outbound> {
        match codec::decode_heartbeat(&body) {
            Ok(hb) if hb.participant == self.id => {
                if let Err(err) = self.participant.send_heartbeat(&hb).await {
                    warn!(participant = %self.id, %err, "heartbeat");
                }
                None
            }
            Ok(hb) => Some(report(
                "error",
                format!(
                    "a heartbeat for {} on {}'s connection",
                    hb.participant, self.id
                ),
            )),
            Err(err) => Some(report("error", format!("heartbeat: {err}"))),
        }
    }

    async fn on_chunk(&self, body: Bytes) -> Option<Outbound> {
        let chunk = match codec::decode_chunk(body.clone()) {
            Ok(chunk) => chunk,
            Err(err) => return Some(report("error", format!("chunk: {err}"))),
        };
        if chunk.source != self.id {
            return Some(report(
                "error",
                format!("a chunk from {} on {}'s connection", chunk.source, self.id),
            ));
        }
        let view = self.participant.authority()?;
        let Some(part) = view.part(chunk.channel).filter(|p| p.command.is_some()) else {
            return Some(report(
                "error",
                format!("{} is not a commandable part", chunk.channel),
            ));
        };
        let verdict = view
            .check_fence(chunk.channel, chunk.lease_token, chunk.epoch)
            .err();
        if let Err(err) = self.publish_chunk(part, &view, body).await {
            return Some(report("error", format!("chunk: {err}")));
        }
        verdict.map(|fence| {
            Outbound::Text(
                json!({
                    "type": "fence",
                    "channel": chunk.channel.0,
                    "token": chunk.lease_token.0,
                    "reason": fence.to_string(),
                    "forwarded": true,
                })
                .to_string(),
            )
        })
    }

    async fn publish_chunk(
        &self,
        part: &PartSpec,
        view: &AuthoritySnapshot,
        body: Bytes,
    ) -> Result<(), SessionError> {
        let mut chunks = self.chunks.lock().await;
        let publisher = match chunks.entry(part.channel) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => {
                let stream = part.stream().map_err(|reason| SessionError::WrongChannel {
                    channel: part.channel,
                    reason,
                })?;
                let qos = stream_qos(&stream, view.clock_mode);
                let publisher = self
                    .gw
                    .transport
                    .session(TrafficClass::of(part.priority))
                    .declare_publisher(self.gw.keys.commands(&part.owner, part.path.as_str()))
                    .priority(qos.priority)
                    .reliability(qos.reliability)
                    .congestion_control(qos.congestion)
                    .express(qos.express)
                    .await?;
                e.insert(publisher)
            }
        };
        publisher.put(ZBytes::from(body)).wait()?;
        Ok(())
    }

    async fn on_request(&self, id: u32, body: &[u8]) -> Outbound {
        let request = match codec::decode_request(body) {
            Ok(request) => request,
            Err(err) => return report("error", format!("request: {err}")),
        };
        let p = &self.participant;
        let result: Result<Vec<crate::wire::Lease>, CtlError> = if request.from != self.id {
            Err(CtlError::new(
                CtlStatus::Forbidden,
                format!("{} cannot send verbs as {}", self.id, request.from),
            ))
        } else {
            match request.command {
                Command::GrantLease {
                    channels,
                    release,
                    ttl_ns,
                    ..
                } => p
                    .grant_leases(channels, release, ttl_ns.map(Duration::from_nanos))
                    .await
                    .map_err(|e| refusal(&e)),
                Command::ReleaseLease { channels, token } => p
                    .release_leases(channels, token)
                    .await
                    .map(|_| Vec::new())
                    .map_err(|e| refusal(&e)),
                Command::Arm => p.arm().await.map(|_| Vec::new()).map_err(|e| refusal(&e)),
                Command::Disarm => p
                    .disarm()
                    .await
                    .map(|_| Vec::new())
                    .map_err(|e| refusal(&e)),
                Command::Estop { reason, .. } => p
                    .estop(reason)
                    .await
                    .map(|_| Vec::new())
                    .map_err(|e| refusal(&e)),
                other => Err(CtlError::new(
                    CtlStatus::Forbidden,
                    format!("the gateway does not forward {}", verb_name(&other)),
                )),
            }
        };
        let Some(state) = p.authority() else {
            return report("error", "no authority snapshot");
        };
        let (result, leases) = match result {
            Ok(leases) => (Ok(()), leases),
            Err(err) => (Err(err), Vec::new()),
        };
        let reply = CtlReply {
            request_id: 0,
            result,
            state,
            leases,
        };
        Outbound::Binary(Bytes::from(frame::with_id(
            frame::REPLY,
            id,
            &codec::encode_reply(&reply),
        )))
    }
}

impl Drop for BrowserLink {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
        info!(participant = %self.id, "browser left");
    }
}

fn verb_name(command: &Command) -> &'static str {
    match command {
        Command::Attach { .. } => "attach",
        Command::Detach => "detach",
        Command::DeclareParts { .. } => "declare_parts",
        Command::DeclareBinding { .. } => "declare_binding",
        Command::DemandAck { .. } => "demand_ack",
        Command::Clear => "clear",
        Command::EnterSafe { .. } => "enter_safe",
        Command::Ready { .. } => "ready",
        Command::Decided { .. } => "decided",
        _ => "this verb",
    }
}

/// A gateway JSON report.
pub fn report(kind: &str, message: impl Into<String>) -> Outbound {
    Outbound::Text(json!({"type": kind, "message": message.into()}).to_string())
}
