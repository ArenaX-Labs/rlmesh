//! The transport: one Zenoh session per traffic class.
//!
//! A congested link must never delay control. Zenoh multiplexes every
//! priority of one session over one link per peer, so over TCP a camera
//! filling the socket buffer delays heartbeats, lease renewals and commands
//! behind it (spike step 3: false lease lapses on one shared link). Zenoh's
//! own fix, several links pinned to priority ranges, needs its
//! `transport_multilink` feature, which pulls `rsa` (RUSTSEC-2023-0071).
//!
//! Instead, a participant's [`Transport`] opens **one Zenoh client session
//! per [`TrafficClass`]**, each its own TCP connection to the host, so each
//! class can only starve itself:
//!
//! | Class | Zenoh priority | Reliability | Congestion | Express | Carries |
//! |---|---|---|---|---|---|
//! | [`TrafficClass::Control`] | RealTime | reliable | BLOCK | yes | liveliness, `ctl`, virtual ticks, the authority state, device views, key deliveries, heartbeats and echoes (those five DROP, see below), chunks of `priority: control` parts |
//! | [`TrafficClass::Realtime`] | InteractiveHigh | best effort | DROP | yes | small state samples, chunks, `_applied` |
//! | [`TrafficClass::Data`] | Data | best effort; reliable for events | DROP; BLOCK for events | no | small custom kinds, events, history replies |
//! | [`TrafficClass::Bulk`] | DataLow | best effort | DROP | no | cameras, depth, large custom kinds |
//!
//! Every session, the host's included, queues up to
//! [`REALTIME_QUEUE_BATCHES`] transmit batches per link for the Realtime
//! class's priority, where Zenoh's default of 2 drops samples in bursts.
//!
//! The host listens on **one** port with one session, which every class
//! session of every participant dials: a host is one routing domain. Each
//! key belongs to one class (both ends pick its session by the same rule),
//! so each class session subscribes only its own class's keys and the host
//! sends each message only on the connection of the class it belongs to.
//!
//! These settings are applied per publisher by
//! [`stream_qos`]. Two refinements: in a virtual-clock session every data
//! stream is reliable with BLOCK (the tick barrier counts on seqs);
//! and heartbeats and their echoes are DROP rather than BLOCK, because a
//! stale timing sample is worthless and the echo is sent from a transport
//! callback, which must never block. The host's own state, views and key
//! deliveries are DROP too: Zenoh fans a put out to every subscriber from the
//! calling thread, so BLOCK would let one peer that stopped reading hold the
//! host task, and each is sent again until it lands. Zenoh 1.10 treats
//! reliability as a marker that selects the link (a reliable link for reliable data, a
//! best-effort link such as UDP when one exists); over the TCP and Unix
//! socket links the session uses, every message rides a reliable link, so
//! best-effort streams get their real-time behaviour from DROP congestion
//! control and from never being retransmitted at the session level.
//!
//! [`Layout::Shared`] uses one session for every class instead: the host's
//! listener, in-process tests, Unix sockets, and QUIC (whose multistream
//! already gives every priority its own stream inside one connection).
//! Priorities are still set on every message, so the shared layout keeps
//! Zenoh's own ordering.
//!
//! The lockstep class arrives with the Gym view (wave 5).

use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use crate::wire::codec::Encoded;
use crate::wire::{ChannelPriority, ClockMode, StreamSpec};
use bytes::Bytes;
use tracing::{debug, warn};
use zenoh::bytes::ZBytes;
use zenoh::qos::{CongestionControl, Priority, Reliability};

use crate::error::SessionError;

mod links;

use links::{Cut, LinkGuard};

/// How long [`Transport::close`] waits on Zenoh before each step that
/// forces it: cutting the links with bytes waiting, then every link, then
/// returning without it.
const CLOSE_STEP: Duration = Duration::from_secs(1);

/// A traffic class: a separate Zenoh session in [`Layout::PerClass`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TrafficClass {
    /// Session control and `priority: control` parts.
    Control,
    /// Small state samples, chunks and `_applied`.
    Realtime,
    /// Events, small custom kinds and history replies.
    Data,
    /// Cameras, depth and large custom kinds.
    Bulk,
}

impl TrafficClass {
    /// Every class, in session order.
    pub const ALL: [Self; 4] = [Self::Control, Self::Realtime, Self::Data, Self::Bulk];

    /// The class a channel priority rides.
    pub fn of(priority: ChannelPriority) -> Self {
        match priority {
            ChannelPriority::Control => Self::Control,
            ChannelPriority::Realtime => Self::Realtime,
            ChannelPriority::Data => Self::Data,
            ChannelPriority::Bulk => Self::Bulk,
        }
    }

    fn index(self) -> usize {
        self as usize
    }

    /// The class's name ("control", "realtime", "data", "bulk").
    pub fn name(self) -> &'static str {
        match self {
            Self::Control => "control",
            Self::Realtime => "realtime",
            Self::Data => "data",
            Self::Bulk => "bulk",
        }
    }

    /// The class's QoS.
    pub fn qos(self) -> ClassQos {
        let (priority, reliability, congestion, express) = match self {
            Self::Control => (
                Priority::RealTime,
                Reliability::Reliable,
                CongestionControl::Block,
                true,
            ),
            Self::Realtime => (
                Priority::InteractiveHigh,
                Reliability::BestEffort,
                CongestionControl::Drop,
                true,
            ),
            Self::Data => (
                Priority::Data,
                Reliability::BestEffort,
                CongestionControl::Drop,
                false,
            ),
            Self::Bulk => (
                Priority::DataLow,
                Reliability::BestEffort,
                CongestionControl::Drop,
                false,
            ),
        };
        ClassQos {
            priority,
            reliability,
            congestion,
            express,
        }
    }
}

/// How one publisher's messages travel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClassQos {
    /// The Zenoh priority.
    pub priority: Priority,
    /// Reliable or best effort.
    pub reliability: Reliability,
    /// What a full queue does.
    pub congestion: CongestionControl,
    /// Skip batching.
    pub express: bool,
}

/// The QoS of a stream's publisher: its class's, except that events are
/// reliable with BLOCK, and so is every stream in a virtual-clock session.
pub fn stream_qos(spec: &StreamSpec, clock: ClockMode) -> ClassQos {
    let mut qos = TrafficClass::of(spec.priority).qos();
    if spec.event || clock == ClockMode::Virtual {
        qos.reliability = Reliability::Reliable;
        qos.congestion = CongestionControl::Block;
    }
    qos
}

/// Declares a publisher on `key` in `class`'s session with `qos`.
pub(crate) async fn declare_publisher(
    transport: &Transport,
    class: TrafficClass,
    key: String,
    qos: ClassQos,
) -> Result<zenoh::pubsub::Publisher<'static>, crate::SessionError> {
    Ok(transport
        .session(class)
        .declare_publisher(key)
        .priority(qos.priority)
        .reliability(qos.reliability)
        .congestion_control(qos.congestion)
        .express(qos.express)
        .await?)
}

/// Heartbeats and their echoes: dropped rather than queued under congestion
/// (a stale timing sample is worthless, and the echo callback must never
/// block), sent at once.
pub(crate) const HEARTBEAT_CONGESTION: CongestionControl = CongestionControl::Drop;
/// Control verbs and virtual ticks: never dropped.
pub(crate) const CONTROL_CONGESTION: CongestionControl = CongestionControl::Block;
/// What the host task publishes and sends again until it lands (the
/// authority state, device views, key deliveries): dropped for a link that
/// stopped reading rather than waited on, which would hold up every other
/// participant.
pub(crate) const RESENT_CONGESTION: CongestionControl = CongestionControl::Drop;
/// The Zenoh priority of session control traffic.
pub(crate) const CONTROL_PRIORITY: Priority = Priority::RealTime;

/// The Zenoh priority a stream's class maps to.
pub fn zenoh_priority(priority: ChannelPriority) -> Priority {
    TrafficClass::of(priority).qos().priority
}

/// Where one session listens and what it dials.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Endpoints {
    /// Zenoh endpoints to listen on (`tcp/0.0.0.0:7447`).
    pub listen: Vec<String>,
    /// Zenoh endpoints to connect to.
    pub connect: Vec<String>,
}

/// How classes map onto Zenoh sessions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Layout {
    /// One session for every class.
    Shared(Endpoints),
    /// One session per class, in [`TrafficClass::ALL`] order. Listening
    /// nowhere, each joins as a client (see [`TransportConfig::per_class_tcp`]).
    PerClass([Endpoints; 4]),
}

/// Link settings applied to every session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransportTuning {
    /// The Zenoh transport lease this side advertises: how long the other
    /// end keeps a link on which this side has gone silent. Zenoh's default
    /// is 10 s; session heartbeats are the real liveness signal, so sessions
    /// keep it short. It does not bound how long this side keeps a peer:
    /// that is the peer's own advertised lease.
    pub lease: Duration,
    /// Keep-alives per lease period.
    pub keep_alive: u32,
    /// How long a DROP message waits for a free transmit batch before it is
    /// dropped (Zenoh's default, 1 ms, stalls a 1 kHz publisher on a
    /// congested link). `None` keeps Zenoh's default.
    pub drop_wait: Option<Duration>,
    /// Join as a Zenoh client: listen nowhere and reach everyone through
    /// the host dialed, which routes for its clients. A peer (the default)
    /// talks to the other peers directly, once the gossip of the peers it
    /// dials tells it where they listen.
    pub client: bool,
    /// How long a client's open keeps dialing a host that does not listen
    /// yet before it fails; `None` fails at once, as Zenoh clients do.
    /// Peers always keep dialing in the background.
    pub connect_timeout: Option<Duration>,
    /// How long one link's open (its handshake, once the TCP connection is
    /// up) may take before Zenoh gives up on it; `None` keeps Zenoh's 10 s.
    pub open_timeout: Option<Duration>,
    /// The embedded host's link gate (Zenoh `usrpwd`).
    pub link: Option<LinkAuth>,
    /// A listening host's bound on the bytes each accepted TCP link holds
    /// unsent (`TCP_NOTSENT_LOWAT`, Linux only): a link that can't keep up
    /// then waits in Zenoh's queues, where congestion control drops what is
    /// stale, instead of in the socket. `None` leaves the kernel's default.
    pub notsent_lowat: Option<u32>,
}

/// The `usrpwd` gate of an embedded host's links: the password
/// comes from the session's key ([`crate::auth::address::link_password`]),
/// so only its holders link to the host at all.
#[derive(Clone, PartialEq, Eq)]
pub struct LinkAuth {
    /// The password this side presents.
    pub password: zeroize::Zeroizing<String>,
    /// A host's `user:password` file, which every link it accepts must
    /// match. Zenoh reads it at open: the caller may delete it then.
    pub dictionary: Option<PathBuf>,
}

impl std::fmt::Debug for LinkAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LinkAuth")
            .field("dictionary", &self.dictionary)
            .finish_non_exhaustive()
    }
}

/// The `usrpwd` user every link of an embedded host presents.
pub const LINK_USER: &str = "rlmesh";

impl Default for TransportTuning {
    fn default() -> Self {
        Self {
            lease: Duration::from_secs(1),
            keep_alive: 4,
            drop_wait: None,
            client: false,
            connect_timeout: None,
            open_timeout: None,
            link: None,
            notsent_lowat: None,
        }
    }
}

/// Everything [`Transport::open`] needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransportConfig {
    /// How classes map onto sessions.
    pub layout: Layout,
    /// Link settings.
    pub tuning: TransportTuning,
}

fn tcp(addr: SocketAddr) -> String {
    format!("tcp/{addr}")
}

impl TransportConfig {
    /// One session for every class.
    pub fn shared(listen: &[&str], connect: &[&str]) -> Self {
        Self {
            layout: Layout::Shared(Endpoints {
                listen: listen.iter().map(ToString::to_string).collect(),
                connect: connect.iter().map(ToString::to_string).collect(),
            }),
            tuning: TransportTuning::default(),
        }
    }

    /// A session host's listener: one session on TCP `listen` (port 0: the
    /// OS picks it), which every class of every participant dials
    /// ([`Self::per_class_tcp`]).
    pub fn host_tcp(listen: SocketAddr) -> Self {
        Self::shared(&[&tcp(listen)], &[])
    }

    /// A participant's link to the host listening at `host`: one client
    /// session per class, each its own TCP connection to that one port. The
    /// usual way to join a session off-host.
    pub fn per_class_tcp(host: SocketAddr) -> Self {
        Self {
            layout: Layout::PerClass(std::array::from_fn(|_| Endpoints {
                listen: Vec::new(),
                connect: vec![tcp(host)],
            })),
            tuning: TransportTuning {
                client: true,
                ..TransportTuning::default()
            },
        }
    }

    /// The same layout with other tuning.
    #[must_use]
    pub fn with_tuning(mut self, tuning: TransportTuning) -> Self {
        self.tuning = tuning;
        self
    }
}

/// Transmit batches each link may queue for [`TrafficClass::Realtime`]
/// traffic (Zenoh's most, against its default of 2).
///
/// Realtime samples are express, so each takes a batch of its own, and a
/// DROP sample that finds none free within the drop wait is dropped, after
/// which Zenoh drops the samples behind it while the queue stays congested:
/// a burst of a few dozen samples back to back lost several on loopback with
/// 2 batches. Batches are allocated as a burst first needs them and kept, a
/// batch being the link's batch size (64 KiB on TCP): up to 1 MiB per link
/// that carries Realtime traffic, 896 KiB more than the default. Only the
/// Realtime priority's queue grows; the other classes keep Zenoh's 2.
pub const REALTIME_QUEUE_BATCHES: usize = 16;

/// The key of `priority`'s queue under `transport/link/tx/queue/size`.
fn queue_name(priority: Priority) -> &'static str {
    match priority {
        Priority::RealTime => "real_time",
        Priority::InteractiveHigh => "interactive_high",
        Priority::InteractiveLow => "interactive_low",
        Priority::DataHigh => "data_high",
        Priority::Data => "data",
        Priority::DataLow => "data_low",
        Priority::Background => "background",
    }
}

/// A peer-mode Zenoh config (client mode when `tuning.client`, which listens
/// nowhere): multicast scouting off, so it listens on `listen` and dials
/// `connect` and nothing else (sessions on one host never discover each
/// other by accident).
pub fn zenoh_config(
    endpoints: &Endpoints,
    tuning: &TransportTuning,
) -> Result<zenoh::Config, SessionError> {
    let mut config = zenoh::Config::default();
    let set = |config: &mut zenoh::Config, key: &str, value: String| {
        config
            .insert_json5(key, &value)
            .map_err(|err| SessionError::Config(format!("zenoh {key}: {err}")))
    };
    let list = |items: &[String]| {
        let quoted: Vec<String> = items.iter().map(|item| format!("{item:?}")).collect();
        format!("[{}]", quoted.join(","))
    };
    let (mode, listen) = if tuning.client {
        ("\"client\"", list(&[]))
    } else {
        ("\"peer\"", list(&endpoints.listen))
    };
    set(&mut config, "mode", mode.into())?;
    set(&mut config, "scouting/multicast/enabled", "false".into())?;
    set(&mut config, "listen/endpoints", listen)?;
    set(&mut config, "connect/endpoints", list(&endpoints.connect))?;
    if let (true, Some(timeout)) = (tuning.client, tuning.connect_timeout) {
        let timeout_ms = u64::try_from(timeout.as_millis())
            .unwrap_or(u64::MAX)
            .max(1);
        set(&mut config, "connect/timeout_ms", timeout_ms.to_string())?;
        // Zenoh's first retry comes after a second; a host coming up is
        // usually there sooner.
        set(
            &mut config,
            "connect/retry",
            "{period_init_ms:100,period_max_ms:1000}".into(),
        )?;
    }
    if let Some(timeout) = tuning.open_timeout {
        let timeout_ms = u64::try_from(timeout.as_millis())
            .unwrap_or(u64::MAX)
            .max(1);
        set(
            &mut config,
            "transport/unicast/open_timeout",
            timeout_ms.to_string(),
        )?;
    }
    let lease_ms = u64::try_from(tuning.lease.as_millis())
        .unwrap_or(u64::MAX)
        .max(1);
    set(&mut config, "transport/link/tx/lease", lease_ms.to_string())?;
    set(
        &mut config,
        "transport/link/tx/keep_alive",
        tuning.keep_alive.max(1).to_string(),
    )?;
    set(
        &mut config,
        &format!(
            "transport/link/tx/queue/size/{}",
            queue_name(TrafficClass::Realtime.qos().priority)
        ),
        REALTIME_QUEUE_BATCHES.to_string(),
    )?;
    if let Some(wait) = tuning.drop_wait {
        set(
            &mut config,
            "transport/link/tx/queue/congestion_control/drop/wait_before_drop",
            wait.as_micros().to_string(),
        )?;
    }
    if let Some(link) = &tuning.link {
        set(
            &mut config,
            "transport/auth/usrpwd/user",
            format!("{LINK_USER:?}"),
        )?;
        let password = zeroize::Zeroizing::new(format!("{:?}", link.password.as_str()));
        set(
            &mut config,
            "transport/auth/usrpwd/password",
            password.to_string(),
        )?;
        if let Some(dictionary) = &link.dictionary {
            set(
                &mut config,
                "transport/auth/usrpwd/dictionary_file",
                format!("{:?}", dictionary.display().to_string()),
            )?;
        }
    }
    Ok(config)
}

/// The open Zenoh sessions of one participant (or host), by class. Cheap to
/// clone: sessions are shared handles.
#[derive(Clone)]
pub struct Transport {
    sessions: Arc<Sessions>,
    shared: bool,
    client: bool,
    /// The link gate it presented, if any ([`Self::link`]).
    link: Option<LinkAuth>,
    /// The sessions of a session host this process runs, which its own
    /// participants ride (see [`Self::is_in_process`]).
    in_process: bool,
    /// What watches the links its listeners accepted, if it listens.
    guard: Option<Arc<LinkGuard>>,
    /// The first endpoint its control session dials, without its protocol
    /// (`10.0.0.2:7500`), if any ([`Self::dialed`]).
    dialed: Option<String>,
}

impl std::fmt::Debug for Transport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transport")
            .field("shared", &self.shared)
            .field(
                "zids",
                &self
                    .sessions
                    .all()
                    .iter()
                    .map(|s| s.zid().to_string())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl Transport {
    /// Opens the sessions `config` describes, off the calling thread (see
    /// [`off_thread`]).
    pub async fn open(config: &TransportConfig) -> Result<Self, SessionError> {
        match tokio::runtime::Handle::try_current() {
            Ok(runtime) if multi_thread(&runtime) => {
                let config = config.clone();
                off_thread(move || runtime.block_on(Self::open_here(&config))).await?
            }
            _ => Self::open_here(config).await,
        }
    }

    async fn open_here(config: &TransportConfig) -> Result<Self, SessionError> {
        match &config.layout {
            Layout::Shared(endpoints) => {
                let session = zenoh::open(zenoh_config(endpoints, &config.tuning)?).await?;
                debug!(zid = %session.zid(), "transport open (shared)");
                let guard = guard(std::slice::from_ref(&session), config).await;
                Ok(Self {
                    client: config.tuning.client,
                    link: config.tuning.link.clone(),
                    guard,
                    dialed: first_dialed(endpoints),
                    ..Self::from_session(session)
                })
            }
            Layout::PerClass(classes) => {
                // Class sessions that listen nowhere are clients: as peers,
                // the host's gossip would link them to each other.
                let tuning = TransportTuning {
                    client: config.tuning.client
                        || classes.iter().all(|class| class.listen.is_empty()),
                    ..config.tuning.clone()
                };
                let mut opened = Vec::with_capacity(4);
                for (class, endpoints) in TrafficClass::ALL.iter().zip(classes) {
                    let session = zenoh::open(zenoh_config(endpoints, &tuning)?).await?;
                    debug!(class = class.name(), zid = %session.zid(), "transport open");
                    opened.push(session);
                }
                let guard = guard(&opened, config).await;
                let sessions: [zenoh::Session; 4] = opened
                    .try_into()
                    .map_err(|_| SessionError::Config("four class sessions".into()))?;
                Ok(Self {
                    sessions: Arc::new(Sessions(Some(sessions))),
                    shared: false,
                    client: tuning.client,
                    link: config.tuning.link.clone(),
                    in_process: false,
                    guard,
                    dialed: first_dialed(&classes[TrafficClass::Control.index()]),
                })
            }
        }
    }

    /// Wraps one already-open session for every class.
    pub fn from_session(session: zenoh::Session) -> Self {
        Self {
            sessions: Arc::new(Sessions(Some([
                session.clone(),
                session.clone(),
                session.clone(),
                session,
            ]))),
            shared: true,
            client: false,
            link: None,
            in_process: false,
            guard: None,
            dialed: None,
        }
    }

    /// Whether every class shares one session.
    pub fn is_shared(&self) -> bool {
        self.shared
    }

    /// Whether these are the sessions of a session host this process runs,
    /// ridden by a participant of it: it opens no links of its own (no
    /// lease a starved process could let lapse) and closes none.
    pub fn is_in_process(&self) -> bool {
        self.in_process
    }

    /// This transport, ridden by a participant of the host it serves.
    pub(crate) fn ridden(&self) -> Self {
        Self {
            in_process: true,
            ..self.clone()
        }
    }

    /// Whether it joined as a Zenoh client (see [`TransportTuning::client`]).
    pub fn is_client(&self) -> bool {
        self.client
    }

    /// The link gate it opened with ([`TransportTuning::link`]): what
    /// another link of this participant to its host presents.
    pub fn link(&self) -> Option<&LinkAuth> {
        self.link.as_ref()
    }

    /// Where its control session dials the host, without the protocol
    /// (`10.0.0.2:7500`): what a message about that host names it by. `None`
    /// for a transport that dials nothing, or rides a host this process runs.
    pub(crate) fn dialed(&self) -> Option<&str> {
        self.dialed.as_deref().filter(|_| !self.in_process)
    }

    /// The session of `class`.
    pub fn session(&self, class: TrafficClass) -> &zenoh::Session {
        &self.sessions.all()[class.index()]
    }

    /// The control session.
    pub fn control(&self) -> &zenoh::Session {
        self.session(TrafficClass::Control)
    }

    /// The session a channel priority rides.
    pub fn for_priority(&self, priority: ChannelPriority) -> &zenoh::Session {
        self.session(TrafficClass::of(priority))
    }

    /// Closes every session (once each, in the shared layout).
    ///
    /// A listening transport's close returns within 3 s whatever its peers
    /// do: Zenoh closes each link with a last message queued behind what the
    /// link still holds, and waits for it without a deadline on a link
    /// nothing reads. So the sessions close on a thread of their own; a link
    /// still holding bytes after a second is cut at the socket (which ends
    /// that wait), every remaining link after another, and after a third the
    /// close returns and leaves Zenoh to finish.
    pub async fn close(&self) -> Result<(), SessionError> {
        let sessions: &[zenoh::Session] = if self.shared {
            &self.sessions.all()[..1]
        } else {
            self.sessions.all()
        };
        let Some(guard) = &self.guard else {
            for session in sessions {
                session.close().await?;
            }
            return Ok(());
        };
        let (done, mut closed) = tokio::sync::oneshot::channel();
        let sessions = sessions.to_vec();
        std::thread::Builder::new()
            .name("rlmesh-close".into())
            .spawn(move || {
                use zenoh::Wait;
                let _ = done.send(sessions.iter().try_for_each(|s| s.close().wait()));
            })?;
        for force in [Some(Cut::Waiting), Some(Cut::All), None] {
            if let Ok(result) = tokio::time::timeout(CLOSE_STEP, &mut closed).await {
                return match result {
                    Ok(closed) => closed.map_err(SessionError::from),
                    Err(_) => Ok(()),
                };
            }
            match force {
                Some(which) => {
                    let cut = guard.cut(which);
                    warn!(
                        cut,
                        ?which,
                        "links still open {CLOSE_STEP:?} into closing: cut"
                    );
                }
                None => warn!("Zenoh did not finish closing: left to it"),
            }
        }
        Ok(())
    }
}

/// The first endpoint `endpoints` dials, without its protocol.
fn first_dialed(endpoints: &Endpoints) -> Option<String> {
    let endpoint = endpoints.connect.first()?;
    let address = endpoint
        .split_once('/')
        .map_or(endpoint.as_str(), |(_, a)| a);
    Some(address.to_string())
}

/// A transport's sessions, closed on a thread of their own when the last
/// handle goes: Zenoh closes a session whose last handle drops in a nested
/// `block_on` on the dropping thread (see [`off_thread`]).
struct Sessions(Option<[zenoh::Session; 4]>);

impl Sessions {
    fn all(&self) -> &[zenoh::Session; 4] {
        self.0
            .as_ref()
            .unwrap_or_else(|| unreachable!("taken only by drop"))
    }
}

impl Drop for Sessions {
    fn drop(&mut self) {
        let Some(sessions) = self.0.take() else {
            return;
        };
        // A thread of its own rather than the runtime's blocking pool: a
        // pool that is shutting down drops what it is handed in place,
        // holding a lock Zenoh's close then takes. A thread that cannot
        // start drops them here.
        let _ = std::thread::Builder::new()
            .name("rlmesh-close".into())
            .spawn(move || drop(sessions));
    }
}

/// Runs `work`, which blocks on Zenoh, on the blocking pool and waits for
/// it.
///
/// Zenoh resolves an open (and a close, and a few queries of its state) with
/// a nested `block_on` on the calling thread. On a thread that is itself
/// blocking on a future (a program's main future, a binding's call, a test),
/// that nested `block_on` parks on the same thread parker as the outer one
/// and takes every wakeup meant for the futures the outer one polls: a reply,
/// a timer or a watch that fires meanwhile is lost, and a future waiting on
/// it is never polled again. A thread of the blocking pool runs no other
/// future, so nothing there can be lost.
pub async fn off_thread<T: Send + 'static>(
    work: impl FnOnce() -> T + Send + 'static,
) -> Result<T, SessionError> {
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|err| SessionError::Zenoh(format!("a blocking Zenoh call failed: {err}").into()))
}

/// Whether `runtime` has a blocking pool Zenoh runs on: Zenoh refuses a
/// current-thread runtime anyway.
fn multi_thread(runtime: &tokio::runtime::Handle) -> bool {
    runtime.runtime_flavor() != tokio::runtime::RuntimeFlavor::CurrentThread
}

/// Opens a Zenoh session off the calling thread (see [`off_thread`]).
pub async fn open_session(config: zenoh::Config) -> Result<zenoh::Session, SessionError> {
    off_thread(move || zenoh::Wait::wait(zenoh::open(config)))
        .await?
        .map_err(SessionError::from)
}

/// Watches the links `sessions` accept on their TCP listeners, when they
/// listen (see [`links`]).
async fn guard(sessions: &[zenoh::Session], config: &TransportConfig) -> Option<Arc<LinkGuard>> {
    let listens = match &config.layout {
        Layout::Shared(endpoints) => !endpoints.listen.is_empty(),
        Layout::PerClass(classes) => classes.iter().any(|c| !c.listen.is_empty()),
    };
    if config.tuning.client || !listens {
        return None;
    }
    let mut ports = Vec::new();
    for session in sessions {
        for locator in session.info().locators().await {
            if locator.protocol().as_str() == "tcp"
                && let Ok(addr) = locator.address().as_str().parse::<SocketAddr>()
            {
                ports.push(addr.port());
            }
        }
    }
    LinkGuard::start(ports, config.tuning.notsent_lowat).map(Arc::new)
}

/// `n` `tcp/127.0.0.1:{port}` endpoints on ports that were free a moment
/// ago, for tests and demos. The OS picks them and they are released before
/// Zenoh binds them, so another process could race for one.
pub fn free_local_tcp_endpoints(n: usize) -> Result<Vec<String>, SessionError> {
    let listeners = (0..n)
        .map(|_| std::net::TcpListener::bind("127.0.0.1:0"))
        .collect::<Result<Vec<_>, _>>()?;
    listeners.iter().map(|l| Ok(tcp(l.local_addr()?))).collect()
}

/// A port on `ip` that was free a moment ago for TCP and UDP both (a host
/// listens on TCP and offers boards UDP on the same number). Same race as
/// [`free_local_tcp_endpoints`].
pub fn free_local_base(ip: IpAddr) -> Result<SocketAddr, SessionError> {
    for _ in 0..64 {
        let tcp = std::net::TcpListener::bind(SocketAddr::new(ip, 0))?;
        let addr = tcp.local_addr()?;
        if std::net::UdpSocket::bind(addr).is_ok() {
            return Ok(addr);
        }
    }
    Err(SessionError::Config("no free local port".into()))
}

/// Opens a session host's listener on a free local port (for tests and
/// demos): the address is what peers dial.
pub async fn open_local_listener(ip: IpAddr) -> Result<(Transport, SocketAddr), SessionError> {
    let mut last = SessionError::Config("no free local port".into());
    for _ in 0..16 {
        let base = free_local_base(ip)?;
        match Transport::open(&TransportConfig::host_tcp(base)).await {
            Ok(transport) => return Ok((transport, base)),
            Err(err) => last = err,
        }
    }
    Err(last)
}

/// An encoded message as one `ZBytes`, gathering its segments without
/// copying them.
pub fn frame_zbytes(encoded: Encoded) -> ZBytes {
    let mut writer = ZBytes::writer();
    for segment in encoded.segments {
        writer.append(ZBytes::from(segment));
    }
    writer.finish()
}

/// Owns a contiguous `ZBytes` for [`Bytes::from_owner`].
struct ZOwner(ZBytes);

impl AsRef<[u8]> for ZOwner {
    fn as_ref(&self) -> &[u8] {
        self.0.slices().next().unwrap_or(&[])
    }
}

/// A received payload as [`Bytes`]: zero-copy when it is one contiguous
/// slice (every payload that crossed a link), one copy otherwise (an
/// in-process delivery of a gathered frame).
pub fn payload_bytes(payload: &ZBytes) -> Bytes {
    let mut slices = payload.slices();
    match (slices.next(), slices.next()) {
        (None, _) => Bytes::new(),
        (Some(_), None) => Bytes::from_owner(ZOwner(payload.clone())),
        (Some(_), Some(_)) => Bytes::from(payload.to_bytes().into_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tuning_lands_in_the_zenoh_config() {
        let tuning = TransportTuning {
            lease: Duration::from_millis(1500),
            keep_alive: 3,
            drop_wait: Some(Duration::from_micros(50)),
            client: false,
            connect_timeout: None,
            open_timeout: Some(Duration::from_secs(25)),
            link: None,
            notsent_lowat: None,
        };
        let endpoints = Endpoints {
            listen: vec!["tcp/127.0.0.1:0".into()],
            connect: vec!["tcp/10.0.0.1:7447".into()],
        };
        let config = zenoh_config(&endpoints, &tuning).expect("every key is valid");
        let get = |key: &str| config.get_json(key).expect(key);
        assert_eq!(get("transport/link/tx/lease"), "1500");
        assert_eq!(get("transport/link/tx/keep_alive"), "3");
        assert_eq!(
            get("transport/link/tx/queue/congestion_control/drop/wait_before_drop"),
            "50"
        );
        assert!(get("connect/endpoints").contains("tcp/10.0.0.1:7447"));
        assert_eq!(get("scouting/multicast/enabled"), "false");
        assert_eq!(get("mode"), "\"peer\"");
        assert_eq!(get("transport/unicast/open_timeout"), "25000");
        assert_eq!(get("transport/link/tx/queue/size/interactive_high"), "16");
        for other in ["control", "real_time", "data", "data_low"] {
            assert_eq!(
                get(&format!("transport/link/tx/queue/size/{other}")),
                "2",
                "{other} keeps Zenoh's default"
            );
        }

        let client = TransportTuning {
            client: true,
            connect_timeout: Some(Duration::from_secs(2)),
            ..TransportTuning::default()
        };
        let config = zenoh_config(&endpoints, &client).expect("every key is valid");
        let get = |key: &str| config.get_json(key).expect(key);
        assert_eq!(get("mode"), "\"client\"");
        assert_eq!(get("listen/endpoints"), "[]");
        assert!(get("connect/endpoints").contains("tcp/10.0.0.1:7447"));
        assert_eq!(get("connect/timeout_ms"), "2000");
    }

    #[test]
    fn every_class_dials_the_one_port_the_host_listens_on() {
        let host = TransportConfig::host_tcp("0.0.0.0:7447".parse().expect("addr"));
        let Layout::Shared(endpoints) = host.layout else {
            panic!("one session");
        };
        assert_eq!(endpoints.listen, vec!["tcp/0.0.0.0:7447"]);
        let joiner = TransportConfig::per_class_tcp("10.0.0.1:7500".parse().expect("addr"));
        assert!(joiner.tuning.client);
        let Layout::PerClass(classes) = joiner.layout else {
            panic!("per class");
        };
        for class in &classes {
            assert_eq!(class.connect, vec!["tcp/10.0.0.1:7500"]);
            assert!(class.listen.is_empty());
        }
        assert_eq!(TrafficClass::of(ChannelPriority::Bulk), TrafficClass::Bulk);
        assert_eq!(zenoh_priority(ChannelPriority::Control), Priority::RealTime);
    }

    #[test]
    fn each_class_maps_to_the_spec_qos_and_events_are_reliable() {
        let stream = |priority| StreamSpec {
            channel: crate::wire::ChannelId(1),
            owner: crate::wire::ParticipantId::new("robot").expect("id"),
            path: "x".into(),
            fields: Vec::new(),
            layout: crate::wire::Layout::Text,
            encoding: crate::wire::Encoding::Raw,
            priority,
            event: false,
            max_age: None,
            applied: false,
        };
        let control = TrafficClass::Control.qos();
        assert_eq!(
            (control.reliability, control.congestion, control.express),
            (Reliability::Reliable, CongestionControl::Block, true)
        );
        let realtime = TrafficClass::Realtime.qos();
        assert_eq!(
            (
                realtime.priority,
                realtime.reliability,
                realtime.congestion,
                realtime.express
            ),
            (
                Priority::InteractiveHigh,
                Reliability::BestEffort,
                CongestionControl::Drop,
                true
            )
        );
        let bulk = TrafficClass::Bulk.qos();
        assert_eq!((bulk.priority, bulk.express), (Priority::DataLow, false));
        let mut event = stream(ChannelPriority::Data);
        assert_eq!(
            stream_qos(&event, ClockMode::Wall).reliability,
            Reliability::BestEffort
        );
        event.event = true;
        let qos = stream_qos(&event, ClockMode::Wall);
        assert_eq!(
            (qos.reliability, qos.congestion, qos.priority),
            (
                Reliability::Reliable,
                CongestionControl::Block,
                Priority::Data
            )
        );
        let camera = stream(ChannelPriority::Bulk);
        assert_eq!(
            stream_qos(&camera, ClockMode::Virtual).congestion,
            CongestionControl::Block,
            "virtual sessions block"
        );
    }

    #[test]
    fn payloads_become_bytes_without_a_copy_when_contiguous() {
        let one = ZBytes::from(vec![1_u8, 2, 3]);
        let bytes = payload_bytes(&one);
        assert_eq!(&bytes[..], &[1, 2, 3]);
        assert_eq!(
            bytes.as_ptr(),
            one.slices().next().expect("slice").as_ptr(),
            "shares the buffer"
        );
        let gathered = frame_zbytes(Encoded {
            segments: vec![
                Bytes::from_static(&[1, 2]),
                Bytes::from_static(&[3, 4]),
                Bytes::from_static(&[5]),
            ],
        });
        assert_eq!(gathered.slices().count(), 3, "segments are not merged");
        assert_eq!(&payload_bytes(&gathered)[..], &[1, 2, 3, 4, 5]);
        assert!(payload_bytes(&ZBytes::new()).is_empty());
    }
}
