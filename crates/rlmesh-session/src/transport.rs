//! The transport: one Zenoh session per traffic class.
//!
//! A congested link must never delay control. Zenoh multiplexes every
//! priority of one session over one link per peer, so over TCP a camera
//! filling the socket buffer delays heartbeats, lease renewals and commands
//! behind it (spike step 3: false lease lapses on one shared link). Zenoh's
//! own fix, several links pinned to priority ranges, needs its
//! `transport_multilink` feature, which pulls `rsa` (RUSTSEC-2023-0071).
//!
//! Instead, a [`Transport`] opens **one Zenoh session per
//! [`TrafficClass`]**, each with its own listen and connect endpoints, so
//! each class rides its own TCP connection and can only starve itself:
//!
//! | Class | Zenoh priority | Reliability | Congestion | Express | Carries |
//! |---|---|---|---|---|---|
//! | [`TrafficClass::Control`] | RealTime | reliable | BLOCK | yes | liveliness, `ctl`, authority state and ticks, heartbeats and echoes (DROP, see below), chunks of `priority: control` parts |
//! | [`TrafficClass::Realtime`] | InteractiveHigh | best effort | DROP | yes | small state samples, chunks, `_applied` |
//! | [`TrafficClass::Data`] | Data | best effort; reliable for events | DROP; BLOCK for events | no | small custom kinds, events, history replies |
//! | [`TrafficClass::Bulk`] | DataLow | best effort | DROP | no | cameras, depth, large custom kinds |
//!
//! That is the v0.2 spec's §10.2 table, applied per publisher by
//! [`stream_qos`]. Two refinements: in a virtual-clock session every data
//! stream is reliable with BLOCK (the tick barrier counts on seqs, §11.3);
//! and heartbeats and their echoes are DROP rather than BLOCK, because a
//! stale timing sample is worthless and the echo is sent from a transport
//! callback, which must never block. Zenoh 1.10 treats reliability as a
//! marker that selects the link (a reliable link for reliable data, a
//! best-effort link such as UDP when one exists); over the TCP and Unix
//! socket links the session uses, every message rides a reliable link, so
//! best-effort streams get their real-time behaviour from DROP congestion
//! control and from never being retransmitted at the session level.
//!
//! [`Layout::Shared`] uses one session for every class instead: in-process
//! tests, Unix sockets, and QUIC (whose multistream already gives every
//! priority its own stream inside one connection). Both ends of a link must
//! use the same layout. Priorities are still set on every message, so the
//! shared layout keeps Zenoh's own ordering.
//!
//! The lockstep class (§10.2, §12.3) arrives with the Gym view (wave 5).

use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::time::Duration;

use crate::wire::codec::Encoded;
use crate::wire::{ChannelPriority, ClockMode, StreamSpec};
use bytes::Bytes;
use tracing::debug;
use zenoh::bytes::ZBytes;
use zenoh::qos::{CongestionControl, Priority, Reliability};

use crate::error::SessionError;

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

    /// The class's QoS (§10.2).
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
/// Control verbs, replies, authority updates and ticks: never dropped.
pub(crate) const CONTROL_CONGESTION: CongestionControl = CongestionControl::Block;
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
    /// One session per class, in [`TrafficClass::ALL`] order.
    PerClass([Endpoints; 4]),
}

/// TLS material for `tls/` and `quic/` endpoints.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TlsFiles {
    /// The CA that signs peers' certificates.
    pub root_ca: PathBuf,
    /// This peer's certificate.
    pub certificate: PathBuf,
    /// Its private key.
    pub private_key: PathBuf,
}

/// Link settings applied to every session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransportTuning {
    /// The Zenoh transport lease: how long a silent peer survives. Zenoh's
    /// default is 10 s; session heartbeats are the real liveness signal, so
    /// sessions keep it short and a hung peer's links are torn down fast.
    pub lease: Duration,
    /// Keep-alives per lease period.
    pub keep_alive: u32,
    /// How long a DROP message waits for a free transmit batch before it is
    /// dropped (Zenoh's default, 1 ms, stalls a 1 kHz publisher on a
    /// congested link). `None` keeps Zenoh's default.
    pub drop_wait: Option<Duration>,
    /// TLS material for `tls/` and `quic/` endpoints.
    pub tls: Option<TlsFiles>,
    /// Join as a Zenoh client: listen nowhere and reach everyone through
    /// the host dialed, which routes for its clients. A peer (the default)
    /// talks to the other peers directly, once the gossip of the peers it
    /// dials tells it where they listen.
    pub client: bool,
    /// How long a client's open keeps dialing a host that does not listen
    /// yet before it fails; `None` fails at once, as Zenoh clients do.
    /// Peers always keep dialing in the background.
    pub connect_timeout: Option<Duration>,
}

impl Default for TransportTuning {
    fn default() -> Self {
        Self {
            lease: Duration::from_secs(1),
            keep_alive: 4,
            drop_wait: None,
            tls: None,
            client: false,
            connect_timeout: None,
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

/// The port offset of each class from a base port in
/// [`TransportConfig::per_class_tcp`].
fn class_port(base: u16, class: TrafficClass) -> Result<u16, SessionError> {
    base.checked_add(class.index() as u16).ok_or_else(|| {
        SessionError::Config(format!("base port {base} leaves no room for 4 classes"))
    })
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

    /// One TCP session per class on consecutive ports: class `i` (in
    /// [`TrafficClass::ALL`] order) listens on `listen.port() + i` and dials
    /// `connect.port() + i`. The usual way to run a session off-host.
    pub fn per_class_tcp(
        listen: Option<SocketAddr>,
        connect: Option<SocketAddr>,
    ) -> Result<Self, SessionError> {
        let mut classes: [Endpoints; 4] = Default::default();
        for class in TrafficClass::ALL {
            let at = |addr: SocketAddr| -> Result<String, SessionError> {
                Ok(tcp(SocketAddr::new(
                    addr.ip(),
                    class_port(addr.port(), class)?,
                )))
            };
            let endpoints = &mut classes[class.index()];
            if let Some(addr) = listen {
                // Port 0 lets the OS pick each class's port.
                endpoints.listen.push(if addr.port() == 0 {
                    tcp(addr)
                } else {
                    at(addr)?
                });
            }
            if let Some(addr) = connect {
                endpoints.connect.push(at(addr)?);
            }
        }
        Ok(Self {
            layout: Layout::PerClass(classes),
            tuning: TransportTuning::default(),
        })
    }

    /// The same layout with other tuning.
    #[must_use]
    pub fn with_tuning(mut self, tuning: TransportTuning) -> Self {
        self.tuning = tuning;
        self
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
    let lease_ms = u64::try_from(tuning.lease.as_millis())
        .unwrap_or(u64::MAX)
        .max(1);
    set(&mut config, "transport/link/tx/lease", lease_ms.to_string())?;
    set(
        &mut config,
        "transport/link/tx/keep_alive",
        tuning.keep_alive.max(1).to_string(),
    )?;
    if let Some(wait) = tuning.drop_wait {
        set(
            &mut config,
            "transport/link/tx/queue/congestion_control/drop/wait_before_drop",
            wait.as_micros().to_string(),
        )?;
    }
    if let Some(tls) = &tuning.tls {
        let path = |p: &PathBuf| format!("{:?}", p.display().to_string());
        set(
            &mut config,
            "transport/link/tls/root_ca_certificate",
            path(&tls.root_ca),
        )?;
        set(
            &mut config,
            "transport/link/tls/listen_certificate",
            path(&tls.certificate),
        )?;
        set(
            &mut config,
            "transport/link/tls/listen_private_key",
            path(&tls.private_key),
        )?;
    }
    Ok(config)
}

/// The open Zenoh sessions of one participant (or host), by class. Cheap to
/// clone: sessions are shared handles.
#[derive(Clone)]
pub struct Transport {
    sessions: [zenoh::Session; 4],
    shared: bool,
    client: bool,
    /// The sessions of a session host this process runs, which its own
    /// participants ride (see [`Self::is_in_process`]).
    in_process: bool,
}

impl std::fmt::Debug for Transport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transport")
            .field("shared", &self.shared)
            .field(
                "zids",
                &self
                    .sessions
                    .iter()
                    .map(|s| s.zid().to_string())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl Transport {
    /// Opens the sessions `config` describes.
    pub async fn open(config: &TransportConfig) -> Result<Self, SessionError> {
        match &config.layout {
            Layout::Shared(endpoints) => {
                let session = zenoh::open(zenoh_config(endpoints, &config.tuning)?).await?;
                debug!(zid = %session.zid(), "transport open (shared)");
                Ok(Self {
                    client: config.tuning.client,
                    ..Self::from_session(session)
                })
            }
            Layout::PerClass(classes) => {
                let mut opened = Vec::with_capacity(4);
                for (class, endpoints) in TrafficClass::ALL.iter().zip(classes) {
                    let session = zenoh::open(zenoh_config(endpoints, &config.tuning)?).await?;
                    debug!(class = class.name(), zid = %session.zid(), "transport open");
                    opened.push(session);
                }
                let sessions: [zenoh::Session; 4] = opened
                    .try_into()
                    .map_err(|_| SessionError::Config("four class sessions".into()))?;
                Ok(Self {
                    sessions,
                    shared: false,
                    client: config.tuning.client,
                    in_process: false,
                })
            }
        }
    }

    /// Wraps one already-open session for every class.
    pub fn from_session(session: zenoh::Session) -> Self {
        Self {
            sessions: [session.clone(), session.clone(), session.clone(), session],
            shared: true,
            client: false,
            in_process: false,
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

    /// The session of `class`.
    pub fn session(&self, class: TrafficClass) -> &zenoh::Session {
        &self.sessions[class.index()]
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
    pub async fn close(&self) -> Result<(), SessionError> {
        let sessions: &[zenoh::Session] = if self.shared {
            &self.sessions[..1]
        } else {
            &self.sessions
        };
        for session in sessions {
            session.close().await?;
        }
        Ok(())
    }
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

/// A base port on `ip` whose next three ports were also free a moment ago
/// (for [`TransportConfig::per_class_tcp`]). Same race as
/// [`free_local_tcp_endpoints`].
pub fn free_local_base(ip: IpAddr) -> Result<SocketAddr, SessionError> {
    for _ in 0..64 {
        let first = std::net::TcpListener::bind(SocketAddr::new(ip, 0))?;
        let base = first.local_addr()?.port();
        if base > u16::MAX - 4 {
            continue;
        }
        let rest: Result<Vec<_>, _> = (1..4)
            .map(|i| std::net::TcpListener::bind(SocketAddr::new(ip, base + i)))
            .collect();
        if rest.is_ok() {
            return Ok(SocketAddr::new(ip, base));
        }
    }
    Err(SessionError::Config(
        "no four consecutive free local ports".into(),
    ))
}

/// Opens per-class TCP listeners on free local ports (for tests and demos):
/// a base from [`free_local_base`], and another one when a port was taken
/// in between. The base is what peers dial.
pub async fn open_local_listeners(ip: IpAddr) -> Result<(Transport, SocketAddr), SessionError> {
    let mut last = SessionError::Config("no free local ports".into());
    for _ in 0..16 {
        let base = free_local_base(ip)?;
        match Transport::open(&TransportConfig::per_class_tcp(Some(base), None)?).await {
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
            tls: Some(TlsFiles {
                root_ca: "ca.pem".into(),
                certificate: "cert.pem".into(),
                private_key: "key.pem".into(),
            }),
            drop_wait: Some(Duration::from_micros(50)),
            client: false,
            connect_timeout: None,
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
        assert_eq!(get("transport/link/tls/root_ca_certificate"), "\"ca.pem\"");
        assert!(get("connect/endpoints").contains("tcp/10.0.0.1:7447"));
        assert_eq!(get("scouting/multicast/enabled"), "false");
        assert_eq!(get("mode"), "\"peer\"");

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
    fn per_class_tcp_uses_consecutive_ports() {
        let config = TransportConfig::per_class_tcp(
            Some("0.0.0.0:7447".parse().expect("addr")),
            Some("10.0.0.1:7500".parse().expect("addr")),
        )
        .expect("config");
        let Layout::PerClass(classes) = config.layout else {
            panic!("per class");
        };
        assert_eq!(classes[0].listen, vec!["tcp/0.0.0.0:7447"]);
        assert_eq!(classes[3].listen, vec!["tcp/0.0.0.0:7450"]);
        assert_eq!(classes[2].connect, vec!["tcp/10.0.0.1:7502"]);
        assert!(
            TransportConfig::per_class_tcp(Some("0.0.0.0:65534".parse().expect("addr")), None)
                .is_err()
        );
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
