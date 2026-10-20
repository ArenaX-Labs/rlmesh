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
//! | Class | Zenoh priority | Carries |
//! |---|---|---|
//! | [`TrafficClass::Control`] | RealTime | liveliness, `ctl`, authority state, heartbeats, CONTROL-priority channels (operator teleop commands) |
//! | [`TrafficClass::Realtime`] | InteractiveHigh, express | REALTIME channels (joints, wrench, commands) |
//! | [`TrafficClass::Data`] | Data | DATA channels, history-store queries |
//! | [`TrafficClass::Bulk`] | DataLow | BULK channels (cameras) |
//!
//! [`Layout::Shared`] uses one session for every class instead: in-process
//! tests, Unix sockets, and QUIC (whose multistream already gives every
//! priority its own stream inside one connection). Both ends of a link must
//! use the same layout.
//!
//! Priorities are still set on every message, so the shared layout keeps
//! Zenoh's own ordering.
//!
//! A channel's `Delivery` is not mapped onto Zenoh reliability: per-publisher
//! reliability is an unstable Zenoh API in 1.10, and over the TCP links the
//! session uses every message is delivered by the link anyway. Best-effort
//! channels get their real-time behaviour from DROP congestion control and
//! from never being retransmitted at the session level.

use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::time::Duration;

use bytes::Bytes;
use rlmesh_wire::codec::Encoded;
use rlmesh_wire::{ChannelPriority, ChannelQos, Congestion};
use tracing::debug;
use zenoh::bytes::ZBytes;
use zenoh::qos::{CongestionControl, Priority};

use crate::error::SessionError;

/// A traffic class: a separate Zenoh session in [`Layout::PerClass`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TrafficClass {
    /// Session control and CONTROL-priority channels.
    Control,
    /// REALTIME channels.
    Realtime,
    /// DATA channels and history.
    Data,
    /// BULK channels.
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
}

/// Heartbeats and their echoes: dropped rather than queued under congestion
/// (a stale timing sample is worthless), sent at once.
pub(crate) const HEARTBEAT_CONGESTION: CongestionControl = CongestionControl::Drop;
/// Control verbs, replies and authority updates: never dropped.
pub(crate) const CONTROL_CONGESTION: CongestionControl = CongestionControl::Block;
/// The Zenoh priority of session control traffic.
pub(crate) const CONTROL_PRIORITY: Priority = Priority::RealTime;

/// The Zenoh priority a channel priority maps to.
pub fn zenoh_priority(priority: ChannelPriority) -> Priority {
    match priority {
        ChannelPriority::Control => Priority::RealTime,
        ChannelPriority::Realtime => Priority::InteractiveHigh,
        ChannelPriority::Data => Priority::Data,
        ChannelPriority::Bulk => Priority::DataLow,
    }
}

/// The Zenoh congestion control of a channel.
pub fn zenoh_congestion(qos: &ChannelQos) -> CongestionControl {
    match qos.congestion {
        Congestion::Drop => CongestionControl::Drop,
        Congestion::Block => CongestionControl::Block,
    }
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
}

impl Default for TransportTuning {
    fn default() -> Self {
        Self {
            lease: Duration::from_secs(1),
            keep_alive: 4,
            drop_wait: None,
            tls: None,
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

/// A peer-mode Zenoh config: multicast scouting off, so it listens on
/// `listen` and dials `connect` and nothing else (sessions on one host never
/// discover each other by accident).
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
    set(&mut config, "mode", "\"peer\"".into())?;
    set(&mut config, "scouting/multicast/enabled", "false".into())?;
    set(&mut config, "listen/endpoints", list(&endpoints.listen))?;
    set(&mut config, "connect/endpoints", list(&endpoints.connect))?;
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
                Ok(Self::from_session(session))
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
                })
            }
        }
    }

    /// Wraps one already-open session for every class.
    pub fn from_session(session: zenoh::Session) -> Self {
        Self {
            sessions: [session.clone(), session.clone(), session.clone(), session],
            shared: true,
        }
    }

    /// Whether every class shares one session.
    pub fn is_shared(&self) -> bool {
        self.shared
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
