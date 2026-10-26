//! Addresses, the one string form a user writes everywhere:
//!
//! ```text
//! participant address  [rlmesh://][host/][session/]participant
//! session address      [rlmesh://][host/][session]
//! ```
//!
//! A host is recognisable on its own: it has a port (`lab:7447`, `:7447`),
//! a dot (`robot-lab.local`, `10.0.0.5`), brackets (`[fe80::1]:7447`), or is
//! `localhost`; otherwise the first segment is a session. No session is the
//! default session; a host without a port is on [`DEFAULT_PORT`].
//!
//! Where a session is reached follows the hosting rule ([`place`]): no host
//! joins the local host on the well-known port ([`well_known_port`],
//! `RLMESH_PORT` overriding 7447), or hosts the session in
//! this process when nothing listens there; an explicit port on this machine
//! hosts in-process on it (`:0` on an OS-assigned port); any other host is
//! joined. A session is named (`bench-3`, [`DEFAULT_SESSION`]) or a UUID; a
//! name identifies its [`SessionId::named`]. The tenant is fixed
//! ([`TENANT`]) and never part of an address.

use std::collections::HashMap;
use std::fmt;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpStream, ToSocketAddrs};
use std::str::FromStr;
use std::time::Duration;

use crate::clock::SystemMonotonicClock;
use crate::error::SessionError;
use crate::host::HostConfig;
use crate::participant::{Participant, ParticipantConfig};
use crate::server::SessionServer;
use crate::transport::{Transport, TransportConfig, TransportTuning, open_local_listeners};
use crate::wire::{ParticipantId, PriorityClass, Role, SessionId, TenantId};

/// The well-known port a session host listens on (control class; the other
/// classes follow it), unless [`PORT_VAR`] names another.
pub const DEFAULT_PORT: u16 = 7447;
/// The environment variable that overrides the well-known port.
pub const PORT_VAR: &str = "RLMESH_PORT";

/// The well-known local port: [`PORT_VAR`] when set to a port, else
/// [`DEFAULT_PORT`].
pub fn well_known_port() -> Result<u16, SessionError> {
    match std::env::var(PORT_VAR) {
        Ok(port) => port
            .trim()
            .parse()
            .map_err(|_| SessionError::Config(format!("{PORT_VAR}={port:?} is not a port"))),
        Err(_) => Ok(DEFAULT_PORT),
    }
}
/// The environment variable (`1`) under which participants join as Zenoh
/// clients of the host they dial: a sandbox sets it on its container.
pub const JOIN_AS_CLIENT_VAR: &str = "RLMESH_JOIN_AS_CLIENT";

static JOIN_AS_CLIENT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Join every later session as a Zenoh client of the host it dials, as
/// [`JOIN_AS_CLIENT_VAR`] does, for a process whose participants all join
/// sessions that sandbox containers host (`rlmesh doctor`, `rlmesh bench`).
pub fn join_as_client() {
    JOIN_AS_CLIENT.store(true, std::sync::atomic::Ordering::Relaxed);
}
/// The session an address names when it names none.
pub const DEFAULT_SESSION: &str = "default";
/// The tenant every local session runs in.
pub const TENANT: &str = "local";
/// The optional scheme, for deep links.
pub const SCHEME: &str = "rlmesh://";
/// How long a lease in a session a participant hosts outlives its holder's
/// last heartbeat (the host's default).
pub const LEASE_GRACE: Duration = Duration::from_millis(250);

/// A host as an address names it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostName {
    /// The name or IP (an IPv6 without its brackets); empty for `:7447`.
    pub name: String,
    /// The port, if the address gives one.
    pub port: Option<u16>,
}

impl fmt::Display for HostName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.name.contains(':') {
            write!(f, "[{}]", self.name)?;
        } else {
            f.write_str(&self.name)?;
        }
        match self.port {
            Some(port) => write!(f, ":{port}"),
            None => Ok(()),
        }
    }
}

/// A parsed address: a session address has no participant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Address {
    /// The host (`None`: the local host, by the hosting rule).
    pub host: Option<HostName>,
    /// The session (`None`: the default session).
    pub session: Option<String>,
    /// The participant (a session address names none).
    pub participant: Option<String>,
}

/// Where the hosting rule puts a session.
///
/// A host serves any number of named sessions: it starts one for its first
/// participant ([`SessionServer`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Placement {
    /// No host given: join the local host on [`well_known_port`], else host the
    /// session in this process there.
    Local,
    /// An explicit port on this machine: host in this process on it (port
    /// 0: an OS-assigned one), or join whoever won the port.
    Host(SocketAddr),
    /// Join the host on another machine.
    Join(String, u16),
}

fn invalid(text: &str, why: impl fmt::Display) -> SessionError {
    SessionError::Config(format!("address {text:?}: {why}"))
}

/// Whether a segment names a host rather than a session.
fn is_host(segment: &str) -> bool {
    segment == "localhost"
        || segment.starts_with('[')
        || segment.contains(':')
        || segment.contains('.')
}

fn host_name(text: &str, segment: &str) -> Result<HostName, SessionError> {
    let port = |port: &str| {
        port.parse::<u16>()
            .map_err(|_| invalid(text, format!("port {port:?}")))
    };
    if let Some(rest) = segment.strip_prefix('[') {
        let (name, after) = rest
            .split_once(']')
            .ok_or_else(|| invalid(text, "an unclosed ["))?;
        let port = match after {
            "" => None,
            after => Some(port(after.strip_prefix(':').ok_or_else(|| {
                invalid(text, "a bracketed host takes :port after the ]")
            })?)?),
        };
        return Ok(HostName {
            name: name.to_string(),
            port,
        });
    }
    match segment.rsplit_once(':') {
        Some((name, _)) if name.contains(':') => {
            Err(invalid(text, "bracket an IPv6 host ([fe80::1]:7447)"))
        }
        Some((name, p)) => Ok(HostName {
            name: name.to_string(),
            port: Some(port(p)?),
        }),
        None => Ok(HostName {
            name: segment.to_string(),
            port: None,
        }),
    }
}

/// The upgrade error of an old `lockstep://` address, with its new form.
fn lockstep_upgrade(text: &str, rest: &str) -> SessionError {
    let segments: Vec<&str> = rest.split('/').collect();
    let new = match segments.as_slice() {
        // {ip}:{port}/{tenant}/{session}/{robot}: the tenant leaves.
        [base, _tenant, session, robot] => format!("{base}/{session}/{robot}"),
        [base, robot] => format!("{base}/{robot}"),
        _ => "[host/][session/]participant".to_string(),
    };
    SessionError::Config(format!(
        "address {text:?}: lockstep:// addresses are gone (a robot's clock is declared, not \
         addressed); use {new:?}"
    ))
}

fn tenant_upgrade(text: &str, new: String) -> SessionError {
    invalid(text, format!("the tenant left addresses; use {new:?}"))
}

impl Address {
    /// Parses a participant address, `[rlmesh://][host/][session/]participant`.
    pub fn participant(text: &str) -> Result<Self, SessionError> {
        Self::parse(text, true)
    }

    /// Parses a session address, `[rlmesh://][host/][session]`.
    pub fn session(text: &str) -> Result<Self, SessionError> {
        Self::parse(text, false)
    }

    fn parse(text: &str, participant: bool) -> Result<Self, SessionError> {
        let trimmed = text.trim();
        if let Some(rest) = trimmed.strip_prefix("lockstep://") {
            return Err(lockstep_upgrade(text, rest));
        }
        let body = trimmed.strip_prefix(SCHEME).unwrap_or(trimmed);
        let mut segments: Vec<&str> = if body.is_empty() {
            Vec::new()
        } else {
            body.split('/').collect()
        };
        if segments.iter().any(|s| s.is_empty()) {
            return Err(invalid(text, "an empty segment"));
        }
        let host = match segments.first() {
            Some(first) if is_host(first) => {
                let host = host_name(text, first)?;
                segments.remove(0);
                Some(host)
            }
            _ => None,
        };
        let prefix = host.as_ref().map(|h| format!("{h}/")).unwrap_or_default();
        let (session, name) = match (participant, segments.as_slice()) {
            (true, [name]) => (None, Some(*name)),
            (true, [session, name]) => (Some(*session), Some(*name)),
            (true, []) => {
                return Err(invalid(
                    text,
                    "names no participant ([host/][session/]participant)",
                ));
            }
            // {ip}:{port}/{tenant}/{session}/{robot}, the old form.
            (true, [_tenant, session, name]) if host.is_some() => {
                return Err(tenant_upgrade(text, format!("{prefix}{session}/{name}")));
            }
            (false, []) => (None, None),
            (false, [session]) => (Some(*session), None),
            // {ip}:{port}/{tenant}/{session}, the old form.
            (false, [_tenant, session]) if host.is_some() => {
                return Err(tenant_upgrade(text, format!("{prefix}{session}")));
            }
            (true, _) => {
                return Err(invalid(text, "expected [host/][session/]participant"));
            }
            (false, _) => return Err(invalid(text, "expected [host/][session]")),
        };
        if let Some(session) = session
            && session.parse::<SessionId>().is_err()
        {
            ParticipantId::new(session)
                .map_err(|_| invalid(text, "a session is a name of [A-Za-z0-9._-] or a UUID"))?;
        }
        if let Some(name) = name {
            ParticipantId::new(name).map_err(|e| invalid(text, e))?;
        }
        Ok(Self {
            host,
            session: session.map(str::to_string),
            participant: name.map(str::to_string),
        })
    }

    /// The session's name ([`DEFAULT_SESSION`] when the address names none).
    pub fn session_name(&self) -> &str {
        self.session.as_deref().unwrap_or(DEFAULT_SESSION)
    }

    /// The session's id: a UUID as given, or the id its name identifies.
    pub fn session_id(&self) -> SessionId {
        session_id(self.session_name())
    }

    /// The session this address names, resolved for joining and never
    /// hosted here: the local host on [`well_known_port`] when it names
    /// none.
    pub fn joining(&self) -> Result<SessionAddress, SessionError> {
        let base = match self.placement() {
            Placement::Local => SocketAddr::from((Ipv4Addr::LOCALHOST, well_known_port()?)),
            Placement::Host(base) => joinable(base),
            Placement::Join(host, port) => resolve(&self.to_string(), &host, port)?,
        };
        SessionAddress::new(base, self.session_name())
    }

    /// Where the hosting rule puts the session.
    pub fn placement(&self) -> Placement {
        let Some(host) = &self.host else {
            return Placement::Local;
        };
        let local_ip = match host.name.as_str() {
            "" | "localhost" => Some(IpAddr::from(Ipv4Addr::LOCALHOST)),
            name => name
                .parse::<IpAddr>()
                .ok()
                .filter(|ip| ip.is_loopback() || ip.is_unspecified()),
        };
        match (local_ip, host.port) {
            (Some(ip), Some(port)) => Placement::Host(SocketAddr::new(ip, port)),
            (Some(_), None) => Placement::Local,
            (None, port) => Placement::Join(host.name.clone(), port.unwrap_or(DEFAULT_PORT)),
        }
    }
}

/// The id a session name (or UUID) identifies.
pub fn session_id(name: &str) -> SessionId {
    name.parse().unwrap_or_else(|_| SessionId::named(name))
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut segments = Vec::new();
        if let Some(host) = &self.host {
            segments.push(host.to_string());
        }
        if let Some(session) = &self.session {
            segments.push(session.clone());
        }
        if let Some(name) = &self.participant {
            segments.push(name.clone());
        }
        f.write_str(&segments.join("/"))
    }
}

/// Whether `text` is a v2 address (a session or participant address) rather
/// than a v1 endpoint (`host:port`, a bare port, or a `scheme://` URL). An
/// old `lockstep://` address counts as v2, so it meets its upgrade error.
pub fn is_v2_address(text: &str) -> bool {
    let text = text.trim();
    if text.starts_with(SCHEME) || text.starts_with("lockstep://") {
        return true;
    }
    if text.contains("://") || text.is_empty() {
        return false;
    }
    match text.split_once('/') {
        Some(_) => true,
        None => !is_host(text) && !text.bytes().all(|b| b.is_ascii_digit()),
    }
}

/// Where a session's host listens: an address resolved to a socket.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionAddress {
    /// The host's control-class listener; the other classes follow it.
    pub base: SocketAddr,
    /// The tenant namespace ([`TENANT`]).
    pub tenant: TenantId,
    /// The session's name (or UUID text), as addresses print it.
    pub name: String,
    /// The session.
    pub session: SessionId,
}

impl SessionAddress {
    /// The session `name` (or UUID) at `base`, in the local tenant.
    pub fn new(base: SocketAddr, name: &str) -> Result<Self, SessionError> {
        Ok(Self {
            base,
            tenant: TenantId::new(TENANT)?,
            name: name.to_string(),
            session: session_id(name),
        })
    }

    /// The participant address of `participant` in this session.
    pub fn participant(&self, participant: &str) -> String {
        format!("{self}/{participant}")
    }
}

fn resolve(text: &str, host: &str, port: u16) -> Result<SocketAddr, SessionError> {
    (host, port)
        .to_socket_addrs()
        .map_err(|e| invalid(text, format!("host {host:?}: {e}")))?
        .next()
        .ok_or_else(|| invalid(text, format!("host {host:?} resolves to nothing")))
}

fn joinable(base: SocketAddr) -> SocketAddr {
    if base.ip().is_unspecified() {
        SocketAddr::from((Ipv4Addr::LOCALHOST, base.port()))
    } else {
        base
    }
}

/// A session address resolved for joining, without hosting: the local host
/// on [`well_known_port`] when it names none.
impl FromStr for SessionAddress {
    type Err = SessionError;

    fn from_str(text: &str) -> Result<Self, SessionError> {
        Address::session(text)?.joining()
    }
}

impl fmt::Display for SessionAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let host = HostName {
            name: self.base.ip().to_string(),
            port: Some(self.base.port()),
        };
        if self.name == DEFAULT_SESSION {
            write!(f, "{host}")
        } else {
            write!(f, "{host}/{}", self.name)
        }
    }
}

impl SessionAddress {
    /// Opens a transport to the host (listening on the host's interface,
    /// any port) and attaches as `id`, on this process's
    /// `CLOCK_MONOTONIC` (so capture times line up with other local
    /// processes').
    pub async fn join(
        &self,
        id: &str,
        roles: Vec<Role>,
        priority: PriorityClass,
        host_timeout: Duration,
    ) -> Result<Participant, SessionError> {
        self.join_declaring(
            id,
            roles,
            priority,
            host_timeout,
            (String::new(), Vec::new()),
        )
        .await
    }

    /// [`Self::join`], declaring the workflow edition the participant runs
    /// at (empty: none) and every one it runs, for the roster.
    pub async fn join_declaring(
        &self,
        id: &str,
        roles: Vec<Role>,
        priority: PriorityClass,
        host_timeout: Duration,
        (workflow_edition, workflow_editions): (String, Vec<String>),
    ) -> Result<Participant, SessionError> {
        let listen_ip = if self.base.ip().is_loopback() {
            self.base.ip()
        } else {
            IpAddr::from([0, 0, 0, 0])
        };
        // A sandbox container's participants sit behind its published ports:
        // the machine outside reaches the container's session host but not
        // their own listeners, and peers do not route for each other. So the
        // sandbox starts its container with JOIN_AS_CLIENT_VAR set, and there
        // participants join as Zenoh clients, which the host routes for.
        // Everywhere else they join as peers, which talk directly.
        let tuning = TransportTuning {
            client: JOIN_AS_CLIENT.load(std::sync::atomic::Ordering::Relaxed)
                || std::env::var_os(JOIN_AS_CLIENT_VAR).is_some_and(|value| value == "1"),
            ..TransportTuning::default()
        };
        let transport = match local_host(self.base) {
            Some(transport) => transport,
            None => {
                Transport::open(
                    &TransportConfig::per_class_tcp(
                        Some(SocketAddr::new(listen_ip, 0)),
                        Some(self.base),
                    )?
                    .with_tuning(tuning),
                )
                .await?
            }
        };
        let mut config = ParticipantConfig::new(
            self.tenant.clone(),
            self.session,
            ParticipantId::new(id)?,
            roles,
            priority,
        );
        config.clock = SystemMonotonicClock.shared();
        config.host_timeout = host_timeout;
        config.workflow_edition = workflow_edition;
        config.workflow_editions = workflow_editions;
        Participant::attach(transport, config)
            .await
            .map_err(|err| match err {
                SessionError::NotServed(why) => SessionError::NotServed(format!(
                    "the host at {} cannot serve session {:?}: {why}",
                    self.base, self.name
                )),
                other => other,
            })
    }
}

/// A session reached by the hosting rule: its address, and the host when
/// this process hosts it (kept with this value, stopped by [`Self::close`]).
pub struct Placed {
    /// Where participants join.
    pub address: SessionAddress,
    hosted: Option<(SessionServer, Transport)>,
}

impl fmt::Debug for Placed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Placed")
            .field("address", &self.address)
            .field("hosted", &self.hosted.is_some())
            .finish()
    }
}

impl Placed {
    /// Whether this process hosts the session.
    pub fn hosts(&self) -> bool {
        self.hosted.is_some()
    }

    /// Stops the host, if this process runs it (participants see it leave).
    pub async fn close(&mut self) -> Result<(), SessionError> {
        if let Some((host, transport)) = self.hosted.take() {
            local_hosts()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(&self.address.base);
            host.shutdown().await;
            transport.close().await?;
        }
        Ok(())
    }
}

/// The transports of the session hosts this process runs, by the address
/// participants join them at. A participant joining one rides its sessions
/// instead of a loopback link: a link has a lease, which a starved process
/// can let lapse, and then loses the host it is running itself.
fn local_hosts() -> &'static std::sync::Mutex<HashMap<SocketAddr, Transport>> {
    static HOSTS: std::sync::OnceLock<std::sync::Mutex<HashMap<SocketAddr, Transport>>> =
        std::sync::OnceLock::new();
    HOSTS.get_or_init(Default::default)
}

fn local_host(base: SocketAddr) -> Option<Transport> {
    local_hosts()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&base)
        .map(Transport::ridden)
}

/// Whether a session host answers on `base` (its control listener accepts).
fn listening(base: SocketAddr) -> bool {
    TcpStream::connect_timeout(&joinable(base), Duration::from_millis(200)).is_ok()
}

/// Hosts session `name` in this process on `base` (port 0: OS-assigned),
/// and every other session of [`TENANT`] its participants name.
pub async fn host_at(
    base: SocketAddr,
    name: &str,
    lease_grace: Duration,
) -> Result<Placed, SessionError> {
    let (transport, base) = if base.port() == 0 {
        open_local_listeners(base.ip()).await?
    } else {
        let transport = Transport::open(&TransportConfig::per_class_tcp(Some(base), None)?).await?;
        (transport, base)
    };
    let address = SessionAddress::new(joinable(base), name)?;
    let mut config = HostConfig::new(address.tenant.clone(), address.session);
    config.clock = SystemMonotonicClock.shared();
    config.lease_grace = lease_grace.max(Duration::from_millis(1));
    config.health_log = None;
    let host = SessionServer::start(transport.clone(), config).await?;
    local_hosts()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(address.base, transport.clone());
    Ok(Placed {
        address,
        hosted: Some((host, transport)),
    })
}

/// Reaches the session `address` names by the hosting rule: joins the local
/// host on [`well_known_port`] (or on the explicit local port) when one
/// listens, else hosts the session in this process there; a bind that
/// loses a race joins the winner. Another machine's host is joined. Leases
/// in a session this process hosts lapse `lease_grace` after their holder's
/// last heartbeat.
pub async fn place(address: &Address, lease_grace: Duration) -> Result<Placed, SessionError> {
    let name = address.session_name();
    let join = |base: SocketAddr| -> Result<Placed, SessionError> {
        Ok(Placed {
            address: SessionAddress::new(joinable(base), name)?,
            hosted: None,
        })
    };
    let base = match address.placement() {
        Placement::Join(host, port) => {
            return join(resolve(&address.to_string(), &host, port)?);
        }
        Placement::Local => {
            let base = SocketAddr::from((Ipv4Addr::LOCALHOST, well_known_port()?));
            if listening(base) {
                return join(base);
            }
            base
        }
        // An explicit port: hosted here, unless another process (or this
        // one) already holds it.
        Placement::Host(base) if base.port() != 0 && listening(base) => return join(base),
        Placement::Host(base) => base,
    };
    match host_at(base, name, lease_grace).await {
        Ok(placed) => Ok(placed),
        // Another process holds the port: join it.
        Err(_) if base.port() != 0 && listening(base) => join(base),
        Err(err) => Err(err),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::SessionHost;

    fn host(name: &str, port: Option<u16>) -> Option<HostName> {
        Some(HostName {
            name: name.into(),
            port,
        })
    }

    fn address(host: Option<HostName>, session: Option<&str>, name: Option<&str>) -> Address {
        Address {
            host,
            session: session.map(str::to_string),
            participant: name.map(str::to_string),
        }
    }

    #[test]
    fn participant_addresses_parse_as_the_sdk_doc_writes_them() {
        let table = [
            ("cartpole", address(None, None, Some("cartpole"))),
            (
                "robot-lab/ur10",
                address(None, Some("robot-lab"), Some("ur10")),
            ),
            (
                "lab:7447/cartpole",
                address(host("lab", Some(7447)), None, Some("cartpole")),
            ),
            (
                "lab:7447/bench-3/cartpole",
                address(host("lab", Some(7447)), Some("bench-3"), Some("cartpole")),
            ),
            (
                "0.0.0.0:7447/cartpole",
                address(host("0.0.0.0", Some(7447)), None, Some("cartpole")),
            ),
            (
                ":7447/cartpole",
                address(host("", Some(7447)), None, Some("cartpole")),
            ),
            (
                "robot-lab.local/ur10",
                address(host("robot-lab.local", None), None, Some("ur10")),
            ),
            (
                "10.0.0.5/bench-3/ur10",
                address(host("10.0.0.5", None), Some("bench-3"), Some("ur10")),
            ),
            (
                "[fe80::1]:7447/ur10",
                address(host("fe80::1", Some(7447)), None, Some("ur10")),
            ),
            (
                "localhost/ur10",
                address(host("localhost", None), None, Some("ur10")),
            ),
            (
                "rlmesh://lab:7447/bench-3/cartpole",
                address(host("lab", Some(7447)), Some("bench-3"), Some("cartpole")),
            ),
            (":0/gym", address(host("", Some(0)), None, Some("gym"))),
        ];
        for (text, want) in table {
            let got = Address::participant(text).unwrap_or_else(|e| panic!("{text}: {e}"));
            assert_eq!(got, want, "{text}");
            assert_eq!(
                got.to_string(),
                text.trim_start_matches(SCHEME),
                "{text} prints back"
            );
        }
    }

    #[test]
    fn session_addresses_parse_and_name_their_session() {
        let table = [
            ("", address(None, None, None)),
            ("robot-lab", address(None, Some("robot-lab"), None)),
            ("lab:7447", address(host("lab", Some(7447)), None, None)),
            (
                "lab:7447/bench-3",
                address(host("lab", Some(7447)), Some("bench-3"), None),
            ),
            ("rlmesh://:0", address(host("", Some(0)), None, None)),
        ];
        for (text, want) in table {
            assert_eq!(Address::session(text).expect(text), want, "{text}");
        }
        let named = Address::session("lab:7447/bench-3").expect("parse");
        assert_eq!(named.session_id(), SessionId::named("bench-3"));
        assert_eq!(
            Address::session("").expect("parse").session_id(),
            SessionId::named(DEFAULT_SESSION)
        );
        let id = SessionId::new();
        let by_id = Address::session(&format!("lab:7447/{id}")).expect("uuid");
        assert_eq!(by_id.session_id(), id);
    }

    #[test]
    fn the_hosting_rule_places_each_form() {
        let place = |text: &str| Address::participant(text).expect(text).placement();
        let local = |port| Placement::Host(SocketAddr::from((Ipv4Addr::LOCALHOST, port)));
        assert_eq!(place("cartpole"), Placement::Local);
        assert_eq!(place("robot-lab/ur10"), Placement::Local);
        assert_eq!(place("localhost/ur10"), Placement::Local);
        assert_eq!(place(":7447/x"), local(7447));
        assert_eq!(place(":0/x"), local(0));
        assert_eq!(place("127.0.0.1:9000/x"), local(9000));
        assert_eq!(
            place("0.0.0.0:7447/x"),
            Placement::Host("0.0.0.0:7447".parse().expect("addr"))
        );
        assert_eq!(place("lab:7447/x"), Placement::Join("lab".into(), 7447));
        assert_eq!(
            place("robot-lab.local/x"),
            Placement::Join("robot-lab.local".into(), DEFAULT_PORT)
        );
        assert_eq!(place("10.0.0.5:9/x"), Placement::Join("10.0.0.5".into(), 9));
    }

    #[test]
    fn old_forms_raise_an_upgrade_error_with_the_new_form() {
        let id = SessionId::new();
        for (old, new) in [
            (
                format!("lockstep://127.0.0.1:7447/local/{id}/gym"),
                format!("127.0.0.1:7447/{id}/gym"),
            ),
            (
                "lockstep://127.0.0.1:0/gym".to_string(),
                "127.0.0.1:0/gym".to_string(),
            ),
            (
                format!("127.0.0.1:7447/local/{id}/gym"),
                format!("127.0.0.1:7447/{id}/gym"),
            ),
        ] {
            let err = Address::participant(&old).expect_err(&old).to_string();
            assert!(err.contains(&format!("{new:?}")), "{old}: {err}");
        }
        let err = Address::session(&format!("127.0.0.1:7447/local/{id}"))
            .expect_err("old session")
            .to_string();
        assert!(err.contains(&format!("\"127.0.0.1:7447/{id}\"")), "{err}");
        for bad in ["", "lab:7447", "a//b", "fe80::1/x", "a/b/c", "bad name"] {
            assert!(Address::participant(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn v2_addresses_stand_apart_from_v1_endpoints() {
        for v2 in [
            "cartpole",
            "robot-lab/ur10",
            ":0/gym",
            "rlmesh://x",
            "lockstep://a:1/b",
        ] {
            assert!(is_v2_address(v2), "{v2}");
        }
        for v1 in [
            "127.0.0.1:50051",
            "50051",
            "tcp://h:1",
            "unix:///s",
            "https://h",
            "lab:7447",
        ] {
            assert!(!is_v2_address(v1), "{v1}");
        }
    }

    #[test]
    fn a_session_address_prints_the_default_session_short() {
        let base: SocketAddr = "127.0.0.1:7447".parse().expect("addr");
        let default = SessionAddress::new(base, DEFAULT_SESSION).expect("default");
        assert_eq!(default.to_string(), "127.0.0.1:7447");
        assert_eq!(default.participant("cartpole"), "127.0.0.1:7447/cartpole");
        let named = SessionAddress::new(base, "bench-3").expect("named");
        assert_eq!(named.to_string(), "127.0.0.1:7447/bench-3");
        let back: SessionAddress = named.to_string().parse().expect("parse");
        assert_eq!(back, named);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_explicit_local_port_hosts_and_a_second_placement_joins_it() {
        let grace = Duration::from_millis(250);
        let mut first = place(&Address::session(":0").expect("parse"), grace)
            .await
            .expect("host");
        assert!(first.hosts());
        let port = first.address.base.port();
        let second = place(
            &Address::session(&format!(":{port}")).expect("parse"),
            grace,
        )
        .await
        .expect("join");
        assert!(!second.hosts(), "the port is taken: join it");
        assert_eq!(second.address, first.address);
        first.close().await.expect("close");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_participant_of_a_host_in_this_process_rides_its_sessions() {
        // No loopback link between a process and itself: no lease a starved
        // process could let lapse, losing the host it runs.
        let mut host = place(
            &Address::session(":0").expect("parse"),
            Duration::from_secs(1),
        )
        .await
        .expect("host");
        let participant = host
            .address
            .join(
                "p",
                vec![Role::Observer],
                PriorityClass::Scripted,
                Duration::from_secs(5),
            )
            .await
            .expect("join");
        let hosting = &host.hosted.as_ref().expect("hosted here").1;
        assert_eq!(
            participant.transport().control().zid(),
            hosting.control().zid()
        );
        participant.close().await;
        host.close().await.expect("close");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn one_host_serves_many_named_sessions_stops_empty_ones_and_refuses_fast_where_it_cannot()
    {
        let grace = Duration::from_millis(250);
        let mut host = place(&Address::session(":0").expect("parse"), grace)
            .await
            .expect("host");
        let join = |base: SocketAddr, name: &str| {
            let session = SessionAddress::new(base, name).expect("session");
            async move {
                session
                    .join(
                        "p",
                        vec![Role::Observer],
                        PriorityClass::Scripted,
                        Duration::from_secs(5),
                    )
                    .await
            }
        };
        let base = host.address.base;
        let bench = join(base, "bench-3")
            .await
            .expect("a new session starts for its first participant");
        let lab = join(base, "robot-lab")
            .await
            .expect("and another beside it");
        let server = &host.hosted.as_ref().expect("hosted here").0;
        let running = server.sessions().await;
        assert!(
            running.contains(&session_id("bench-3")) && running.contains(&session_id("robot-lab"))
        );
        // A session stops when its last participant leaves.
        bench.close().await;
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while server.sessions().await.contains(&session_id("bench-3")) {
            assert!(
                std::time::Instant::now() < deadline,
                "bench-3 never stopped"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(server.sessions().await.contains(&session_id("robot-lab")));
        drop(lab);
        host.close().await.expect("close");

        // A plain single-session host cannot start sessions: refused at once.
        let (transport, base) = open_local_listeners(IpAddr::from(Ipv4Addr::LOCALHOST))
            .await
            .expect("listen");
        let only = SessionAddress::new(base, "only").expect("session");
        let single = SessionHost::start(
            transport.clone(),
            HostConfig::new(only.tenant.clone(), only.session),
        )
        .await
        .expect("single host");
        let started = std::time::Instant::now();
        let Err(err) = join(base, "elsewhere").await else {
            panic!("not served");
        };
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "fails fast: {err}"
        );
        let text = err.to_string();
        assert!(
            text.contains(&base.to_string()) && text.contains("elsewhere"),
            "{text}"
        );
        single.shutdown().await;
        transport.close().await.expect("close");
    }
}
