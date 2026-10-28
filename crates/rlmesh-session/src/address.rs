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
//!
//! **The key.** A session hosted in a process (an embedded host) draws its
//! own keys, and the address it hands out carries the credential, as a
//! Jupyter URL carries its token: `127.0.0.1:7447/bench-3#{local
//! secret}.{host key}` (each 32 bytes, base32: [`AddressKey`]), and a
//! participant address appends `/{participant}` after it. Holding the
//! address is the invite (design §2): its holder attaches as device
//! `local`. A process joining a host on this machine without the key finds
//! it where the host left it for this user ([`AddressKey`] files, 0600), so
//! two scripts still meet by name.

use std::collections::HashMap;
use std::fmt;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpStream, ToSocketAddrs};
use std::str::FromStr;
use std::time::Duration;

use rlmesh_session_core::auth::{Credential, HostIdentity};
use zeroize::Zeroizing;

use crate::clock::SystemMonotonicClock;
use crate::error::SessionError;
use crate::host::HostConfig;
use crate::participant::{Participant, ParticipantConfig};
use crate::server::SessionServer;
use crate::transport::{Transport, TransportConfig, TransportTuning, open_local_listeners};
use crate::wire::{CtlError, CtlStatus, ParticipantId, PriorityClass, Role, SessionId, TenantId};

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

/// The key an embedded host's address carries (`#{secret}.{host}`): its
/// local secret, from which a holder derives the `local` device's key, and
/// its static public key. A secret: never logged.
#[derive(Clone, PartialEq, Eq)]
pub struct AddressKey {
    secret: Zeroizing<[u8; 32]>,
    host: [u8; 32],
}

impl fmt::Debug for AddressKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AddressKey(..)")
    }
}

/// RFC 4648 base32, lowercase, unpadded: 52 characters for 32 bytes.
const BASE32: &[u8; 32] = b"abcdefghijklmnopqrstuvwxyz234567";

fn base32(bytes: &[u8; 32]) -> String {
    let mut out = String::with_capacity(52);
    let (mut acc, mut bits) = (0_u32, 0_u32);
    for &b in bytes {
        acc = (acc << 8) | u32::from(b);
        bits += 8;
        while bits >= 5 {
            bits -= 5;
            out.push(char::from(BASE32[((acc >> bits) & 31) as usize]));
        }
    }
    if bits > 0 {
        out.push(char::from(BASE32[((acc << (5 - bits)) & 31) as usize]));
    }
    out
}

fn unbase32(text: &str) -> Option<[u8; 32]> {
    if text.len() != 52 {
        return None;
    }
    let mut out = [0_u8; 32];
    let (mut acc, mut bits, mut at) = (0_u32, 0_u32, 0);
    for c in text.bytes() {
        let value = BASE32.iter().position(|&b| b == c.to_ascii_lowercase())?;
        acc = (acc << 5) | u32::try_from(value).ok()?;
        bits += 5;
        if bits >= 8 {
            bits -= 8;
            *out.get_mut(at)? = (acc >> bits) as u8;
            at += 1;
        }
    }
    // The 4 padding bits of the last character are zero.
    (at == 32 && acc & ((1 << bits) - 1) == 0).then_some(out)
}

impl AddressKey {
    /// The key of `identity`'s address.
    pub fn of(identity: &HostIdentity) -> Self {
        Self {
            secret: Zeroizing::new(*identity.local_secret()),
            host: *identity.static_public(),
        }
    }

    /// The credential of its holder (device `local`).
    pub fn credential(&self) -> Credential {
        Credential::local(&self.secret, self.host)
    }

    fn parse(text: &str) -> Option<Self> {
        let (secret, host) = text.split_once('.')?;
        Some(Self {
            secret: Zeroizing::new(unbase32(secret)?),
            host: unbase32(host)?,
        })
    }
}

impl FromStr for AddressKey {
    type Err = SessionError;

    /// `{secret}.{host key}`, as an address carries it after `#`.
    fn from_str(text: &str) -> Result<Self, SessionError> {
        Self::parse(text.trim()).ok_or_else(|| invalid(text, "not {secret}.{host key}"))
    }
}

impl fmt::Display for AddressKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", base32(&self.secret), base32(&self.host))
    }
}

/// The environment variable a launcher passes the keys of the host it
/// starts in (a sandbox container) through, as [`launch_identity`] makes
/// them: the workload spec of design §10. [`host_at`] hosts under them.
pub const HOST_IDENTITY_VAR: &str = "RLMESH_HOST_IDENTITY";

/// Fresh keys for a host another process will run (a container): the text
/// to pass it in [`HOST_IDENTITY_VAR`] (`{static}.{sign}.{local}`, each
/// base32; a secret), and the key of the address its participants join by.
pub fn launch_identity() -> Result<(Zeroizing<String>, AddressKey), SessionError> {
    let mut secrets = Zeroizing::new([[0_u8; 32]; 3]);
    for secret in secrets.iter_mut() {
        rlmesh_session_core::crypto::Rng::fill_bytes(&mut crate::host::OsRng, secret)
            .map_err(|err| SessionError::Config(err.to_string()))?;
    }
    let text = Zeroizing::new(secrets.iter().map(base32).collect::<Vec<_>>().join("."));
    let [mut noise, mut sign, mut local] = *secrets;
    let identity = HostIdentity::from_secrets(&mut noise, &mut sign, &mut local);
    Ok((text, AddressKey::of(&identity)))
}

/// The host keys [`HOST_IDENTITY_VAR`] passes this process, if any.
fn launched_identity() -> Result<Option<HostIdentity>, SessionError> {
    let Ok(text) = std::env::var(HOST_IDENTITY_VAR) else {
        return Ok(None);
    };
    let text = Zeroizing::new(text);
    let bad = || {
        SessionError::Config(format!(
            "{HOST_IDENTITY_VAR} is not {{static}}.{{sign}}.{{local}}"
        ))
    };
    let mut parts = text.trim().split('.').map(unbase32);
    let (Some(Some(noise)), Some(Some(sign)), Some(Some(local)), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(bad());
    };
    let (mut noise, mut sign, mut local) = (
        Zeroizing::new(noise),
        Zeroizing::new(sign),
        Zeroizing::new(local),
    );
    Ok(Some(HostIdentity::from_secrets(
        &mut noise, &mut sign, &mut local,
    )))
}

/// A parsed address: a session address has no participant. Its `Display`
/// and `Debug` leave the key out (`#…`); [`Address::invite`] has it.
#[derive(Clone, PartialEq, Eq)]
pub struct Address {
    /// The host (`None`: the local host, by the hosting rule).
    pub host: Option<HostName>,
    /// The session (`None`: the default session).
    pub session: Option<String>,
    /// The participant (a session address names none).
    pub participant: Option<String>,
    /// The key it carries, if any (`#...`).
    pub key: Option<AddressKey>,
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
    let text = redact(text);
    SessionError::Config(format!("address {text:?}: {why}"))
}

/// `text` with its key, if any, left out: what logs and errors show. The
/// key is a credential (it attaches as device `local`).
fn redact(text: &str) -> String {
    match text.split_once('#') {
        Some((before, rest)) => {
            let after = &rest[rest.find('/').unwrap_or(rest.len())..];
            format!("{before}#…{after}")
        }
        None => text.to_string(),
    }
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
    let text = redact(text);
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
        let (body, key) = split_key(trimmed.strip_prefix(SCHEME).unwrap_or(trimmed));
        let key = match key {
            Some(key) => Some(
                AddressKey::parse(key)
                    .ok_or_else(|| invalid(text, "the key after # is not {secret}.{host key}"))?,
            ),
            None => None,
        };
        let body = body.as_str();
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
            key,
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
            Placement::Join(host, port) => resolve(&self.text(false), &host, port)?,
        };
        Ok(SessionAddress {
            key: self.key.clone(),
            ..SessionAddress::new(base, self.session_name())?
        })
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

impl Address {
    /// Its text, the key in full (`full`) or left out. The key follows the
    /// session (or the host, for the default session).
    fn text(&self, full: bool) -> String {
        let mut segments = Vec::new();
        if let Some(host) = &self.host {
            segments.push(host.to_string());
        }
        if let Some(session) = &self.session {
            segments.push(session.clone());
        }
        if let Some(key) = &self.key {
            let key = if full { key.to_string() } else { "…".into() };
            match segments.last_mut() {
                Some(last) => *last = format!("{last}#{key}"),
                None => segments.push(format!("#{key}")),
            }
        }
        if let Some(name) = &self.participant {
            segments.push(name.clone());
        }
        segments.join("/")
    }

    /// The address with its key: the invite one process hands another (a
    /// credential, since its holder attaches as device `local`).
    pub fn invite(&self) -> String {
        self.text(true)
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text(false))
    }
}

impl fmt::Debug for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Address").field(&self.text(false)).finish()
    }
}

/// `text` without its key, and the key: it runs from `#` to the next `/`
/// (base32 has none), so a participant name appended to a keyed session
/// address still parses.
fn split_key(text: &str) -> (String, Option<&str>) {
    match text.split_once('#') {
        Some((before, rest)) => {
            let (key, after) = rest.split_at(rest.find('/').unwrap_or(rest.len()));
            (format!("{before}{after}"), Some(key))
        }
        None => (text.to_string(), None),
    }
}

/// Whether `text` is a v2 address (a session or participant address) rather
/// than a v1 endpoint (`host:port`, a bare port, or a `scheme://` URL). An
/// old `lockstep://` address counts as v2, so it meets its upgrade error,
/// and a keyed `host:port#key` is v2.
pub fn is_v2_address(text: &str) -> bool {
    let (text, key) = split_key(text.trim());
    if key.is_some() {
        return true;
    }
    let text = text.as_str();
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

/// Where a session's host listens: an address resolved to a socket. Its
/// `Display` and `Debug` leave the key out (`#…`), for logs, titles and
/// errors; [`SessionAddress::invite`] has it.
#[derive(Clone, PartialEq, Eq)]
pub struct SessionAddress {
    /// The host's control-class listener; the other classes follow it.
    pub base: SocketAddr,
    /// The tenant namespace ([`TENANT`]).
    pub tenant: TenantId,
    /// The session's name (or UUID text), as addresses print it.
    pub name: String,
    /// The session.
    pub session: SessionId,
    /// The key its host's address carries (`None`: found at join, for a
    /// host in this process or one that left its key for this user).
    pub key: Option<AddressKey>,
}

impl SessionAddress {
    /// The session `name` (or UUID) at `base`, in the local tenant, without
    /// a key.
    pub fn new(base: SocketAddr, name: &str) -> Result<Self, SessionError> {
        Ok(Self {
            base,
            tenant: TenantId::new(TENANT)?,
            name: name.to_string(),
            session: session_id(name),
            key: None,
        })
    }

    /// The credential its participants attach with: see [`credential`].
    pub fn credential(&self) -> Result<Credential, SessionError> {
        credential(self.base, self.key.as_ref())
    }

    /// The address with its key: the invite one process hands another (a
    /// launcher, a sandbox, a printed address). A credential: its holder
    /// attaches as device `local`.
    pub fn invite(&self) -> String {
        self.text(true)
    }

    /// The participant address of `participant` in this session, with the
    /// key: an invite, as [`Self::invite`].
    pub fn participant_invite(&self, participant: &str) -> String {
        format!("{}/{participant}", self.invite())
    }

    /// Its text, the key in full (`full`) or left out.
    fn text(&self, full: bool) -> String {
        let host = HostName {
            name: self.base.ip().to_string(),
            port: Some(self.base.port()),
        };
        let mut text = if self.name == DEFAULT_SESSION {
            host.to_string()
        } else {
            format!("{host}/{}", self.name)
        };
        match &self.key {
            Some(key) if full => text.push_str(&format!("#{key}")),
            Some(_) => text.push_str("#…"),
            None => {}
        }
        text
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
        f.write_str(&self.text(false))
    }
}

impl fmt::Debug for SessionAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("SessionAddress")
            .field(&self.text(false))
            .finish()
    }
}

impl SessionAddress {
    /// Joins the host as a Zenoh client (riding its transport when this
    /// process runs it), waiting up to `host_timeout` for it to listen and
    /// answer, and attaches as `id`, on this process's
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
        // A participant joins a host another process runs as a Zenoh
        // client: its one link is to the host, which routes for it. Joined
        // as peers, they were introduced to each other by the host's gossip,
        // so every participant that joined it, of any session, dialed every
        // other one (and a participant behind a sandbox's published ports
        // advertises addresses the other side cannot reach). Zenoh 1.10's
        // peer routing then lost declarations and replies, and runs of many
        // routes slowed and hung. The price: two participants that both join
        // one host reach each other through it rather than directly.
        let deadline = tokio::time::Instant::now() + host_timeout;
        let transport = match local_host(self.base) {
            Some(transport) => transport,
            None => {
                let tuning = TransportTuning {
                    client: true,
                    connect_timeout: Some(host_timeout),
                    ..TransportTuning::default()
                };
                Transport::open(
                    &TransportConfig::per_class_tcp(None, Some(self.base))?.with_tuning(tuning),
                )
                .await
                .map_err(|err| match err {
                    SessionError::Zenoh(_) => SessionError::Timeout(
                        host_timeout,
                        format!("the session host at {}", self.base),
                    ),
                    other => other,
                })?
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
        config.host_timeout = deadline.saturating_duration_since(tokio::time::Instant::now());
        config.workflow_edition = workflow_edition;
        config.workflow_editions = workflow_editions;
        config.credential = Some(self.credential()?);
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
    /// Where participants join, with the key when this process hosts it.
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
            forget_local_host(&self.address);
            host.shutdown().await;
            transport.close().await?;
        }
        Ok(())
    }
}

impl Drop for Placed {
    /// A placed session dropped unclosed (its owner cancelled or unwound)
    /// still stops the host it runs, on the runtime it is dropped on.
    fn drop(&mut self) {
        let Some((host, transport)) = self.hosted.take() else {
            return;
        };
        forget_local_host(&self.address);
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                host.shutdown().await;
                let _ = transport.close().await;
            });
        }
    }
}

/// The session hosts this process runs, by the address participants join
/// them at: the transport and the key. A participant joining one rides its
/// sessions instead of a loopback link: a link has a lease, which a starved
/// process can let lapse, and then loses the host it is running itself.
type LocalHosts = HashMap<SocketAddr, (Transport, AddressKey)>;

fn local_hosts() -> &'static std::sync::Mutex<LocalHosts> {
    static HOSTS: std::sync::OnceLock<std::sync::Mutex<LocalHosts>> = std::sync::OnceLock::new();
    HOSTS.get_or_init(Default::default)
}

fn local_host(base: SocketAddr) -> Option<Transport> {
    local_hosts()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&base)
        .map(|(transport, _)| Transport::ridden(transport))
}

/// The credential of a participant of the host at `base`: `key` (an
/// address's), else the key of a host on this machine (one this process
/// runs, or one that left its key for this user).
pub fn credential(base: SocketAddr, key: Option<&AddressKey>) -> Result<Credential, SessionError> {
    key.cloned()
        .or_else(|| local_key(base))
        .map(|key| key.credential())
        .ok_or_else(|| {
            SessionError::Rejected(CtlError::new(
                CtlStatus::Unauthenticated,
                format!(
                    "no key for the session at {base}: join with the address its host printed \
                     (it ends in #key)"
                ),
            ))
        })
}

/// The key of a host on this machine: one this process runs, or the file
/// one left for this user ([`key_file`]).
fn local_key(base: SocketAddr) -> Option<AddressKey> {
    if let Some((_, key)) = local_hosts()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&base)
    {
        return Some(key.clone());
    }
    if !base.ip().is_loopback() {
        return None;
    }
    let text = std::fs::read_to_string(key_file(base.port())?).ok()?;
    AddressKey::parse(text.trim())
}

/// Where an embedded host on `port` leaves its key for this user's other
/// processes (as Jupyter leaves its connection files): the user's runtime
/// directory, else their cache directory.
// TODO: rlmeshd's local.key (step 11) replaces these files for the
// daemon's sessions; embedded hosts keep them.
fn key_file(port: u16) -> Option<std::path::PathBuf> {
    let dir = dirs::runtime_dir().or_else(dirs::cache_dir)?;
    Some(dir.join("rlmesh").join("hosts").join(port.to_string()))
}

/// Leaves `key` for this user's processes joining a host on `port`: the
/// directory 0700 and the file 0600, written whole (renamed into place).
fn leave_key(port: u16, key: &AddressKey) -> std::io::Result<()> {
    let path = key_file(port).ok_or_else(|| std::io::Error::other("no home directory"))?;
    let dir = path
        .parent()
        .ok_or_else(|| std::io::Error::other("no directory"))?;
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    builder.create(dir)?;
    let partial = dir.join(format!(".{port}.{}", std::process::id()));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    std::io::Write::write_all(&mut options.open(&partial)?, key.to_string().as_bytes())?;
    std::fs::rename(&partial, &path)
}

/// This process stopped hosting at `address`: forget its transport, and
/// remove the key file it left, unless another host has replaced it.
fn forget_local_host(address: &SessionAddress) {
    local_hosts()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(&address.base);
    if let (Some(path), Some(key)) = (key_file(address.base.port()), &address.key)
        && std::fs::read_to_string(&path).is_ok_and(|text| text.trim() == key.to_string())
    {
        let _ = std::fs::remove_file(path);
    }
}

/// Whether a session host answers on `base` (its control listener accepts).
fn listening(base: SocketAddr) -> bool {
    TcpStream::connect_timeout(&joinable(base), Duration::from_millis(200)).is_ok()
}

/// Hosts session `name` in this process on `base` (port 0: OS-assigned),
/// and every other session of [`TENANT`] its participants name, under fresh
/// keys (or the ones its launcher passed in [`HOST_IDENTITY_VAR`]): the
/// returned address carries them, and on a named port they are also left
/// for this user's other processes.
pub async fn host_at(
    base: SocketAddr,
    name: &str,
    lease_grace: Duration,
) -> Result<Placed, SessionError> {
    serve_at(
        base,
        name,
        lease_grace,
        #[cfg(feature = "record")]
        None,
    )
    .await
}

/// [`host_at`], recording every session it hosts into `rolling`
/// ([`SessionServer::start_recording`]).
#[cfg(feature = "record")]
pub async fn host_recording_at(
    base: SocketAddr,
    name: &str,
    lease_grace: Duration,
    rolling: crate::record::Rolling,
) -> Result<Placed, SessionError> {
    serve_at(base, name, lease_grace, Some(rolling)).await
}

async fn serve_at(
    base: SocketAddr,
    name: &str,
    lease_grace: Duration,
    #[cfg(feature = "record")] rolling: Option<crate::record::Rolling>,
) -> Result<Placed, SessionError> {
    // Only a port someone can name is worth a key file: an OS-assigned one
    // is reached by the address this returns.
    let named_port = base.port() != 0;
    let (transport, base) = if base.port() == 0 {
        open_local_listeners(base.ip()).await?
    } else {
        let transport = Transport::open(&TransportConfig::per_class_tcp(Some(base), None)?).await?;
        (transport, base)
    };
    let mut address = SessionAddress::new(joinable(base), name)?;
    let mut config = HostConfig::new(address.tenant.clone(), address.session);
    config.clock = SystemMonotonicClock.shared();
    config.lease_grace = lease_grace.max(Duration::from_millis(1));
    config.health_log = None;
    if let Some(identity) = launched_identity()? {
        config.identity = std::sync::Arc::new(identity);
    }
    let key = AddressKey::of(&config.identity);
    address.key = Some(key.clone());
    #[cfg(feature = "record")]
    let host = match rolling {
        Some(rolling) => SessionServer::start_recording(transport.clone(), config, rolling).await?,
        None => SessionServer::start(transport.clone(), config).await?,
    };
    #[cfg(not(feature = "record"))]
    let host = SessionServer::start(transport.clone(), config).await?;
    if named_port && let Err(err) = leave_key(address.base.port(), &key) {
        tracing::warn!(%err, "the session's key was not left for local joiners: share its address");
    }
    local_hosts()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(address.base, (transport.clone(), key));
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
        let base = joinable(base);
        Ok(Placed {
            address: SessionAddress {
                key: address.key.clone().or_else(|| local_key(base)),
                ..SessionAddress::new(base, name)?
            },
            hosted: None,
        })
    };
    let base = match address.placement() {
        Placement::Join(host, port) => {
            return join(resolve(&address.text(false), &host, port)?);
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
            key: None,
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
        assert_eq!(
            default.participant_invite("cartpole"),
            "127.0.0.1:7447/cartpole"
        );
        let named = SessionAddress::new(base, "bench-3").expect("named");
        assert_eq!(named.to_string(), "127.0.0.1:7447/bench-3");
        let back: SessionAddress = named.to_string().parse().expect("parse");
        assert_eq!(back, named);
    }

    #[test]
    fn a_keyed_address_carries_its_key_through_a_participant_name() {
        let config = HostConfig::new(TenantId::new(TENANT).expect("tenant"), SessionId::new());
        let key = AddressKey::of(&config.identity);
        let base: SocketAddr = "127.0.0.1:7447".parse().expect("addr");
        for name in [DEFAULT_SESSION, "bench-3"] {
            let session = SessionAddress {
                key: Some(key.clone()),
                ..SessionAddress::new(base, name).expect("session")
            };
            let text = session.invite();
            assert!(text.ends_with(&format!("#{key}")), "{text}");
            assert!(is_v2_address(&text), "{text}");
            assert_eq!(text.parse::<SessionAddress>().expect("parse"), session);
            // Scripts append the participant: the key stays with its session.
            let invite = session.participant_invite("arm1");
            let participant = Address::participant(&invite).expect("parse");
            assert_eq!(participant.participant.as_deref(), Some("arm1"));
            assert_eq!(participant.joining().expect("joining"), session);
            assert_eq!(participant.invite(), invite);
            // Display and Debug, for logs, titles and errors, leave it out.
            let secret = key.to_string();
            let secret = secret.split('.').next().expect("secret");
            for shown in [
                session.to_string(),
                format!("{session:?}"),
                participant.to_string(),
                format!("{participant:?}"),
            ] {
                assert!(!shown.contains(secret) && shown.contains("#…"), "{shown}");
            }
        }
        assert_eq!(
            key.credential().device_public(),
            &config.identity.local_public(),
            "a holder derives the local device's key"
        );
        assert!(format!("{key:?}").len() < 20, "never logged");
        // A launcher's keys: the host it starts hosts under them.
        let (text, launched) = launch_identity().expect("keys");
        let mut parts = text.split('.').map(|p| unbase32(p).expect("base32"));
        let (mut a, mut b, mut c) = (
            parts.next().expect("static"),
            parts.next().expect("sign"),
            parts.next().expect("local"),
        );
        assert_eq!(
            AddressKey::of(&HostIdentity::from_secrets(&mut a, &mut b, &mut c)),
            launched
        );
        let err = Address::session("127.0.0.1:7447#nope").expect_err("bad key");
        assert!(err.to_string().contains("#"), "{err}");
        // Errors never show a key, well-formed or not.
        let secret = key.to_string();
        let secret = secret.split('.').next().expect("secret");
        let malformed = format!("127.0.0.1:7447/bench 3#{key}");
        let err = Address::session(&malformed).expect_err("a bad name");
        assert!(!err.to_string().contains(secret), "{err}");
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
        // A host this process placed needs no key in the address.
        let join = |base: SocketAddr, name: &str, key: Option<AddressKey>| {
            let session = SessionAddress {
                key,
                ..SessionAddress::new(base, name).expect("session")
            };
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
        let bench = join(base, "bench-3", None)
            .await
            .expect("a new session starts for its first participant");
        let lab = join(base, "robot-lab", None)
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
        let config = HostConfig::new(only.tenant.clone(), only.session);
        let key = AddressKey::of(&config.identity);
        let single = SessionHost::start(transport.clone(), config)
            .await
            .expect("single host");
        let started = std::time::Instant::now();
        let Err(err) = join(base, "elsewhere", Some(key)).await else {
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
