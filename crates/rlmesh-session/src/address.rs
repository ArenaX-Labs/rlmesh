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
//! address is the invite: its holder attaches as device
//! `local`. A process joining a host on this machine without the key uses
//! rlmeshd's ([`LOCAL_KEY`] in [`daemon_dir`], 0600), so this user's
//! scripts meet by name at the daemon.

mod credentials;
mod hosting;

#[cfg(feature = "fleet")]
pub use credentials::remember_credential;
pub use credentials::{LOCAL_KEY, STATE_DIR_VAR, credential, daemon_dir, link_auth, write_private};
#[cfg(feature = "record")]
pub use hosting::host_recording_at;
pub use hosting::{
    HostingOptions, Placed, connect, host_at, host_at_with, open_listeners, open_listeners_with,
    place, place_with,
};

use std::fmt;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, ToSocketAddrs};
use std::str::FromStr;
use std::time::Duration;

pub use rlmesh_session_core::auth::AddressKey;
use rlmesh_session_core::auth::address::{base32, unbase32};
use rlmesh_session_core::auth::{Credential, HostIdentity};
use zeroize::Zeroizing;

#[cfg(feature = "fleet")]
use crate::address::credentials::remembered;
use crate::address::hosting::{listening, local_host, open_client};
use crate::clock::SystemMonotonicClock;
use crate::error::SessionError;
use crate::participant::{Participant, ParticipantConfig};
#[cfg(doc)]
use crate::server::SessionServer;
use crate::transport::{LinkAuth, Transport};
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

/// The environment variable a launcher passes the keys of the host it
/// starts in (a sandbox container) through, as [`launch_identity`] makes
/// them. [`host_at`] hosts under them.
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

/// The key of the address participants join a host by when it runs under
/// `identity` ([`launch_identity`]'s text, a secret), for a launcher that
/// made the identity elsewhere: the key [`launch_identity`] returned with
/// it.
pub fn identity_key(identity: &str) -> Result<AddressKey, SessionError> {
    Ok(AddressKey::of(&host_identity(identity)?))
}

/// The address of the session `session` (`host:port[/session]`) on a host
/// run under `identity` ([`launch_identity`]'s text), with the host's key:
/// set its `participant` for a participant's, and print it with
/// [`Address::invite`] (a credential).
pub fn launched_address(identity: &str, session: &str) -> Result<Address, SessionError> {
    let mut address = Address::session(session)?;
    if address.host.is_none() {
        return Err(invalid(session, "names no host (host:port[/session])"));
    }
    if address.key.is_some() {
        return Err(invalid(session, "carries a key already"));
    }
    address.key = Some(identity_key(identity)?);
    Ok(address)
}

/// The host keys [`HOST_IDENTITY_VAR`] passes this process, if any.
fn launched_identity() -> Result<Option<HostIdentity>, SessionError> {
    let Ok(text) = std::env::var(HOST_IDENTITY_VAR) else {
        return Ok(None);
    };
    host_identity(&Zeroizing::new(text)).map(Some)
}

/// The host keys of identity text (`{static}.{sign}.{local}`), which the
/// error never repeats.
fn host_identity(text: &str) -> Result<HostIdentity, SessionError> {
    let bad = || {
        SessionError::Config(format!(
            "a host identity ({HOST_IDENTITY_VAR}) is {{static}}.{{sign}}.{{local}}, each base32"
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
    Ok(HostIdentity::from_secrets(
        &mut noise, &mut sign, &mut local,
    ))
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
    /// Host it beyond loopback anyway ([`host_at`]); never in its text.
    pub insecure: bool,
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
pub fn redact(text: &str) -> String {
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
                key.parse::<AddressKey>()
                    .map_err(|_| invalid(text, "the key after # is not {secret}.{host key}"))?,
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
        // `_lobby` and the like are the daemon's own.
        if session.is_some_and(|s| s.starts_with('_')) {
            return Err(invalid(text, "session names starting with _ are reserved"));
        }
        if let Some(name) = name {
            ParticipantId::new(name).map_err(|e| invalid(text, e))?;
        }
        Ok(Self {
            insecure: false,
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
    /// What its participants attach with, ahead of the key and any found
    /// by `base`: a paired machine's device key against the host its host
    /// file names, or against a satellite's run host.
    /// Never in its text.
    pub credential: Option<Credential>,
}

/// How long one try at a loopback link gets while a host in this process
/// may still be starting ([`SessionAddress::link`]).
const LOCAL_HOST_TRY: Duration = Duration::from_millis(500);

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
            credential: None,
        })
    }

    /// The credential its participants attach with: its own, else the one
    /// this process was given for its session (`remember_credential`,
    /// feature `fleet`), else see [`credential`].
    pub fn credential(&self) -> Result<Credential, SessionError> {
        if let Some(credential) = self.credential.clone() {
            return Ok(credential);
        }
        #[cfg(feature = "fleet")]
        if let Some(credential) = remembered()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&(self.base, self.session))
            .cloned()
        {
            return Ok(credential);
        }
        credential(self.base, self.key.as_ref())
    }

    /// The address with its key: the invite one process hands another (a
    /// launcher, a sandbox, a printed address). A credential: its holder
    /// attaches as device `local`.
    pub fn invite(&self) -> String {
        self.text(true)
    }

    /// What its links to the host present: see [`link_auth`].
    pub fn link_auth(&self) -> Option<LinkAuth> {
        link_auth(self.base, self.key.as_ref())
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
    // A name that does not resolve (yet: a container's peer before it starts)
    // is the lookup failing, not the address: an I/O error, which a retry may
    // get past, where a malformed address is a configuration error.
    let unresolved = |kind, why: String| {
        SessionError::Io(std::io::Error::new(
            kind,
            format!("address {:?}: host {host:?}: {why}", redact(text)),
        ))
    };
    (host, port)
        .to_socket_addrs()
        .map_err(|e| unresolved(e.kind(), e.to_string()))?
        .next()
        .ok_or_else(|| unresolved(std::io::ErrorKind::NotFound, "resolves to nothing".into()))
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
        self.join_with(id, roles, priority, host_timeout, |config| {
            config.workflow_edition = workflow_edition;
            config.workflow_editions = workflow_editions;
        })
        .await
    }

    /// Whether the key it joins with would come from a host on this machine
    /// (it carries none, and no host in this process has brought one).
    fn awaits_host_key(&self) -> bool {
        self.credential.is_none() && self.key.is_none() && local_host(self.base).is_none()
    }

    /// [`Self::join`], its config finished by `finish`.
    pub(crate) async fn join_with(
        &self,
        id: &str,
        roles: Vec<Role>,
        priority: PriorityClass,
        host_timeout: Duration,
        finish: impl FnOnce(&mut ParticipantConfig),
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
        let deadline = crate::clock::deadline_after(host_timeout);
        // A host on this machine not up yet brings its key when it is (one
        // this process runs, or rlmeshd's). rlmeshd's key is on disk whether
        // or not rlmeshd is the host at this address, so holding it alone
        // doesn't end the wait: a host listening does.
        while self.base.ip().is_loopback()
            && tokio::time::Instant::now() < deadline
            && (self.credential().is_err()
                || (self.awaits_host_key() && !listening(self.base).await))
        {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        // Without a key there is nothing to link with: say so before
        // dialing. Taken again once linked, when a host this process runs
        // may have brought its own.
        self.credential()?;
        let transport = self.link(deadline, host_timeout).await?;
        let mut config = ParticipantConfig::new(
            self.tenant.clone(),
            self.session,
            ParticipantId::new(id)?,
            roles,
            priority,
        );
        config.clock = SystemMonotonicClock.shared();
        config.host_timeout = deadline.saturating_duration_since(tokio::time::Instant::now());
        finish(&mut config);
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

impl SessionAddress {
    /// A link to the host by `deadline`, as a Zenoh client (riding its
    /// transport when this process runs it): through the password gate of a
    /// loopback or launched host, on any network, with the address's key,
    /// and without one to a host that gates no link (rlmeshd, an `insecure`
    /// host), which the handshake alone admits. `waited` names the wait in a
    /// timeout.
    pub async fn link(
        &self,
        deadline: tokio::time::Instant,
        waited: Duration,
    ) -> Result<Transport, SessionError> {
        loop {
            if let Some(transport) = local_host(self.base) {
                return Ok(transport);
            }
            // A host this process is still starting listens before it is
            // known here, and gates its links: try briefly, then look again.
            let starting = self.base.ip().is_loopback() && self.awaits_host_key();
            let until = if starting {
                deadline.min(tokio::time::Instant::now() + LOCAL_HOST_TRY)
            } else {
                deadline
            };
            match open_client(self.base, self.link_auth(), until).await {
                // Not yet, or refused by a host this process is still
                // registering: look again. A refusal at the deadline is
                // final ([`hosting::refused_link`]); running out of time
                // isn't, a slow or starting host may answer the next try.
                Err(
                    SessionError::Zenoh(_) | SessionError::Rejected(_) | SessionError::Timeout(..),
                ) if until < deadline => {}
                Err(SessionError::Zenoh(_)) => {
                    return Err(SessionError::Timeout(
                        waited,
                        format!("the session host at {}", self.base),
                    ));
                }
                // Its class ports unanswered ([`hosting::unanswered_ports`]).
                Err(SessionError::Timeout(_, what)) => {
                    return Err(SessionError::Timeout(waited, what));
                }
                other => return other,
            }
        }
    }
}

#[cfg(test)]
mod tests;
