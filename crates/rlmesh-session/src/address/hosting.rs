//! The hosting rule: joining a host, or hosting the session in this process.

use std::collections::HashMap;
use std::fmt;
use std::net::{Ipv4Addr, SocketAddr};
use std::time::Duration;

use rlmesh_session_core::auth::AddressKey;

use crate::address::HOST_IDENTITY_VAR;
use crate::address::credentials::{hosted_key, write_private};
use crate::address::host_keys::KeyFile;
use crate::address::{
    Address, Placement, SessionAddress, TENANT, joinable, launched_identity, resolve, session_id,
    well_known_port,
};
use crate::clock::SystemMonotonicClock;
use crate::error::SessionError;
use crate::host::{BoardData, HostConfig};
use crate::server::SessionServer;
use crate::transport::{
    LINK_USER, LinkAuth, Transport, TransportConfig, TransportTuning, free_local_base,
};
use crate::wire::TenantId;

/// A session reached by the hosting rule: its address, and the host when
/// this process hosts it (kept with this value, stopped by [`Self::close`]).
pub struct Placed {
    /// Where participants join, with the key when this process hosts it.
    pub address: SessionAddress,
    pub(super) hosted: Option<(SessionServer, Transport)>,
    /// The key it left for this user's other processes, when it hosts on
    /// loopback.
    key_file: Option<KeyFile>,
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
    /// The session at `address`, joined rather than hosted: what a
    /// participant that must not host in another's place (a driver dialing
    /// a keyed address) reaches it as.
    pub fn joined(address: SessionAddress) -> Self {
        Self {
            address,
            hosted: None,
            key_file: None,
        }
    }

    /// Whether this process hosts the session.
    pub fn hosts(&self) -> bool {
        self.hosted.is_some()
    }

    /// Its participants' link health, when this process hosts it (empty
    /// otherwise).
    pub async fn health(&self) -> Vec<crate::ParticipantHealth> {
        let Some((server, _)) = &self.hosted else {
            return Vec::new();
        };
        let session = self.address.session;
        server
            .health()
            .await
            .into_iter()
            .find(|(id, _)| *id == session)
            .map(|(_, rows)| rows)
            .unwrap_or_default()
    }

    /// Stops the host, if this process runs it (participants see it leave).
    pub async fn close(&mut self) -> Result<(), SessionError> {
        self.key_file.take();
        if let Some((host, transport)) = self.hosted.take() {
            forget_local_host(&self.address);
            host.shutdown().await;
            transport.close().await?;
        }
        Ok(())
    }
}

impl Drop for Placed {
    fn drop(&mut self) {
        self.key_file.take();
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

type LocalHosts = HashMap<SocketAddr, (Transport, AddressKey)>;

pub(super) fn local_hosts() -> &'static std::sync::Mutex<LocalHosts> {
    static HOSTS: std::sync::OnceLock<std::sync::Mutex<LocalHosts>> = std::sync::OnceLock::new();
    HOSTS.get_or_init(Default::default)
}

pub(super) fn local_host(base: SocketAddr) -> Option<Transport> {
    local_hosts()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&base)
        .map(|(transport, _)| Transport::ridden(transport))
}

pub(super) fn local_host_keyed(key: &AddressKey) -> Option<Transport> {
    local_hosts()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .values()
        .find(|(_, hosted)| hosted == key)
        .map(|(transport, _)| Transport::ridden(transport))
}

fn forget_local_host(address: &SessionAddress) {
    local_hosts()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(&address.base);
}

pub(super) async fn listening(base: SocketAddr) -> bool {
    accepts(joinable(base)).await
}

async fn accepts(addr: SocketAddr) -> bool {
    matches!(
        tokio::time::timeout(LISTEN_PROBE, tokio::net::TcpStream::connect(addr)).await,
        Ok(Ok(_))
    )
}

const LISTEN_PROBE: Duration = Duration::from_secs(2);

const HOLD_PROBE: Duration = Duration::from_millis(250);

async fn holds(addr: SocketAddr) -> bool {
    use tokio::io::AsyncReadExt as _;
    let connect = tokio::time::timeout(LISTEN_PROBE, tokio::net::TcpStream::connect(addr));
    let Ok(Ok(mut stream)) = connect.await else {
        return false;
    };
    let mut byte = [0; 1];
    !matches!(
        tokio::time::timeout(HOLD_PROBE, stream.read(&mut byte)).await,
        Ok(Ok(0) | Err(_))
    )
}

const REFUSED_ROUNDS: u32 = 2;

const OPEN_ATTEMPT: Duration = Duration::from_secs(15);

pub(crate) fn refused_link(base: SocketAddr) -> SessionError {
    SessionError::Rejected(crate::wire::CtlError::new(
        crate::wire::CtlStatus::Unauthenticated,
        format!(
            "the session host at {base} listens but refused this participant's link: the \
             address's key is not that host's (a wrong or stale key, or none). Join with the \
             address its host printed or its launcher made (it ends in #key)"
        ),
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Foreign {
    Http2,
    Other,
}

const H2_PREFACE: &[u8] = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";

pub(super) const PROBE_AFTER: Duration = Duration::from_secs(1);

const PREFACE_PROBE: Duration = Duration::from_secs(1);

pub(crate) fn foreign(head: &[u8]) -> Foreign {
    match head {
        [_, _, _, 0x04, _, 0, 0, 0, 0, ..] => Foreign::Http2,
        _ => Foreign::Other,
    }
}

async fn first_bytes(stream: &mut tokio::net::TcpStream, within: Duration) -> Vec<u8> {
    use tokio::io::AsyncReadExt as _;
    let mut head = [0; 9];
    let mut read = 0;
    let deadline = tokio::time::Instant::now() + within;
    while read < head.len() {
        match tokio::time::timeout_at(deadline, stream.read(&mut head[read..])).await {
            Ok(Ok(n)) if n > 0 => read += n,
            _ => break,
        }
    }
    head[..read].to_vec()
}

async fn probe(addr: SocketAddr) -> Option<Foreign> {
    use tokio::io::AsyncWriteExt as _;
    let connect = tokio::time::timeout(LISTEN_PROBE, tokio::net::TcpStream::connect(addr));
    let mut stream = connect.await.ok()?.ok()?;
    stream.write_all(H2_PREFACE).await.ok()?;
    let head = first_bytes(&mut stream, PREFACE_PROBE).await;
    (!head.is_empty()).then(|| foreign(&head))
}

pub(super) async fn or_not_session(base: SocketAddr, error: SessionError) -> SessionError {
    if !listening(base).await {
        return error;
    }
    match probe(joinable(base)).await {
        Some(foreign) => not_session(base, foreign),
        None => error,
    }
}

pub(crate) fn not_session(base: SocketAddr, foreign: Foreign) -> SessionError {
    SessionError::NotSession(match foreign {
        Foreign::Http2 => format!(
            "the server at {base} doesn't speak the RLMesh session protocol: it answers \
             HTTP/2, as an RLMesh 0.1 (gRPC) server does. RLMesh 0.2 can't connect to 0.1; \
             serve it with RLMesh 0.2"
        ),
        Foreign::Other => format!(
            "the server at {base} doesn't speak the RLMesh session protocol; is it an RLMesh \
             0.1 (gRPC) server, or something else on that port? RLMesh 0.2 can't connect to 0.1"
        ),
    })
}

/// Links to the host at `base` as a Zenoh client within `timeout`, with no
/// password (a LAN host's, as rlmeshd's pairing endpoint is reached).
pub async fn connect(base: SocketAddr, timeout: Duration) -> Result<Transport, SessionError> {
    open_client(base, None, tokio::time::Instant::now() + timeout)
        .await
        .map_err(|err| match err {
            SessionError::Zenoh(_) => SessionError::Timeout(timeout, format!("a host at {base}")),
            SessionError::Timeout(_, what) => SessionError::Timeout(timeout, what),
            other => other,
        })
}

pub(super) async fn open_client(
    base: SocketAddr,
    link: Option<LinkAuth>,
    deadline: tokio::time::Instant,
) -> Result<Transport, SessionError> {
    let delayed = async {
        tokio::time::sleep(PROBE_AFTER).await;
        while !listening(base).await {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        match probe(joinable(base)).await {
            Some(foreign) => foreign,
            None => std::future::pending().await,
        }
    };
    tokio::select! {
        biased;
        joined = join_client(base, link, deadline) => match joined {
            Ok(transport) => Ok(transport),
            Err(error) => Err(or_not_session(base, error).await),
        },
        foreign = delayed => Err(not_session(base, foreign)),
    }
}

async fn join_client(
    base: SocketAddr,
    link: Option<LinkAuth>,
    deadline: tokio::time::Instant,
) -> Result<Transport, SessionError> {
    let open = |link, until: tokio::time::Instant| {
        let left = until.saturating_duration_since(tokio::time::Instant::now());
        let tuning = TransportTuning {
            client: true,
            open_timeout: Some(left.max(Duration::from_millis(1))),
            link,
            ..TransportTuning::default()
        };
        async move {
            let config = TransportConfig::per_class_tcp(base).with_tuning(tuning);
            tokio::time::timeout_at(until, Transport::open(&config))
                .await
                .unwrap_or_else(|_| Err(SessionError::Zenoh("no answer in time".into())))
        }
    };
    let attempts = [link, None];
    let attempts = if attempts[0].is_some() {
        &attempts[..]
    } else {
        &attempts[1..]
    };
    let mut refused = 0;
    loop {
        while tokio::time::Instant::now() < deadline {
            if listening(base).await {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let mut last = None;
        let mut answered = true;
        for attempt in attempts {
            let until = deadline.min(tokio::time::Instant::now() + OPEN_ATTEMPT);
            match open(attempt.clone(), until).await {
                Ok(transport) => return Ok(transport),
                Err(err) => last = Some(err),
            }
            if tokio::time::Instant::now() >= until {
                answered = false;
            }
            if tokio::time::Instant::now() >= deadline {
                break;
            }
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(last.unwrap_or_else(|| SessionError::Zenoh("no answer in time".into())));
        }
        let by_host = answered
            && tokio::time::timeout_at(deadline, holds(joinable(base)))
                .await
                .unwrap_or(false);
        refused = if by_host { refused + 1 } else { 0 };
        if refused >= REFUSED_ROUNDS {
            return Err(refused_link(base));
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Hosts session `name` in this process on `base` (port 0: OS-assigned),
/// and every other session of [`TENANT`] its participants name, under fresh
/// keys (or the ones its launcher passed in [`HOST_IDENTITY_VAR`]): the
/// returned address carries them, the invite for other processes.
///
/// It listens on loopback only, unless `insecure` or a launcher passed its keys. On loopback,
/// and when launched on any interface, every link must present the password the key derives;
/// an `insecure` host beyond loopback gates no link, and the handshake alone admits. Boards
/// are offered the datagram path as [`BOARD_DATA_VAR`](crate::host::BOARD_DATA_VAR) says, else
/// as [`BoardData::default`] does, on UDP at `base`'s port number.
pub async fn host_at(
    base: SocketAddr,
    name: &str,
    lease_grace: Duration,
    insecure: bool,
) -> Result<Placed, SessionError> {
    host_at_with(
        base,
        name,
        lease_grace,
        insecure,
        &HostingOptions::default(),
    )
    .await
}

/// How a host this process starts runs, beyond its address and lease
/// grace ([`host_at_with`], [`place_with`]). The default is what
/// [`host_at`] and [`place`] do.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HostingOptions {
    /// How boards carry their realtime parts (`None`: as
    /// [`BOARD_DATA_VAR`](crate::host::BOARD_DATA_VAR) says, else [`BoardData::default`]).
    pub board_data: Option<BoardData>,
}

impl HostingOptions {
    /// The board data path in force: the one chosen, else the
    /// environment's, else the default.
    ///
    /// # Errors
    /// [`SessionError::Config`] for a [`BOARD_DATA_VAR`](crate::host::BOARD_DATA_VAR) value `rlmesh host`
    /// would refuse, with its message.
    pub fn board_data(&self) -> Result<BoardData, SessionError> {
        match self.board_data {
            Some(chosen) => Ok(chosen),
            None => BoardData::from_env()
                .map(Option::unwrap_or_default)
                .map_err(SessionError::Config),
        }
    }
}

/// [`host_at`] with `options`.
///
/// # Errors
/// As [`host_at`], and [`HostingOptions::board_data`]'s.
pub async fn host_at_with(
    base: SocketAddr,
    name: &str,
    lease_grace: Duration,
    insecure: bool,
    options: &HostingOptions,
) -> Result<Placed, SessionError> {
    serve_at(
        base,
        name,
        lease_grace,
        insecure,
        options,
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
    insecure: bool,
    rolling: crate::record::Rolling,
) -> Result<Placed, SessionError> {
    serve_at(
        base,
        name,
        lease_grace,
        insecure,
        &HostingOptions::default(),
        Some(rolling),
    )
    .await
}

async fn serve_at(
    base: SocketAddr,
    name: &str,
    lease_grace: Duration,
    insecure: bool,
    options: &HostingOptions,
    #[cfg(feature = "record")] rolling: Option<crate::record::Rolling>,
) -> Result<Placed, SessionError> {
    let board_data = options.board_data()?;
    let launched = launched_identity()?;
    if !base.ip().is_loopback() && !insecure && launched.is_none() {
        return Err(SessionError::Config(format!(
            "an embedded host listens on loopback only, not {base}, unless its launcher passes \
             it a host identity: set {HOST_IDENTITY_VAR} to the identity \
             `rlmesh host-identity new --address HOST:PORT --participant NAME` (or \
             rlmesh.launch) makes, through the environment, and dial the keyed address it \
             prints; or listen on 127.0.0.1:{port} for this machine only. For a session other \
             machines join, run `rlmesh host`; to accept that anyone who can reach {base} can \
             try this host, pass insecure (Host(insecure=True))",
            port = base.port(),
        )));
    }
    let mut config = HostConfig::new(TenantId::new(TENANT)?, session_id(name));
    let launched = match launched {
        Some(identity) => {
            config.identity = std::sync::Arc::new(identity);
            true
        }
        None => false,
    };
    let gated = base.ip().is_loopback() || launched;
    let password = gated.then(|| config.identity.link_password());
    let (transport, base) = open_listener(base, password.as_ref()).await?;
    config.datagram = board_data.open(base).await?;
    config.datagram_strict = board_data.strict();
    let mut address = SessionAddress::new(joinable(base), name)?;
    config.clock = SystemMonotonicClock.shared();
    config.lease_grace = lease_grace.max(Duration::from_millis(1));
    config.health_log = None;
    let key = AddressKey::of(&config.identity);
    address.key = Some(key.clone());
    #[cfg(feature = "record")]
    let host = match rolling {
        Some(rolling) => SessionServer::start_recording(transport.clone(), config, rolling).await?,
        None => SessionServer::start(transport.clone(), config).await?,
    };
    #[cfg(not(feature = "record"))]
    let host = SessionServer::start(transport.clone(), config).await?;
    local_hosts()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(address.base, (transport.clone(), key.clone()));
    let key_file = (base.ip().is_loopback() && !launched)
        .then(|| KeyFile::write(address.base, &key))
        .flatten();
    Ok(Placed {
        address,
        hosted: Some((host, transport)),
        key_file,
    })
}

/// Opens a host's listener on `base` (port 0: a free local port), which
/// every class of every participant dials, every link gated by `password`
/// (Zenoh `usrpwd`) when there is one.
pub async fn open_listener(
    base: SocketAddr,
    password: Option<&zeroize::Zeroizing<String>>,
) -> Result<(Transport, SocketAddr), SessionError> {
    open_listener_with(base, password, TransportTuning::default()).await
}

/// [`open_listener`] with `tuning` for every session (its `link` is the
/// gate `password` makes).
///
/// # Errors
/// As [`open_listener`].
pub async fn open_listener_with(
    base: SocketAddr,
    password: Option<&zeroize::Zeroizing<String>>,
    tuning: TransportTuning,
) -> Result<(Transport, SocketAddr), SessionError> {
    let dictionary = password.map(dictionary).transpose()?;
    let tuning = TransportTuning {
        link: password
            .zip(dictionary.clone())
            .map(|(password, dictionary)| LinkAuth {
                password: password.clone(),
                dictionary: Some(dictionary),
            }),
        ..tuning
    };
    let open = |base| TransportConfig::host_tcp(base).with_tuning(tuning.clone());
    let opened = if base.port() == 0 {
        let mut last = SessionError::Config("no free local ports".into());
        let mut found = None;
        for _ in 0..16 {
            let free = free_local_base(base.ip())?;
            match Transport::open(&open(free)).await {
                Ok(transport) => {
                    found = Some((transport, free));
                    break;
                }
                Err(err) => last = err,
            }
        }
        found.ok_or(last)
    } else {
        Transport::open(&open(base))
            .await
            .map(|transport| (transport, base))
    };
    if let Some(dictionary) = dictionary {
        discard_dictionary(&dictionary);
    }
    opened
}

pub(crate) const RUNTIME_DIR_VAR: &str = "RLMESH_RUNTIME_DIR";

const PRIVATE_DIR_PREFIX: &str = ".rlmesh-link-";

fn dictionary(password: &zeroize::Zeroizing<String>) -> Result<std::path::PathBuf, SessionError> {
    static OPENED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = OPENED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let name = format!(".link.{}.{n}", std::process::id());
    let contents = zeroize::Zeroizing::new(format!("{LINK_USER}:{}", password.as_str()));
    if let Some(dir) = std::env::var_os(RUNTIME_DIR_VAR) {
        let path = std::path::Path::new(&dir).join(&name);
        return write_private(&path, contents.as_bytes())
            .map(|()| path)
            .map_err(|err| {
                SessionError::Config(format!(
                    "the link gate in {RUNTIME_DIR_VAR}={}: {err}",
                    std::path::Path::new(&dir).display()
                ))
            });
    }
    gate_in(
        default_runtime_dirs(),
        &std::env::temp_dir(),
        &name,
        contents.as_bytes(),
    )
}

fn default_runtime_dirs() -> impl Iterator<Item = std::path::PathBuf> {
    [dirs::runtime_dir(), dirs::cache_dir()]
        .into_iter()
        .flatten()
        .map(|dir| dir.join("rlmesh"))
}

pub(super) fn runtime_dirs() -> Vec<std::path::PathBuf> {
    match std::env::var_os(RUNTIME_DIR_VAR) {
        Some(dir) => vec![dir.into()],
        None => default_runtime_dirs().collect(),
    }
}

fn gate_in(
    dirs: impl IntoIterator<Item = std::path::PathBuf>,
    temp: &std::path::Path,
    name: &str,
    contents: &[u8],
) -> Result<std::path::PathBuf, SessionError> {
    for dir in dirs {
        let path = dir.join(name);
        if write_private(&path, contents).is_ok() {
            return Ok(path);
        }
    }
    let written = private_temp_dir(temp).and_then(|dir| {
        let path = dir.join(name);
        write_private(&path, contents)
            .map(|()| path)
            .inspect_err(|_| {
                let _ = std::fs::remove_dir_all(&dir);
            })
    });
    written.map_err(|err| {
        SessionError::Config(format!(
            "no writable directory for the link gate (the runtime and cache directories, then \
             {}: {err}); set {RUNTIME_DIR_VAR} to a directory only this user can write",
            temp.display()
        ))
    })
}

fn private_temp_dir(temp: &std::path::Path) -> std::io::Result<std::path::PathBuf> {
    let mut random = [0_u8; 16];
    rlmesh_session_core::crypto::Rng::fill_bytes(&mut crate::host::OsRng, &mut random)
        .map_err(|err| std::io::Error::other(err.to_string()))?;
    let random: String = random.iter().map(|b| format!("{b:02x}")).collect();
    let dir = temp.join(format!(
        "{PRIVATE_DIR_PREFIX}{}-{random}",
        std::process::id()
    ));
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    builder.create(&dir)?;
    Ok(dir)
}

fn discard_dictionary(path: &std::path::Path) {
    let _ = std::fs::remove_file(path);
    if let Some(dir) = path.parent().filter(|dir| {
        dir.file_name()
            .is_some_and(|name| name.to_string_lossy().starts_with(PRIVATE_DIR_PREFIX))
    }) {
        let _ = std::fs::remove_dir(dir);
    }
}

/// Reaches the session `address` names by the hosting rule: joins the local
/// host on [`well_known_port`] (or the address's explicit local port) when one
/// listens, else hosts the session in this process there; a bind that loses a
/// race joins the winner. Another machine's host is joined. Leases in a session
/// this process hosts lapse `lease_grace` after their holder's last heartbeat.
pub async fn place(address: &Address, lease_grace: Duration) -> Result<Placed, SessionError> {
    place_with(address, lease_grace, &HostingOptions::default()).await
}

/// [`place`], a session hosted here run with `options` (a joined one
/// ignores them).
///
/// # Errors
/// As [`place`], and [`HostingOptions::board_data`]'s when it hosts.
pub async fn place_with(
    address: &Address,
    lease_grace: Duration,
    options: &HostingOptions,
) -> Result<Placed, SessionError> {
    let name = address.session_name();
    let join = |base: SocketAddr| -> Result<Placed, SessionError> {
        let base = joinable(base);
        Ok(Placed {
            address: SessionAddress {
                key: address.key.clone().or_else(|| hosted_key(base)),
                ..SessionAddress::new(base, name)?
            },
            hosted: None,
            key_file: None,
        })
    };
    let base = match address.placement() {
        Placement::Join(host, port) => {
            return join(resolve(&address.text(false), &host, port)?);
        }
        Placement::Local => {
            let base = SocketAddr::from((Ipv4Addr::LOCALHOST, well_known_port()?));
            if listening(base).await {
                return join(base);
            }
            base
        }
        Placement::Host(base) if base.port() != 0 && listening(base).await => return join(base),
        Placement::Host(base) => base,
    };
    match host_at_with(base, name, lease_grace, address.insecure, options).await {
        Ok(placed) => Ok(placed),
        Err(_) if base.port() != 0 && listening(base).await => join(base),
        Err(err) => Err(err),
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::fs::PermissionsExt;
    use std::path::Path;
    use std::process::Command;

    use super::*;

    const CHILD: &str = "RLMESH_TEST_LINK_GATE_CHILD";

    fn mode(path: &Path) -> u32 {
        std::fs::metadata(path).expect("stat").permissions().mode() & 0o777
    }

    fn blocked(scratch: &Path, name: &str) -> std::path::PathBuf {
        let path = scratch.join(name);
        std::fs::write(&path, b"").expect("write a blocking file");
        path
    }

    #[test]
    fn a_gate_with_no_default_directory_gets_a_private_temp_dir() {
        let scratch = tempfile::tempdir().unwrap();
        let temp = scratch.path().join("tmp");
        std::fs::create_dir(&temp).unwrap();
        let unwritable = [
            blocked(scratch.path(), "runtime"),
            blocked(scratch.path(), "cache"),
        ]
        .map(|dir| dir.join("rlmesh"));

        let first = gate_in(unwritable.clone(), &temp, ".link.1", b"u:p").unwrap();
        let second = gate_in(unwritable, &temp, ".link.1", b"u:p").unwrap();
        let dir = first.parent().unwrap();
        assert_eq!(dir.parent(), Some(temp.as_path()));
        assert_ne!(
            dir,
            second.parent().unwrap(),
            "each gate gets a directory of its own"
        );
        assert_eq!(mode(dir), 0o700);
        assert_eq!(mode(&first), 0o600);
        assert_eq!(std::fs::read(&first).unwrap(), b"u:p");

        discard_dictionary(&first);
        discard_dictionary(&second);
        assert_eq!(std::fs::read_dir(&temp).unwrap().count(), 0);
    }

    #[test]
    fn a_writable_default_is_used_and_kept() {
        let scratch = tempfile::tempdir().unwrap();
        let cache = scratch.path().join("cache").join("rlmesh");
        let gate = gate_in([cache.clone()], scratch.path(), ".link.2", b"u:p").unwrap();
        assert_eq!(gate, cache.join(".link.2"));
        discard_dictionary(&gate);
        assert!(cache.is_dir(), "a shared directory outlives its gates");
    }

    #[test]
    fn nothing_writable_names_the_override() {
        let scratch = tempfile::tempdir().unwrap();
        let temp = blocked(scratch.path(), "tmp");
        let error = gate_in([], &temp, ".link.3", b"u:p").unwrap_err();
        assert!(error.to_string().contains(RUNTIME_DIR_VAR), "{error}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn gate_child() {
        if std::env::var_os(CHILD).is_none() {
            return;
        }
        let base = SocketAddr::from((Ipv4Addr::LOCALHOST, 0));
        let mut placed = host_at(base, "gate", Duration::from_millis(250), false)
            .await
            .expect("host on loopback");
        placed.close().await.expect("close");
    }

    fn host_without_home(temp: &Path, cache: &Path, runtime_dir: Option<&Path>) -> String {
        let mut command = Command::new(std::env::current_exe().expect("test binary"));
        command
            .args([
                "--exact",
                "address::hosting::tests::gate_child",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env_remove("HOME")
            .env_remove("XDG_RUNTIME_DIR")
            .env_remove(crate::address::HOST_IDENTITY_VAR)
            .env_remove(RUNTIME_DIR_VAR)
            .env("XDG_CACHE_HOME", cache)
            .env("TMPDIR", temp);
        if let Some(dir) = runtime_dir {
            command.env(RUNTIME_DIR_VAR, dir);
        }
        let output = command.output().expect("run the child");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            output.status.success(),
            !text.contains("panicked"),
            "{text}"
        );
        text
    }

    const KEYLESS_CHILD: &str = "RLMESH_TEST_KEYLESS_CHILD";

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn keyless_child() {
        use crate::address::credentials::credential;
        use crate::wire::CtlStatus;
        if std::env::var_os(KEYLESS_CHILD).is_none() {
            return;
        }
        let daemon: AddressKey = std::fs::read_to_string(
            crate::address::daemon_dir()
                .expect("state dir")
                .join(crate::address::LOCAL_KEY),
        )
        .expect("the scratch daemon key")
        .parse()
        .expect("a key");
        let wide = SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0));
        let mut host = host_at(wide, "lan", Duration::from_millis(250), true)
            .await
            .expect("insecure host");
        let port = host.address.base.port();
        let hosted = host.address.key.clone().expect("hosted key");
        let loopback = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
        assert_eq!(
            credential(loopback, None).expect("hosted key"),
            hosted.credential()
        );
        let other = SocketAddr::from((Ipv4Addr::LOCALHOST, port.wrapping_add(7)));
        assert_eq!(
            credential(other, None).expect("daemon key"),
            daemon.credential()
        );
        let remote = SocketAddr::from(([192, 0, 2, 7], port));
        let refused = credential(remote, None).expect_err("no key beyond loopback");
        assert_eq!(
            refused.refusal().map(|r| r.status),
            Some(CtlStatus::Unauthenticated),
            "{refused}"
        );
        assert_eq!(
            crate::address::link_auth(remote, Some(&hosted)).map(|link| link.password),
            Some(hosted.link_password())
        );
        assert!(crate::address::link_auth(remote, None).is_none());
        host.close().await.expect("close");
    }

    #[test]
    fn a_keyless_join_takes_a_key_only_for_a_loopback_host() {
        let scratch = tempfile::tempdir().unwrap();
        let (_, key) = crate::address::launch_identity().unwrap();
        crate::address::write_private(
            &scratch.path().join(crate::address::LOCAL_KEY),
            key.to_string().as_bytes(),
        )
        .unwrap();
        let output = Command::new(std::env::current_exe().expect("test binary"))
            .args([
                "--exact",
                "address::hosting::tests::keyless_child",
                "--nocapture",
            ])
            .env(KEYLESS_CHILD, "1")
            .env(crate::address::STATE_DIR_VAR, scratch.path())
            .env_remove(crate::address::HOST_IDENTITY_VAR)
            .output()
            .expect("run the child");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            output.status.success() && text.contains("1 passed"),
            "{text}"
        );
    }

    #[test]
    fn gate_without_home() {
        let scratch = tempfile::tempdir().unwrap();
        let cache = blocked(scratch.path(), "cache");
        let temp = scratch.path().join("tmp");
        std::fs::create_dir(&temp).unwrap();

        let text = host_without_home(&temp, &cache, None);
        assert!(!text.contains("panicked"), "{text}");
        assert_eq!(
            std::fs::read_dir(&temp).unwrap().count(),
            0,
            "the gate is cleaned up"
        );

        let blocked_temp = blocked(scratch.path(), "no-tmp");
        let text = host_without_home(&blocked_temp, &cache, None);
        assert!(text.contains(RUNTIME_DIR_VAR), "{text}");

        let runtime = scratch.path().join("runtime");
        let text = host_without_home(&blocked_temp, &cache, Some(&runtime));
        assert!(!text.contains("panicked"), "{text}");
        assert!(runtime.is_dir());
        let left: Vec<_> = std::fs::read_dir(&runtime)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(left, ["hosts"], "{left:?}");
        assert_eq!(std::fs::read_dir(runtime.join("hosts")).unwrap().count(), 0);
    }
}
