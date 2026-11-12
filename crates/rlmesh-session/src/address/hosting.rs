//! The hosting rule: joining a host, or hosting the session in this process.

use std::collections::HashMap;
use std::fmt;
use std::net::{Ipv4Addr, SocketAddr};
use std::time::Duration;

use rlmesh_session_core::auth::AddressKey;

use crate::address::HOST_IDENTITY_VAR;
use crate::address::credentials::{hosted_key, write_private};
use crate::address::{
    Address, Placement, SessionAddress, TENANT, joinable, launched_identity, resolve, session_id,
    well_known_port,
};
use crate::clock::SystemMonotonicClock;
use crate::error::SessionError;
use crate::host::{BoardData, HostConfig};
use crate::server::SessionServer;
use crate::transport::{
    LINK_USER, LinkAuth, TrafficClass, Transport, TransportConfig, TransportTuning, class_port,
    free_local_base,
};
use crate::wire::TenantId;

/// A session reached by the hosting rule: its address, and the host when
/// this process hosts it (kept with this value, stopped by [`Self::close`]).
pub struct Placed {
    /// Where participants join, with the key when this process hosts it.
    pub address: SessionAddress,
    pub(super) hosted: Option<(SessionServer, Transport)>,
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

/// This process stopped hosting at `address`: forget its transport.
fn forget_local_host(address: &SessionAddress) {
    local_hosts()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(&address.base);
}

/// Whether a session host answers on `base` (its control listener accepts).
/// Generous for a slow link, where a TCP connect takes a round trip; a port
/// nothing listens on refuses at once. Asynchronous: a connect that waits
/// out a dropped SYN must not hold the thread polling the join.
pub(super) async fn listening(base: SocketAddr) -> bool {
    accepts(joinable(base)).await
}

/// Whether a TCP connect to `addr` succeeds within [`LISTEN_PROBE`].
async fn accepts(addr: SocketAddr) -> bool {
    matches!(
        tokio::time::timeout(LISTEN_PROBE, tokio::net::TcpStream::connect(addr)).await,
        Ok(Ok(_))
    )
}

/// How long [`listening`] waits for a TCP connect.
const LISTEN_PROBE: Duration = Duration::from_secs(2);

/// How many rounds a host listening on every class's port may refuse every
/// link before [`open_client`] calls it [`refused_link`]: a second round
/// rules out a host still opening its listeners.
const REFUSED_ROUNDS: u32 = 2;

/// How long one attempt at a link may take before [`open_client`] gives up
/// on it and tries again, whatever the join's own deadline: an open whose
/// handshake reply never comes (a lost segment, a host that dropped it) is
/// not left waiting out a join timeout of many minutes. A slowed link opens
/// its four classes in a few seconds.
const OPEN_ATTEMPT: Duration = Duration::from_secs(15);

/// A host that listens at `base` but refused every link offered it: the
/// key this participant joins with is not that host's (a wrong or stale
/// one, or none where it gates links). Final, unlike a host that does not
/// listen yet: retrying with the same key cannot link.
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

/// Joins the host at `base` as a Zenoh client by `deadline`, presenting
/// `link` on any network. Whether a host gates its links isn't known from
/// its address: a launched or loopback host gates every link, and one bound
/// beyond loopback otherwise (`insecure`, rlmeshd's LAN listener) gates
/// none, since Zenoh can't gate one listener and not another, and refuses a
/// password it never asked for. The handshake that follows authenticates
/// the host either way.
///
/// Once the host listens, each round makes one attempt with the password and
/// one without (only the latter with no `link`), each allowed until
/// `deadline` or [`OPEN_ATTEMPT`], whichever comes first, so a slow host is
/// never cut short and a lost handshake is tried again. A refusal comes back
/// within a round trip, a slow link only later: an attempt that fails before
/// its bound was refused. A host listening on every class's port that
/// refuses every attempt, round after round, gates its links with another
/// key: [`refused_link`], final. Running out of time is a timeout (a Zenoh
/// error), which a retry may get past, or, when the base port answered and
/// some class's port did not, [`unanswered_ports`].
pub(super) async fn open_client(
    base: SocketAddr,
    link: Option<LinkAuth>,
    deadline: tokio::time::Instant,
) -> Result<Transport, SessionError> {
    // One attempt, as Zenoh clients make, bounded by `until`: Zenoh's own
    // handshake timeout is lifted to it, so it never ends a slow open early,
    // which would read as a refusal.
    let open = |link, until: tokio::time::Instant| {
        let left = until.saturating_duration_since(tokio::time::Instant::now());
        let tuning = TransportTuning {
            client: true,
            open_timeout: Some(left.max(Duration::from_millis(1))),
            link,
            ..TransportTuning::default()
        };
        async move {
            let config = TransportConfig::per_class_tcp(None, Some(base))?.with_tuning(tuning);
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
    // A base port that leaves no room for the classes never links.
    TransportConfig::per_class_tcp(None, Some(base))?;
    let started = tokio::time::Instant::now();
    let mut refused = 0;
    // The class ports that did not answer, as last probed once the base did:
    // kept when the deadline cuts a round's probe short.
    let mut silent = Vec::new();
    loop {
        let mut base_answered = false;
        while tokio::time::Instant::now() < deadline {
            if listening(base).await {
                base_answered = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        if base_answered
            && let Ok(probed) = tokio::time::timeout_at(deadline, silent_class_ports(base)).await
        {
            silent = probed;
        }
        let mut last = None;
        // Whether every attempt failed before its bound: refused, not slow.
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
            if !silent.is_empty() {
                return Err(unanswered_ports(base, &silent, started.elapsed()));
            }
            return Err(last.unwrap_or_else(|| SessionError::Zenoh("no answer in time".into())));
        }
        refused = if answered && silent.is_empty() {
            refused + 1
        } else {
            0
        };
        if refused >= REFUSED_ROUNDS {
            return Err(refused_link(base));
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// The class ports of the host at `base` that accept no TCP connect, probed
/// at once: empty when it listens on every one, so a link it refuses was
/// refused by the host, not by a port not yet (or never) open.
async fn silent_class_ports(base: SocketAddr) -> Vec<u16> {
    let base = joinable(base);
    let probe = |class| async move {
        let port = class_port(base.port(), class).ok()?;
        (!accepts(SocketAddr::new(base.ip(), port)).await).then_some(port)
    };
    let [a, b, c, d] = TrafficClass::ALL.map(probe);
    let (a, b, c, d) = tokio::join!(a, b, c, d);
    [a, b, c, d].into_iter().flatten().collect()
}

/// A join that ran out of time while the host at `base` answered on its base
/// port but not on `silent`: typically a container that publishes one port.
fn unanswered_ports(base: SocketAddr, silent: &[u16], waited: Duration) -> SessionError {
    let ports = silent
        .iter()
        .map(u16::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let noun = if silent.len() == 1 { "port" } else { "ports" };
    let first = base.port();
    let last = first.saturating_add(TrafficClass::ALL.len() as u16 - 1);
    SessionError::Timeout(
        waited,
        format!(
            "the session host at {base}: {noun} {ports} did not answer; a participant must reach \
             all four of its ports, {first} to {last}, one per traffic class (a container must \
             publish each)"
        ),
    )
}

/// Hosts session `name` in this process on `base` (port 0: OS-assigned),
/// and every other session of [`TENANT`] its participants name, under fresh
/// keys (or the ones its launcher passed in [`HOST_IDENTITY_VAR`]): the
/// returned address carries them, the invite for other processes.
///
/// It listens on loopback only unless `insecure`,
/// or a launcher passed its keys (a sandbox's container, whose published
/// ports its launcher's participants dial with them, from this machine or
/// another): LAN sessions are rlmeshd's. On loopback, and when launched on
/// any interface, every link must present the password the key derives, so
/// only holders of the keyed address link at all. An `insecure` host beyond
/// loopback gates no link: Zenoh gates every listener of a router or none,
/// and a board's zenoh-pico has no `usrpwd`, so there the handshake alone
/// admits, as on any LAN link.
///
/// It offers boards the datagram path as [`BoardData::default`] says, on
/// UDP `base` + 1; a port that won't open leaves them on TCP.
pub async fn host_at(
    base: SocketAddr,
    name: &str,
    lease_grace: Duration,
    insecure: bool,
) -> Result<Placed, SessionError> {
    serve_at(
        base,
        name,
        lease_grace,
        insecure,
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
    serve_at(base, name, lease_grace, insecure, Some(rolling)).await
}

async fn serve_at(
    base: SocketAddr,
    name: &str,
    lease_grace: Duration,
    insecure: bool,
    #[cfg(feature = "record")] rolling: Option<crate::record::Rolling>,
) -> Result<Placed, SessionError> {
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
    let (transport, base) = open_listeners(base, password.as_ref()).await?;
    let board_data = BoardData::default();
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
        .insert(address.base, (transport.clone(), key));
    Ok(Placed {
        address,
        hosted: Some((host, transport)),
    })
}

/// Opens a host's per-class listeners on `base` (port 0: free local
/// ports), every link gated by `password` (Zenoh `usrpwd`) when
/// there is one.
pub async fn open_listeners(
    base: SocketAddr,
    password: Option<&zeroize::Zeroizing<String>>,
) -> Result<(Transport, SocketAddr), SessionError> {
    open_listeners_with(base, password, TransportTuning::default()).await
}

/// [`open_listeners`] with `tuning` for every session (its `link` is the
/// gate `password` makes).
///
/// # Errors
/// As [`open_listeners`].
pub async fn open_listeners_with(
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
    let open = |base| {
        TransportConfig::per_class_tcp(Some(base), None).map(|c| c.with_tuning(tuning.clone()))
    };
    let opened = if base.port() == 0 {
        let mut last = SessionError::Config("no free local ports".into());
        let mut found = None;
        for _ in 0..16 {
            let free = free_local_base(base.ip())?;
            match Transport::open(&open(free)?).await {
                Ok(transport) => {
                    found = Some((transport, free));
                    break;
                }
                Err(err) => last = err,
            }
        }
        found.ok_or(last)
    } else {
        Transport::open(&open(base)?)
            .await
            .map(|transport| (transport, base))
    };
    if let Some(dictionary) = dictionary {
        discard_dictionary(&dictionary);
    }
    opened
}

/// The variable naming a directory, private to this user, for the link gate
/// a session host writes ([`dictionary`]), ahead of every default.
pub(crate) const RUNTIME_DIR_VAR: &str = "RLMESH_RUNTIME_DIR";

/// The name prefix of the directory a link gate gets of its own in the
/// system temp directory.
const PRIVATE_DIR_PREFIX: &str = ".rlmesh-link-";

/// Writes the `user:password` file a gated host's Zenoh reads at open; the
/// caller removes it right after ([`discard_dictionary`]). It holds the
/// link password, so it lives only where no other user can read it or plant
/// a link: [`RUNTIME_DIR_VAR`], else the user's runtime or cache directory,
/// else a fresh directory of its own in the system temp directory (a
/// container with no home and a read-only root has neither of the others).
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
    let defaults = [dirs::runtime_dir(), dirs::cache_dir()]
        .into_iter()
        .flatten()
        .map(|dir| dir.join("rlmesh"));
    gate_in(defaults, &std::env::temp_dir(), &name, contents.as_bytes())
}

/// Writes the link gate `name` into the first of `dirs` that takes it, else
/// into a private directory of its own in `temp`.
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

/// A new directory in `temp` that only this user can enter, under a name no
/// one can predict. Created rather than opened, so whatever another user
/// planted under that name (a link) is refused, never followed.
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

/// Removes a link gate [`dictionary`] wrote, with the directory it made of
/// its own for it.
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
                key: address.key.clone().or_else(|| hosted_key(base)),
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
            if listening(base).await {
                return join(base);
            }
            base
        }
        // An explicit port: hosted here, unless another process (or this
        // one) already holds it.
        Placement::Host(base) if base.port() != 0 && listening(base).await => return join(base),
        Placement::Host(base) => base,
    };
    match host_at(base, name, lease_grace, address.insecure).await {
        Ok(placed) => Ok(placed),
        // Another process holds the port: join it.
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

    /// Set in the child process [`gate_without_home`] runs.
    const CHILD: &str = "RLMESH_TEST_LINK_GATE_CHILD";

    fn mode(path: &Path) -> u32 {
        std::fs::metadata(path).expect("stat").permissions().mode() & 0o777
    }

    /// A path no directory can be made under, whoever runs the test: a
    /// plain file stands where the directory would.
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

    /// The child half of [`gate_without_home`]: hosts on loopback, which
    /// writes the gate, in whatever environment the parent gave it.
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

    /// The child half of [`a_keyless_join_takes_a_key_only_for_a_loopback_host`]:
    /// runs with a scratch rlmeshd state directory holding a `local` key.
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
        // A host this process runs beyond loopback (as a launched container's is).
        let wide = SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0));
        let mut host = host_at(wide, "lan", Duration::from_millis(250), true)
            .await
            .expect("insecure host");
        let port = host.address.base.port();
        let hosted = host.address.key.clone().expect("hosted key");
        let loopback = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
        // Dialed on loopback, a keyless join finds this process's own key...
        assert_eq!(
            credential(loopback, None).expect("hosted key"),
            hosted.credential()
        );
        // ...and for a loopback port nothing here hosts, rlmeshd's.
        let other = SocketAddr::from((Ipv4Addr::LOCALHOST, port.wrapping_add(7)));
        assert_eq!(
            credential(other, None).expect("daemon key"),
            daemon.credential()
        );
        // Dialed at any other address, the same host, the same port: no key
        // stands in, so the join is refused before it reaches the host.
        let remote = SocketAddr::from(([192, 0, 2, 7], port));
        let refused = credential(remote, None).expect_err("no key beyond loopback");
        assert_eq!(
            refused.refusal().map(|r| r.status),
            Some(CtlStatus::Unauthenticated),
            "{refused}"
        );
        // A key the address carries is presented on any network; without
        // one, nothing stands in for it there.
        assert_eq!(
            crate::address::link_auth(remote, Some(&hosted)).map(|link| link.password),
            Some(hosted.link_password())
        );
        assert!(crate::address::link_auth(remote, None).is_none());
        host.close().await.expect("close");
    }

    /// Whether a join without a key is accepted is decided by the joining
    /// process alone, from the address it dials: on loopback it may use a
    /// key it already holds (a host it runs, rlmeshd's 0600 file); beyond
    /// loopback, none. The host never waives the handshake for anyone
    /// (`an_embedded_host_stays_on_loopback_and_links_only_with_its_key`).
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
        assert_eq!(std::fs::read_dir(&runtime).unwrap().count(), 0);
    }
}
