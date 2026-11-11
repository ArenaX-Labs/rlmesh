//! The hosting rule: joining a host, or hosting the session in this process.

use std::collections::HashMap;
use std::fmt;
use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::time::Duration;

use rlmesh_session_core::auth::AddressKey;

#[cfg(doc)]
use crate::address::HOST_IDENTITY_VAR;
use crate::address::credentials::{hosted_key, write_private};
use crate::address::{
    Address, Placement, SessionAddress, TENANT, joinable, launched_identity, resolve, session_id,
    well_known_port,
};
use crate::clock::SystemMonotonicClock;
use crate::error::SessionError;
use crate::host::HostConfig;
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
pub(super) fn listening(base: SocketAddr) -> bool {
    TcpStream::connect_timeout(&joinable(base), Duration::from_millis(200)).is_ok()
}

/// How long a link presenting the password gets, once the host listens,
/// before [`open_client`] tries one without.
const GATED_TRY: Duration = Duration::from_secs(1);

/// Links to the host at `base` as a Zenoh client within `timeout`, with no
/// password (a LAN host's, as rlmeshd's pairing endpoint is reached).
pub async fn connect(base: SocketAddr, timeout: Duration) -> Result<Transport, SessionError> {
    open_client(base, None, tokio::time::Instant::now() + timeout)
        .await
        .map_err(|err| match err {
            SessionError::Zenoh(_) => SessionError::Timeout(timeout, format!("a host at {base}")),
            other => other,
        })
}

/// Joins the host at `base` as a Zenoh client by `deadline`, presenting
/// `link`. A host bound beyond loopback (`insecure`, rlmeshd's LAN listener)
/// gates no link, since Zenoh can't gate one listener and not another, and refuses a
/// password it never asked for: when a gated
/// link to a listening host fails, the join links without one, and the
/// handshake alone admits it, as on any LAN link.
pub(super) async fn open_client(
    base: SocketAddr,
    link: Option<LinkAuth>,
    deadline: tokio::time::Instant,
) -> Result<Transport, SessionError> {
    let open = |link, until: tokio::time::Instant| {
        let tuning = TransportTuning {
            client: true,
            connect_timeout: Some(until.saturating_duration_since(tokio::time::Instant::now())),
            link,
            ..TransportTuning::default()
        };
        async move {
            Transport::open(&TransportConfig::per_class_tcp(None, Some(base))?.with_tuning(tuning))
                .await
        }
    };
    let Some(link) = link else {
        return open(None, deadline).await;
    };
    while !listening(base) && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let gated = open(
        Some(link),
        deadline.min(tokio::time::Instant::now() + GATED_TRY),
    )
    .await;
    if gated.is_ok() || !listening(base) {
        return gated;
    }
    open(None, deadline).await
}

/// Hosts session `name` in this process on `base` (port 0: OS-assigned),
/// and every other session of [`TENANT`] its participants name, under fresh
/// keys (or the ones its launcher passed in [`HOST_IDENTITY_VAR`]): the
/// returned address carries them, the invite for other processes.
///
/// It listens on loopback only unless `insecure`,
/// or a launcher passed its keys (a sandbox's container, whose published
/// ports its launcher dials with them): LAN sessions are rlmeshd's. On
/// loopback, and in a launched container, every link must present the
/// password the key derives. Beyond loopback no link does:
/// Zenoh gates every listener of a router or none, and a board's zenoh-pico
/// has no `usrpwd`, so there the handshake alone admits, as on any LAN link.
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
            "an embedded host listens on loopback only, not {base}: for a session other \
             machines join, run `rlmesh host`; to accept that anyone who can reach {base} can \
             try this host, pass insecure (Host(insecure=True))"
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
    let dictionary = password.map(dictionary).transpose()?;
    let tuning = TransportTuning {
        link: password
            .zip(dictionary.clone())
            .map(|(password, dictionary)| LinkAuth {
                password: password.clone(),
                dictionary: Some(dictionary),
            }),
        ..TransportTuning::default()
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
        let _ = std::fs::remove_file(dictionary);
    }
    opened
}

/// Writes the `user:password` file a gated host's Zenoh reads at open; the
/// caller removes it right after.
fn dictionary(password: &zeroize::Zeroizing<String>) -> Result<std::path::PathBuf, SessionError> {
    let dir = dirs::runtime_dir()
        .or_else(dirs::cache_dir)
        .map(|dir| dir.join("rlmesh"))
        .ok_or_else(|| SessionError::Config("no directory for the link gate".into()))?;
    static OPENED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = OPENED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = dir.join(format!(".link.{}.{n}", std::process::id()));
    write_private(
        &path,
        format!("{LINK_USER}:{}", password.as_str()).as_bytes(),
    )
    .map_err(|err| SessionError::Config(format!("the link gate: {err}")))?;
    Ok(path)
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
    match host_at(base, name, lease_grace, address.insecure).await {
        Ok(placed) => Ok(placed),
        // Another process holds the port: join it.
        Err(_) if base.port() != 0 && listening(base) => join(base),
        Err(err) => Err(err),
    }
}
