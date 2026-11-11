//! `rlmesh host` (installed as `rlmeshd` too): this machine's session host,
//! in the foreground or as a service. It hosts real-time sessions on the
//! LAN under one host identity that survives restarts, and admits only the
//! devices of its device table.
//!
//! There is no key file to pass: its state directory
//! ([`address::daemon_dir`], `--state-dir`) holds
//! `host.key` (its static X25519 key and Ed25519 seed), `local.key` (the key
//! of device `local`: this user's processes on this machine attach with it)
//! and `devices.toml` ([`crate::devices`]). The keys are made on first
//! start, 0600, and kept, so its identity survives restarts. It re-reads
//! `devices.toml` every [`RELOAD`] and applies a change to every session at
//! once; a robot name's first embodiment by a device allowed to pin it is
//! written there as its pin.
//!
//! It records every session it hosts into a rolling recording unless
//! `--no-record`.
//!
//! With the unreleased `fleet` feature and `--lobby` it also hosts the
//! lobby, `_lobby` (`crate::lobby`), where robots provisioned to it wait
//! for a scene's run, runs scenes, is the satellite of the hosts this
//! machine paired with (`crate::satellite`), and advertises itself over
//! mDNS (`discovery::advertise`).
//!
//! Its pairing endpoint (`rlmeshd/pair`, [`PairService`]) pairs machines,
//! which the owner approves on this machine through the local channel
//! ([`crate::pair`]), as `devices provision` enrolls boards. Its host id
//! ([`HostIdentity::host_id`]) is in its first line.
//!
//! Unlike an embedded host it gates no link (`usrpwd`): Zenoh gates every
//! listener of a router or none, and a board's zenoh-pico has no `usrpwd`,
//! so the handshake alone admits, here as on any LAN link.

use std::io::Write;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use rlmesh_session::address::{
    self, AddressKey, DEFAULT_SESSION, LEASE_GRACE, LOCAL_KEY, TENANT, session_id, write_private,
};
use rlmesh_session::auth::HostIdentity;
use rlmesh_session::auth::address::{base32, unbase32};
use rlmesh_session::authority::SealPolicy;
#[cfg(feature = "fleet")]
use rlmesh_session::discovery;
use rlmesh_session::pairing::PairService;
use rlmesh_session::record::Rolling;
use rlmesh_session::server::SessionServer;
use rlmesh_session::wire::{PairInfo, Role, TenantId};
use rlmesh_session::{HostConfig, SystemMonotonicClock};
use zeroize::Zeroizing;

use crate::cli::HostArgs;
use crate::devices::{DEVICES, Pin, Table};
#[cfg(feature = "fleet")]
use crate::lobby::Home;

/// The host's own keys in the state directory.
pub const HOST_KEY: &str = "host.key";
/// How often `devices.toml` is read again.
const RELOAD: Duration = Duration::from_millis(500);

/// The host's own warnings, on stderr: `RUST_LOG` filters them, `warn` when
/// unset. A process that already has a logger (the Python package's) keeps
/// it.
fn init_logging() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .try_init();
}

/// The state directory: `dir`, else [`address::daemon_dir`].
pub fn state_dir(dir: Option<&Path>) -> Result<PathBuf> {
    dir.map(Path::to_path_buf)
        .or_else(address::daemon_dir)
        .context("no state directory for rlmesh host: pass --state-dir")
}

/// The two 32-byte secrets of `path` (`{a}.{b}`, base32), if it exists.
fn read_pair(path: &Path) -> Result<Option<Zeroizing<[[u8; 32]; 2]>>> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => Zeroizing::new(text),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err).with_context(|| path.display().to_string()),
    };
    let bad = || format!("{} is not two base32 keys", path.display());
    let (a, b) = text.trim().split_once('.').with_context(bad)?;
    let pair = [
        unbase32(a).with_context(bad)?,
        unbase32(b).with_context(bad)?,
    ];
    Ok(Some(Zeroizing::new(pair)))
}

/// The host identity in `dir`, made on first start (`host.key` and
/// `local.key`, 0600). A missing `local.key` gets a new local secret.
pub fn identity(dir: &Path) -> Result<HostIdentity> {
    let fresh = || -> Result<Arc<HostIdentity>> {
        Ok(HostConfig::new(TenantId::new(TENANT)?, session_id(DEFAULT_SESSION)).identity)
    };
    let (mut noise, mut sign) = (Zeroizing::new([0_u8; 32]), Zeroizing::new([0_u8; 32]));
    let host_path = dir.join(HOST_KEY);
    match read_pair(&host_path)? {
        Some(pair) => {
            noise.copy_from_slice(&pair[0]);
            sign.copy_from_slice(&pair[1]);
        }
        None => {
            fresh()?.write_secrets(&mut noise, &mut sign);
            let text = Zeroizing::new(format!(
                "{}.{}",
                Zeroizing::new(base32(&noise)).as_str(),
                Zeroizing::new(base32(&sign)).as_str()
            ));
            write_private(&host_path, text.as_bytes())
                .with_context(|| host_path.display().to_string())?;
        }
    }
    let local_path = dir.join(LOCAL_KEY);
    let mut local = match read_pair(&local_path)? {
        Some(pair) => Zeroizing::new(pair[0]),
        None => Zeroizing::new(*fresh()?.local_secret()),
    };
    let identity = HostIdentity::from_secrets(&mut noise, &mut sign, &mut local);
    let key = AddressKey::of(&identity);
    let held = std::fs::read_to_string(&local_path)
        .ok()
        .map(Zeroizing::new)
        .and_then(|text| text.parse::<AddressKey>().ok());
    if held.as_ref() != Some(&key) {
        let text = Zeroizing::new(key.to_string());
        write_private(&local_path, text.as_bytes())
            .with_context(|| local_path.display().to_string())?;
    }
    Ok(identity)
}

/// The base addresses rlmeshd gives paired machines and boards: `base`,
/// or for every interface (`0.0.0.0`) the address of the interface its
/// default route leaves by.
// TODO: one address per host; list every interface's when a robot is on
// more than one network (mDNS finds the others).
pub fn advertised(base: SocketAddr) -> Vec<String> {
    if !base.ip().is_unspecified() {
        return vec![base.to_string()];
    }
    // A UDP connect sends nothing: it only picks the route.
    let routed = std::net::UdpSocket::bind((base.ip(), 0))
        .and_then(|socket| {
            let probe: SocketAddr = if base.is_ipv4() {
                "192.0.2.1:9".parse().map_err(std::io::Error::other)?
            } else {
                "[2001:db8::1]:9".parse().map_err(std::io::Error::other)?
            };
            socket.connect(probe)?;
            socket.local_addr()
        })
        .map(|local| local.ip());
    match routed {
        Ok(ip) if !ip.is_unspecified() => vec![SocketAddr::new(ip, base.port()).to_string()],
        _ => Vec::new(),
    }
}

/// `rlmesh host`: hosts until SIGINT or SIGTERM.
pub async fn host(args: &HostArgs, stdout: &mut impl Write, stderr: &mut impl Write) -> Result<()> {
    init_logging();
    let dir = state_dir(args.state_dir.as_deref())?;
    let identity = Arc::new(identity(&dir)?);
    let table = Table::read(&dir)?;
    let (devices, pins) = table.host()?;
    let base = match args.listen {
        Some(base) => base,
        None => SocketAddr::from((Ipv4Addr::UNSPECIFIED, address::well_known_port()?)),
    };
    let mut config = HostConfig::new(TenantId::new(TENANT)?, session_id(DEFAULT_SESSION));
    config.identity = identity.clone();
    config.devices = devices;
    config.pins = pins;
    config.clock = SystemMonotonicClock.shared();
    config.lease_grace = args.lease_grace.unwrap_or(LEASE_GRACE);
    if let Some(grace) = args.presence_grace {
        config.presence_grace = grace;
    }
    config.rearm_robots = args.rearm_robots;
    // Beyond loopback every part is sealed: anyone on the LAN sees the
    // traffic.
    if !base.ip().is_loopback() && !args.no_seal {
        config.seal = SealPolicy::All;
    }
    config.health_log = None;
    let (transport, base) = address::open_listeners(base, None)
        .await
        .with_context(|| format!("could not listen on {base} (is a host already there?)"))?;
    let recordings = recordings(args)?;
    let server = Arc::new(match &recordings {
        Some(dir) => {
            SessionServer::start_recording(
                transport.clone(),
                config.clone(),
                rolling(args, dir.clone()),
            )
            .await?
        }
        None => SessionServer::start(transport.clone(), config.clone()).await?,
    });
    server.limit_sessions(usize::try_from(args.max_sessions).unwrap_or(usize::MAX));
    let service = Arc::new(PairService::start(&transport, identity.clone()).await?);
    #[cfg(feature = "fleet")]
    let fleet = match args.lobby {
        true => Some(Fleet::start(&server, &dir, config, base, stderr).await?),
        false => None,
    };
    #[cfg(not(feature = "fleet"))]
    drop(config);
    let (apply, mut applying) = tokio::sync::mpsc::channel(4);
    #[cfg(unix)]
    let local = crate::pair::serve_local(crate::pair::Local {
        dir: dir.clone(),
        service: service.clone(),
        host: PairInfo {
            device: String::new(),
            host_key: *identity.static_public(),
            host_id: identity.host_id(),
            addresses: advertised(base),
        },
        apply,
    })?;
    #[cfg(not(unix))]
    drop((apply, service));
    // One line: a reader that takes it and closes the pipe must not fail a
    // second write.
    let reach = match advertised(base).first() {
        Some(address) if base.ip().is_unspecified() => format!(", reachable at {address}"),
        Some(_) => String::new(),
        None => {
            writeln!(
                stderr,
                "no network route to pick an address from: boards can't be provisioned until \
                 rlmesh host runs with --listen <LAN IP>:<port>"
            )?;
            String::new()
        }
    };
    writeln!(
        stdout,
        "rlmesh host on {base} as {}, state in {}{reach}",
        identity.host_id(),
        dir.display()
    )?;
    if let Some(dir) = &recordings {
        writeln!(stdout, "recording to {}", dir.display())?;
    }
    stdout.flush()?;
    // The table applied, and the day it was: an entry may expire overnight.
    let mut applied = (table, crate::devices::today());
    let mut refused: Option<String> = None;
    let mut shutdown = std::pin::pin!(shutdown_signal());
    loop {
        let now = tokio::select! {
            result = &mut shutdown => {
                result?;
                break;
            }
            Some(done) = applying.recv() => Some(done),
            () = tokio::time::sleep(RELOAD) => None,
        };
        if now.is_none()
            && let Err(err) = pin_first_embodiments(&dir, &server, stderr).await
        {
            let _ = writeln!(stderr, "rlmesh host: robot pins not written: {err:#}");
        }
        if let Some(table) = reload(&dir, &server, &mut applied, &mut refused, stderr).await {
            #[cfg(feature = "fleet")]
            if let Some(fleet) = &fleet {
                fleet.home.devices_changed(&table);
            }
            #[cfg(not(feature = "fleet"))]
            drop(table);
        }
        if let Some(done) = now {
            let _ = done.send(());
        }
    }
    #[cfg(unix)]
    {
        local.abort();
        let _ = std::fs::remove_file(dir.join(crate::pair::SOCKET));
    }
    #[cfg(feature = "fleet")]
    if let Some(fleet) = fleet {
        fleet.stop();
    }
    server.shutdown().await;
    transport.close().await?;
    Ok(())
}

/// Where to record (`None`: `--no-record`).
/// The rolling budgets, with the flags' overrides.
fn rolling(args: &HostArgs, dir: PathBuf) -> Rolling {
    let mut rolling = Rolling::new(dir);
    if let Some(keep) = args.record_keep {
        rolling.max_age = keep;
    }
    if let Some(bytes) = args.record_max_size {
        rolling.max_bytes = bytes;
    }
    if let Some(bytes) = args.record_min_free {
        rolling.min_free_bytes = bytes;
    }
    if let Some(keep) = args.incidents_keep {
        rolling.incident_max_age = keep;
    }
    if let Some(bytes) = args.incidents_max_size {
        rolling.incident_max_bytes = bytes;
    }
    rolling
}

fn recordings(args: &HostArgs) -> Result<Option<PathBuf>> {
    if args.no_record {
        return Ok(None);
    }
    match &args.record_dir {
        Some(dir) => Ok(Some(dir.clone())),
        None => Rolling::default_dir()
            .context("no user data directory to record into (pass --record-dir or --no-record)")
            .map(Some),
    }
}

/// Applies `devices.toml` to every session if it changed since `applied`,
/// or the day did (a device may have expired), and returns the table it
/// applied.
async fn reload(
    dir: &Path,
    server: &SessionServer,
    applied: &mut (Table, i64),
    refused: &mut Option<String>,
    stderr: &mut impl Write,
) -> Option<Table> {
    match Table::read(dir) {
        Ok(table) if table != applied.0 || crate::devices::today() != applied.1 => {
            // Read checked it.
            let Ok((devices, pins)) = table.host() else {
                return None;
            };
            server.set_devices(devices, pins).await;
            let _ = writeln!(stderr, "rlmesh host: {DEVICES} applied");
            *applied = (table.clone(), crate::devices::today());
            *refused = None;
            return Some(table);
        }
        Ok(_) => *refused = None,
        Err(err) => {
            // Once per broken edit, not every read.
            let err = format!("{err:#}");
            if refused.as_ref() != Some(&err) {
                let _ = writeln!(stderr, "rlmesh host: {DEVICES} not applied: {err}");
            }
            *refused = Some(err);
        }
    }
    None
}

/// The daemon's fleet services (feature `fleet`): the lobby and its
/// scenes, its satellite side and its mDNS advertisement.
#[cfg(feature = "fleet")]
struct Fleet {
    home: Arc<Home>,
    satellite: Vec<tokio::task::JoinHandle<()>>,
    advertised: Option<discovery::Advertisement>,
}

#[cfg(feature = "fleet")]
impl Fleet {
    async fn start(
        server: &Arc<SessionServer>,
        dir: &Path,
        config: HostConfig,
        base: SocketAddr,
        stderr: &mut impl Write,
    ) -> Result<Self> {
        let identity = config.identity.clone();
        let home = Home::start(server, dir.to_path_buf(), config, advertised(base)).await?;
        let satellite = crate::satellite::Satellite::start(
            server.clone(),
            rlmesh_session::pairing::hosts_dir(),
            &identity.host_id(),
            advertised(base),
        );
        // Without multicast it is found at its stored addresses only.
        let advertised = discovery::advertise(&identity.host_id(), base)
            .map_err(|err| {
                let _ = writeln!(stderr, "rlmesh host: not advertised over mDNS: {err}");
            })
            .ok();
        Ok(Self {
            home,
            satellite,
            advertised,
        })
    }

    fn stop(self) {
        drop(self.advertised);
        for task in self.satellite {
            task.abort();
        }
    }
}

/// Pins every unpinned robot name embodied by a device allowed to pin it
/// ([`Table::may_pin`]).
async fn pin_first_embodiments(
    dir: &Path,
    server: &SessionServer,
    stderr: &mut impl Write,
) -> Result<()> {
    let table = Table::read(dir)?;
    let mut new = Vec::new();
    for snapshot in server.snapshots().await {
        for robot in &snapshot.roster {
            let name = robot.id.as_str();
            if robot.present
                && robot.has_role(Role::Embodiment)
                && snapshot.parts.iter().any(|p| p.owner == robot.id)
                && !table.robots.contains_key(name)
                && table.may_pin(name, &robot.device)
            {
                new.push((name.to_string(), robot.device.clone()));
            }
        }
    }
    if new.is_empty() {
        return Ok(());
    }
    crate::devices::edit(dir, |table| {
        for (robot, device) in new {
            let _ = writeln!(stderr, "rlmesh host: pinned robot {robot} to {device}");
            // TODO: a tethered robot's first embodiment pins its device
            // alone, since authority state doesn't carry board_id; record the
            // declared board once the bridge sends one (scene step 10).
            table.robots.entry(robot).or_insert(Pin {
                device,
                board_id: None,
            });
        }
        Ok(())
    })
}

#[cfg(unix)]
pub(crate) async fn shutdown_signal() -> std::io::Result<()> {
    use tokio::signal::unix::{SignalKind, signal};
    let mut term = signal(SignalKind::terminate())?;
    tokio::select! {
        result = tokio::signal::ctrl_c() => result?,
        _ = term.recv() => {}
    }
    Ok(())
}

#[cfg(not(unix))]
pub(crate) async fn shutdown_signal() -> std::io::Result<()> {
    tokio::signal::ctrl_c().await
}
