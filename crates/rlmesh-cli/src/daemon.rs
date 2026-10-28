//! `rlmesh daemon` (installed as `rlmeshd`): this machine's host identity,
//! and v2 sessions hosted under it on the LAN (design §2, §3, §5).
//!
//! Its state directory ([`address::daemon_dir`], `--state-dir`) holds
//! `host.key` (its static X25519 key and Ed25519 seed), `local.key` (the key
//! of device `local`: this user's processes on this machine attach with it)
//! and `devices.toml` ([`crate::devices`]). The keys are made on first
//! start, 0600, and kept, so its identity survives restarts. It re-reads
//! `devices.toml` every [`RELOAD`] and applies a change to every session at
//! once; a robot name's first embodiment by a device allowed to pin it is
//! written there as its pin.
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
use rlmesh_session::server::SessionServer;
use rlmesh_session::wire::{Role, TenantId};
use rlmesh_session::{HostConfig, SystemMonotonicClock};
use zeroize::Zeroizing;

use crate::cli::DaemonArgs;
use crate::devices::{DEVICES, Pin, Table};

/// The host's own keys in the state directory.
pub const HOST_KEY: &str = "host.key";
/// How often `devices.toml` is read again.
const RELOAD: Duration = Duration::from_millis(500);

/// The state directory: `dir`, else [`address::daemon_dir`].
pub fn state_dir(dir: Option<&Path>) -> Result<PathBuf> {
    dir.map(Path::to_path_buf)
        .or_else(address::daemon_dir)
        .context("no state directory for rlmeshd: pass --state-dir")
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

/// `rlmesh daemon`: hosts until SIGINT or SIGTERM.
pub async fn daemon(
    args: &DaemonArgs,
    stdout: &mut impl Write,
    stderr: &mut impl Write,
) -> Result<()> {
    let dir = state_dir(args.state_dir.as_deref())?;
    let identity = identity(&dir)?;
    let table = Table::read(&dir)?;
    let (devices, pins) = table.host()?;
    let base = match args.listen {
        Some(base) => base,
        None => SocketAddr::from((Ipv4Addr::UNSPECIFIED, address::well_known_port()?)),
    };
    let mut config = HostConfig::new(TenantId::new(TENANT)?, session_id(DEFAULT_SESSION));
    config.identity = Arc::new(identity);
    config.devices = devices;
    config.pins = pins;
    config.clock = SystemMonotonicClock.shared();
    config.lease_grace = LEASE_GRACE;
    config.health_log = None;
    // TODO: no pairing endpoint yet; step 12 adds `rlmeshd/pair` and its
    // pending exchanges here.
    let (transport, base) = address::open_listeners(base, None)
        .await
        .with_context(|| format!("could not listen on {base} (is a host already there?)"))?;
    let server = SessionServer::start(transport.clone(), config).await?;
    writeln!(stdout, "rlmeshd on {base}, state in {}", dir.display())?;
    stdout.flush()?;
    let mut applied = table;
    let mut refused: Option<String> = None;
    let mut shutdown = std::pin::pin!(crate::host::shutdown_signal());
    loop {
        tokio::select! {
            result = &mut shutdown => {
                result?;
                break;
            }
            () = tokio::time::sleep(RELOAD) => {}
        }
        if let Err(err) = pin_first_embodiments(&dir, &server, stderr).await {
            let _ = writeln!(stderr, "rlmeshd: robot pins not written: {err:#}");
        }
        match Table::read(&dir) {
            Ok(table) if table != applied => {
                // Read checked it.
                let Ok((devices, pins)) = table.host() else {
                    continue;
                };
                server.set_devices(devices, pins).await;
                let _ = writeln!(stderr, "rlmeshd: {DEVICES} applied");
                applied = table;
                refused = None;
            }
            Ok(_) => refused = None,
            Err(err) => {
                // Once per broken edit, not every read.
                let err = format!("{err:#}");
                if refused.as_ref() != Some(&err) {
                    let _ = writeln!(stderr, "rlmeshd: {DEVICES} not applied: {err}");
                }
                refused = Some(err);
            }
        }
    }
    server.shutdown().await;
    transport.close().await?;
    Ok(())
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
            let _ = writeln!(stderr, "rlmeshd: pinned robot {robot} to {device}");
            table.robots.entry(robot).or_insert(Pin {
                device,
                board_id: None,
            });
        }
        Ok(())
    })
}
