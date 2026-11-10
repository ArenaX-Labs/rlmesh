//! Credentials for a session: remembered, paired, from the address's key or the
//! daemon's, and the private files they live in.

#[cfg(feature = "fleet")]
use std::collections::HashMap;
use std::net::SocketAddr;

use rlmesh_session_core::auth::AddressKey;
use rlmesh_session_core::auth::Credential;
use zeroize::Zeroizing;

#[cfg(feature = "fleet")]
use crate::address::SessionAddress;
use crate::address::hosting::local_hosts;
use crate::error::SessionError;
use crate::transport::LinkAuth;
#[cfg(feature = "fleet")]
use crate::wire::SessionId;
use crate::wire::{CtlError, CtlStatus};

/// The credentials this process holds for sessions on hosts that aren't
/// its home's, by base and session ([`remember_credential`]).
#[cfg(feature = "fleet")]
type Remembered = HashMap<(SocketAddr, SessionId), Credential>;

#[cfg(feature = "fleet")]
pub(super) fn remembered() -> &'static std::sync::Mutex<Remembered> {
    static REMEMBERED: std::sync::OnceLock<std::sync::Mutex<Remembered>> =
        std::sync::OnceLock::new();
    REMEMBERED.get_or_init(Default::default)
}

/// Keeps `credential` for `address`'s session in this process, so its text
/// (which never carries a device key) joins with it too: a satellite's run,
/// whose address a binding passes to its controller as text.
// TODO: kept until the process exits; forget ended runs if a long-lived
// process runs thousands of satellite-hosted scenes.
#[cfg(feature = "fleet")]
pub fn remember_credential(address: &SessionAddress, credential: Credential) {
    remembered()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert((address.base, address.session), credential);
}

/// This machine's key from pairing with the host at `base`.
fn paired(base: SocketAddr) -> Option<Credential> {
    crate::pairing::hosts_dir().and_then(|dir| crate::pairing::paired_credential(&dir, base))
}

/// The credential of a participant of the host at `base`: `key` (an
/// address's), else the key of a host on this machine (one this process
/// runs, or rlmeshd's), else this machine's key from pairing with the host
/// ([`crate::pairing::paired_credential`]).
pub fn credential(base: SocketAddr, key: Option<&AddressKey>) -> Result<Credential, SessionError> {
    key.cloned()
        .or_else(|| hosted_key(base))
        .or_else(|| base.ip().is_loopback().then(daemon_key).flatten())
        .map(|key| key.credential())
        .or_else(|| paired(base))
        .ok_or_else(|| {
            SessionError::Rejected(CtlError::new(
                CtlStatus::Unauthenticated,
                format!(
                    "no key for the session at {base}: join with the address its host printed \
                     (it ends in #key), run `rlmesh host` on this machine, or pair this machine with \
                     the robot's (rlmesh pair HOST)"
                ),
            ))
        })
}

/// What a link to the host at `base` presents: on loopback, the
/// password `key` derives, else the key of a host this process runs; on
/// the LAN, and to rlmeshd (a LAN host), nothing.
pub fn link_auth(base: SocketAddr, key: Option<&AddressKey>) -> Option<LinkAuth> {
    if !base.ip().is_loopback() {
        return None;
    }
    let key = key.cloned().or_else(|| hosted_key(base))?;
    Some(LinkAuth {
        password: key.link_password(),
        dictionary: None,
    })
}

/// The key of a host this process runs at `base`.
pub(super) fn hosted_key(base: SocketAddr) -> Option<AddressKey> {
    local_hosts()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&base)
        .map(|(_, key)| key.clone())
}

/// The variable that names rlmeshd's state directory ([`daemon_dir`]).
pub const STATE_DIR_VAR: &str = "RLMESH_STATE_DIR";

/// The file in [`daemon_dir`] that holds the key of rlmeshd's device
/// `local`: its address key (`{local secret}.{host key}`, [`AddressKey`]),
/// 0600. This user's processes on its machine attach as `local` with it.
pub const LOCAL_KEY: &str = "local.key";

/// Where rlmeshd keeps its identity and device table:
/// [`STATE_DIR_VAR`], else `rlmesh/daemon` in the user's state directory
/// (`$XDG_STATE_HOME`, `~/.local/state`), or their local data directory
/// where there is none.
pub fn daemon_dir() -> Option<std::path::PathBuf> {
    if let Some(dir) = std::env::var_os(STATE_DIR_VAR) {
        return Some(dir.into());
    }
    let dir = dirs::state_dir().or_else(dirs::data_local_dir)?;
    Some(dir.join("rlmesh").join("daemon"))
}

/// rlmeshd's `local` key, when this user can read it.
fn daemon_key() -> Option<AddressKey> {
    let text = Zeroizing::new(std::fs::read_to_string(daemon_dir()?.join(LOCAL_KEY)).ok()?);
    text.parse().ok()
}

/// Writes `contents` to `path` for this user only: the directory 0700 and
/// the file 0600, written whole (renamed into place).
pub fn write_private(path: &std::path::Path, contents: &[u8]) -> std::io::Result<()> {
    let dir = path
        .parent()
        .ok_or_else(|| std::io::Error::other("no directory"))?;
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    builder.create(dir)?;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let partial = dir.join(format!(".{name}.{}.partial", std::process::id()));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    std::io::Write::write_all(&mut options.open(&partial)?, contents)?;
    std::fs::rename(&partial, path)
}
