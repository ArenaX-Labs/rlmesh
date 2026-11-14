//! The moment a served env or model can answer: the address a peer dials it
//! at, and `RLMESH_READY_FILE`, the file a launcher waits on.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

/// The environment variable naming the ready file.
pub const READY_FILE_ENV: &str = "RLMESH_READY_FILE";

/// A served participant that can answer requests. No key or identity: the
/// address is redacted.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Ready {
    /// `env` or `model`.
    pub kind: &'static str,
    /// Its participant name.
    pub participant: String,
    /// Where a peer dials it, key left out (`host:port#…/name`).
    pub address: String,
    /// Its session's port.
    pub port: Option<u16>,
    /// This process.
    pub pid: u32,
}

impl Ready {
    /// `requested` is the address serving was asked for, `invite` the keyed
    /// address it serves at: bound to every interface, the invite's
    /// loopback host becomes this machine's.
    pub(crate) fn new(
        kind: &'static str,
        participant: &str,
        requested: &str,
        invite: &str,
    ) -> Ready {
        let address = dialable(requested, &rlmesh_session::address::redact(invite));
        Ready {
            kind,
            participant: participant.to_owned(),
            port: port_of(&address),
            address,
            pid: std::process::id(),
        }
    }
}

/// A serve to announce once it can answer: the address it was asked to
/// serve at, and the ready file to write ([`ready_file_from_env`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Announce {
    /// The address serving was asked for (`0.0.0.0:7447/env`).
    pub requested: String,
    /// Where to write the ready file.
    pub file: Option<PathBuf>,
    /// Print the serving line, timed from this start (a native factory
    /// serve's; a binding prints its own from the hook).
    pub(crate) banner: Option<std::time::Instant>,
}

impl Announce {
    /// Announce a serve of `requested`, with the ready file a launcher set.
    pub fn from_env(requested: &str) -> Announce {
        Announce {
            requested: requested.to_owned(),
            file: ready_file_from_env(),
            banner: None,
        }
    }

    /// Say that `participant` (an `env` or `model`) answers at `invite`.
    pub(crate) fn ready(self, kind: &'static str, participant: &str, invite: &str) -> Announced {
        let ready = Ready::new(kind, participant, &self.requested, invite);
        if let Some(started) = self.banner {
            print_serving(&ready, started);
        }
        announce(&ready, self.file)
    }
}

type Hook = Arc<dyn Fn(&Ready) + Send + Sync>;

static HOOK: Mutex<Option<Hook>> = Mutex::new(None);

/// Call `hook` as each served participant becomes ready (`None` clears it):
/// a binding prints its serving line there.
pub fn set_ready_hook(hook: Option<Hook>) {
    *HOOK.lock().unwrap_or_else(PoisonError::into_inner) = hook;
}

/// The ready file this process's served participants write, if a launcher
/// set one.
pub fn ready_file_from_env() -> Option<PathBuf> {
    std::env::var_os(READY_FILE_ENV)
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
}

/// Removes the ready file when the serve that wrote it stops.
#[derive(Debug)]
pub(crate) struct Announced(Option<PathBuf>);

impl Drop for Announced {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Say `ready`: the hook, then `file`, written whole or not at all.
pub(crate) fn announce(ready: &Ready, file: Option<PathBuf>) -> Announced {
    let hook = HOOK.lock().unwrap_or_else(PoisonError::into_inner).clone();
    if let Some(hook) = hook {
        hook(ready);
    }
    let Some(path) = file else {
        return Announced(None);
    };
    match write_atomically(&path, ready) {
        Ok(()) => Announced(Some(path)),
        Err(err) => {
            tracing::warn!(path = %path.display(), %err, "could not write {READY_FILE_ENV}");
            Announced(None)
        }
    }
}

#[allow(clippy::print_stdout)]
fn print_serving(ready: &Ready, started: std::time::Instant) {
    use std::io::Write;
    println!(
        "RLMesh serving {} on {} (startup: ready {:.1}s)",
        ready.kind,
        ready.address,
        started.elapsed().as_secs_f64()
    );
    let _ = std::io::stdout().flush();
}

fn write_atomically(path: &Path, ready: &Ready) -> std::io::Result<()> {
    let body = serde_json::to_vec(ready).map_err(std::io::Error::other)?;
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    // A sibling, so the rename stays on one filesystem.
    let partial = path.with_file_name(format!(".{name}.{}.partial", ready.pid));
    std::fs::write(&partial, body)?;
    std::fs::rename(&partial, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&partial);
    })
}

/// The host and port segment of an address (`rlmesh://` and a key aside).
fn host_port(address: &str) -> &str {
    let address = address.strip_prefix("rlmesh://").unwrap_or(address);
    let first = address.split('/').next().unwrap_or_default();
    first.split('#').next().unwrap_or_default()
}

fn split_host(host_port: &str) -> (&str, Option<&str>) {
    if let Some(rest) = host_port.strip_prefix('[') {
        let (host, port) = rest.split_once(']').unwrap_or((rest, ""));
        return (host, port.strip_prefix(':'));
    }
    match host_port.rsplit_once(':') {
        Some((host, port)) if !host.contains(':') => (host, Some(port)),
        _ => (host_port, None),
    }
}

fn port_of(address: &str) -> Option<u16> {
    split_host(host_port(address)).1?.parse().ok()
}

/// `invite` with this machine's host in place of loopback when `requested`
/// binds every interface: what another machine or container can dial.
fn dialable(requested: &str, invite: &str) -> String {
    let (host, _) = split_host(host_port(requested));
    if !matches!(host, "0.0.0.0" | "::") {
        return invite.to_owned();
    }
    let Some(machine) = machine_host() else {
        return invite.to_owned();
    };
    let served = host_port(invite);
    let Some(port) = split_host(served).1 else {
        return invite.to_owned();
    };
    let machine = if machine.contains(':') {
        format!("[{machine}]")
    } else {
        machine
    };
    invite.replacen(served, &format!("{machine}:{port}"), 1)
}

/// This machine as a peer reaches it: its hostname's address when that is
/// not loopback (a container's own address), else the hostname.
fn machine_host() -> Option<String> {
    use std::net::ToSocketAddrs;
    let name = hostname()?;
    let address = (name.as_str(), 0)
        .to_socket_addrs()
        .ok()
        .and_then(|mut found| found.find(|a| a.is_ipv4() && !a.ip().is_loopback()));
    Some(address.map_or(name, |a| a.ip().to_string()))
}

#[cfg(unix)]
#[allow(unsafe_code)]
fn hostname() -> Option<String> {
    let mut buffer = [0u8; 256];
    // SAFETY: the buffer is valid for its length; gethostname writes a
    // NUL-terminated name into it (truncated names are refused below).
    let status = unsafe { libc::gethostname(buffer.as_mut_ptr().cast(), buffer.len()) };
    if status != 0 {
        return None;
    }
    let end = buffer.iter().position(|&b| b == 0)?;
    let name = std::str::from_utf8(&buffer[..end]).ok()?.trim();
    (!name.is_empty()).then(|| name.to_owned())
}

#[cfg(not(unix))]
fn hostname() -> Option<String> {
    std::env::var("COMPUTERNAME")
        .ok()
        .filter(|name| !name.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_narrow_bind_keeps_its_address() {
        assert_eq!(
            dialable("127.0.0.1:7447/env", "127.0.0.1:7447#…/env"),
            "127.0.0.1:7447#…/env"
        );
        assert_eq!(
            dialable(":0/env", "127.0.0.1:41000/env"),
            "127.0.0.1:41000/env"
        );
    }

    #[test]
    fn a_wide_bind_names_this_machine() {
        let shown = dialable("0.0.0.0:7447/env", "127.0.0.1:7447#…/env");
        if machine_host().is_some() {
            assert!(!shown.starts_with("127.0.0.1"), "{shown}");
        }
        assert!(shown.ends_with(":7447#…/env"), "{shown}");
        assert_eq!(port_of(&shown), Some(7447));
    }

    #[test]
    fn the_file_is_written_whole_and_removed_with_the_serve() {
        let dir = std::env::temp_dir().join(format!("rlmesh-ready-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ready.json");
        let ready = Ready::new("env", "env", ":0/env", "127.0.0.1:41000#secretkey/env");
        let announced = announce(&ready, Some(path.clone()));
        let body: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(body["kind"], "env");
        assert_eq!(body["participant"], "env");
        assert_eq!(body["address"], "127.0.0.1:41000#…/env");
        assert_eq!(body["port"], 41000);
        assert_eq!(body["pid"], std::process::id());
        assert!(
            !String::from_utf8(std::fs::read(&path).unwrap())
                .unwrap()
                .contains("secretkey")
        );
        assert_eq!(
            std::fs::read_dir(&dir).unwrap().count(),
            1,
            "no partial file left"
        );
        drop(announced);
        assert!(!path.exists());
        let _ = std::fs::remove_dir(&dir);
    }
}
