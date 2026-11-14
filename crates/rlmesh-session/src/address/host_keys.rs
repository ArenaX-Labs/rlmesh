//! The key a loopback host in this process leaves for this user's other
//! processes on the machine, so a join that names only the address
//! (`127.0.0.1:5587/cartpole`) finds it: one file per listening address in
//! `hosts` under the runtime directory ([`runtime_dirs`]), 0600 in a 0700
//! directory, removed when the host closes.
//!
//! The host holds a shared lock on its file while it runs. A file nothing
//! holds is a leftover of a host that is gone (its process crashed or was
//! killed) and is passed over, so a host that later takes the port (rlmeshd,
//! another user's) is joined as if no file were there.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use rlmesh_session_core::auth::AddressKey;
use zeroize::Zeroizing;

use crate::address::hosting::runtime_dirs;
use crate::error::SessionError;

/// A host's key file, removed when dropped (the host closed).
pub(super) struct KeyFile {
    path: PathBuf,
    /// Open, with the shared lock that says its host runs.
    file: File,
}

impl std::fmt::Debug for KeyFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("KeyFile").field(&self.path).finish()
    }
}

/// A key found on file for a host, and where.
#[derive(Debug)]
pub(super) struct Filed {
    pub key: AddressKey,
    pub path: PathBuf,
}

/// The file name of the key of the host listening at `base`.
fn file_name(base: SocketAddr) -> String {
    // An IPv6 address's colons are no file name on Windows.
    format!(
        "{}-{}.key",
        base.ip().to_string().replace(':', "_"),
        base.port()
    )
}

/// The directories key files are written to and looked for in, in order.
fn key_dirs() -> Vec<PathBuf> {
    runtime_dirs()
        .into_iter()
        .map(|dir| dir.join("hosts"))
        .collect()
}

impl KeyFile {
    /// Leaves `key` for this user's other processes as the key of the host
    /// listening at `base` (loopback), in the first of the key directories
    /// that takes it. `None` (logged) where none does: other processes then
    /// join with the keyed address.
    pub(super) fn write(base: SocketAddr, key: &AddressKey) -> Option<Self> {
        let mut last = None;
        for dir in key_dirs() {
            match Self::write_in(&dir, base, key) {
                Ok(file) => return Some(file),
                Err(err) => last = Some((dir, err)),
            }
        }
        if let Some((dir, err)) = last {
            tracing::warn!(
                "the host at {base} left no key for this user's other processes ({}: {err}); \
                 they join it with the address it prints",
                dir.display()
            );
        }
        None
    }

    /// [`Self::write`] in `dir`: written to a new 0600 file, locked, then
    /// renamed into place, so the key is never readable by anyone else nor
    /// found half-written or unlocked.
    pub(super) fn write_in(
        dir: &Path,
        base: SocketAddr,
        key: &AddressKey,
    ) -> std::io::Result<Self> {
        let mut builder = std::fs::DirBuilder::new();
        builder.recursive(true);
        #[cfg(unix)]
        std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
        builder.create(dir)?;
        static WRITTEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = WRITTEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let name = file_name(base);
        let partial = dir.join(format!(".{name}.{}.{n}.partial", std::process::id()));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
        }
        let mut file = options.open(&partial)?;
        let written = (|| {
            file.write_all(Zeroizing::new(key.to_string()).as_bytes())?;
            file.lock_shared()?;
            let path = dir.join(&name);
            std::fs::rename(&partial, &path)?;
            Ok(path)
        })();
        match written {
            Ok(path) => Ok(Self { path, file }),
            Err(err) => {
                let _ = std::fs::remove_file(&partial);
                Err(err)
            }
        }
    }

    #[cfg(test)]
    pub(super) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for KeyFile {
    /// Removes the file, unless another host has since put its own there.
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let ours = |m: &std::fs::Metadata| (m.dev(), m.ino());
            let (Ok(open), Ok(named)) =
                (self.file.metadata(), std::fs::symlink_metadata(&self.path))
            else {
                return;
            };
            if ours(&open) != ours(&named) {
                return;
            }
        }
        let _ = std::fs::remove_file(&self.path);
    }
}

/// The key this user's host listening at `base` left on file, when `base`
/// is loopback and a host that still runs left one. A file this user
/// doesn't own, that others can read, or that isn't a key is refused with
/// an error, never read past.
pub(super) fn filed_key(base: SocketAddr) -> Result<Option<Filed>, SessionError> {
    if !base.ip().is_loopback() {
        return Ok(None);
    }
    for dir in key_dirs() {
        if let Some(filed) = filed_in(&dir, base)? {
            return Ok(Some(filed));
        }
    }
    Ok(None)
}

/// [`filed_key`] in `dir`.
pub(super) fn filed_in(dir: &Path, base: SocketAddr) -> Result<Option<Filed>, SessionError> {
    let path = dir.join(file_name(base));
    let refused = |why: String| {
        SessionError::Config(format!(
            "refused the key on file for the session host at {base} ({}): {why}; remove it, or \
             join with the address its host printed (it ends in #key)",
            path.display()
        ))
    };
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::custom_flags(&mut options, libc::O_NOFOLLOW);
    let mut file = match options.open(&path) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(refused(format!("cannot open it ({err}), or it is a link"))),
    };
    let meta = file
        .metadata()
        .map_err(|err| refused(format!("cannot stat it ({err})")))?;
    if !meta.is_file() {
        return Err(refused("it is not a file".into()));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        trusted(meta.uid(), meta.mode(), effective_uid()).map_err(refused)?;
    }
    // Its host holds a shared lock while it runs: a file this takes an
    // exclusive one on outlived its host.
    match file.try_lock() {
        Ok(()) => {
            let _ = file.unlock();
            return Ok(None);
        }
        Err(std::fs::TryLockError::WouldBlock) => {}
        Err(std::fs::TryLockError::Error(err)) => {
            return Err(refused(format!(
                "cannot tell whether its host runs ({err})"
            )));
        }
    }
    let mut text = Zeroizing::new(String::new());
    file.read_to_string(&mut text)
        .map_err(|err| refused(format!("cannot read it ({err})")))?;
    let key = text
        .parse()
        .map_err(|_| refused("it holds no address key".into()))?;
    Ok(Some(Filed { key, path }))
}

/// Whether a key file owned by `uid` with `mode` may be read by user `me`:
/// theirs, and no one else's to read or write.
#[cfg(unix)]
fn trusted(uid: u32, mode: u32, me: u32) -> Result<(), String> {
    if uid != me {
        return Err(format!("it is owned by uid {uid}, not by this user"));
    }
    if mode & 0o077 != 0 {
        return Err(format!(
            "its mode {:o} lets others read or write it (a key file is 0600)",
            mode & 0o777
        ));
    }
    Ok(())
}

#[cfg(unix)]
#[allow(unsafe_code)]
fn effective_uid() -> u32 {
    // SAFETY: geteuid takes no arguments, cannot fail and touches no memory.
    unsafe { libc::geteuid() }
}

#[cfg(all(test, unix))]
mod tests;
