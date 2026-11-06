//! Pairing over the network (design §3): a machine's (the laptop's side,
//! [`pair`]; the host's pairing endpoint, [`PairService`]; the host file a
//! paired machine keeps, [`HostFile`], which its participants attach with,
//! [`paired_credential`]), and a browser token's (design §5), whose key
//! stays on the machine whose gateway serves the browser, beside a hash of
//! the token ([`BrowserToken`]).

use std::path::{Path, PathBuf};

use rlmesh_session_core::auth::Credential;
use rlmesh_session_core::auth::address::{base32, unbase32};
use rlmesh_session_core::crypto::lp::kdf;
use rlmesh_session_core::crypto::{Key, Rng, StaticKeypair};
use zeroize::Zeroizing;

use crate::address;
use crate::error::SessionError;
use crate::host::OsRng;
use crate::wire::PairInfo;

mod machine;
pub use machine::{HostFile, PairService, pair, paired_credential};

/// A machine rlmeshd approved: what it stores.
pub struct Paired {
    /// rlmeshd's answer, its host key checked against the one XX
    /// authenticated.
    pub info: PairInfo,
    /// The key this machine attaches with from now on.
    pub device: StaticKeypair,
}

/// Where a paired machine keeps its host files: `rlmesh/hosts` in
/// `$XDG_CONFIG_HOME`, else in `~/.config` (design §9, on every OS).
pub fn hosts_dir() -> Option<PathBuf> {
    let config = std::env::var_os("XDG_CONFIG_HOME")
        .filter(|dir| !dir.is_empty())
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".config")))?;
    Some(config.join("rlmesh").join("hosts"))
}

/// Where a gateway on this machine finds the browser tokens paired here
/// (`rlmesh pair HOST --browser`): `rlmesh/browsers.toml` beside
/// [`hosts_dir`].
pub fn browsers_path() -> Option<PathBuf> {
    Some(hosts_dir()?.with_file_name("browsers.toml"))
}

const TOKEN_LABEL: &[u8] = b"rlmesh/v2 browser token";

/// What a gateway keeps of a browser token instead of the token itself.
pub fn token_hash(token: &str) -> [u8; 32] {
    kdf(&[0; 32], TOKEN_LABEL, &[token.as_bytes()])
        .unwrap_or_else(|_| unreachable!("a short label"))
}

/// A new browser token: 32 random bytes, base32 (52 characters a
/// WebSocket subprotocol can carry).
pub fn new_token() -> Result<String, SessionError> {
    let mut bytes = Zeroizing::new([0_u8; 32]);
    OsRng
        .fill_bytes(&mut *bytes)
        .map_err(|err| SessionError::Key(err.to_string()))?;
    Ok(base32(&bytes))
}

/// A browser token paired with a host, as the gateway that serves the
/// browser keeps it (`[browsers.{device}]` in [`browsers_path`], 0600):
/// the device rlmeshd enrolled it as, the host's id and static key, the
/// token's hash ([`token_hash`]), never the token, and the device's key,
/// which the gateway attaches the browser with (design §5).
///
/// ```toml
/// [browsers.web-josh]
/// host_id = "rlm-3f9a2c"
/// host_key = "…"      # base32
/// token = "…"         # the token's hash, base32
/// device_key = "…"    # base32, a secret
/// ```
pub struct BrowserToken {
    /// Its device id on the host.
    pub device: String,
    /// The host id.
    pub host_id: String,
    host_key: [u8; 32],
    hash: [u8; 32],
    device_key: Zeroizing<[u8; 32]>,
}

impl std::fmt::Debug for BrowserToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BrowserToken")
            .field("device", &self.device)
            .field("host_id", &self.host_id)
            .finish_non_exhaustive()
    }
}

impl BrowserToken {
    /// What a gateway keeps of `token`, which `paired` enrolled.
    pub fn new(paired: &Paired, token: &str) -> Self {
        let mut device_key = Zeroizing::new([0_u8; 32]);
        paired.device.write_private(&mut device_key);
        Self {
            device: paired.info.device.clone(),
            host_id: paired.info.host_id.clone(),
            host_key: paired.info.host_key,
            hash: token_hash(token),
            device_key,
        }
    }

    /// The credential the gateway attaches its browser with.
    pub fn credential(&self) -> Credential {
        let mut private = self.device_key.clone();
        Credential::new(StaticKeypair::from_private(&mut private), self.host_key)
    }

    /// Adds it to the file at `path` (made 0600 if new), replacing an entry
    /// of the same device.
    pub fn add(&self, path: &Path) -> Result<(), SessionError> {
        let mut table = read_browsers(path)?;
        let mut entry = toml::Table::new();
        entry.insert("host_id".into(), self.host_id.clone().into());
        entry.insert("host_key".into(), base32(&self.host_key).into());
        entry.insert("token".into(), base32(&self.hash).into());
        entry.insert(
            "device_key".into(),
            Zeroizing::new(base32(&self.device_key)).as_str().into(),
        );
        table.insert(self.device.clone(), entry.into());
        let mut file = toml::Table::new();
        file.insert("browsers".into(), table.into());
        let text = Zeroizing::new(
            toml::to_string(&file).map_err(|err| SessionError::Config(err.to_string()))?,
        );
        address::write_private(path, text.as_bytes())?;
        Ok(())
    }

    /// The entry of the file at `path` whose hash is `token`'s, if any.
    pub fn find(path: &Path, token: &str) -> Result<Option<Self>, SessionError> {
        let want = Key::from_bytes(token_hash(token));
        let bad = |device: &str| {
            SessionError::Config(format!("{}: browser {device} is malformed", path.display()))
        };
        for (device, entry) in read_browsers(path)? {
            let text = |key: &str| entry.get(key).and_then(toml::Value::as_str);
            let key = |name: &str| text(name).and_then(unbase32);
            let hash = key("token").ok_or_else(|| bad(&device))?;
            if !Key::from_bytes(hash).ct_eq(&want) {
                continue;
            }
            return Ok(Some(Self {
                host_id: text("host_id").unwrap_or_default().to_string(),
                host_key: key("host_key").ok_or_else(|| bad(&device))?,
                hash,
                device_key: Zeroizing::new(key("device_key").ok_or_else(|| bad(&device))?),
                device,
            }));
        }
        Ok(None)
    }
}

/// The `[browsers]` table of the file at `path` (none yet: empty).
fn read_browsers(path: &Path) -> Result<toml::Table, SessionError> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => Zeroizing::new(text),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(toml::Table::new()),
        Err(err) => return Err(err.into()),
    };
    let mut file: toml::Table = text
        .parse()
        .map_err(|_| SessionError::Config(format!("{}: not TOML", path.display())))?;
    match file.remove("browsers") {
        Some(toml::Value::Table(table)) => Ok(table),
        None => Ok(toml::Table::new()),
        Some(_) => Err(SessionError::Config(format!(
            "{}: browsers is not a table",
            path.display()
        ))),
    }
}
