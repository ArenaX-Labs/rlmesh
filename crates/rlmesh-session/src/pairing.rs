//! Machine pairing over the network (design §3): the laptop's side
//! ([`pair`]), rlmeshd's pairing endpoint ([`PairService`]), and the host
//! file a paired machine keeps ([`HostFile`]), which its participants
//! attach with ([`paired_credential`]). A browser token pairs the same way
//! (design §5): its key stays on the machine whose gateway serves the
//! browser, beside a hash of the token ([`BrowserToken`]). The exchanges
//! themselves are [`rlmesh_session_core::auth::pairing`]'s; this module
//! moves their bytes on [`PAIR_KEY`].

use std::collections::HashMap;
use std::net::{SocketAddr, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rlmesh_session_core::auth::address::{base32, unbase32};
use rlmesh_session_core::auth::pairing::{
    self, ApproveError, Approved, Handled, PairError, Pairing, Pairings, Pending, Progress,
};
use rlmesh_session_core::auth::{Credential, HostIdentity};
use rlmesh_session_core::crypto::lp::kdf;
use rlmesh_session_core::crypto::{Key, Rng, Sas, StaticKeypair};
use tokio_util::sync::CancellationToken;
use zenoh::Wait;
use zenoh::query::{ConsolidationMode, Query};
use zeroize::Zeroizing;

use crate::address;
use crate::error::SessionError;
use crate::host::OsRng;
use crate::transport::Transport;
use crate::wire::{CtlError, PAIR_KEY, PairInfo};

/// How long a step before the code waits for its answer before sending
/// it again.
const STEP_TRY: Duration = Duration::from_secs(2);
/// How long a `PairWait` waits before it is sent again.
const WAIT_TRY: Duration = Duration::from_secs(10);
/// How long the laptop tries to reach rlmeshd's pairing endpoint.
const REACH: Duration = Duration::from_secs(15);
/// How long the laptop waits for the owner: the daemon's expiry.
const APPROVAL: Duration = Duration::from_nanos(pairing::EXPIRY_NS.unsigned_abs());

/// A machine rlmeshd approved: what it stores.
pub struct Paired {
    /// rlmeshd's answer, its host key checked against the one XX
    /// authenticated.
    pub info: PairInfo,
    /// The key this machine attaches with from now on.
    pub device: StaticKeypair,
}

fn pair_failed(err: PairError) -> SessionError {
    match err {
        PairError::Refused(err) => SessionError::Rejected(err),
        other => SessionError::Key(format!("pairing failed: {other}")),
    }
}

/// Pairs this machine with the rlmeshd at `base` as `name`, of `kind`
/// ([`pairing::MACHINE`], or [`pairing::BROWSER`] for a browser token):
/// `show` gets the code to type on the robot (`rlmesh pair approve`), and
/// the result arrives once the owner has.
///
/// # Errors
/// [`SessionError::Rejected`] when rlmeshd or the owner refused,
/// [`SessionError::Timeout`] when rlmeshd never answered or no one
/// approved in time, [`SessionError::Key`] when the exchange failed its
/// checks (someone in the middle).
pub async fn pair(
    base: SocketAddr,
    name: &str,
    kind: &'static str,
    mut show: impl FnMut(Sas),
) -> Result<Paired, SessionError> {
    let key_error = |err: rlmesh_session_core::crypto::RngError| SessionError::Key(err.to_string());
    let device = StaticKeypair::generate(&mut OsRng).map_err(key_error)?;
    let mut exchange = [0_u8; 16];
    OsRng.fill_bytes(&mut exchange).map_err(key_error)?;
    let mut pairing =
        Pairing::start(&mut OsRng, &device, exchange, name, kind).map_err(pair_failed)?;
    let transport = address::connect(base, REACH).await?;
    let control = transport.control().clone();
    let mut deadline = Instant::now() + REACH;
    let mut waiting = false;
    let outcome = loop {
        if Instant::now() >= deadline {
            break Err(SessionError::Timeout(
                if waiting { APPROVAL } else { REACH },
                if waiting {
                    "the owner to approve the pairing on the robot".to_string()
                } else {
                    format!("rlmeshd's pairing endpoint at {base}")
                },
            ));
        }
        let wait = if waiting { WAIT_TRY } else { STEP_TRY };
        let Some(reply) = ask(&control, pairing.request(), wait).await? else {
            // Nothing answered: not up yet, or the query was lost.
            tokio::time::sleep(Duration::from_millis(200)).await;
            continue;
        };
        match pairing.on_reply(&mut OsRng, &reply) {
            Ok(Progress::Next | Progress::Ignored) => {}
            Ok(Progress::Show(sas)) => {
                show(sas);
                waiting = true;
                deadline = Instant::now() + APPROVAL;
            }
            Ok(Progress::Paired(info)) => break Ok(Paired { info, device }),
            Err(err) => break Err(pair_failed(err)),
        }
    };
    transport.close().await?;
    outcome
}

/// The first reply to a pairing query, or `None` when none came in `wait`.
async fn ask(
    control: &zenoh::Session,
    request: &[u8],
    wait: Duration,
) -> Result<Option<Vec<u8>>, SessionError> {
    let replies = control
        .get(PAIR_KEY)
        .payload(request.to_vec())
        .consolidation(ConsolidationMode::None)
        .timeout(wait)
        .await?;
    while let Ok(reply) = replies.recv_async().await {
        if let Ok(sample) = reply.result() {
            return Ok(Some(sample.payload().to_bytes().into_owned()));
        }
    }
    Ok(None)
}

struct Desk {
    pairings: Pairings,
    /// Held `PairWait` queries by exchange.
    held: HashMap<[u8; 16], Vec<Query>>,
}

impl Desk {
    /// Answers the held queries the core's decisions answered.
    fn deliver(&mut self) {
        for (exchange, reply) in self.pairings.take_results() {
            for query in self.held.remove(&exchange).unwrap_or_default() {
                answer(query, reply.clone());
            }
        }
    }
}

fn answer(query: Query, reply: Vec<u8>) {
    let key = query.key_expr().clone();
    let _ = query.reply(key, reply).wait();
}

/// rlmeshd's pairing endpoint (`rlmeshd/pair`): it answers laptops'
/// exchanges, and takes the owner's approvals from the daemon's local
/// channel. Dropping it stops answering.
pub struct PairService {
    desk: Arc<Mutex<Desk>>,
    started: Instant,
    cancel: CancellationToken,
    _queryable: zenoh::query::Queryable<()>,
}

fn lock(desk: &Mutex<Desk>) -> std::sync::MutexGuard<'_, Desk> {
    desk.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl PairService {
    /// Serves pairing on `transport` as `identity`.
    pub async fn start(
        transport: &Transport,
        identity: Arc<HostIdentity>,
    ) -> Result<Self, SessionError> {
        let started = Instant::now();
        let desk = Arc::new(Mutex::new(Desk {
            pairings: Pairings::new(),
            held: HashMap::new(),
        }));
        let now = move || i64::try_from(started.elapsed().as_nanos()).unwrap_or(i64::MAX);
        let queryable = transport
            .control()
            .declare_queryable(PAIR_KEY)
            .callback({
                let desk = desk.clone();
                move |query| {
                    let Some(raw) = query.payload().map(|p| p.to_bytes().into_owned()) else {
                        return;
                    };
                    let mut desk = lock(&desk);
                    match desk.pairings.handle(&mut OsRng, &identity, now(), &raw) {
                        Handled::Reply(reply) => answer(query, reply),
                        Handled::Wait(exchange) => {
                            desk.held.entry(exchange).or_default().push(query);
                        }
                        Handled::Ignore => {}
                    }
                }
            })
            .await?;
        let cancel = CancellationToken::new();
        tokio::spawn({
            let (desk, cancel) = (desk.clone(), cancel.clone());
            async move {
                loop {
                    tokio::select! {
                        () = cancel.cancelled() => return,
                        () = tokio::time::sleep(Duration::from_secs(1)) => {}
                    }
                    let mut desk = lock(&desk);
                    for exchange in desk.pairings.expire(now()) {
                        for query in desk.held.remove(&exchange).unwrap_or_default() {
                            answer(query, pairing::expired_reply());
                        }
                    }
                }
            }
        });
        Ok(Self {
            desk,
            started,
            cancel,
            _queryable: queryable,
        })
    }

    fn now(&self) -> i64 {
        i64::try_from(self.started.elapsed().as_nanos()).unwrap_or(i64::MAX)
    }

    /// The exchanges showing a code, waiting for the owner.
    pub fn pending(&self) -> Vec<Pending> {
        lock(&self.desk).pairings.pending(self.now())
    }

    /// The owner typed `code`: the exchange to enroll, then
    /// [`decide`](Self::decide).
    ///
    /// # Errors
    /// [`ApproveError`]: nothing approved (and the exchanges it refused
    /// are told).
    pub fn approve(&self, code: u32) -> Result<Approved, ApproveError> {
        let mut desk = lock(&self.desk);
        let approved = desk.pairings.approve(code);
        desk.deliver();
        approved
    }

    /// The outcome of an approved exchange, sent to its machine: whether
    /// the exchange was still there (not expired).
    pub fn decide(&self, exchange: [u8; 16], result: Result<PairInfo, CtlError>) -> bool {
        let mut desk = lock(&self.desk);
        let kept = desk.pairings.decide(exchange, result);
        desk.deliver();
        kept
    }
}

impl Drop for PairService {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
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

/// What a paired machine knows of one host (`{host id}.toml` in
/// [`hosts_dir`], 0600): the host's static key, id and addresses, and this
/// machine's device id and key there.
///
/// ```toml
/// host_id = "rlm-3f9a2c"
/// host_key = "…"                # base32
/// addresses = ["192.168.1.20:7447", "robot.local:7447"]
/// device = "laptop-josh"
/// device_key = "…"              # base32, a secret
/// ```
pub struct HostFile {
    /// The host id.
    pub host_id: String,
    /// The host's static X25519 key.
    pub host_key: [u8; 32],
    /// Its base addresses, `host:port`.
    pub addresses: Vec<String>,
    /// This machine's device id there.
    pub device: String,
    device_key: Zeroizing<[u8; 32]>,
}

impl std::fmt::Debug for HostFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostFile")
            .field("host_id", &self.host_id)
            .field("addresses", &self.addresses)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl HostFile {
    /// What `paired` stores; `dialed`, the address pairing reached the
    /// host at, comes first unless the host gave it too.
    pub fn new(paired: &Paired, dialed: &str) -> Self {
        let mut addresses = vec![dialed.to_string()];
        for address in &paired.info.addresses {
            if !addresses.contains(address) {
                addresses.push(address.clone());
            }
        }
        let mut device_key = Zeroizing::new([0_u8; 32]);
        paired.device.write_private(&mut device_key);
        Self {
            host_id: paired.info.host_id.clone(),
            host_key: paired.info.host_key,
            addresses,
            device: paired.info.device.clone(),
            device_key,
        }
    }

    /// Writes it to `{host id}.toml` in `dir` (0600, the directory 0700):
    /// its path.
    pub fn write(&self, dir: &Path) -> Result<PathBuf, SessionError> {
        if self.host_id.is_empty()
            || !self
                .host_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-')
        {
            return Err(SessionError::Config(format!(
                "host id {:?} is not a file name",
                self.host_id
            )));
        }
        let mut table = toml::Table::new();
        table.insert("host_id".into(), self.host_id.clone().into());
        table.insert("host_key".into(), base32(&self.host_key).into());
        table.insert(
            "addresses".into(),
            toml::Value::Array(self.addresses.iter().cloned().map(Into::into).collect()),
        );
        table.insert("device".into(), self.device.clone().into());
        table.insert(
            "device_key".into(),
            Zeroizing::new(base32(&self.device_key)).as_str().into(),
        );
        let text = Zeroizing::new(
            toml::to_string(&table).map_err(|err| SessionError::Config(err.to_string()))?,
        );
        let path = dir.join(format!("{}.toml", self.host_id));
        address::write_private(&path, text.as_bytes())?;
        Ok(path)
    }

    /// The host file at `path`.
    pub fn read(path: &Path) -> Result<Self, SessionError> {
        let bad = |why: &str| SessionError::Config(format!("{}: {why}", path.display()));
        let text = Zeroizing::new(std::fs::read_to_string(path)?);
        let table: toml::Table = text.parse().map_err(|_| bad("not TOML"))?;
        let text = |key: &str| {
            table
                .get(key)
                .and_then(toml::Value::as_str)
                .ok_or_else(|| bad(&format!("no {key}")))
        };
        let key = |name: &str| {
            text(name).and_then(|t| unbase32(t).ok_or_else(|| bad(&format!("{name} is not a key"))))
        };
        let addresses = table
            .get("addresses")
            .and_then(toml::Value::as_array)
            .map(|list| {
                list.iter()
                    .filter_map(|a| a.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        Ok(Self {
            host_id: text("host_id")?.to_string(),
            host_key: key("host_key")?,
            addresses,
            device: text("device")?.to_string(),
            device_key: Zeroizing::new(key("device_key")?),
        })
    }

    /// The credential its participants attach with.
    pub fn credential(&self) -> Credential {
        let mut private = self.device_key.clone();
        Credential::new(StaticKeypair::from_private(&mut private), self.host_key)
    }

    /// Whether one of its addresses resolves to `base`.
    fn reaches(&self, base: SocketAddr) -> bool {
        self.addresses.iter().any(|address| {
            address
                .to_socket_addrs()
                .is_ok_and(|mut found| found.any(|a| a == base))
        })
    }
}

/// The credential of the host file in `dir` one of whose addresses is
/// `base`, if any.
pub fn paired_credential(dir: &Path, base: SocketAddr) -> Option<Credential> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "toml"))
        .collect();
    paths.sort();
    paths
        .iter()
        .filter_map(|path| HostFile::read(path).ok())
        .find(|file| file.reaches(base))
        .map(|file| file.credential())
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
