//! Remote environment clients over one core supporting any number of lanes.

use std::sync::Arc;
use std::time::Duration;

use rlmesh_grpc::env::{ResetRequest as ProtoResetRequest, StepRequest as ProtoStepRequest};
use rlmesh_grpc::wire::{
    decode_batched_partial_values, encode_batched_partial_values, meta_map_to_proto,
    render_request_to_proto, render_result_from_proto,
};
use rlmesh_spaces::{MetaMap, MetaValue};
use tokio_util::sync::CancellationToken;

use super::server::Transport;
use super::types::{
    RenderRequest, ResetRequest as VectorResetRequest, ResetResult as VectorResetResult,
    StepRequest as VectorStepRequest, StepResult as VectorStepResult,
};
use super::wire::{proto_episode_metadata_to_public, validate_action_count, validate_count};
use crate::run::drive::{
    CancelWatch, block_on_library, in_async_context, in_library, refuse_in_async_context, runtime,
};
use crate::spaces::{Space, SpaceSpec};
use crate::value::{meta_codec, space_codec};
use crate::{
    ConnectAddress, EnvContract, EnvironmentError, Error, InfoDict, Metadata, Result, Value,
    VectorSeed, binding::spaces,
};

/// How long `close` waits for the server when the client has no request
/// timeout: past it the client detaches locally.
const CLOSE_GRACE: Duration = Duration::from_secs(5);

/// The runtime-reserved info keys.
const RUNTIME_INFO_KEYS: [&str; 2] = ["episode_ids", "completed_episodes"];

/// Options for connecting to a single or vector environment.
///
/// `address` excludes the helpers (`host`, `port`, `path`, `transport`);
/// without either the constructor fails with the ValueError class. A zero
/// timeout means no timeout, as `None` does. `Debug` never prints the
/// `api_key`.
#[derive(Clone, Default)]
pub struct RemoteEnvOptions {
    /// The endpoint (`tcp://host:port`, `host:port`, `https://host[:port]`
    /// or `unix:///path`).
    pub address: Option<String>,
    /// TCP host helper (default `127.0.0.1`); needs `port`.
    pub host: Option<String>,
    /// TCP port helper.
    pub port: Option<u16>,
    /// Unix socket path helper; excludes `host` and `port`.
    pub path: Option<String>,
    /// Explicit transport; `Unix` needs `path`, `Tcp` refuses it.
    pub transport: Option<Transport>,
    /// Bound on the dial and handshake.
    pub connect_timeout: Option<Duration>,
    /// Bound on each `reset`, `step`, `render` and `shutdown`, also sent to
    /// the server as the request deadline; `close` uses it too.
    pub request_timeout: Option<Duration>,
    /// The workflow edition this client declares on the handshake; empty or
    /// whitespace declares nothing and resolution continues.
    pub workflow_edition: Option<String>,
    /// The key sent as `authorization: Bearer <key>` on every request; `None`
    /// reads `RLMESH_API_KEY` for an `https://` address only, `""` sends none.
    pub api_key: Option<String>,
    /// A PEM bundle an `https://` dial trusts instead of the system roots.
    pub tls_ca_file: Option<std::path::PathBuf>,
}

impl std::fmt::Debug for RemoteEnvOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteEnvOptions")
            .field("address", &self.address)
            .field("host", &self.host)
            .field("port", &self.port)
            .field("path", &self.path)
            .field("transport", &self.transport)
            .field("connect_timeout", &self.connect_timeout)
            .field("request_timeout", &self.request_timeout)
            .field("workflow_edition", &self.workflow_edition)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("tls_ca_file", &self.tls_ca_file)
            .finish()
    }
}

impl RemoteEnvOptions {
    /// Set `address`.
    pub fn address(mut self, address: impl Into<String>) -> Self {
        self.address = Some(address.into());
        self
    }

    /// Set `host`.
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.host = Some(host.into());
        self
    }

    /// Set `port`.
    pub fn port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }

    /// Set `path`.
    pub fn path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Set `transport`.
    pub fn transport(mut self, transport: Transport) -> Self {
        self.transport = Some(transport);
        self
    }

    /// Set `connect_timeout`.
    pub fn connect_timeout(mut self, timeout: impl Into<Duration>) -> Self {
        self.connect_timeout = Some(timeout.into());
        self
    }

    /// Set `request_timeout`.
    pub fn request_timeout(mut self, timeout: impl Into<Duration>) -> Self {
        self.request_timeout = Some(timeout.into());
        self
    }

    /// Set `workflow_edition`.
    pub fn workflow_edition(mut self, edition: impl Into<String>) -> Self {
        self.workflow_edition = Some(edition.into());
        self
    }

    /// Set `api_key`.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set `tls_ca_file`.
    pub fn tls_ca_file(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.tls_ca_file = Some(path.into());
        self
    }
}

/// Normalize a connect address: one connect string from an address or the
/// helpers, every refusal the ValueError class.
pub(crate) fn normalize_connect_address(
    address: Option<&str>,
    host: Option<&str>,
    port: Option<u16>,
    path: Option<&str>,
    transport: Option<Transport>,
) -> Result<String> {
    let helpers = host.is_some() || port.is_some() || path.is_some() || transport.is_some();
    if let Some(address) = address {
        if address.is_empty() {
            return Err(Error::invalid_argument("address cannot be empty"));
        }
        if helpers {
            return Err(Error::invalid_argument(
                "address cannot be combined with host, port, path, or transport",
            ));
        }
        return Ok(address.to_string());
    }
    let tcp_helpers = host.is_some() || port.is_some();
    if path.is_some() && tcp_helpers {
        return Err(Error::invalid_argument(
            "path cannot be combined with host or port",
        ));
    }
    if path.is_some() && transport == Some(Transport::Tcp) {
        return Err(Error::invalid_argument(
            "path cannot be combined with transport='tcp'",
        ));
    }
    if transport == Some(Transport::Unix) && tcp_helpers {
        return Err(Error::invalid_argument(
            "unix transport cannot be combined with host or port",
        ));
    }
    if path.is_some() || transport == Some(Transport::Unix) {
        let Some(path) = path else {
            return Err(Error::invalid_argument(
                "path is required when transport='unix'",
            ));
        };
        if cfg!(windows) {
            return Err(Error::invalid_argument(
                "unix sockets are not supported on Windows; use tcp://host:port instead",
            ));
        }
        return Ok(format!("unix://{path}"));
    }
    if !tcp_helpers {
        return Err(Error::invalid_argument("address or host/port is required"));
    }
    let Some(port) = port else {
        return Err(Error::invalid_argument(
            "port is required when connecting with host/path helpers",
        ));
    };
    Ok(format!(
        "tcp://{}:{port}",
        host.filter(|host| !host.is_empty()).unwrap_or("127.0.0.1")
    ))
}

/// Zero means no timeout, as `None` does.
pub(crate) fn normalize_timeout(timeout: Option<Duration>) -> Option<Duration> {
    timeout.filter(|timeout| !timeout.is_zero())
}

/// A timeout given in seconds (the bindings' `*_timeout_seconds`): `None`
/// and zero are no timeout; negative or non-finite is the ValueError class.
pub(crate) fn timeout_from_secs(value: Option<f64>, field: &str) -> Result<Option<Duration>> {
    match value {
        None => Ok(None),
        Some(seconds) => Duration::try_from_secs_f64(seconds)
            .map(|timeout| normalize_timeout(Some(timeout)))
            .map_err(|_| {
                Error::invalid_argument(format!(
                    "{field} must be a non-negative finite float or None"
                ))
            }),
    }
}

/// The wire `timeout_ms` for a request deadline: rounded up to whole
/// milliseconds, 0 for none.
fn wire_timeout_ms(timeout: Option<Duration>) -> u64 {
    timeout.map_or(0, |timeout| {
        u64::try_from(timeout.as_nanos().div_ceil(1_000_000)).unwrap_or(u64::MAX)
    })
}

/// Bound `future` by `timeout`: a reply handled at or after the deadline is
/// discarded and the call fails with the TimeoutError class.
async fn bounded<T>(
    timeout: Option<Duration>,
    future: impl std::future::Future<Output = Result<T>>,
) -> Result<T> {
    match timeout {
        Some(timeout) => {
            let deadline = tokio::time::Instant::now() + timeout;
            rlmesh_grpc::env::client::with_request_deadline(
                deadline,
                tokio::time::timeout_at(deadline, future),
            )
            .await
            .map_err(|_| Error::Timeout(timeout))?
        }
        None => future.await,
    }
}

/// A contract's observation and action spaces; one without them is a
/// malformed handshake.
fn spaces_of(
    contract: &spaces::EnvContract,
) -> Result<(Arc<spaces::SpaceSpec>, Arc<spaces::SpaceSpec>)> {
    match (&contract.observation_space, &contract.action_space) {
        (Some(observation), Some(action)) => {
            Ok((Arc::new(observation.clone()), Arc::new(action.clone())))
        }
        _ => Err(Error::Protocol(
            "remote environment contract is missing its spaces".to_string(),
        )),
    }
}

/// A lane index for the wire (`uint32`): above `u32::MAX` is the ValueError
/// class, before any I/O.
fn wire_lane(index: u64) -> Result<u32> {
    u32::try_from(index).map_err(|_| {
        Error::invalid_argument(format!(
            "env_index {index} is out of range: lane indices are at most {}",
            u32::MAX
        ))
    })
}

/// A shared read handle on a client's selected workflow edition: the
/// handshake selection until a pin replaces it.
#[derive(Clone, Debug)]
pub struct EditionCell(Edition);

#[derive(Clone, Debug)]
enum Edition {
    Grpc(rlmesh_grpc::env::PinnedEdition),
    #[cfg(feature = "lockstep")]
    Local(Arc<std::sync::Mutex<String>>),
}

impl EditionCell {
    /// The current selection.
    pub fn get(&self) -> String {
        match &self.0 {
            Edition::Grpc(pinned) => pinned.get(),
            #[cfg(feature = "lockstep")]
            Edition::Local(edition) => edition
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone(),
        }
    }
}

/// How a client reaches its env: a v1 gRPC Join stream, or (with the
/// `lockstep` feature) a v2 lockstep driver (`lockstep://`, §12). Both take
/// and answer native values; the proto conversion is the gRPC arm's.
#[derive(Clone)]
enum Link {
    Grpc {
        client: Box<rlmesh_grpc::EnvClient>,
        observation_space: Arc<spaces::SpaceSpec>,
        action_space: Arc<spaces::SpaceSpec>,
        num_envs: usize,
    },
    #[cfg(feature = "lockstep")]
    Lockstep {
        client: crate::lockstep::LockstepClient,
        address: String,
        /// The selection; lockstep sends no pin (the env runs its declared
        /// edition), so a pin is only recorded.
        edition: Arc<std::sync::Mutex<String>>,
        episodes: Episodes,
    },
}

#[cfg(feature = "lockstep")]
fn lockstep_error(error: crate::lockstep::GymError) -> Error {
    use crate::lockstep::GymError;
    match error {
        GymError::Timeout(timeout) => Error::Timeout(timeout),
        GymError::Closed(why) => Error::Connection(why),
        GymError::Session(error) => Error::Connection(error.to_string()),
        GymError::Map(why) | GymError::Config(why) | GymError::Value(why) => {
            Error::invalid_argument(why)
        }
        other => Error::Environment(other.to_failure()),
    }
}

impl Link {
    /// The gRPC client, unless this link is lockstep.
    fn grpc(&mut self) -> Option<&mut rlmesh_grpc::EnvClient> {
        match self {
            Self::Grpc { client, .. } => Some(client),
            #[cfg(feature = "lockstep")]
            Self::Lockstep { .. } => None,
        }
    }

    fn address(&self) -> &str {
        match self {
            Self::Grpc { client, .. } => client.address(),
            #[cfg(feature = "lockstep")]
            Self::Lockstep { address, .. } => address,
        }
    }

    /// Resets `lanes` (empty: every lane): the observations and the info.
    async fn reset(
        &mut self,
        lanes: Vec<u32>,
        seeds: Vec<i64>,
        options: Option<MetaMap>,
        episode_ids: Vec<String>,
        timeout: Option<Duration>,
    ) -> Result<(Vec<spaces::SpaceValue>, Option<MetaMap>)> {
        match self {
            Self::Grpc {
                client,
                observation_space,
                num_envs,
                ..
            } => {
                let request = ProtoResetRequest {
                    seeds,
                    options: options.as_ref().map(meta_map_to_proto),
                    timeout_ms: wire_timeout_ms(timeout),
                    env_indices: lanes,
                    episode_ids,
                };
                let response = client.reset(request).await.map_err(Error::from_grpc)?;
                let observations = decode_batched_partial_values(
                    response.observation.as_ref(),
                    observation_space,
                    *num_envs,
                )
                .map_err(|err| Error::Protocol(err.to_string()))?;
                Ok((
                    observations,
                    response.infos.map(rlmesh_grpc::wire::meta_map_from_proto),
                ))
            }
            #[cfg(feature = "lockstep")]
            Self::Lockstep {
                client, episodes, ..
            } => {
                let reset = client
                    .reset(&lanes, &seeds, options.as_ref(), timeout)
                    .await
                    .map_err(lockstep_error)?;
                episodes.reset(&lanes, &seeds, &episode_ids);
                Ok((reset.observations, reset.info))
            }
        }
    }

    /// Steps every lane with one action each.
    async fn step(
        &mut self,
        actions: &[spaces::SpaceValue],
        episode_ids: Vec<String>,
        timeout: Option<Duration>,
    ) -> Result<VectorStepResult> {
        match self {
            Self::Grpc {
                client,
                observation_space,
                action_space,
                num_envs,
            } => {
                let action = encode_batched_partial_values(actions, action_space)
                    .map_err(|err| Error::invalid_argument(err.to_string()))?;
                let request = ProtoStepRequest {
                    action: Some(action),
                    timeout_ms: wire_timeout_ms(timeout),
                    // Full-width step; subset-stepping is reserved-but-deferred.
                    env_indices: Vec::new(),
                    episode_ids,
                };
                let response = client.step(request).await.map_err(Error::from_grpc)?;
                let observations = decode_batched_partial_values(
                    response.observation.as_ref(),
                    observation_space,
                    *num_envs,
                )
                .map_err(|err| Error::Protocol(err.to_string()))?;
                let flags = |mask: &[u8]| mask.iter().map(|value| *value != 0).collect();
                Ok(VectorStepResult {
                    observations,
                    rewards: response.rewards,
                    terminated: flags(&response.terminated_mask),
                    truncated: flags(&response.truncated_mask),
                    info: response.infos.map(rlmesh_grpc::wire::meta_map_from_proto),
                    completed_episodes: response
                        .completed_episodes
                        .into_iter()
                        .map(proto_episode_metadata_to_public)
                        .collect::<std::result::Result<Vec<_>, _>>()
                        .map_err(Error::Protocol)?,
                    episode_ids: Vec::new(),
                })
            }
            #[cfg(feature = "lockstep")]
            Self::Lockstep {
                client, episodes, ..
            } => {
                let step = client
                    .step(&[], actions, timeout)
                    .await
                    .map_err(lockstep_error)?;
                let mut result = VectorStepResult {
                    observations: step.observations,
                    rewards: step.rewards,
                    terminated: step.terminated,
                    truncated: step.truncated,
                    info: step.info,
                    completed_episodes: Vec::new(),
                    episode_ids: Vec::new(),
                };
                let mode = client.info().contract.autoreset_mode;
                result.completed_episodes = episodes.step(&result, &episode_ids, mode);
                Ok(result)
            }
        }
    }

    /// Lane `lane`'s frame as PNG bytes, if it has one.
    async fn render(&mut self, lane: u32, timeout: Option<Duration>) -> Result<Option<Vec<u8>>> {
        match self {
            Self::Grpc { client, .. } => {
                let request = render_request_to_proto(&RenderRequest {
                    env_index: Some(lane as usize),
                    timeout_ms: i64::try_from(wire_timeout_ms(timeout)).unwrap_or(i64::MAX),
                });
                let response = client.render(request).await.map_err(Error::from_grpc)?;
                let result = render_result_from_proto(response)
                    .map_err(|err| Error::Protocol(err.to_string()))?;
                Ok(result.frame.map(|frame| frame.frame))
            }
            #[cfg(feature = "lockstep")]
            Self::Lockstep { client, .. } => client
                .render(lane, timeout)
                .await
                .map(|frame| frame.map(|frame| frame.to_vec()))
                .map_err(lockstep_error),
        }
    }
}

/// The episodes a lockstep client's lanes run. A lockstep reply carries no
/// episode metadata, so the client, which holds the episode ids, keeps the
/// accounting the v1 env server reports in `completed_episodes`: an episode
/// ends on a done step or when a reset replaces it, and under NEXT_STEP the
/// lane's next step starts the autoreset episode.
#[cfg(feature = "lockstep")]
#[derive(Clone, Debug, Default)]
struct Episodes {
    lanes: Vec<Lane>,
    /// Episodes a reset replaced, reported by the next step.
    interrupted: Vec<super::types::EpisodeMetadata>,
}

#[cfg(feature = "lockstep")]
#[derive(Clone, Debug, Default)]
enum Lane {
    #[default]
    Idle,
    Active(Episode),
    /// Done under NEXT_STEP (the next step is the autoreset observation) or
    /// SAME_STEP (the next step is the new episode's first).
    Pending {
        same_step: bool,
    },
}

#[cfg(feature = "lockstep")]
#[derive(Clone, Debug)]
struct Episode {
    id: String,
    seed: Option<i64>,
    steps: i64,
    reward: f64,
    start_ns: i64,
}

#[cfg(feature = "lockstep")]
fn unix_ns() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_nanos()).unwrap_or(i64::MAX))
}

#[cfg(feature = "lockstep")]
impl Episode {
    fn new(id: String, seed: Option<i64>) -> Self {
        Self {
            id,
            seed,
            steps: 0,
            reward: 0.0,
            start_ns: unix_ns(),
        }
    }

    fn end(self, lane: usize, terminated: bool, truncated: bool) -> super::types::EpisodeMetadata {
        let end_ns = unix_ns();
        super::types::EpisodeMetadata {
            episode_id: self.id,
            seed: self.seed,
            env_index: i32::try_from(lane).unwrap_or(i32::MAX),
            step_count: self.steps,
            cumulative_reward: self.reward,
            terminated,
            truncated,
            start_timestamp_ns: self.start_ns,
            end_timestamp_ns: end_ns,
            duration_ms: (end_ns - self.start_ns).max(0) / 1_000_000,
            final_info: None,
        }
    }
}

#[cfg(feature = "lockstep")]
impl Episodes {
    fn new(num_envs: usize) -> Self {
        Self {
            lanes: (0..num_envs).map(|_| Lane::Idle).collect(),
            interrupted: Vec::new(),
        }
    }

    /// `lanes` (empty: every lane) start episodes `ids`, seeded by `seeds`.
    fn reset(&mut self, lanes: &[u32], seeds: &[i64], ids: &[String]) {
        let all: Vec<u32>;
        let lanes = if lanes.is_empty() {
            all = (0..self.lanes.len() as u32).collect();
            &all
        } else {
            lanes
        };
        for (pos, &lane) in lanes.iter().enumerate() {
            let Some(slot) = self.lanes.get_mut(lane as usize) else {
                continue;
            };
            let id = ids.get(pos).cloned().unwrap_or_default();
            let started = Lane::Active(Episode::new(id, seeds.get(pos).copied()));
            if let Lane::Active(old) = std::mem::replace(slot, started) {
                self.interrupted.push(old.end(lane as usize, false, true));
            }
        }
    }

    /// A full-width step: the episodes it completed (`ids`: the lanes'
    /// current ids, which an autoreset episode adopts).
    fn step(
        &mut self,
        result: &VectorStepResult,
        ids: &[String],
        mode: spaces::types::AutoresetMode,
    ) -> Vec<super::types::EpisodeMetadata> {
        use spaces::types::AutoresetMode;
        let mut completed = std::mem::take(&mut self.interrupted);
        for (lane, slot) in self.lanes.iter_mut().enumerate() {
            let terminated = result.terminated.get(lane).copied().unwrap_or(false);
            let truncated = result.truncated.get(lane).copied().unwrap_or(false);
            let id = || ids.get(lane).cloned().unwrap_or_default();
            let mut episode = match std::mem::take(slot) {
                Lane::Active(episode) => episode,
                // The autoreset observation: the episode the env rolled.
                Lane::Pending { same_step: false } => {
                    *slot = Lane::Active(Episode::new(id(), None));
                    continue;
                }
                Lane::Pending { same_step: true } => Episode::new(id(), None),
                Lane::Idle => continue,
            };
            episode.steps += 1;
            episode.reward += result.rewards.get(lane).copied().unwrap_or(0.0);
            if terminated || truncated {
                completed.push(episode.end(lane, terminated, truncated));
                *slot = match mode {
                    AutoresetMode::NextStep => Lane::Pending { same_step: false },
                    AutoresetMode::SameStep => Lane::Pending { same_step: true },
                    AutoresetMode::Disabled => Lane::Idle,
                };
            } else {
                *slot = Lane::Active(episode);
            }
        }
        completed
    }
}

/// How [`EnvClientCore::connect_with`] dials.
#[derive(Clone, Debug, Default)]
pub struct CoreConnect<'a> {
    /// Endpoint token (empty: none).
    pub token: &'a str,
    /// The certificate authorities an `https://` dial trusts.
    pub trust: rlmesh_grpc::TlsTrust,
    /// The workflow edition this side declares (already resolved).
    pub declared: Option<&'a str>,
    /// Bound on the dial and handshake.
    pub connect_timeout: Option<Duration>,
    /// Default bound on each request; also `close`'s bound.
    pub request_timeout: Option<Duration>,
}

/// The any-arity remote env client both [`RemoteEnv`] and
/// [`RemoteVectorEnv`] wrap, and the Python extension drives directly
/// (binding plumbing, exempt from semver).
pub struct EnvClientCore {
    inner: Link,
    env_contract: spaces::EnvContract,
    observation_space: Arc<spaces::SpaceSpec>,
    action_space: Arc<spaces::SpaceSpec>,
    num_envs: usize,
    /// The env's negotiation offer captured at handshake.
    session_offer: rlmesh_proto::SessionOffer,
    /// This connection's container id (UUIDv7), minted at connect.
    env_id: String,
    /// Per-lane current episode ids; this client is the id authority on the
    /// direct path.
    episode_ids: Vec<String>,
    /// The edition this leg negotiated, or the last pin.
    selected_workflow_edition: String,
    request_timeout: Option<Duration>,
    closed: bool,
    warned_shadowed_info_key: bool,
    /// Whether `info` carries the runtime-reserved keys.
    runtime_info: bool,
    /// The client class errors name (`RemoteEnv`, `RemoteVectorEnv`).
    name: &'static str,
}

impl std::fmt::Debug for EnvClientCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnvClientCore")
            .field("address", &self.address())
            .field("num_envs", &self.num_envs)
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}

impl EnvClientCore {
    /// Dial `address` and handshake, with no token, declaration or bounds.
    pub async fn connect(address: &str) -> Result<Self> {
        Self::connect_with(address, CoreConnect::default()).await
    }

    /// Dial `address` sending `token` on every request.
    pub async fn connect_with_token(address: &str, token: &str) -> Result<Self> {
        Self::connect_with(
            address,
            CoreConnect {
                token,
                ..CoreConnect::default()
            },
        )
        .await
    }

    /// Dial `address` and handshake. A dial failure is the ConnectionError
    /// class, a refused handshake the ProtocolException class, an expired
    /// `connect_timeout` the TimeoutError class.
    pub async fn connect_with(address: &str, options: CoreConnect<'_>) -> Result<Self> {
        let parsed = ConnectAddress::parse(address)?;
        check_token(options.token)?;
        let credentials = rlmesh_grpc::Credentials {
            token: options.token.to_string(),
            trust: options.trust,
        };
        let dial = Self::dial(parsed, &credentials, options.declared);
        let mut core = match normalize_timeout(options.connect_timeout) {
            Some(timeout) => tokio::time::timeout(timeout, dial)
                .await
                .map_err(|_| Error::Timeout(timeout))?,
            None => dial.await,
        }
        .map_err(|err| {
            crate::error::dial_failure(err, "an env", address, "is the EnvServer running?")
        })?;
        core.request_timeout = normalize_timeout(options.request_timeout);
        Ok(core)
    }

    async fn dial(
        address: ConnectAddress,
        credentials: &rlmesh_grpc::Credentials,
        declared: Option<&str>,
    ) -> Result<Self> {
        let (inner, env_contract, spaces, num_envs, session_offer, selected) = match address {
            #[cfg(feature = "lockstep")]
            ConnectAddress::Lockstep(address) => {
                let client = crate::lockstep::connect(&address, Default::default())
                    .await
                    .map_err(lockstep_error)?;
                let info = client.info();
                let (offer, selected, _) = crate::editions::negotiate_env(info, declared)
                    .map_err(|why| lockstep_error(crate::lockstep::GymError::Config(why)))?;
                let env_contract = info.contract.clone();
                let spaces = spaces_of(&env_contract)?;
                let num_envs = (env_contract.num_envs as usize).max(1);
                let link = Link::Lockstep {
                    client,
                    address,
                    edition: Arc::new(std::sync::Mutex::new(selected.clone())),
                    episodes: Episodes::new(num_envs),
                };
                (link, env_contract, spaces, num_envs, offer, selected)
            }
            #[cfg(not(feature = "lockstep"))]
            ConnectAddress::Lockstep(address) => {
                return Err(Error::invalid_argument(format!(
                    "{address}: this build has no v2 lockstep transport (the `lockstep` feature)"
                )));
            }
            address => {
                let mut client =
                    rlmesh_grpc::EnvClient::connect_with(&address.to_string(), credentials)
                        .await
                        .map_err(Error::from_grpc)?;
                client.declare_workflow_edition(declared.map(str::to_string));
                let handshake = client.handshake().await.map_err(Error::from_grpc)?;
                let session_offer = handshake.session_offer();
                // The handshake refuses a contract without spaces; one that
                // does not decode is a malformed handshake too.
                let env_contract = rlmesh_grpc::wire::env_contract_from_proto(
                    handshake.env_contract,
                )
                .map_err(|error| {
                    Error::Protocol(format!("invalid spaces spec from remote env: {error}"))
                })?;
                let spaces = spaces_of(&env_contract)?;
                let link = Link::Grpc {
                    client: Box::new(client),
                    observation_space: Arc::clone(&spaces.0),
                    action_space: Arc::clone(&spaces.1),
                    num_envs: handshake.num_envs,
                };
                (
                    link,
                    env_contract,
                    spaces,
                    handshake.num_envs,
                    session_offer,
                    handshake.selected_workflow_edition,
                )
            }
        };
        Ok(Self {
            inner,
            env_contract,
            observation_space: spaces.0,
            action_space: spaces.1,
            num_envs,
            session_offer,
            env_id: crate::mint_id(),
            episode_ids: vec![String::new(); num_envs],
            selected_workflow_edition: selected,
            request_timeout: None,
            closed: false,
            warned_shadowed_info_key: false,
            runtime_info: true,
            name: "the env client",
        })
    }

    /// Refuse an endpoint with more than one env (`RemoteEnv`'s arity).
    pub fn require_single(&self) -> Result<()> {
        if self.num_envs != 1 {
            return Err(Error::invalid_argument(format!(
                "Endpoint {:?} serves {} environments. Use RemoteVectorEnv instead.",
                self.address(),
                self.num_envs
            )));
        }
        Ok(())
    }

    /// Refuse an endpoint with fewer than two envs (`RemoteVectorEnv`'s
    /// arity).
    pub fn require_vector(&self) -> Result<()> {
        if self.num_envs < 2 {
            return Err(Error::invalid_argument(
                "RemoteVectorEnv requires an endpoint with num_envs >= 2",
            ));
        }
        Ok(())
    }

    /// The canonical connect string of the dialed endpoint.
    pub fn address(&self) -> &str {
        self.inner.address()
    }

    /// This connection's container id (UUIDv7).
    pub fn env_id(&self) -> &str {
        &self.env_id
    }

    /// The handshake's wire contract.
    pub fn env_contract(&self) -> &spaces::EnvContract {
        &self.env_contract
    }

    /// The wire observation space.
    pub fn observation_space(&self) -> &spaces::SpaceSpec {
        &self.observation_space
    }

    /// The wire action space.
    pub fn action_space(&self) -> &spaces::SpaceSpec {
        &self.action_space
    }

    /// Lanes the endpoint steps in lockstep.
    pub fn num_envs(&self) -> usize {
        self.num_envs
    }

    /// The request bound every call uses unless given its own.
    pub fn request_timeout(&self) -> Option<Duration> {
        self.request_timeout
    }

    /// Whether `close` (or an accepted `shutdown`) ran.
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// The edition the handshake selected, or the last pin.
    pub fn selected_workflow_edition(&self) -> &str {
        &self.selected_workflow_edition
    }

    /// Pin the env to the session edition a model leg settled on (the
    /// three-way floor), replacing the handshake selection as the first Join
    /// message.
    pub fn pin_workflow_edition(&mut self, edition: &str) {
        self.selected_workflow_edition = edition.trim().to_string();
        match &mut self.inner {
            Link::Grpc { client, .. } => {
                client.pin_workflow_edition(self.selected_workflow_edition.clone());
            }
            #[cfg(feature = "lockstep")]
            Link::Lockstep { edition, .. } => {
                *edition
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) =
                    self.selected_workflow_edition.clone();
            }
        }
    }

    /// The env's negotiation offer captured at handshake.
    pub fn session_offer(&self) -> &rlmesh_proto::SessionOffer {
        &self.session_offer
    }

    /// A shared handle on the selected edition, reading every pin.
    pub fn edition_cell(&self) -> EditionCell {
        EditionCell(match &self.inner {
            Link::Grpc { client, .. } => Edition::Grpc(client.pinned_edition()),
            #[cfg(feature = "lockstep")]
            Link::Lockstep { edition, .. } => Edition::Local(Arc::clone(edition)),
        })
    }

    /// Leave `info` exactly as the env produced it, without the
    /// runtime-reserved keys (a Gymnasium face's `info`).
    pub fn without_runtime_info(&mut self) {
        self.runtime_info = false;
    }

    /// Name the client class closed-handle errors report.
    pub fn set_name(&mut self, name: &'static str) {
        self.name = name;
    }

    fn check_open(&self) -> Result<()> {
        if self.closed {
            return Err(Error::Runtime(format!("{} is closed", self.name)));
        }
        Ok(())
    }

    /// Set a runtime-reserved info key unless the env reported its own; the
    /// first collision per client is logged.
    fn insert_runtime_info(&mut self, info: &mut MetaMap, key: &'static str, value: MetaValue) {
        if !self.runtime_info {
            return;
        }
        if !info.contains_key(key) {
            info.insert(key.to_string(), value);
            return;
        }
        if !self.warned_shadowed_info_key {
            self.warned_shadowed_info_key = true;
            tracing::warn!(
                address = %self.address(),
                "env info key {key:?} shadows the runtime-reserved info key of the same \
                 name; keeping the env's value (reserved keys: {RUNTIME_INFO_KEYS:?}). \
                 Rename the env's key to read the runtime's value. Logged once per client."
            );
        }
    }

    fn episode_ids_value(&self) -> MetaValue {
        MetaValue::List(
            self.episode_ids
                .iter()
                .cloned()
                .map(MetaValue::String)
                .collect(),
        )
    }

    /// The bound for one call: the request's own `timeout_ms` when set,
    /// else the client's request timeout.
    fn call_timeout(&self, timeout_ms: i64) -> Option<Duration> {
        match u64::try_from(timeout_ms) {
            Ok(ms) if ms > 0 => Some(Duration::from_millis(ms)),
            _ => self.request_timeout,
        }
    }

    /// Reset lanes (`env_indices` empty: all of them) and return the wire
    /// result; `info` carries `episode_ids`. A nonzero `timeout_ms` bounds
    /// this call instead of the client's request timeout.
    pub async fn reset(&mut self, req: VectorResetRequest) -> Result<VectorResetResult> {
        self.check_open()?;
        let timeout = self.call_timeout(req.timeout_ms);
        let env_indices = req
            .env_indices
            .into_iter()
            .map(|index| {
                u64::try_from(index)
                    .map_err(|_| {
                        Error::invalid_argument(format!(
                            "partial reset: env_index {index} must be a non-negative lane offset"
                        ))
                    })
                    .and_then(wire_lane)
            })
            .collect::<Result<Vec<u32>>>()?;
        self.reset_lanes(req.seeds, req.options, env_indices, timeout)
            .await
    }

    async fn reset_lanes(
        &mut self,
        seeds: Vec<i64>,
        options: Option<MetaMap>,
        env_indices: Vec<u32>,
        timeout: Option<Duration>,
    ) -> Result<VectorResetResult> {
        // Mint authoritative episode ids for the lanes this reset starts and
        // push them down (the env adopts them; it never mints).
        let pushed_ids: Vec<String> = if env_indices.is_empty() {
            let ids: Vec<String> = (0..self.num_envs).map(|_| crate::mint_id()).collect();
            self.episode_ids = ids.clone();
            ids
        } else {
            let ids: Vec<String> = env_indices.iter().map(|_| crate::mint_id()).collect();
            for (lane, id) in env_indices.iter().zip(&ids) {
                if let Some(slot) = self.episode_ids.get_mut(*lane as usize) {
                    *slot = id.clone();
                }
            }
            ids
        };
        let inner = &mut self.inner;
        let (observations, info) = bounded(
            timeout,
            inner.reset(env_indices, seeds, options, pushed_ids, timeout),
        )
        .await?;
        validate_count(&observations, self.num_envs, "observations")
            .map_err(|error| Error::Environment(EnvironmentError::from_wire(error)))?;
        let mut info = info.unwrap_or_default();
        let ids = self.episode_ids_value();
        self.insert_runtime_info(&mut info, "episode_ids", ids);
        Ok(VectorResetResult {
            observations,
            info: Some(info),
            episode_ids: self.episode_ids.clone(),
        })
    }

    /// Apply one action per lane and return the wire transition; `info`
    /// carries `episode_ids` and `completed_episodes`.
    pub async fn step(&mut self, req: VectorStepRequest) -> Result<VectorStepResult> {
        self.check_open()?;
        let timeout = self.call_timeout(req.timeout_ms);
        validate_action_count(&req.actions, self.num_envs)
            .map_err(|error| Error::Environment(EnvironmentError::from_wire(error)))?;
        // Push the authoritative per-lane ids (the env adopts a rolled id on a
        // NEXT_STEP autoreset).
        let pushed_ids = self.episode_ids.clone();
        let inner = &mut self.inner;
        let VectorStepResult {
            observations,
            rewards,
            terminated,
            truncated,
            info,
            completed_episodes,
            ..
        } = bounded(timeout, inner.step(&req.actions, pushed_ids, timeout)).await?;

        let env_count = self.num_envs;
        for check in [
            validate_count(&observations, env_count, "observations"),
            validate_count(&terminated, env_count, "terminated values"),
            validate_count(&truncated, env_count, "truncated values"),
            validate_count(&rewards, env_count, "rewards values"),
        ] {
            check.map_err(|error| Error::Environment(EnvironmentError::from_wire(error)))?;
        }

        let mut info = info.unwrap_or_default();
        let ids = self.episode_ids_value();
        self.insert_runtime_info(&mut info, "episode_ids", ids);
        // A single env reports 1 on the step that ends its episode; a vector
        // env the count of lanes whose episode ended.
        let completed = if self.num_envs == 1 {
            usize::from(terminated[0] || truncated[0]).max(completed_episodes.len())
        } else {
            completed_episodes.len()
        };
        self.insert_runtime_info(
            &mut info,
            "completed_episodes",
            MetaValue::Int(i64::try_from(completed).unwrap_or(i64::MAX)),
        );

        // Surface the ids as of this step (the terminal obs of a completing
        // lane still belongs to the episode that just ended).
        let result = VectorStepResult {
            observations,
            rewards,
            terminated,
            truncated,
            info: Some(info),
            episode_ids: self.episode_ids.clone(),
            completed_episodes,
        };
        // Under NEXT_STEP the env autoresets each completed lane next step and
        // adopts the fresh id pushed then, and under SAME_STEP the lane's next
        // step is already the new episode's, so roll it now. Under DISABLED
        // the lane stays done until an explicit reset re-mints.
        if self.env_contract.autoreset_mode != spaces::types::AutoresetMode::Disabled {
            for completed in &result.completed_episodes {
                if let Some(slot) = self.episode_ids.get_mut(completed.env_index as usize) {
                    *slot = crate::mint_id();
                }
            }
        }
        Ok(result)
    }

    /// A render-only view of this already-connected channel. Context lifetime
    /// and operation admission are enforced by the owning session context.
    pub(crate) fn render_callback(&self) -> Arc<dyn Fn() -> Result<Value> + Send + Sync> {
        let inner = self.inner.clone();
        let timeout = self.request_timeout;
        Arc::new(move || {
            crate::run::drive::refuse_in_async_context("RunContext render")?;
            let mut inner = inner.clone();
            crate::run::drive::run_sync(async move {
                match bounded(timeout, inner.render(0, timeout)).await? {
                    Some(png) => decode_render_png(&png),
                    None => Ok(Value::None),
                }
            })
        })
    }
    /// Request a render frame of lane `env_index`: the PNG bytes, or `None`
    /// when the env has no frame.
    pub async fn render(&mut self, env_index: u64) -> Result<Option<Vec<u8>>> {
        self.check_open()?;
        let lane = wire_lane(env_index)?;
        let timeout = self.request_timeout;
        let inner = &mut self.inner;
        bounded(timeout, inner.render(lane, timeout)).await
    }

    /// Close this client's session (idempotent). Bounded by the request
    /// timeout, else a 5 s grace; on expiry or a lost connection it detaches
    /// locally and succeeds. Other errors are returned, and the client is
    /// closed either way.
    pub async fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        let bound = self.request_timeout.unwrap_or(CLOSE_GRACE);
        #[cfg(feature = "lockstep")]
        if let Link::Lockstep { client, .. } = &self.inner {
            // Releases the lease, so the robot re-arms for the next driver.
            if tokio::time::timeout(bound, client.close()).await.is_err() {
                tracing::debug!("lockstep detach did not complete within {bound:?}");
            }
            return Ok(());
        }
        let Some(inner) = self.inner.grpc() else {
            return Ok(());
        };
        let outcome = tokio::time::timeout(bound, inner.close()).await;
        match outcome {
            Ok(Ok(_)) => Ok(()),
            Err(_) => {
                tracing::debug!("close did not complete within {bound:?}; detaching locally");
                inner.detach();
                Ok(())
            }
            Ok(Err(err)) => {
                let err = Error::from_grpc(err);
                inner.detach();
                match err {
                    Error::Connection(_) | Error::Timeout(_) => {
                        tracing::debug!("close did not complete ({err}); detaching locally");
                        Ok(())
                    }
                    other => Err(other),
                }
            }
        }
    }

    /// Register owned cleanup before it is queued by a nonblocking Drop.
    pub(crate) fn prepare_close(&self) {
        match &self.inner {
            Link::Grpc { client, .. } => client.prepare_close(),
            #[cfg(feature = "lockstep")]
            Link::Lockstep { .. } => {}
        }
    }

    /// Ask the server to shut itself down; an accepted request closes this
    /// client locally.
    pub async fn shutdown<'r>(&mut self, reason: impl Into<Option<&'r str>>) -> Result<bool> {
        self.check_open()?;
        let timeout = self.request_timeout;
        let reason = reason.into().unwrap_or("owner shutdown").to_string();
        let Some(inner) = self.inner.grpc() else {
            // A lockstep robot is its server's to stop; a driver cannot.
            return Ok(false);
        };
        let response = bounded(timeout, async {
            inner.shutdown(reason).await.map_err(Error::from_grpc)
        })
        .await?;
        if response.accepted {
            self.closed = true;
        }
        Ok(response.accepted)
    }

    /// Reset as [`reset`](Self::reset) and decode to shared values: one
    /// value per lane and the info.
    pub(crate) async fn reset_values(
        &mut self,
        seeds: Vec<i64>,
        options: Option<&InfoDict>,
        env_indices: Vec<u32>,
    ) -> Result<(Vec<Value>, InfoDict)> {
        self.check_open()?;
        let options = options.map(meta_codec::metadata_to_wire).transpose()?;
        let timeout = self.request_timeout;
        let result = self
            .reset_lanes(seeds, options, env_indices, timeout)
            .await?;
        let lanes = result
            .observations
            .into_iter()
            .map(|lane| space_codec::value_from_space(&self.observation_space, lane))
            .collect::<Result<Vec<_>>>()?;
        Ok((
            lanes,
            meta_codec::metadata_from_wire(result.info.unwrap_or_default()),
        ))
    }

    /// Step as [`step`](Self::step) with one shared value per lane.
    #[allow(clippy::type_complexity)]
    pub(crate) async fn step_values(
        &mut self,
        actions: Vec<Value>,
    ) -> Result<(Vec<Value>, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)> {
        let actions = actions
            .iter()
            .map(|action| space_codec::value_to_space(&self.action_space, action))
            .collect::<Result<Vec<_>>>()?;
        let result = self
            .step(VectorStepRequest {
                actions,
                timeout_ms: 0,
            })
            .await?;
        let lanes = result
            .observations
            .into_iter()
            .map(|lane| space_codec::value_from_space(&self.observation_space, lane))
            .collect::<Result<Vec<_>>>()?;
        Ok((
            lanes,
            result.rewards,
            result.terminated,
            result.truncated,
            meta_codec::metadata_from_wire(result.info.unwrap_or_default()),
        ))
    }

    /// The transport client, for tests that reach below the facade.
    #[cfg(test)]
    pub(crate) fn grpc(&mut self) -> &mut rlmesh_grpc::EnvClient {
        self.inner.grpc().expect("a gRPC client")
    }

    /// Render lane `env_index` decoded to a uint8 frame.
    pub(crate) async fn render_value(&mut self, env_index: u64) -> Result<Option<Value>> {
        match self.render(env_index).await? {
            Some(png) => decode_render_png(&png).map(Some),
            None => Ok(None),
        }
    }
}

/// Decode a wire PNG frame to `(shape, raw uint8 bytes)`: `[H, W]`, `[H, W,
/// 3]` or `[H, W, 4]`, other formats as RGBA. An undecodable frame is the
/// ProtocolException class.
pub(crate) fn decode_render_frame(png: &[u8]) -> Result<(Vec<usize>, Vec<u8>)> {
    use image::{DynamicImage, ImageFormat};

    let image = image::load_from_memory_with_format(png, ImageFormat::Png)
        .map_err(|err| Error::Protocol(format!("render frame does not decode: {err}")))?;
    let (width, height) = (image.width() as usize, image.height() as usize);
    let (channels, raw) = match image {
        DynamicImage::ImageLuma8(buf) => (1usize, buf.into_raw()),
        DynamicImage::ImageRgb8(buf) => (3, buf.into_raw()),
        DynamicImage::ImageRgba8(buf) => (4, buf.into_raw()),
        other => (4, other.to_rgba8().into_raw()),
    };
    let shape = if channels == 1 {
        vec![height, width]
    } else {
        vec![height, width, channels]
    };
    Ok((shape, raw))
}

fn decode_render_png(png: &[u8]) -> Result<Value> {
    use crate::ext::TensorExt;
    let (shape, raw) = decode_render_frame(png)?;
    Ok(Value::Tensor(crate::Tensor::from_vec(
        raw,
        &shape,
        crate::DType::Uint8,
    )?))
}

/// Resolve the client's own declaration (Core's chain: the call, then the
/// process and project surfaces) and dial with `options`.
async fn connect_core(options: RemoteEnvOptions) -> Result<EnvClientCore> {
    let address = normalize_connect_address(
        options.address.as_deref(),
        options.host.as_deref(),
        options.port,
        options.path.as_deref(),
        options.transport,
    )?;
    ConnectAddress::parse(&address)?;
    let auth = crate::credentials::resolve(
        &address,
        options.api_key.as_deref(),
        options.tls_ca_file.as_deref(),
        true,
    )?;
    let declared = crate::editions::resolve_native(
        crate::editions::EditionSources {
            call: options.workflow_edition.as_deref(),
            option: None,
            declared: None,
        },
        false,
    )?;
    EnvClientCore::connect_with(
        &address,
        CoreConnect {
            token: auth.token(),
            trust: auth.trust().clone(),
            declared: declared.as_deref(),
            connect_timeout: options.connect_timeout,
            request_timeout: options.request_timeout,
        },
    )
    .await
}

/// Fail `future` with the cancellation class when `token` fires.
async fn cancellable<T>(
    subscriptions: Vec<crate::ext::CancelSubscription>,
    future: impl std::future::Future<Output = Result<T>>,
) -> Result<T> {
    let driver = CancellationToken::new();
    let _watch = CancelWatch::from_subscriptions(subscriptions, driver.clone())?;
    let result = tokio::select! {
        biased;
        () = driver.cancelled() => Err(Error::Cancelled),
        out = future => out,
    };
    _watch.finish(result)
}

/// Close an unclosed core at scope exit: bounded on this thread outside a
/// runtime, on a detached library task inside one.
fn teardown(mut core: Box<EnvClientCore>) {
    if core.is_closed() {
        return;
    }
    if in_async_context() {
        core.prepare_close();
        runtime().spawn(async move {
            if let Err(err) = core.close().await {
                tracing::warn!("closing a dropped remote env client failed: {err}");
            }
        });
    } else if let Err(err) = block_on_library(core.close()) {
        tracing::warn!("closing a dropped remote env client failed: {err}");
    }
}

/// What both clients keep outside the per-call core, so accessors never wait
/// on a call.
struct Shared {
    address: String,
    env_id: String,
    contract: EnvContract,
    metadata: Metadata,
    observation_space: Space,
    action_space: Space,
    num_envs: u64,
}

impl Shared {
    fn of(core: &EnvClientCore) -> Result<Shared> {
        let contract = EnvContract::from_wire(core.env_contract())
            .map_err(|err| Error::Protocol(err.to_string()))?;
        Ok(Shared {
            address: core.address().to_string(),
            env_id: core.env_id().to_string(),
            metadata: contract.metadata().cloned().unwrap_or_default(),
            observation_space: Space::new(contract.observation_space().clone()),
            action_space: Space::new(contract.action_space().clone()),
            num_envs: core.num_envs() as u64,
            contract,
        })
    }
}

/// The client half a session can borrow: its core moves into the session for
/// the borrow and comes back when the session ends.
pub(crate) trait Lendable {
    /// The handle's canonical connect string.
    fn remote_address(&self) -> &str;
    /// Move the core out for a session's borrow.
    fn lend(&mut self) -> Result<Box<EnvClientCore>>;
    /// Return a lent core.
    fn give_back(&mut self, core: Box<EnvClientCore>);
}

macro_rules! client_common {
    ($Client:ident, $name:literal) => {
        impl $Client {
            fn core_mut(&mut self) -> Result<&mut EnvClientCore> {
                self.core.as_deref_mut().ok_or_else(|| {
                    Error::Runtime(concat!($name, " is in use by a session").to_string())
                })
            }

            /// Run one I/O call on the core, cancellable through the handle's
            /// token.
            async fn io<T>(
                &mut self,
                op: impl for<'c> FnOnce(
                    &'c mut EnvClientCore,
                ) -> std::pin::Pin<
                    Box<dyn std::future::Future<Output = Result<T>> + Send + 'c>,
                >,
            ) -> Result<T> {
                let subscriptions = self.admit(None)?;
                self.io_subscribed(subscriptions, op).await
            }

            fn admit(
                &mut self,
                cancel: Option<&crate::ext::CancelToken>,
            ) -> Result<Vec<crate::ext::CancelSubscription>> {
                let core = self.core_mut()?;
                core.check_open()?;
                let subscriptions: Vec<_> = std::iter::once(&self.cancel)
                    .chain(cancel)
                    .map(crate::ext::CancelToken::subscribe)
                    .collect();
                if subscriptions
                    .iter()
                    .any(crate::ext::CancelSubscription::fired)
                {
                    return Err(Error::Cancelled);
                }
                Ok(subscriptions)
            }

            async fn io_subscribed<T>(
                &mut self,
                subscriptions: Vec<crate::ext::CancelSubscription>,
                op: impl for<'c> FnOnce(
                    &'c mut EnvClientCore,
                ) -> std::pin::Pin<
                    Box<dyn std::future::Future<Output = Result<T>> + Send + 'c>,
                >,
            ) -> Result<T> {
                let core = self.core_mut()?;
                cancellable(subscriptions, op(core)).await
            }

            /// The canonical connect string of the dialed endpoint.
            pub fn address(&self) -> &str {
                &self.shared.address
            }

            /// This connection's container id (UUIDv7), distinct from the
            /// human env name (`env_contract().id()`).
            pub fn env_id(&self) -> &str {
                &self.shared.env_id
            }

            /// The workflow edition this connection negotiated, or the last
            /// edition a session pinned it to.
            pub fn selected_workflow_edition(&self) -> &str {
                match &self.core {
                    Some(core) => core.selected_workflow_edition(),
                    None => &self.lent_edition,
                }
            }

            /// The contract the server reported at handshake.
            pub fn env_contract(&self) -> &EnvContract {
                &self.shared.contract
            }

            /// The env's render mode, if any.
            pub fn render_mode(&self) -> Option<&str> {
                self.shared.contract.render_mode()
            }

            /// The endpoint's metadata (empty when it reports none).
            pub fn metadata(&self) -> &Metadata {
                &self.shared.metadata
            }

            /// The observation space spec reported by the endpoint.
            pub fn observation_space_spec(&self) -> &SpaceSpec {
                self.shared.contract.observation_space()
            }

            /// The action space spec reported by the endpoint.
            pub fn action_space_spec(&self) -> &SpaceSpec {
                self.shared.contract.action_space()
            }

            /// This handle's cancel token: firing it fails the call in
            /// flight, or the next call.
            pub fn cancel_handle(&self) -> crate::ext::CancelToken {
                self.cancel.clone()
            }

            /// Render lane `env_index` (`None` = 0) as a uint8 frame, or
            /// `None` when the env has no frame.
            pub fn render(&mut self, env_index: Option<u64>) -> Result<Option<Value>> {
                refuse_in_async_context(concat!($name, "::render"))?;
                block_on_library(self.render_in(env_index))
            }

            /// The async twin of [`render`](Self::render).
            pub async fn render_async(&mut self, env_index: Option<u64>) -> Result<Option<Value>> {
                in_library(self.render_in(env_index)).await
            }

            async fn render_in(&mut self, env_index: Option<u64>) -> Result<Option<Value>> {
                let index = env_index.unwrap_or(0);
                self.io(move |core| Box::pin(core.render_value(index)))
                    .await
            }

            /// Close this client's session (idempotent); the server keeps
            /// running. Errors other than a lost connection or an expired
            /// bound are returned.
            pub fn close(&mut self) -> Result<()> {
                refuse_in_async_context(concat!($name, "::close"))?;
                block_on_library(self.close_in())
            }

            /// The async twin of [`close`](Self::close).
            pub async fn close_async(&mut self) -> Result<()> {
                in_library(self.close_in()).await
            }

            async fn close_in(&mut self) -> Result<()> {
                self.core_mut()?.close().await
            }

            /// Ask the server to shut itself down (`None` = `"owner
            /// shutdown"`); true when it accepted, which closes this client.
            pub fn shutdown(&mut self, reason: Option<&str>) -> Result<bool> {
                refuse_in_async_context(concat!($name, "::shutdown"))?;
                block_on_library(self.shutdown_in(reason))
            }

            /// The async twin of [`shutdown`](Self::shutdown).
            pub async fn shutdown_async(&mut self, reason: Option<&str>) -> Result<bool> {
                in_library(self.shutdown_in(reason)).await
            }

            async fn shutdown_in(&mut self, reason: Option<&str>) -> Result<bool> {
                let reason = reason.map(str::to_string);
                self.io(move |core| Box::pin(async move { core.shutdown(reason.as_deref()).await }))
                    .await
            }

            pub(crate) fn core_ref(&self) -> Option<&EnvClientCore> {
                self.core.as_deref()
            }
        }

        impl Lendable for $Client {
            fn remote_address(&self) -> &str {
                &self.shared.address
            }

            fn lend(&mut self) -> Result<Box<EnvClientCore>> {
                let core = self.core.take().ok_or_else(|| {
                    Error::Runtime(concat!($name, " is in use by a session").to_string())
                })?;
                self.lent_edition = core.selected_workflow_edition().to_string();
                Ok(core)
            }

            fn give_back(&mut self, core: Box<EnvClientCore>) {
                self.core = Some(core);
            }
        }

        impl Drop for $Client {
            fn drop(&mut self) {
                if let Some(core) = self.core.take() {
                    teardown(core);
                }
            }
        }

        impl std::fmt::Debug for $Client {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct($name)
                    .field("address", &self.shared.address)
                    .field("env_id", &self.shared.env_id)
                    .finish_non_exhaustive()
            }
        }
    };
}

/// A client for one remote env.
///
/// Construction dials and handshakes; the env's exclusive session slot is
/// taken by the first `reset`, `step` or `render`. Calls block (each has an
/// `_async` twin); `&mut self` serializes them. Dropping an unclosed client
/// closes it with `close`'s bound.
pub struct RemoteEnv {
    core: Option<Box<EnvClientCore>>,
    shared: Shared,
    lent_edition: String,
    cancel: crate::ext::CancelToken,
}

client_common!(RemoteEnv, "RemoteEnv");

impl RemoteEnv {
    pub(crate) fn reset_cancel(
        &mut self,
        seed: Option<i64>,
        options: Option<&InfoDict>,
        cancel: Option<crate::ext::CancelToken>,
    ) -> Result<(Value, InfoDict)> {
        refuse_in_async_context("RemoteEnv::reset")?;
        let subscriptions = self.admit(cancel.as_ref())?;
        block_on_library(self.reset_in(seed, options, Some(subscriptions)))
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn step_cancel(
        &mut self,
        action: Value,
        cancel: Option<crate::ext::CancelToken>,
    ) -> Result<(Value, f64, bool, bool, InfoDict)> {
        refuse_in_async_context("RemoteEnv::step")?;
        let subscriptions = self.admit(cancel.as_ref())?;
        block_on_library(self.step_in(action, Some(subscriptions)))
    }

    /// Dial and handshake. Invalid options are the ValueError class, a dial
    /// failure the ConnectionError class, a refused handshake the
    /// ProtocolException class; an endpoint with more than one env is
    /// refused.
    pub fn new(options: RemoteEnvOptions) -> Result<RemoteEnv> {
        refuse_in_async_context("RemoteEnv::new")?;
        block_on_library(Self::new_in(options))
    }

    /// The async twin of [`new`](Self::new).
    pub async fn new_async(options: RemoteEnvOptions) -> Result<RemoteEnv> {
        in_library(Self::new_in(options)).await
    }

    async fn new_in(options: RemoteEnvOptions) -> Result<RemoteEnv> {
        let mut core = connect_core(options).await?;
        core.require_single()?;
        core.set_name("RemoteEnv");
        Ok(RemoteEnv {
            shared: Shared::of(&core)?,
            lent_edition: String::new(),
            core: Some(Box::new(core)),
            cancel: crate::ext::CancelToken::new(),
        })
    }

    /// The observation space (built once).
    pub fn observation_space(&self) -> &Space {
        &self.shared.observation_space
    }

    /// The action space (built once).
    pub fn action_space(&self) -> &Space {
        &self.shared.action_space
    }

    /// Start an episode: `(observation, info)`. `info` carries the
    /// runtime-reserved `episode_ids`.
    pub fn reset(
        &mut self,
        seed: Option<i64>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        refuse_in_async_context("RemoteEnv::reset")?;
        block_on_library(self.reset_in(seed, options, None))
    }

    /// The async twin of [`reset`](Self::reset).
    pub async fn reset_async(
        &mut self,
        seed: Option<i64>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        in_library(self.reset_in(seed, options, None)).await
    }

    async fn reset_in(
        &mut self,
        seed: Option<i64>,
        options: Option<&InfoDict>,
        subscriptions: Option<Vec<crate::ext::CancelSubscription>>,
    ) -> Result<(Value, InfoDict)> {
        let subscriptions = match subscriptions {
            Some(s) => s,
            None => self.admit(None)?,
        };
        let options = options.cloned();
        self.io_subscribed(subscriptions, move |core| {
            Box::pin(async move {
                let (lanes, info) = core
                    .reset_values(seed.into_iter().collect(), options.as_ref(), Vec::new())
                    .await?;
                Ok((lanes.into_iter().next().unwrap_or(Value::None), info))
            })
        })
        .await
    }

    /// Apply one action: `(observation, reward, terminated, truncated,
    /// info)`; `info` carries `episode_ids` and `completed_episodes`.
    #[allow(clippy::type_complexity)]
    pub fn step(&mut self, action: Value) -> Result<(Value, f64, bool, bool, InfoDict)> {
        refuse_in_async_context("RemoteEnv::step")?;
        block_on_library(self.step_in(action, None))
    }

    /// The async twin of [`step`](Self::step).
    #[allow(clippy::type_complexity)]
    pub async fn step_async(
        &mut self,
        action: Value,
    ) -> Result<(Value, f64, bool, bool, InfoDict)> {
        in_library(self.step_in(action, None)).await
    }

    #[allow(clippy::type_complexity)]
    async fn step_in(
        &mut self,
        action: Value,
        subscriptions: Option<Vec<crate::ext::CancelSubscription>>,
    ) -> Result<(Value, f64, bool, bool, InfoDict)> {
        let subscriptions = match subscriptions {
            Some(s) => s,
            None => self.admit(None)?,
        };
        self.io_subscribed(subscriptions, move |core| {
            Box::pin(async move {
                let (lanes, rewards, terminated, truncated, info) =
                    core.step_values(vec![action]).await?;
                Ok((
                    lanes.into_iter().next().unwrap_or(Value::None),
                    rewards.first().copied().unwrap_or_default(),
                    terminated.first().copied().unwrap_or_default(),
                    truncated.first().copied().unwrap_or_default(),
                    info,
                ))
            })
        })
        .await
    }
}

/// A client for a remote vector env (two or more lanes stepped in lockstep):
/// A remote vector environment. Observations and actions are one batched
/// [`Value`]; rewards and flags are per-lane vectors.
pub struct RemoteVectorEnv {
    core: Option<Box<EnvClientCore>>,
    shared: Shared,
    lent_edition: String,
    cancel: crate::ext::CancelToken,
}

client_common!(RemoteVectorEnv, "RemoteVectorEnv");

impl RemoteVectorEnv {
    pub(crate) fn reset_cancel(
        &mut self,
        seed: Option<VectorSeed>,
        options: Option<&InfoDict>,
        cancel: Option<crate::ext::CancelToken>,
    ) -> Result<(Value, InfoDict)> {
        refuse_in_async_context("RemoteVectorEnv::reset")?;
        let subscriptions = self.admit(cancel.as_ref())?;
        block_on_library(self.reset_in(seed, options, Vec::new(), Some(subscriptions)))
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn step_cancel(
        &mut self,
        action: Value,
        cancel: Option<crate::ext::CancelToken>,
    ) -> Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)> {
        refuse_in_async_context("RemoteVectorEnv::step")?;
        let lanes = self.unstack_actions(action)?;
        let subscriptions = self.admit(cancel.as_ref())?;
        block_on_library(self.step_lanes_in(lanes, subscriptions))
    }

    /// Dial and handshake. An endpoint with fewer than two envs is refused
    /// (the ValueError class).
    pub fn new(options: RemoteEnvOptions) -> Result<RemoteVectorEnv> {
        refuse_in_async_context("RemoteVectorEnv::new")?;
        block_on_library(Self::new_in(options))
    }

    /// The async twin of [`new`](Self::new).
    pub async fn new_async(options: RemoteEnvOptions) -> Result<RemoteVectorEnv> {
        in_library(Self::new_in(options)).await
    }

    async fn new_in(options: RemoteEnvOptions) -> Result<RemoteVectorEnv> {
        let mut core = connect_core(options).await?;
        core.require_vector()?;
        core.set_name("RemoteVectorEnv");
        Ok(RemoteVectorEnv {
            shared: Shared::of(&core)?,
            lent_edition: String::new(),
            core: Some(Box::new(core)),
            cancel: crate::ext::CancelToken::new(),
        })
    }

    /// Lanes the endpoint steps in lockstep.
    pub fn num_envs(&self) -> u64 {
        self.shared.num_envs
    }

    /// One lane's observation space (built once).
    pub fn single_observation_space(&self) -> &Space {
        &self.shared.observation_space
    }

    /// One lane's action space (built once).
    pub fn single_action_space(&self) -> &Space {
        &self.shared.action_space
    }

    /// The same object as [`single_observation_space`](Self::single_observation_space).
    pub fn observation_space(&self) -> &Space {
        &self.shared.observation_space
    }

    /// The same object as [`single_action_space`](Self::single_action_space).
    pub fn action_space(&self) -> &Space {
        &self.shared.action_space
    }

    /// Reset every lane: the batched observation and the info (with one
    /// `episode_ids` entry per lane). An int seed is sent as a one-element
    /// list.
    pub fn reset(
        &mut self,
        seed: Option<VectorSeed>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        refuse_in_async_context("RemoteVectorEnv::reset")?;
        block_on_library(self.reset_in(seed, options, Vec::new(), None))
    }

    /// The async twin of [`reset`](Self::reset).
    pub async fn reset_async(
        &mut self,
        seed: Option<VectorSeed>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        in_library(self.reset_in(seed, options, Vec::new(), None)).await
    }

    pub(crate) async fn reset_in(
        &mut self,
        seed: Option<VectorSeed>,
        options: Option<&InfoDict>,
        env_indices: Vec<u32>,
        subscriptions: Option<Vec<crate::ext::CancelSubscription>>,
    ) -> Result<(Value, InfoDict)> {
        let subscriptions = match subscriptions {
            Some(s) => s,
            None => self.admit(None)?,
        };
        let seeds = match seed {
            None => Vec::new(),
            Some(VectorSeed::One(seed)) => vec![seed],
            Some(VectorSeed::Each(seeds)) => seeds,
        };
        let options = options.cloned();
        let observation = self.shared.contract.observation_space().wire().clone();
        self.io_subscribed(subscriptions, move |core| {
            Box::pin(async move {
                let (lanes, info) = core
                    .reset_values(seeds, options.as_ref(), env_indices)
                    .await?;
                Ok((super::batch::stack_lanes(&observation, lanes)?, info))
            })
        })
        .await
    }

    /// Apply one batched action: the batched observation, per-lane rewards,
    /// terminated and truncated flags, and the info (`completed_episodes` is
    /// the count of lanes whose episode ended).
    #[allow(clippy::type_complexity)]
    pub fn step(
        &mut self,
        actions: Value,
    ) -> Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)> {
        refuse_in_async_context("RemoteVectorEnv::step")?;
        block_on_library(self.step_in(actions))
    }

    /// The async twin of [`step`](Self::step).
    #[allow(clippy::type_complexity)]
    pub async fn step_async(
        &mut self,
        actions: Value,
    ) -> Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)> {
        in_library(self.step_in(actions)).await
    }

    fn unstack_actions(&self, actions: Value) -> Result<Vec<Value>> {
        let action = self.shared.contract.action_space().wire().clone();
        super::batch::unstack_lanes(&action, actions, self.shared.num_envs as usize)
    }

    #[allow(clippy::type_complexity)]
    async fn step_in(
        &mut self,
        actions: Value,
    ) -> Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)> {
        let lanes = self.unstack_actions(actions)?;
        let subscriptions = self.admit(None)?;
        self.step_lanes_in(lanes, subscriptions).await
    }

    #[allow(clippy::type_complexity)]
    async fn step_lanes_in(
        &mut self,
        lanes: Vec<Value>,
        subscriptions: Vec<crate::ext::CancelSubscription>,
    ) -> Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)> {
        let observation = self.shared.contract.observation_space().wire().clone();
        self.io_subscribed(subscriptions, move |core| {
            Box::pin(async move {
                let (lanes, rewards, terminated, truncated, info) = core.step_values(lanes).await?;
                Ok((
                    super::batch::stack_lanes(&observation, lanes)?,
                    rewards,
                    terminated,
                    truncated,
                    info,
                ))
            })
        })
        .await
    }
}

/// `ext::reset_lanes`' body.
pub(crate) async fn reset_lanes_in(
    env: &mut RemoteVectorEnv,
    env_indices: &[u64],
    seed: Option<VectorSeed>,
    options: Option<&InfoDict>,
) -> Result<(Value, InfoDict)> {
    let lanes = env_indices
        .iter()
        .map(|index| wire_lane(*index))
        .collect::<Result<Vec<u32>>>()?;
    env.reset_in(seed, options, lanes, None).await
}

/// Refuse a token the authorization header cannot carry.
fn check_token(token: &str) -> Result<()> {
    if !token.is_empty()
        && let Some(problem) = rlmesh_grpc::helpers::auth::token_header_problem(token)
    {
        return Err(Error::invalid_argument(format!(
            "the endpoint token cannot be sent in the authorization header: it {problem}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_watch_keeps_a_firing_between_admission_and_first_poll() {
        let token = crate::ext::CancelToken::new();
        let subscriptions = vec![token.subscribe()];
        // Deterministic hook at the exact handoff used by io_subscribed.
        token.cancel();
        let result = block_on_library(cancellable(subscriptions, async {
            panic!("a fired admission must refuse before any I/O");
            #[allow(unreachable_code)]
            Ok(())
        }));
        assert_eq!(result, Err(Error::Cancelled));
        token.cancel();
        assert!(token.take());
    }

    #[test]
    fn connect_address_follows_python_endpoint_rules() {
        let ok = |a: Option<&str>, h: Option<&str>, p: Option<u16>, path: Option<&str>, t| {
            normalize_connect_address(a, h, p, path, t)
        };
        assert_eq!(
            ok(Some("tcp://x:1"), None, None, None, None).unwrap(),
            "tcp://x:1"
        );
        assert_eq!(
            ok(None, None, Some(5), None, None).unwrap(),
            "tcp://127.0.0.1:5"
        );
        assert_eq!(
            ok(None, Some("h"), Some(5), None, Some(Transport::Tcp)).unwrap(),
            "tcp://h:5"
        );
        for (args, text) in [
            (
                (Some(""), None, None, None, None),
                "address cannot be empty",
            ),
            ((Some("x"), None, Some(1), None, None), "cannot be combined"),
            (
                (None, None, None, None, None),
                "address or host/port is required",
            ),
            (
                (None, None, None, None, Some(Transport::Tcp)),
                "address or host/port is required",
            ),
            ((None, Some("h"), None, None, None), "port is required"),
            (
                (None, None, None, Some("/s"), Some(Transport::Tcp)),
                "transport='tcp'",
            ),
            (
                (None, Some("h"), None, Some("/s"), None),
                "path cannot be combined",
            ),
            (
                (None, None, None, None, Some(Transport::Unix)),
                "path is required",
            ),
        ] {
            let err = ok(args.0, args.1, args.2, args.3, args.4).unwrap_err();
            assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
            assert!(err.to_string().contains(text), "{err} lacks {text:?}");
        }
        #[cfg(unix)]
        assert_eq!(ok(None, None, None, Some("/s"), None).unwrap(), "unix:///s");
    }

    #[test]
    fn timeouts_read_zero_as_none_and_round_up_on_the_wire() {
        assert_eq!(normalize_timeout(Some(Duration::ZERO)), None);
        assert_eq!(timeout_from_secs(Some(0.0), "t").unwrap(), None);
        assert!(timeout_from_secs(Some(-1.0), "t").is_err());
        assert!(timeout_from_secs(Some(f64::NAN), "t").is_err());
        assert_eq!(wire_timeout_ms(None), 0);
        assert_eq!(wire_timeout_ms(Some(Duration::from_micros(1))), 1);
        assert_eq!(wire_timeout_ms(Some(Duration::from_micros(1500))), 2);
        assert_eq!(wire_timeout_ms(Some(Duration::from_secs(5))), 5000);
    }

    #[test]
    fn lane_indices_above_u32_are_value_errors() {
        assert_eq!(wire_lane(7).unwrap(), 7);
        let err = wire_lane(u64::from(u32::MAX) + 1).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn render_frames_decode_by_channel_count() {
        use image::{ColorType, ImageEncoder, codecs::png::PngEncoder};
        let png = |raw: &[u8], w, h, color: ColorType| {
            let mut out = Vec::new();
            PngEncoder::new(&mut out)
                .write_image(raw, w, h, color.into())
                .unwrap();
            out
        };
        let (shape, raw) =
            decode_render_frame(&png(&[1, 2, 3, 4, 5, 6], 2, 1, ColorType::Rgb8)).unwrap();
        assert_eq!((shape, raw), (vec![1, 2, 3], vec![1, 2, 3, 4, 5, 6]));
        let (shape, _) =
            decode_render_frame(&png(&[1, 2, 3, 4, 5, 6, 7, 8], 2, 1, ColorType::Rgba8)).unwrap();
        assert_eq!(shape, vec![1, 2, 4]);
        let (shape, _) = decode_render_frame(&png(&[1, 2, 3, 4], 2, 2, ColorType::L8)).unwrap();
        assert_eq!(shape, vec![2, 2]);
        let err = decode_render_frame(b"not a png").unwrap_err();
        assert!(matches!(err, Error::Protocol(_)), "{err:?}");
    }

    #[test]
    fn invalid_tokens_are_refused_before_any_dial() {
        assert!(check_token("").is_ok());
        assert!(check_token("s3cret").is_ok());
        let err = check_token("s3cret\n").unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
        assert!(!err.to_string().contains("s3cret"), "{err}");
    }

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn clients_are_send_and_sync() {
        assert_send_sync::<RemoteEnv>();
        assert_send_sync::<RemoteVectorEnv>();
        assert_send_sync::<EnvClientCore>();
    }
}
