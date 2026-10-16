//! The remote env clients: Python's `RemoteEnv` and `RemoteVectorEnv`, over
//! one any-arity core the bindings drive directly (`binding::EnvClientCore`).

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
/// timeout (RMT-4): past it the client detaches locally.
const CLOSE_GRACE: Duration = Duration::from_secs(5);

/// The runtime-reserved info keys (RMT-18).
const RUNTIME_INFO_KEYS: [&str; 2] = ["episode_ids", "completed_episodes"];

/// The keywords of Python's `RemoteEnv(address=None, *, ...)` and
/// `RemoteVectorEnv(...)` (RMT-3).
///
/// `address` excludes the helpers (`host`, `port`, `path`, `transport`);
/// without either the constructor fails with the ValueError class. A zero
/// timeout means no timeout, as `None` does (RMT-4).
#[derive(Clone, Debug, Default)]
pub struct RemoteEnvOptions {
    /// The endpoint (`tcp://host:port`, `host:port` or `unix:///path`).
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
}

/// Python's `normalize_connect_address`: one connect string from an address
/// or the helpers, every refusal the ValueError class.
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

/// RMT-4: zero means no timeout, as `None` does.
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

/// The wire `timeout_ms` for a request deadline (RMT-21): rounded up to
/// whole milliseconds, 0 for none.
fn wire_timeout_ms(timeout: Option<Duration>) -> u64 {
    timeout.map_or(0, |timeout| {
        u64::try_from(timeout.as_nanos().div_ceil(1_000_000)).unwrap_or(u64::MAX)
    })
}

/// Bound `future` by `timeout`: a reply handled at or after the deadline is
/// discarded and the call fails with the TimeoutError class (RMT-21).
async fn bounded<T>(
    timeout: Option<Duration>,
    future: impl std::future::Future<Output = Result<T>>,
) -> Result<T> {
    match timeout {
        Some(timeout) => tokio::time::timeout(timeout, future)
            .await
            .map_err(|_| Error::Timeout(timeout))?,
        None => future.await,
    }
}

/// A lane index for the wire (`uint32`): above `u32::MAX` is the ValueError
/// class, before any I/O (RMT-P11).
fn wire_lane(index: u64) -> Result<u32> {
    u32::try_from(index).map_err(|_| {
        Error::invalid_argument(format!(
            "env_index {index} is out of range: lane indices are at most {}",
            u32::MAX
        ))
    })
}

/// A shared read handle on a client's selected workflow edition: the
/// handshake selection until a pin replaces it (RMT-20).
#[derive(Clone, Debug)]
pub struct EditionCell(rlmesh_grpc::env::PinnedEdition);

impl EditionCell {
    /// The current selection.
    pub fn get(&self) -> String {
        self.0.get()
    }
}

/// How [`EnvClientCore::connect_with`] dials.
#[derive(Clone, Debug, Default)]
pub struct CoreConnect<'a> {
    /// Endpoint token (empty: none).
    pub token: &'a str,
    /// The workflow edition this side declares (already resolved).
    pub declared: Option<&'a str>,
    /// Bound on the dial and handshake.
    pub connect_timeout: Option<Duration>,
    /// Default bound on each request; also `close`'s bound.
    pub request_timeout: Option<Duration>,
}

/// The any-arity remote env client both [`RemoteEnv`] and
/// [`RemoteVectorEnv`] wrap, and the Python extension drives directly
/// (R-25: binding plumbing, exempt from semver).
pub struct EnvClientCore {
    inner: rlmesh_grpc::EnvClient,
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
    /// `connect_timeout` the TimeoutError class (RMT-12).
    pub async fn connect_with(address: &str, options: CoreConnect<'_>) -> Result<Self> {
        let parsed = ConnectAddress::parse(address)?;
        check_token(options.token)?;
        let dial = Self::dial(parsed, options.token, options.declared);
        let mut core = match normalize_timeout(options.connect_timeout) {
            Some(timeout) => tokio::time::timeout(timeout, dial)
                .await
                .map_err(|_| Error::Timeout(timeout))?,
            None => dial.await,
        }
        .map_err(|err| match err {
            Error::Connection(message) => Error::Connection(format!(
                "could not connect to an env at '{address}'; is the EnvServer running? \
                 ({message})"
            )),
            other => other,
        })?;
        core.request_timeout = normalize_timeout(options.request_timeout);
        Ok(core)
    }

    async fn dial(address: ConnectAddress, token: &str, declared: Option<&str>) -> Result<Self> {
        let mut inner = rlmesh_grpc::EnvClient::connect_with_token(&address.to_string(), token)
            .await
            .map_err(Error::from_grpc)?;
        inner.declare_workflow_edition(declared.map(str::to_string));
        let handshake = inner.handshake().await.map_err(Error::from_grpc)?;
        let session_offer = handshake.session_offer();
        // The handshake refuses a contract without spaces; one that does not
        // decode is a malformed handshake too.
        let env_contract = rlmesh_grpc::wire::env_contract_from_proto(handshake.env_contract)
            .map_err(|error| {
                Error::Protocol(format!("invalid spaces spec from remote env: {error}"))
            })?;
        let (Some(observation_space), Some(action_space)) = (
            env_contract.observation_space.clone(),
            env_contract.action_space.clone(),
        ) else {
            return Err(Error::Protocol(
                "remote environment contract is missing its spaces".to_string(),
            ));
        };
        Ok(Self {
            inner,
            env_contract,
            observation_space: Arc::new(observation_space),
            action_space: Arc::new(action_space),
            num_envs: handshake.num_envs,
            session_offer,
            env_id: crate::mint_id(),
            episode_ids: vec![String::new(); handshake.num_envs],
            selected_workflow_edition: handshake.selected_workflow_edition,
            request_timeout: None,
            closed: false,
            warned_shadowed_info_key: false,
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
    /// message (RMT-20).
    pub fn pin_workflow_edition(&mut self, edition: &str) {
        self.selected_workflow_edition = edition.trim().to_string();
        self.inner
            .pin_workflow_edition(self.selected_workflow_edition.clone());
    }

    /// The env's negotiation offer captured at handshake.
    pub fn session_offer(&self) -> &rlmesh_proto::SessionOffer {
        &self.session_offer
    }

    /// A shared handle on the selected edition, reading every pin.
    pub fn edition_cell(&self) -> EditionCell {
        EditionCell(self.inner.pinned_edition())
    }

    /// Name the client class closed-handle errors report (RMT-13).
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
    /// first collision per client is logged (RMT-18).
    fn insert_runtime_info(&mut self, info: &mut MetaMap, key: &'static str, value: MetaValue) {
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
        let observation_space = Arc::clone(&self.observation_space);
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
        let request = ProtoResetRequest {
            seeds,
            options: options.as_ref().map(meta_map_to_proto),
            timeout_ms: wire_timeout_ms(timeout),
            env_indices,
            episode_ids: pushed_ids,
        };
        let inner = &mut self.inner;
        let response = bounded(timeout, async {
            inner.reset(request).await.map_err(Error::from_grpc)
        })
        .await?;

        let observations = decode_batched_partial_values(
            response.observation.as_ref(),
            &observation_space,
            self.num_envs,
        )
        .map_err(|err| Error::Protocol(err.to_string()))?;
        validate_count(&observations, self.num_envs, "observations")
            .map_err(|error| Error::Environment(EnvironmentError::from_wire(error)))?;
        let mut info = response
            .infos
            .map(rlmesh_grpc::wire::meta_map_from_proto)
            .unwrap_or_default();
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
        let action_space = Arc::clone(&self.action_space);
        let observation_space = Arc::clone(&self.observation_space);
        validate_action_count(&req.actions, self.num_envs)
            .map_err(|error| Error::Environment(EnvironmentError::from_wire(error)))?;
        let action = encode_batched_partial_values(&req.actions, &action_space)
            .map_err(|err| Error::invalid_argument(err.to_string()))?;
        let request = ProtoStepRequest {
            action: Some(action),
            timeout_ms: wire_timeout_ms(timeout),
            // Full-width step; subset-stepping is reserved-but-deferred.
            env_indices: Vec::new(),
            // Push the authoritative per-lane ids (the env adopts a rolled id
            // on a NEXT_STEP autoreset).
            episode_ids: self.episode_ids.clone(),
        };
        let inner = &mut self.inner;
        let response = bounded(timeout, async {
            inner.step(request).await.map_err(Error::from_grpc)
        })
        .await?;

        let observations = decode_batched_partial_values(
            response.observation.as_ref(),
            &observation_space,
            self.num_envs,
        )
        .map_err(|err| Error::Protocol(err.to_string()))?;
        let terminated = response
            .terminated_mask
            .iter()
            .map(|value| *value != 0)
            .collect::<Vec<_>>();
        let truncated = response
            .truncated_mask
            .iter()
            .map(|value| *value != 0)
            .collect::<Vec<_>>();
        let completed_episodes = response
            .completed_episodes
            .into_iter()
            .map(proto_episode_metadata_to_public)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Error::Protocol)?;

        let env_count = self.num_envs;
        for check in [
            validate_count(&observations, env_count, "observations"),
            validate_count(&terminated, env_count, "terminated values"),
            validate_count(&truncated, env_count, "truncated values"),
            validate_count(&response.rewards, env_count, "rewards values"),
        ] {
            check.map_err(|error| Error::Environment(EnvironmentError::from_wire(error)))?;
        }

        let mut info = response
            .infos
            .map(rlmesh_grpc::wire::meta_map_from_proto)
            .unwrap_or_default();
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
            rewards: response.rewards,
            terminated,
            truncated,
            info: Some(info),
            episode_ids: self.episode_ids.clone(),
            completed_episodes,
        };
        // Under NEXT_STEP the env autoresets each completed lane next step and
        // adopts the fresh id pushed then, so roll it now. Under any other
        // mode the lane stays done until an explicit reset re-mints.
        if self.env_contract.autoreset_mode == spaces::types::AutoresetMode::NextStep {
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
                let request = render_request_to_proto(&RenderRequest {
                    env_index: Some(0),
                    timeout_ms: i64::try_from(wire_timeout_ms(timeout)).unwrap_or(i64::MAX),
                });
                let response = bounded(timeout, async {
                    inner.render(request).await.map_err(Error::from_grpc)
                })
                .await?;
                let result = render_result_from_proto(response)
                    .map_err(|e| Error::Protocol(e.to_string()))?;
                match result.frame {
                    Some(frame) => decode_render_png(&frame.frame),
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
        let request = render_request_to_proto(&RenderRequest {
            env_index: Some(lane as usize),
            timeout_ms: i64::try_from(wire_timeout_ms(timeout)).unwrap_or(i64::MAX),
        });
        let inner = &mut self.inner;
        let response = bounded(timeout, async {
            inner.render(request).await.map_err(Error::from_grpc)
        })
        .await?;
        let result =
            render_result_from_proto(response).map_err(|err| Error::Protocol(err.to_string()))?;
        Ok(result.frame.map(|frame| frame.frame))
    }

    /// Close this client's session (idempotent). Bounded by the request
    /// timeout, else a 5 s grace; on expiry or a lost connection it detaches
    /// locally and succeeds (RMT-4). Other errors are returned, and the
    /// client is closed either way.
    pub async fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        let bound = self.request_timeout.unwrap_or(CLOSE_GRACE);
        let outcome = tokio::time::timeout(bound, self.inner.close()).await;
        match outcome {
            Ok(Ok(_)) => Ok(()),
            Err(_) => {
                tracing::debug!("close did not complete within {bound:?}; detaching locally");
                self.inner.detach();
                Ok(())
            }
            Ok(Err(err)) => {
                let err = Error::from_grpc(err);
                self.inner.detach();
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

    /// Ask the server to shut itself down; an accepted request closes this
    /// client locally.
    pub async fn shutdown<'r>(&mut self, reason: impl Into<Option<&'r str>>) -> Result<bool> {
        self.check_open()?;
        let timeout = self.request_timeout;
        let reason = reason.into().unwrap_or("owner shutdown").to_string();
        let inner = &mut self.inner;
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
        &mut self.inner
    }

    /// Render lane `env_index` decoded to a uint8 frame (RMT-16).
    pub(crate) async fn render_value(&mut self, env_index: u64) -> Result<Option<Value>> {
        match self.render(env_index).await? {
            Some(png) => decode_render_png(&png).map(Some),
            None => Ok(None),
        }
    }
}

/// Decode a wire PNG frame to `(shape, raw uint8 bytes)`: `[H, W]`,
/// `[H, W, 3]` or `[H, W, 4]`, other formats as RGBA. An undecodable frame
/// is the ProtocolException class (RMT-P3).
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
async fn connect_core(options: RemoteEnvOptions, token: &str) -> Result<EnvClientCore> {
    let address = normalize_connect_address(
        options.address.as_deref(),
        options.host.as_deref(),
        options.port,
        options.path.as_deref(),
        options.transport,
    )?;
    ConnectAddress::parse(&address)?;
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
            token,
            declared: declared.as_deref(),
            connect_timeout: options.connect_timeout,
            request_timeout: options.request_timeout,
        },
    )
    .await
}

/// Fail `future` with the cancellation class when `token` fires (R-18).
async fn cancellable<T>(
    token: &crate::ext::CancelToken,
    future: impl std::future::Future<Output = Result<T>>,
) -> Result<T> {
    let driver = CancellationToken::new();
    let _watch = CancelWatch::start(vec![token.clone()], driver.clone())?;
    tokio::select! {
        biased;
        () = driver.cancelled() => Err(Error::Cancelled),
        out = future => out,
    }
}

/// Close an unclosed core at scope exit (RMT-8): bounded on this thread
/// outside a runtime, on a detached library task inside one.
fn teardown(mut core: Box<EnvClientCore>) {
    if core.is_closed() {
        return;
    }
    if in_async_context() {
        runtime().spawn(async move {
            if let Err(err) = core.close().await {
                tracing::warn!("closing a dropped remote env client failed: {err}");
            }
        });
    } else if let Err(err) = block_on_library(core.close()) {
        tracing::warn!("closing a dropped remote env client failed: {err}");
    }
}

/// What both clients keep outside the per-call core, so accessors never
/// wait on a call (RMT-17).
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
/// the borrow and comes back when the session ends (R-17).
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
                let token = self.cancel.clone();
                let core = self.core_mut()?;
                cancellable(&token, op(core)).await
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
            /// edition a session pinned it to (RMT-20).
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

            /// This handle's cancel token (R-18): firing it fails the call in
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
            /// bound are returned (RMT-4).
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

/// A client for one remote env: Python's `RemoteEnv`.
///
/// Construction dials and handshakes; the env's exclusive session slot is
/// taken by the first `reset`, `step` or `render`. Calls block (each has an
/// `_async` twin, R-16); `&mut self` serializes them. Dropping an unclosed
/// client closes it with `close`'s bound (RMT-8).
pub struct RemoteEnv {
    core: Option<Box<EnvClientCore>>,
    shared: Shared,
    lent_edition: String,
    cancel: crate::ext::CancelToken,
}

client_common!(RemoteEnv, "RemoteEnv");

impl RemoteEnv {
    /// Dial and handshake: Python's `RemoteEnv(...)`. Invalid options are
    /// the ValueError class, a dial failure the ConnectionError class, a
    /// refused handshake the ProtocolException class (RMT-12); an endpoint
    /// with more than one env is refused.
    pub fn new(options: RemoteEnvOptions) -> Result<RemoteEnv> {
        refuse_in_async_context("RemoteEnv::new")?;
        block_on_library(Self::new_in(options, ""))
    }

    /// The async twin of [`new`](Self::new).
    pub async fn new_async(options: RemoteEnvOptions) -> Result<RemoteEnv> {
        in_library(Self::new_in(options, "")).await
    }

    pub(crate) async fn new_in(options: RemoteEnvOptions, token: &str) -> Result<RemoteEnv> {
        let mut core = connect_core(options, token).await?;
        core.require_single()?;
        core.set_name("RemoteEnv");
        Ok(RemoteEnv {
            shared: Shared::of(&core)?,
            lent_edition: String::new(),
            core: Some(Box::new(core)),
            cancel: crate::ext::CancelToken::new(),
        })
    }

    /// The observation space (built once; RMT-9).
    pub fn observation_space(&self) -> &Space {
        &self.shared.observation_space
    }

    /// The action space (built once; RMT-9).
    pub fn action_space(&self) -> &Space {
        &self.shared.action_space
    }

    /// Start an episode: `(observation, info)`. `info` carries the
    /// runtime-reserved `episode_ids` (RMT-18).
    pub fn reset(
        &mut self,
        seed: Option<i64>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        refuse_in_async_context("RemoteEnv::reset")?;
        block_on_library(self.reset_in(seed, options))
    }

    /// The async twin of [`reset`](Self::reset).
    pub async fn reset_async(
        &mut self,
        seed: Option<i64>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        in_library(self.reset_in(seed, options)).await
    }

    async fn reset_in(
        &mut self,
        seed: Option<i64>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        let options = options.cloned();
        let (lanes, info) = self
            .io(move |core| {
                Box::pin(async move {
                    core.reset_values(seed.into_iter().collect(), options.as_ref(), Vec::new())
                        .await
                })
            })
            .await?;
        Ok((lanes.into_iter().next().unwrap_or(Value::None), info))
    }

    /// Apply one action: `(observation, reward, terminated, truncated,
    /// info)`; `info` carries `episode_ids` and `completed_episodes`.
    #[allow(clippy::type_complexity)]
    pub fn step(&mut self, action: Value) -> Result<(Value, f64, bool, bool, InfoDict)> {
        refuse_in_async_context("RemoteEnv::step")?;
        block_on_library(self.step_in(action))
    }

    /// The async twin of [`step`](Self::step).
    #[allow(clippy::type_complexity)]
    pub async fn step_async(
        &mut self,
        action: Value,
    ) -> Result<(Value, f64, bool, bool, InfoDict)> {
        in_library(self.step_in(action)).await
    }

    #[allow(clippy::type_complexity)]
    async fn step_in(&mut self, action: Value) -> Result<(Value, f64, bool, bool, InfoDict)> {
        let (lanes, rewards, terminated, truncated, info) = self
            .io(move |core| Box::pin(core.step_values(vec![action])))
            .await?;
        Ok((
            lanes.into_iter().next().unwrap_or(Value::None),
            rewards.first().copied().unwrap_or_default(),
            terminated.first().copied().unwrap_or_default(),
            truncated.first().copied().unwrap_or_default(),
            info,
        ))
    }
}

/// A client for a remote vector env (two or more lanes stepped in lockstep):
/// Python's `RemoteVectorEnv`. Observations and actions are one batched
/// [`Value`]; rewards and flags are per-lane vectors.
pub struct RemoteVectorEnv {
    core: Option<Box<EnvClientCore>>,
    shared: Shared,
    lent_edition: String,
    cancel: crate::ext::CancelToken,
}

client_common!(RemoteVectorEnv, "RemoteVectorEnv");

impl RemoteVectorEnv {
    /// Dial and handshake: Python's `RemoteVectorEnv(...)`. An endpoint with
    /// fewer than two envs is refused (the ValueError class).
    pub fn new(options: RemoteEnvOptions) -> Result<RemoteVectorEnv> {
        refuse_in_async_context("RemoteVectorEnv::new")?;
        block_on_library(Self::new_in(options, ""))
    }

    /// The async twin of [`new`](Self::new).
    pub async fn new_async(options: RemoteEnvOptions) -> Result<RemoteVectorEnv> {
        in_library(Self::new_in(options, "")).await
    }

    pub(crate) async fn new_in(options: RemoteEnvOptions, token: &str) -> Result<RemoteVectorEnv> {
        let mut core = connect_core(options, token).await?;
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

    /// One lane's observation space (built once; RMT-9).
    pub fn single_observation_space(&self) -> &Space {
        &self.shared.observation_space
    }

    /// One lane's action space (built once; RMT-9).
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
    /// list, as Python does.
    pub fn reset(
        &mut self,
        seed: Option<VectorSeed>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        refuse_in_async_context("RemoteVectorEnv::reset")?;
        block_on_library(self.reset_in(seed, options, Vec::new()))
    }

    /// The async twin of [`reset`](Self::reset).
    pub async fn reset_async(
        &mut self,
        seed: Option<VectorSeed>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        in_library(self.reset_in(seed, options, Vec::new())).await
    }

    pub(crate) async fn reset_in(
        &mut self,
        seed: Option<VectorSeed>,
        options: Option<&InfoDict>,
        env_indices: Vec<u32>,
    ) -> Result<(Value, InfoDict)> {
        let seeds = match seed {
            None => Vec::new(),
            Some(VectorSeed::One(seed)) => vec![seed],
            Some(VectorSeed::Each(seeds)) => seeds,
        };
        let options = options.cloned();
        let observation = self.shared.contract.observation_space().wire().clone();
        let (lanes, info) = self
            .io(move |core| {
                Box::pin(async move {
                    core.reset_values(seeds, options.as_ref(), env_indices)
                        .await
                })
            })
            .await?;
        Ok((super::batch::stack_lanes(&observation, lanes)?, info))
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

    #[allow(clippy::type_complexity)]
    async fn step_in(
        &mut self,
        actions: Value,
    ) -> Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)> {
        let action = self.shared.contract.action_space().wire().clone();
        let observation = self.shared.contract.observation_space().wire().clone();
        let lanes = super::batch::unstack_lanes(&action, actions, self.shared.num_envs as usize)?;
        let (lanes, rewards, terminated, truncated, info) = self
            .io(move |core| Box::pin(core.step_values(lanes)))
            .await?;
        Ok((
            super::batch::stack_lanes(&observation, lanes)?,
            rewards,
            terminated,
            truncated,
            info,
        ))
    }
}

/// `ext::reset_lanes`' body (RMT-14).
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
    env.reset_in(seed, options, lanes).await
}

/// The ext constructors' token check and dial (RMT-5).
pub(crate) fn check_token(token: &str) -> Result<()> {
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
