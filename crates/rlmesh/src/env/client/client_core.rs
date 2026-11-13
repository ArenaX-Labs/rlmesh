//! The any-arity client core both clients wrap: connect, reset, step, render
//! and close over a [`Link`], and its teardown.

use std::sync::Arc;
use std::time::Duration;

use rlmesh_spaces::{MetaMap, MetaValue};
use tokio_util::sync::CancellationToken;

use super::episodes::Episodes;
use super::link::{Edition, EditionCell, Link, lockstep_error, spaces_of, wire_lane};
use super::{
    CLOSE_GRACE, RUNTIME_INFO_KEYS, RemoteEnvOptions, bounded, normalize_connect_address,
    normalize_timeout,
};
#[cfg(doc)]
use super::{RemoteEnv, RemoteVectorEnv};
use crate::env::types::{
    ResetRequest as VectorResetRequest, ResetResult as VectorResetResult,
    StepRequest as VectorStepRequest, StepResult as VectorStepResult,
};
use crate::env::wire::{validate_action_count, validate_count};
use crate::run::drive::{CancelWatch, block_on_library, in_async_context, runtime};
use crate::value::{meta_codec, space_codec};
use crate::{ConnectAddress, EnvironmentError, Error, InfoDict, Result, Value, binding::spaces};

/// How [`EnvClientCore::connect_with`] dials.
#[derive(Clone, Debug, Default)]
pub struct CoreConnect<'a> {
    /// The workflow edition this side declares (already resolved).
    pub declared: Option<&'a str>,
    /// Bound on the dial and handshake.
    pub connect_timeout: Option<Duration>,
    /// Default bound on each request; also `close`'s bound.
    pub request_timeout: Option<Duration>,
}

/// How long a dial bounded by `connect_timeout` waits for the session: the
/// whole bound, as the caller asked, less a margin, so a join that runs out
/// says why (ports that did not answer, a host not up) before the bare
/// bound expires.
fn join_bound(connect_timeout: Duration) -> Duration {
    connect_timeout.saturating_sub((connect_timeout / 10).min(Duration::from_millis(250)))
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
            .field("address", &self.shown_address())
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

    /// Dial `address` and handshake. A dial failure is the ConnectionError
    /// class (a join that ran out of time too, saying why, as with no
    /// bound), a refused handshake the ProtocolException class, an expired
    /// `connect_timeout` the TimeoutError class.
    pub async fn connect_with(address: &str, options: CoreConnect<'_>) -> Result<Self> {
        let parsed = ConnectAddress::parse(address)?;
        let connect_timeout = normalize_timeout(options.connect_timeout);
        let dial = Self::dial(parsed, options.declared, connect_timeout.map(join_bound));
        let mut core = match connect_timeout {
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
        declared: Option<&str>,
        join_within: Option<Duration>,
    ) -> Result<Self> {
        let (inner, env_contract, spaces, num_envs, session_offer, selected) = match address {
            ConnectAddress::Lockstep(address) => {
                let mut options = crate::lockstep::ConnectOptions::default();
                if let Some(within) = join_within {
                    options.timeout = within;
                }
                let client = match crate::lockstep::connect_driver(&address, options)
                    .await
                    .map_err(lockstep_error)?
                {
                    crate::lockstep::Driven::Gym(client) => client,
                    crate::lockstep::Driven::View(view) => {
                        return Ok(Self::of_view(view, address, declared));
                    }
                };
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

    /// A client over a v2-native robot's Gym view: one lane, its spaces
    /// keyed by part path, the edition this build runs for `declared`.
    fn of_view(view: crate::lockstep::GymView, address: String, declared: Option<&str>) -> Self {
        let (observation_space, action_space) = view.spaces();
        let session_offer = rlmesh_proto::SessionOffer::this_build(declared);
        let selected = session_offer.preferred.clone().unwrap_or_default();
        let robot = address.rsplit('/').next().unwrap_or_default().to_string();
        Self {
            inner: Link::View {
                view: Arc::new(std::sync::Mutex::new(Some(Arc::new(view)))),
                address,
                edition: Arc::new(std::sync::Mutex::new(selected.clone())),
            },
            env_contract: spaces::EnvContract {
                id: robot,
                observation_space: Some(observation_space.clone()),
                action_space: Some(action_space.clone()),
                num_envs: 1,
                ..Default::default()
            },
            observation_space: Arc::new(observation_space),
            action_space: Arc::new(action_space),
            num_envs: 1,
            session_offer,
            env_id: crate::mint_id(),
            episode_ids: vec![String::new()],
            selected_workflow_edition: selected,
            request_timeout: None,
            closed: false,
            warned_shadowed_info_key: false,
            runtime_info: true,
            name: "the env client",
        }
    }

    /// Refuse an endpoint with more than one env (`RemoteEnv`'s arity).
    pub fn require_single(&self) -> Result<()> {
        if self.num_envs != 1 {
            return Err(Error::invalid_argument(format!(
                "Endpoint {:?} serves {} environments. Use RemoteVectorEnv instead.",
                self.shown_address(),
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

    /// The address as errors and logs show it: an invite's key, a
    /// credential, left out.
    fn shown_address(&self) -> String {
        rlmesh_session::address::redact(self.address())
    }

    /// The session the env is in (its address), on the v2 path.
    pub fn session(&self) -> Option<&str> {
        match &self.inner {
            Link::Lockstep { client, .. } => client.session(),
            _ => None,
        }
    }

    /// Whether the endpoint is a v2-native robot driven through its Gym
    /// view (not a served Gym env).
    pub fn drives_view(&self) -> bool {
        if let Link::View { .. } = self.inner {
            return true;
        }
        false
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

    /// Whether `close` ran.
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
            Link::Lockstep { edition, .. } | Link::View { edition, .. } => {
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
            Link::Lockstep { edition, .. } | Link::View { edition, .. } => {
                Edition::Local(Arc::clone(edition))
            }
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

    pub(super) fn check_open(&self) -> Result<()> {
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
                address = %self.shown_address(),
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
        match &self.inner {
            Link::Lockstep { client, .. } => {
                if tokio::time::timeout(bound, client.close()).await.is_err() {
                    tracing::debug!("lockstep detach did not complete within {bound:?}");
                }
            }
            Link::View { view, .. } => {
                view.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take();
            }
        }
        Ok(())
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

pub(crate) fn decode_render_png(png: &[u8]) -> Result<Value> {
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
pub(super) async fn connect_core(options: RemoteEnvOptions) -> Result<EnvClientCore> {
    let address = normalize_connect_address(options.address.as_deref())?;
    let declared = crate::editions::resolve_native(crate::editions::EditionSources {
        call: options.workflow_edition.as_deref(),
        option: None,
        declared: None,
    })?;
    EnvClientCore::connect_with(
        &address,
        CoreConnect {
            declared: declared.as_deref(),
            connect_timeout: options.connect_timeout,
            request_timeout: options.request_timeout,
        },
    )
    .await
}

/// Fail `future` with the cancellation class when `token` fires.
pub(super) async fn cancellable<T>(
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
pub(super) fn teardown(mut core: Box<EnvClientCore>) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::client::{RemoteEnv, RemoteVectorEnv, timeout_from_secs};

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
    fn timeouts_read_zero_as_none() {
        assert_eq!(normalize_timeout(Some(Duration::ZERO)), None);
        assert_eq!(timeout_from_secs(Some(0.0), "t").unwrap(), None);
        assert!(timeout_from_secs(Some(-1.0), "t").is_err());
        assert!(timeout_from_secs(Some(f64::NAN), "t").is_err());
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

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn clients_are_send_and_sync() {
        assert_send_sync::<RemoteEnv>();
        assert_send_sync::<RemoteVectorEnv>();
        assert_send_sync::<EnvClientCore>();
    }
}
