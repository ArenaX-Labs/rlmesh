//! The sync Gymnasium envs on the wire: each lane (or vector env) is driven
//! on one thread for the server's life, and its values cross the wire's
//! request/result shape here.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc;

use async_trait::async_trait;
use rlmesh_spaces::AutoresetMode;
use rlmesh_spaces::errors::EnvRuntimeError;
use tokio::sync::oneshot;

use super::batch::{encode_frame, first_item, stack_lanes, unstack_lanes};
use super::legacy;
use super::types::{
    CloseResult as VectorCloseResult, ResetRequest as VectorResetRequest,
    ResetResult as VectorResetResult, StepRequest as VectorStepRequest,
    StepResult as VectorStepResult,
};
use super::{Env, VectorEnv, VectorSeed};
use crate::binding::spaces::{
    self, CloseRequest, RenderFrame, RenderRequest, RenderResult,
    request::{CloseResult, ResetRequest, ResetResult, StepRequest, StepResult},
};
use crate::ext::TensorExt;
use crate::specs::EnvContract;
use crate::value::{meta_codec, space_codec};
use crate::{Error, InfoDict, Metadata, Result, Value};

/// The id an env with none serves (ENV-6).
pub(crate) const UNKNOWN_ENV_ID: &str = "UnknownEnv-v1";

const AUTORESET_KEY: &str = "autoreset_mode";

/// ENV-27: the autoreset mode an env's metadata declares.
pub(crate) fn autoreset_mode(metadata: Option<&Metadata>) -> Result<AutoresetMode> {
    let Some(value) = metadata.and_then(|metadata| metadata.get(AUTORESET_KEY)) else {
        return Ok(AutoresetMode::Disabled);
    };
    match value {
        Value::None => Err(Error::invalid_argument(
            "metadata[\"autoreset_mode\"] is present but None; set it to a gymnasium \
             AutoresetMode enum or one of \"NextStep\"/\"SameStep\"/\"Disabled\", or omit the \
             key entirely to default to DISABLED",
        )),
        Value::Text(mode) => match mode.as_str() {
            "NextStep" => Ok(AutoresetMode::NextStep),
            "Disabled" => Ok(AutoresetMode::Disabled),
            "SameStep" => Err(Error::invalid_argument(
                "SAME_STEP autoreset mode is not yet supported by the rlmesh runtime; \
                 construct the environment with NEXT_STEP or DISABLED autoreset",
            )),
            other => Err(Error::invalid_argument(format!(
                "unrecognized metadata[\"autoreset_mode\"] {other:?}; expected a gymnasium \
                 AutoresetMode enum or one of \"NextStep\"/\"SameStep\"/\"Disabled\""
            ))),
        },
        other => Err(Error::invalid_argument(format!(
            "metadata[\"autoreset_mode\"] must be a gymnasium AutoresetMode enum or one of \
             \"NextStep\"/\"SameStep\"/\"Disabled\", got {other:?}"
        ))),
    }
}

/// Construction step 7: the contract a scalar env serves as one lane.
pub(crate) fn env_contract_of(env: &dyn Env) -> Result<EnvContract> {
    let metadata = env.metadata();
    let autoreset = autoreset_mode(metadata.as_ref())?;
    Ok(EnvContract::new(
        env.id()
            .filter(|id| !id.is_empty())
            .unwrap_or_else(|| UNKNOWN_ENV_ID.to_string()),
        env.observation_space().spec().clone(),
        env.action_space().spec().clone(),
        env.render_mode(),
        1,
        metadata,
        autoreset,
    ))
}

/// The contract a vector env serves (its single spaces at `num_envs`).
pub(crate) fn vector_contract_of(env: &dyn VectorEnv) -> Result<EnvContract> {
    let metadata = env.metadata();
    let autoreset = autoreset_mode(metadata.as_ref())?;
    Ok(EnvContract::new(
        env.id()
            .filter(|id| !id.is_empty())
            .unwrap_or_else(|| UNKNOWN_ENV_ID.to_string()),
        env.single_observation_space().spec().clone(),
        env.single_action_space().spec().clone(),
        env.render_mode(),
        env.num_envs(),
        metadata,
        autoreset,
    ))
}

/// ENV-35: an authored `{op}` failure as the client receives it, code
/// `INTERNAL` and the text `"{op} failed: {message}"`, recoverable only for a
/// recoverable Environment-class error.
pub(crate) fn authored_failure(op: &str, err: &Error) -> EnvRuntimeError {
    let (message, recoverable) = match err {
        Error::Environment(env) => (env.message.clone(), env.is_recoverable),
        other => (other.to_string(), false),
    };
    let text = format!("{op} failed: {message}");
    if recoverable {
        EnvRuntimeError::Recoverable(text)
    } else {
        EnvRuntimeError::Runtime(text)
    }
}

/// A value the env produced that the wire cannot carry (a structural
/// deviation from its declared space).
fn encode_failure(op: &str, err: Error) -> EnvRuntimeError {
    EnvRuntimeError::Runtime(format!(
        "{op} returned a value the wire cannot carry: {err}"
    ))
}

type Job<T> = Box<dyn FnOnce(&mut T) + Send>;

/// One thread that owns a `T` and runs jobs on it in order.
pub(crate) struct Actor<T> {
    tx: mpsc::Sender<Job<T>>,
}

/// The mailbox of an actor whose thread is the caller's (a foreground serve's
/// lane 0): [`run`](Self::run) drains it until the actor is dropped.
pub(crate) struct ActorRunner<T> {
    rx: mpsc::Receiver<Job<T>>,
    target: T,
}

impl<T: Send + 'static> Actor<T> {
    /// An actor on its own dedicated plain thread.
    pub(crate) fn spawn(
        name: String,
        target: T,
    ) -> Result<(Actor<T>, std::thread::JoinHandle<()>)> {
        let (actor, runner) = Actor::foreground(target);
        let thread = std::thread::Builder::new()
            .name(name)
            .spawn(move || runner.run())
            .map_err(|err| Error::Internal(format!("failed to spawn an env thread: {err}")))?;
        Ok((actor, thread))
    }

    /// An actor whose jobs the returned runner executes.
    pub(crate) fn foreground(target: T) -> (Actor<T>, ActorRunner<T>) {
        let (tx, rx) = mpsc::channel();
        (Actor { tx }, ActorRunner { rx, target })
    }

    /// Run `job` on the actor's thread and wait for its result.
    pub(crate) async fn call<R: Send + 'static>(
        &self,
        op: &'static str,
        job: impl FnOnce(&mut T) -> std::result::Result<R, EnvRuntimeError> + Send + 'static,
    ) -> std::result::Result<R, EnvRuntimeError> {
        let (reply, rx) = oneshot::channel();
        let boxed: Job<T> = Box::new(move |target: &mut T| {
            let result = catch_unwind(AssertUnwindSafe(|| job(target)))
                .unwrap_or_else(|_| Err(EnvRuntimeError::Runtime(format!("{op} task panicked"))));
            let _ = reply.send(result);
        });
        self.tx.send(boxed).map_err(|_| {
            EnvRuntimeError::Runtime(format!("{op}: the serving thread has stopped"))
        })?;
        rx.await.unwrap_or_else(|_| {
            Err(EnvRuntimeError::Runtime(format!(
                "{op}: the serving thread has stopped"
            )))
        })
    }
}

impl<T> ActorRunner<T> {
    /// Run jobs until every handle to the actor is dropped, then drop the
    /// target on this thread.
    pub(crate) fn run(mut self) {
        for job in self.rx {
            job(&mut self.target);
        }
    }
}

/// The wire shapes of one scalar lane: its spaces, contract and codecs.
struct LaneShape {
    observation: spaces::SpaceSpec,
    action: spaces::SpaceSpec,
    contract: spaces::EnvContract,
}

/// How a lane's calls reach its env: inline on the lane thread, or relayed to
/// the thread that called a foreground `serve()`.
enum LaneCalls {
    Inline(Box<dyn Env + Send>),
    Relayed(Actor<Box<dyn Env + Send>>),
}

/// A sync [`Env`] as one lane of the wire's lane host.
pub(crate) struct SyncLane {
    shape: LaneShape,
    calls: LaneCalls,
}

impl SyncLane {
    /// `contract` is the lane's derived contract (construction step 7).
    pub(crate) fn new(env: Box<dyn Env + Send>, contract: &EnvContract) -> Result<SyncLane> {
        Ok(SyncLane {
            shape: lane_shape(contract)?,
            calls: LaneCalls::Inline(env),
        })
    }

    /// A lane whose env runs on the runner's thread.
    pub(crate) fn relayed(
        env: Box<dyn Env + Send>,
        contract: &EnvContract,
    ) -> Result<(SyncLane, ActorRunner<Box<dyn Env + Send>>)> {
        let (actor, runner) = Actor::foreground(env);
        Ok((
            SyncLane {
                shape: lane_shape(contract)?,
                calls: LaneCalls::Relayed(actor),
            },
            runner,
        ))
    }

    async fn run<R: Send + 'static>(
        &mut self,
        op: &'static str,
        job: impl FnOnce(&mut Box<dyn Env + Send>) -> std::result::Result<R, EnvRuntimeError>
        + Send
        + 'static,
    ) -> std::result::Result<R, EnvRuntimeError> {
        match &mut self.calls {
            LaneCalls::Inline(env) => catch_unwind(AssertUnwindSafe(|| job(env)))
                .unwrap_or_else(|_| Err(EnvRuntimeError::Runtime(format!("{op} task panicked")))),
            LaneCalls::Relayed(actor) => actor.call(op, job).await,
        }
    }
}

fn lane_shape(contract: &EnvContract) -> Result<LaneShape> {
    Ok(LaneShape {
        observation: contract.observation_space().wire().clone(),
        action: contract.action_space().wire().clone(),
        contract: contract.to_wire()?,
    })
}

fn info_to_wire(
    op: &str,
    info: &InfoDict,
) -> std::result::Result<Option<spaces::MetaMap>, EnvRuntimeError> {
    if info.is_empty() {
        return Ok(None);
    }
    meta_codec::metadata_to_wire(info)
        .map(Some)
        .map_err(|err| encode_failure(op, err))
}

fn frame_result(frame: Option<Value>) -> std::result::Result<RenderResult, EnvRuntimeError> {
    let frame = match frame {
        Some(frame) => encode_frame(&frame).map_err(|err| encode_failure("render", err))?,
        None => None,
    };
    Ok(RenderResult {
        frame: frame.map(|frame| RenderFrame { frame }),
    })
}

#[async_trait]
impl legacy::Env for SyncLane {
    fn observation_space(&self) -> &spaces::SpaceSpec {
        &self.shape.observation
    }

    fn action_space(&self) -> &spaces::SpaceSpec {
        &self.shape.action
    }

    fn env_contract(&self) -> &spaces::EnvContract {
        &self.shape.contract
    }

    async fn reset(
        &mut self,
        req: ResetRequest,
    ) -> std::result::Result<ResetResult, EnvRuntimeError> {
        let space = self.shape.observation.clone();
        self.run("reset", move |env| {
            let options = req.options.map(meta_codec::metadata_from_wire);
            let (observation, info) = env
                .reset(req.seed, options.as_ref())
                .map_err(|err| authored_failure("reset", &err))?;
            Ok(ResetResult {
                observation: Some(
                    space_codec::value_to_space(&space, &observation)
                        .map_err(|err| encode_failure("reset", err))?,
                ),
                info: info_to_wire("reset", &info)?,
                episode_id: None,
            })
        })
        .await
    }

    async fn step(&mut self, req: StepRequest) -> std::result::Result<StepResult, EnvRuntimeError> {
        let observation_space = self.shape.observation.clone();
        let action_space = self.shape.action.clone();
        self.run("step", move |env| {
            let action = req
                .action
                .map(|action| space_codec::value_from_space(&action_space, action))
                .transpose()
                .map_err(|err| EnvRuntimeError::InvalidValue(err.to_string()))?;
            let (observation, reward, terminated, truncated, info) = env
                .step(action)
                .map_err(|err| authored_failure("step", &err))?;
            Ok(StepResult {
                observation: Some(
                    space_codec::value_to_space(&observation_space, &observation)
                        .map_err(|err| encode_failure("step", err))?,
                ),
                reward,
                terminated,
                truncated,
                info: info_to_wire("step", &info)?,
            })
        })
        .await
    }

    async fn render(
        &mut self,
        _req: RenderRequest,
    ) -> std::result::Result<RenderResult, EnvRuntimeError> {
        self.run("render", |env| {
            frame_result(
                env.render()
                    .map_err(|err| authored_failure("render", &err))?,
            )
        })
        .await
    }

    async fn close(
        &mut self,
        _req: CloseRequest,
    ) -> std::result::Result<CloseResult, EnvRuntimeError> {
        self.run("close", |env| {
            env.close()
                .map(|()| CloseResult)
                .map_err(|err| authored_failure("close", &err))
        })
        .await
    }
}

/// A sync [`VectorEnv`] on the wire, driven on one dedicated thread (or the
/// caller's, for a foreground serve).
pub(crate) struct SyncVector {
    actor: Actor<Box<dyn VectorEnv + Send>>,
    observation: spaces::SpaceSpec,
    action: spaces::SpaceSpec,
    contract: spaces::EnvContract,
    num_envs: usize,
}

impl SyncVector {
    pub(crate) fn new(
        actor: Actor<Box<dyn VectorEnv + Send>>,
        contract: &EnvContract,
    ) -> Result<SyncVector> {
        Ok(SyncVector {
            actor,
            observation: contract.observation_space().wire().clone(),
            action: contract.action_space().wire().clone(),
            contract: contract.to_wire()?,
            num_envs: usize::try_from(contract.num_envs()).unwrap_or(usize::MAX),
        })
    }
}

fn vector_observations(
    op: &str,
    space: &spaces::SpaceSpec,
    batch: Value,
    num_envs: usize,
) -> std::result::Result<Vec<spaces::SpaceValue>, EnvRuntimeError> {
    unstack_lanes(space, batch, num_envs)
        .and_then(|lanes| {
            lanes
                .iter()
                .map(|lane| space_codec::value_to_space(space, lane))
                .collect::<Result<Vec<_>>>()
        })
        .map_err(|err| encode_failure(op, err))
}

fn vector_seed(seeds: Vec<i64>) -> Option<VectorSeed> {
    // ENV-4: no seed is None, one wire seed one int, more a list.
    match seeds.as_slice() {
        [] => None,
        [seed] => Some(VectorSeed::One(*seed)),
        _ => Some(VectorSeed::Each(seeds)),
    }
}

#[async_trait]
impl legacy::VectorEnv for SyncVector {
    fn observation_space(&self) -> &spaces::SpaceSpec {
        &self.observation
    }

    fn action_space(&self) -> &spaces::SpaceSpec {
        &self.action
    }

    fn num_envs(&self) -> usize {
        self.num_envs
    }

    fn env_contract(&self) -> &spaces::EnvContract {
        &self.contract
    }

    async fn reset(
        &mut self,
        req: VectorResetRequest,
    ) -> std::result::Result<VectorResetResult, EnvRuntimeError> {
        let space = self.observation.clone();
        let num_envs = self.num_envs;
        self.actor
            .call("reset", move |env| {
                let options = req.options.map(meta_codec::metadata_from_wire);
                let (observation, info) = env
                    .reset(vector_seed(req.seeds), options.as_ref())
                    .map_err(|err| authored_failure("reset", &err))?;
                Ok(VectorResetResult {
                    observations: vector_observations("reset", &space, observation, num_envs)?,
                    info: info_to_wire("reset", &info)?,
                    episode_ids: Vec::new(),
                })
            })
            .await
    }

    async fn step(
        &mut self,
        req: VectorStepRequest,
    ) -> std::result::Result<VectorStepResult, EnvRuntimeError> {
        let observation_space = self.observation.clone();
        let action_space = self.action.clone();
        let num_envs = self.num_envs;
        self.actor
            .call("step", move |env| {
                let lanes = req
                    .actions
                    .into_iter()
                    .map(|action| space_codec::value_from_space(&action_space, action))
                    .collect::<crate::Result<Vec<_>>>()
                    .map_err(|err| EnvRuntimeError::InvalidValue(err.to_string()))?;
                let actions = stack_lanes(&action_space, lanes)
                    .map_err(|err| EnvRuntimeError::InvalidValue(err.to_string()))?;
                let (observation, rewards, terminated, truncated, info) = env
                    .step(actions)
                    .map_err(|err| authored_failure("step", &err))?;
                Ok(VectorStepResult {
                    observations: vector_observations(
                        "step",
                        &observation_space,
                        observation,
                        num_envs,
                    )?,
                    rewards,
                    terminated,
                    truncated,
                    info: info_to_wire("step", &info)?,
                    completed_episodes: Vec::new(),
                    episode_ids: Vec::new(),
                })
            })
            .await
    }

    async fn render(
        &mut self,
        req: RenderRequest,
    ) -> std::result::Result<RenderResult, EnvRuntimeError> {
        let lane = req.env_index.unwrap_or(0);
        self.actor
            .call("render", move |env| {
                let frames = env
                    .render()
                    .map_err(|err| authored_failure("render", &err))?;
                let frame = match frames {
                    Some(Value::List(items) | Value::Tuple(items)) => items.into_iter().nth(lane),
                    Some(Value::Tensor(tensor)) => tensor
                        .unstack()
                        .map_err(|err| encode_failure("render", err))?
                        .into_iter()
                        .nth(lane)
                        .map(Value::Tensor),
                    Some(other) => {
                        Some(first_item(other).map_err(|err| encode_failure("render", err))?)
                    }
                    None => None,
                };
                frame_result(frame)
            })
            .await
    }

    async fn close(
        &mut self,
        _req: CloseRequest,
    ) -> std::result::Result<VectorCloseResult, EnvRuntimeError> {
        self.actor
            .call("close", |env| {
                env.close()
                    .map(|()| VectorCloseResult::default())
                    .map_err(|err| authored_failure("close", &err))
            })
            .await
    }
}
