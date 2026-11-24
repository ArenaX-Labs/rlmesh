//! [`LaneEnv`]: N independent scalar [`Env`]s, one actor thread each.
//!
//! The instances are the lanes of one `num_envs = N` endpoint. Each lane's thread owns its env
//! and runs its jobs in order, so a lane's reset and step always run on the same OS thread
//! (as OSMesa and Vulkan contexts expect), and lanes run concurrently with no shared lock.
//! `N = 1` is the ordinary scalar server.

use std::collections::BTreeMap;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use tokio::sync::oneshot;

use super::legacy::Env;
use crate::binding::spaces::{self, CloseRequest, MetaMap, MetaValue, RenderRequest, RenderResult};
use rlmesh_proto::{EndpointPhases, elapsed_ns};

pub(super) type LaneReply<T> = Result<(T, EndpointPhases), spaces::EnvRuntimeError>;

enum Job {
    Reset(
        spaces::request::ResetRequest,
        oneshot::Sender<LaneReply<spaces::request::ResetResult>>,
    ),
    Step(
        spaces::request::StepRequest,
        oneshot::Sender<LaneReply<spaces::request::StepResult>>,
    ),
    Render(RenderRequest, oneshot::Sender<LaneReply<RenderResult>>),
    Close(
        CloseRequest,
        oneshot::Sender<LaneReply<spaces::request::CloseResult>>,
    ),
}

struct Envelope {
    job: Job,
    enqueued: Instant,
}

const LANE_JOIN_GRACE: Duration = Duration::from_secs(5);

const LANE_JOIN_POLL: Duration = Duration::from_millis(1);

pub(super) type JoinThreads = std::sync::Arc<std::sync::Mutex<Vec<std::thread::JoinHandle<()>>>>;

struct LaneActor {
    tx: mpsc::Sender<Envelope>,
    thread: std::thread::JoinHandle<()>,
}

/// N scalar environments served as the lanes of one vector endpoint.
pub struct LaneEnv {
    lanes: Vec<LaneActor>,
    abandoned: Option<JoinThreads>,
    observation_space: spaces::SpaceSpec,
    action_space: spaces::SpaceSpec,
    env_contract: spaces::EnvContract,
}

impl LaneEnv {
    /// Host `envs` as lanes `0..envs.len()`, each on its own thread. Spaces
    /// and the contract are served from lane 0, so every lane must carry the
    /// same contract (spaces, metadata, tags).
    ///
    /// # Errors
    /// [`EnvRuntimeError::Runtime`](spaces::EnvRuntimeError::Runtime) when a lane's
    /// contract differs from lane 0's.
    ///
    /// # Panics
    /// Panics on an empty `envs`.
    pub fn new<E: Env + 'static>(envs: Vec<E>) -> Result<Self, spaces::EnvRuntimeError> {
        let first = envs.first().expect("LaneEnv needs at least one lane");
        let observation_space = first.observation_space().clone();
        let action_space = first.action_space().clone();
        let mut env_contract = first.env_contract().clone();
        for (index, env) in envs.iter().enumerate().skip(1) {
            if env.env_contract() != &env_contract {
                return Err(spaces::EnvRuntimeError::Runtime(format!(
                    "lane {index} disagrees with lane 0 on the env contract (spaces, metadata, \
                     tags): every lane of one endpoint must be the same make()"
                )));
            }
        }
        env_contract.autoreset_mode = spaces::types::AutoresetMode::Disabled;
        let lanes = envs
            .into_iter()
            .enumerate()
            .map(|(index, env)| LaneActor::spawn(index, env))
            .collect();
        Ok(Self {
            lanes,
            abandoned: None,
            observation_space,
            action_space,
            env_contract,
        })
    }

    pub(crate) fn track_abandoned(&mut self, threads: JoinThreads) {
        self.abandoned = Some(threads);
    }

    /// The lane count (the endpoint's vector width).
    pub fn num_envs(&self) -> usize {
        self.lanes.len()
    }

    /// One lane's observation space.
    pub fn observation_space(&self) -> &spaces::SpaceSpec {
        &self.observation_space
    }

    /// One lane's action space.
    pub fn action_space(&self) -> &spaces::SpaceSpec {
        &self.action_space
    }

    /// One lane's contract (spaces, id, render mode, metadata).
    pub fn env_contract(&self) -> &spaces::EnvContract {
        &self.env_contract
    }

    fn lane(&self, index: usize) -> Result<&LaneActor, spaces::EnvRuntimeError> {
        self.lanes.get(index).ok_or_else(|| {
            spaces::EnvRuntimeError::Runtime(format!(
                "lane {index} out of range for {} lanes",
                self.lanes.len()
            ))
        })
    }

    /// Reset one lane.
    pub async fn reset(
        &self,
        lane: usize,
        req: spaces::request::ResetRequest,
    ) -> LaneReply<spaces::request::ResetResult> {
        let (tx, rx) = oneshot::channel();
        self.lane(lane)?.send(Job::Reset(req, tx), lane)?;
        await_reply(rx, lane).await
    }

    /// Step one lane.
    pub async fn step(
        &self,
        lane: usize,
        req: spaces::request::StepRequest,
    ) -> LaneReply<spaces::request::StepResult> {
        let (tx, rx) = oneshot::channel();
        self.lane(lane)?.send(Job::Step(req, tx), lane)?;
        await_reply(rx, lane).await
    }

    /// Render one lane.
    pub async fn render(&self, lane: usize, req: RenderRequest) -> LaneReply<RenderResult> {
        let (tx, rx) = oneshot::channel();
        self.lane(lane)?.send(Job::Render(req, tx), lane)?;
        await_reply(rx, lane).await
    }

    /// Close every lane, in lane order.
    pub async fn close(&self, req: CloseRequest) -> Result<(), spaces::EnvRuntimeError> {
        for (lane, actor) in self.lanes.iter().enumerate() {
            let (tx, rx) = oneshot::channel();
            actor.send(Job::Close(req.clone(), tx), lane)?;
            await_reply(rx, lane).await?;
        }
        Ok(())
    }
}

impl Drop for LaneEnv {
    fn drop(&mut self) {
        let (senders, threads): (Vec<_>, Vec<_>) = self
            .lanes
            .drain(..)
            .map(|LaneActor { tx, thread }| (tx, thread))
            .unzip();
        // Close every mailbox first so lanes wind down concurrently; the bounded join ensures each env is dropped
        // before this returns (an unjoined lane dropping a Python env after interpreter finalization is a crash).
        drop(senders);
        let deadline = Instant::now() + LANE_JOIN_GRACE;
        for thread in threads {
            if thread.thread().id() == std::thread::current().id() {
                continue;
            }
            while !thread.is_finished() && Instant::now() < deadline {
                std::thread::sleep(LANE_JOIN_POLL);
            }
            if thread.is_finished() {
                let _ = thread.join();
            } else if let Some(abandoned) = &self.abandoned {
                abandoned
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(thread);
            }
        }
    }
}

impl LaneActor {
    fn spawn<E: Env + 'static>(index: usize, mut env: E) -> Self {
        let (tx, rx) = mpsc::channel::<Envelope>();
        let thread = std::thread::Builder::new()
            .name(format!("rlmesh-lane-{index}"))
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(1)
                    .enable_all()
                    .build()
                    .expect("lane runtime");
                // The env's calls run outside any `block_on`: a callback may block, start another runtime or
                // block in place (Zenoh dropping a session), none of which a runtime context allows.
                let _handle = runtime.enter();
                for Envelope { job, enqueued } in rx {
                    let queue_ns = elapsed_ns(enqueued);
                    let stamp = |env: &mut E| {
                        let mut phases = env.take_last_phases();
                        phases.queue_ns = queue_ns;
                        phases
                    };
                    match job {
                        Job::Reset(req, reply) => {
                            let result = futures::executor::block_on(env.reset(req));
                            let phases = stamp(&mut env);
                            let _ = reply.send(result.map(|ok| (ok, phases)));
                        }
                        Job::Step(req, reply) => {
                            let result = futures::executor::block_on(env.step(req));
                            let phases = stamp(&mut env);
                            let _ = reply.send(result.map(|ok| (ok, phases)));
                        }
                        Job::Render(req, reply) => {
                            let result = futures::executor::block_on(env.render(req));
                            let phases = stamp(&mut env);
                            let _ = reply.send(result.map(|ok| (ok, phases)));
                        }
                        Job::Close(req, reply) => {
                            let result = futures::executor::block_on(env.close(req));
                            let phases = stamp(&mut env);
                            let _ = reply.send(result.map(|ok| (ok, phases)));
                        }
                    }
                }
            })
            .expect("spawn lane thread");
        Self { tx, thread }
    }

    fn send(&self, job: Job, lane: usize) -> Result<(), spaces::EnvRuntimeError> {
        self.tx
            .send(Envelope {
                job,
                enqueued: Instant::now(),
            })
            .map_err(|_| {
                spaces::EnvRuntimeError::Runtime(format!("lane {lane} is no longer running"))
            })
    }
}

async fn await_reply<T>(rx: oneshot::Receiver<LaneReply<T>>, lane: usize) -> LaneReply<T> {
    rx.await.unwrap_or_else(|_| {
        Err(spaces::EnvRuntimeError::Runtime(format!(
            "lane {lane} stopped before answering (its thread panicked?)"
        )))
    })
}

pub(super) fn batch_infos(infos: Vec<Option<MetaMap>>, done: Option<&[bool]>) -> Option<MetaMap> {
    if infos.len() == 1 {
        return infos.into_iter().next().flatten();
    }
    if infos.iter().all(Option::is_none) {
        return None;
    }
    let mask = done.map_or_else(|| vec![true; infos.len()], <[bool]>::to_vec);
    let mut batched = BTreeMap::new();
    batched.insert(
        "final_info".to_string(),
        MetaValue::List(
            infos
                .into_iter()
                .map(|info| MetaValue::Map(info.unwrap_or_default()))
                .collect(),
        ),
    );
    batched.insert(
        "_final_info".to_string(),
        MetaValue::List(mask.into_iter().map(MetaValue::Bool).collect()),
    );
    Some(batched)
}

pub(super) fn fold_phases(lane_phases: Vec<EndpointPhases>) -> EndpointPhases {
    if lane_phases.len() == 1 {
        return lane_phases.into_iter().next().unwrap_or_default();
    }
    let mut folded = EndpointPhases::default();
    let mut min_user = u64::MAX;
    let mut max_user = 0;
    for phases in &lane_phases {
        folded.decode_ns += phases.decode_ns;
        folded.user_ns += phases.user_ns;
        folded.encode_ns += phases.encode_ns;
        folded.queue_ns = folded.queue_ns.max(phases.queue_ns);
        min_user = min_user.min(phases.user_ns);
        max_user = max_user.max(phases.user_ns);
    }
    if !lane_phases.is_empty() && max_user > 0 {
        folded.lane_skew_ns = Some(max_user.saturating_sub(min_user));
    }
    folded
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::time::Duration;

    struct SleepyEnv {
        obs_space: spaces::SpaceSpec,
        action_space: spaces::SpaceSpec,
        env_contract: spaces::EnvContract,
        delay: Duration,
        threads: std::sync::Arc<std::sync::Mutex<Vec<std::thread::ThreadId>>>,
        drop_block: Duration,
        blocks_in_place: bool,
        blocks_on_a_runtime: bool,
    }

    impl Drop for SleepyEnv {
        fn drop(&mut self) {
            std::thread::sleep(self.drop_block);
        }
    }

    impl SleepyEnv {
        fn new(
            delay: Duration,
            threads: std::sync::Arc<std::sync::Mutex<Vec<std::thread::ThreadId>>>,
        ) -> Self {
            let obs_space = spaces::spaces::BoxSpaceBuilder::scalar(0.0, 255.0, vec![1])
                .dtype(spaces::DType::Uint8)
                .build()
                .unwrap();
            let action_space = spaces::spaces::BoxSpaceBuilder::scalar(0.0, 1.0, vec![1])
                .dtype(spaces::DType::Uint8)
                .build()
                .unwrap();
            let env_contract = spaces::EnvContract {
                id: "SleepyEnv-v0".to_string(),
                autoreset_mode: Default::default(),
                observation_space: Some(obs_space.clone()),
                action_space: Some(action_space.clone()),
                metadata: None,
                render_mode: String::new(),
                num_envs: 1,
            };
            Self {
                obs_space,
                action_space,
                env_contract,
                delay,
                threads,
                drop_block: Duration::ZERO,
                blocks_in_place: false,
                blocks_on_a_runtime: false,
            }
        }

        fn obs() -> spaces::SpaceValue {
            spaces::SpaceValue::Box(
                spaces::Tensor::from_vec(vec![0], vec![1], spaces::DType::Uint8).unwrap(),
            )
        }
    }

    #[async_trait]
    impl Env for SleepyEnv {
        fn observation_space(&self) -> &spaces::SpaceSpec {
            &self.obs_space
        }

        fn action_space(&self) -> &spaces::SpaceSpec {
            &self.action_space
        }

        fn env_contract(&self) -> &spaces::EnvContract {
            &self.env_contract
        }

        async fn reset(
            &mut self,
            _req: spaces::request::ResetRequest,
        ) -> std::result::Result<spaces::request::ResetResult, spaces::EnvRuntimeError> {
            self.threads
                .lock()
                .unwrap()
                .push(std::thread::current().id());
            Ok(spaces::request::ResetResult {
                observation: Some(Self::obs()),
                info: None,
                episode_id: None,
            })
        }

        async fn step(
            &mut self,
            _req: spaces::request::StepRequest,
        ) -> std::result::Result<spaces::request::StepResult, spaces::EnvRuntimeError> {
            self.threads
                .lock()
                .unwrap()
                .push(std::thread::current().id());
            std::thread::sleep(self.delay);
            if self.blocks_in_place {
                tokio::task::block_in_place(|| ());
            }
            if self.blocks_on_a_runtime {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("a client's runtime")
                    .block_on(tokio::time::sleep(Duration::from_millis(1)));
            }
            Ok(spaces::request::StepResult {
                observation: Some(Self::obs()),
                reward: 1.0,
                terminated: false,
                truncated: false,
                info: None,
            })
        }

        async fn render(
            &mut self,
            _req: RenderRequest,
        ) -> std::result::Result<RenderResult, spaces::EnvRuntimeError> {
            Ok(RenderResult::default())
        }

        async fn close(
            &mut self,
            _req: CloseRequest,
        ) -> std::result::Result<spaces::request::CloseResult, spaces::EnvRuntimeError> {
            Ok(spaces::request::CloseResult)
        }
    }

    #[test]
    fn lanes_must_agree_with_lane_0_on_the_env_contract() {
        let threads = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut odd = SleepyEnv::new(Duration::ZERO, std::sync::Arc::clone(&threads));
        odd.env_contract.metadata = Some(MetaMap::from([(
            "action_role".to_string(),
            MetaValue::String("delta".to_string()),
        )]));
        let envs = vec![
            SleepyEnv::new(Duration::ZERO, std::sync::Arc::clone(&threads)),
            odd,
        ];
        let err = LaneEnv::new(envs)
            .err()
            .expect("mismatched lanes must be refused");
        assert!(
            err.to_string().contains("lane 1 disagrees with lane 0"),
            "{err}"
        );
    }

    #[test]
    fn dropping_the_endpoint_joins_the_lane_threads_that_own_the_envs() {
        let threads = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let envs = (0..2)
            .map(|_| SleepyEnv::new(Duration::ZERO, std::sync::Arc::clone(&threads)))
            .collect();
        let lanes = LaneEnv::new(envs).expect("matching lanes");

        drop(lanes);

        assert_eq!(
            std::sync::Arc::strong_count(&threads),
            1,
            "lane threads must be joined (and their envs dropped) by LaneEnv::drop"
        );
    }

    #[test]
    fn dropping_the_endpoint_gives_up_on_a_wedged_lane_instead_of_hanging() {
        let threads = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut wedged = SleepyEnv::new(Duration::ZERO, std::sync::Arc::clone(&threads));
        wedged.drop_block = Duration::from_secs(600);
        let lanes = LaneEnv::new(vec![wedged]).expect("one lane");

        let started = Instant::now();
        drop(lanes);
        let elapsed = started.elapsed();

        assert!(
            elapsed < LANE_JOIN_GRACE * 2,
            "LaneEnv::drop must give up on a wedged lane after the grace, took {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn a_lane_call_may_block_in_place() {
        // A Python env callback can drop a Zenoh session, which blocks in place; that must not panic the lane.
        let threads = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut env = SleepyEnv::new(Duration::ZERO, threads);
        env.blocks_in_place = true;
        let lanes = LaneEnv::new(vec![env]).expect("one lane");
        let (result, _) = lanes
            .step(0, spaces::request::StepRequest::default())
            .await
            .expect("step");
        assert_eq!(result.reward, 1.0);
    }

    #[tokio::test]
    async fn a_lane_call_may_run_a_runtime_of_its_own() {
        let threads = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut env = SleepyEnv::new(Duration::ZERO, threads);
        env.blocks_on_a_runtime = true;
        let lanes = LaneEnv::new(vec![env]).expect("one lane");
        let (result, _) = lanes
            .step(0, spaces::request::StepRequest::default())
            .await
            .expect("step");
        assert_eq!(result.reward, 1.0);
    }
}
