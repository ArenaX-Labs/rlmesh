//! [`LaneEnv`]: N independent scalar [`Env`]s, one actor thread each.
//!
//! The server calls the author's `make()` N times and hosts the instances as
//! the lanes of one `num_envs = N` endpoint. Each lane is an actor: a thread
//! that owns its env and runs its jobs in order, so a lane's reset and step
//! always execute on the same OS thread (what OSMesa and Vulkan contexts
//! expect) and the lanes run concurrently with no shared lock. `N = 1` is
//! the ordinary scalar server: same wire, same code path.

use std::collections::BTreeMap;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use tokio::sync::oneshot;

use super::Env;
use crate::spaces::{self, CloseRequest, MetaMap, MetaValue, RenderRequest, RenderResult};
use rlmesh_proto::{EndpointPhases, elapsed_ns};

/// The op result plus the lane's phase split for it (queue wait included).
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

/// How long [`LaneEnv::drop`] waits, across all lanes, for the lane threads to
/// finish dropping their envs. Matches the server's teardown grace: a lane that
/// outlasts it is leaked so that shutdown stays bounded.
const LANE_JOIN_GRACE: Duration = Duration::from_secs(5);

/// How often that wait re-checks a lane thread.
const LANE_JOIN_POLL: Duration = Duration::from_millis(1);

/// One lane: its actor thread's mailbox and the thread itself.
struct LaneActor {
    tx: mpsc::Sender<Envelope>,
    thread: std::thread::JoinHandle<()>,
}

/// N scalar environments served as the lanes of one vector endpoint.
pub struct LaneEnv {
    lanes: Vec<LaneActor>,
    observation_space: spaces::SpaceSpec,
    action_space: spaces::SpaceSpec,
    env_contract: spaces::EnvContract,
}

impl LaneEnv {
    /// Host `envs` as lanes `0..envs.len()`, each on its own thread. Spaces
    /// and the contract are served from lane 0, so every lane must carry the
    /// same contract (spaces, metadata, tags): a lane that disagrees is an
    /// error, since the model resolves its adapter against lane 0 alone.
    ///
    /// # Panics
    /// Panics on an empty `envs`.
    pub fn new<E: Env + 'static>(envs: Vec<E>) -> Result<Self, spaces::EnvRuntimeError> {
        let first = envs.first().expect("LaneEnv needs at least one lane");
        let observation_space = first.observation_space().clone();
        let action_space = first.action_space().clone();
        let env_contract = first.env_contract().clone();
        for (index, env) in envs.iter().enumerate().skip(1) {
            if env.env_contract() != &env_contract {
                return Err(spaces::EnvRuntimeError::Runtime(format!(
                    "lane {index} disagrees with lane 0 on the env contract (spaces, metadata, \
                     tags): every lane of one endpoint must be the same make()"
                )));
            }
        }
        let lanes = envs
            .into_iter()
            .enumerate()
            .map(|(index, env)| LaneActor::spawn(index, env))
            .collect();
        Ok(Self {
            lanes,
            observation_space,
            action_space,
            env_contract,
        })
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
        // A lane thread owns its env, so the endpoint should not outlive its
        // threads: dropping the mailbox ends the actor loop, and the join is
        // what guarantees the env is fully dropped before this returns. For an
        // env whose drop runs foreign code (a Python env dropped after the
        // interpreter has finalized) an unjoined lane thread is a crash.
        //
        // Every mailbox closes first, so the lanes wind down concurrently and
        // share one deadline. The wait is bounded because teardown is bounded:
        // a lane still inside a wedged user `close()` is left to the OS rather
        // than hanging shutdown (and process exit) on it forever.
        let (senders, threads): (Vec<_>, Vec<_>) = self
            .lanes
            .drain(..)
            .map(|LaneActor { tx, thread }| (tx, thread))
            .unzip();
        drop(senders);
        let deadline = Instant::now() + LANE_JOIN_GRACE;
        for thread in threads {
            while !thread.is_finished() && Instant::now() < deadline {
                std::thread::sleep(LANE_JOIN_POLL);
            }
            if thread.is_finished() {
                let _ = thread.join();
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
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("lane runtime");
                for Envelope { job, enqueued } in rx {
                    // The wait in the mailbox is this op's queue time.
                    let queue_ns = elapsed_ns(enqueued);
                    let stamp = |env: &mut E| {
                        let mut phases = env.take_last_phases();
                        phases.queue_ns = queue_ns;
                        phases
                    };
                    match job {
                        Job::Reset(req, reply) => {
                            let result = runtime.block_on(env.reset(req));
                            let phases = stamp(&mut env);
                            let _ = reply.send(result.map(|ok| (ok, phases)));
                        }
                        Job::Step(req, reply) => {
                            let result = runtime.block_on(env.step(req));
                            let phases = stamp(&mut env);
                            let _ = reply.send(result.map(|ok| (ok, phases)));
                        }
                        Job::Render(req, reply) => {
                            let result = runtime.block_on(env.render(req));
                            let phases = stamp(&mut env);
                            let _ = reply.send(result.map(|ok| (ok, phases)));
                        }
                        Job::Close(req, reply) => {
                            let result = runtime.block_on(env.close(req));
                            let phases = stamp(&mut env);
                            let _ = reply.send(result.map(|ok| (ok, phases)));
                        }
                    }
                }
                // Mailbox closed: the LaneEnv is gone, drop the env with the thread.
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

/// Batch per-lane info maps. One lane passes its info through unchanged (the
/// scalar shape every consumer already reads). Several lanes use the gym vector
/// convention the wire layer decodes per lane: `final_info` is a list with one
/// map per lane and `_final_info` masks the lanes that completed this step.
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

/// The phase split of a concurrent lane batch: the lanes' own splits summed
/// (their work overlapped, so this is CPU, not wall), the longest queue wait,
/// and `lane_skew_ns` as the spread between the fastest and slowest lane's
/// work. A lone lane's split passes through as is.
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
    use crate::env::WireLaneAdapter;
    use async_trait::async_trait;
    use rlmesh_grpc::wire::encode_batched_partial_values;
    use rlmesh_proto::env::v1::{ResetRequest, StepRequest};
    use rlmesh_proto::has_capability;
    use std::time::Duration;
    use tokio::net::TcpListener;
    use tokio_stream::wrappers::TcpListenerStream;

    /// A scalar env whose step takes `delay` and records the thread it ran on.
    struct SleepyEnv {
        obs_space: spaces::SpaceSpec,
        action_space: spaces::SpaceSpec,
        env_contract: spaces::EnvContract,
        delay: Duration,
        threads: std::sync::Arc<std::sync::Mutex<Vec<std::thread::ThreadId>>>,
        /// How long this env's drop blocks: a stand-in for a wedged `close()`.
        drop_block: Duration,
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
        // The endpoint serves lane 0's contract and the model resolves its
        // adapter against it, so a lane whose contract differs (here: tags in
        // metadata) is a startup error, not a silent adoption of lane 0's.
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
        // Each env lives on its lane thread, so the endpoint's drop has to wait
        // for those threads: a lane still dropping its env after the endpoint
        // is gone is what aborts the interpreter when the env is a Python one.
        // The env holds `threads`, so a strong count back down to 1 proves the
        // lane thread finished dropping it before drop() returned.
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
        // The join above is bounded: a lane stuck in a user `close()` that never
        // returns must not hold the endpoint's drop -- and with it the server's
        // shutdown and process exit -- open forever. Give up on it instead.
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
    async fn lanes_step_concurrently_on_their_own_threads_over_one_join_stream() {
        // Wide enough that even a slow runner's overhead cannot make two
        // overlapping steps look sequential.
        let delay = Duration::from_millis(300);
        let lane_threads: Vec<_> = (0..2)
            .map(|_| std::sync::Arc::new(std::sync::Mutex::new(Vec::new())))
            .collect();
        let envs = lane_threads
            .iter()
            .map(|threads| SleepyEnv::new(delay, std::sync::Arc::clone(threads)))
            .collect();
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = format!("tcp://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(rlmesh_grpc::env::env_service(
                    WireLaneAdapter::new(envs).unwrap(),
                ))
                .serve_with_incoming(TcpListenerStream::new(listener))
                .await
                .unwrap()
        });

        let mut client = rlmesh_grpc::EnvClient::connect(&address).await.unwrap();
        let handshake = client.handshake().await.unwrap();
        assert_eq!(handshake.num_envs, 2);
        assert!(has_capability(
            &handshake.capabilities,
            rlmesh_proto::capabilities::ENV_SUBSET_STEP,
        ));
        let contract =
            rlmesh_grpc::wire::env_contract_from_proto(handshake.env_contract.clone()).unwrap();
        let action_space = contract.action_space.unwrap();

        // One handle per lane on the same session: reset each lane, then step
        // both at once. The steps overlap on the server, so the pair completes
        // in about one env delay rather than two.
        let mut lanes: Vec<rlmesh_grpc::EnvClient> = vec![client.clone(), client];
        for (lane, handle) in lanes.iter_mut().enumerate() {
            let reset = handle
                .reset(ResetRequest {
                    seeds: vec![lane as i64],
                    env_indices: vec![lane as u32],
                    episode_ids: vec![format!("ep-{lane}")],
                    ..Default::default()
                })
                .await
                .unwrap();
            assert!(reset.observation.is_some());
        }
        let action = spaces::SpaceValue::Box(
            spaces::Tensor::from_vec(vec![1], vec![1], spaces::DType::Uint8).unwrap(),
        );
        let started = Instant::now();
        let steps = lanes
            .into_iter()
            .enumerate()
            .map(|(lane, mut handle)| {
                let action =
                    encode_batched_partial_values(std::slice::from_ref(&action), &action_space)
                        .unwrap();
                tokio::spawn(async move {
                    handle
                        .step(StepRequest {
                            action: Some(action),
                            env_indices: vec![lane as u32],
                            episode_ids: vec![format!("ep-{lane}")],
                            ..Default::default()
                        })
                        .await
                        .unwrap()
                })
            })
            .collect::<Vec<_>>();
        for (lane, step) in steps.into_iter().enumerate() {
            let response = step.await.unwrap();
            assert_eq!(response.env_indices, vec![lane as u32]);
            assert_eq!(response.rewards, vec![1.0]);
            assert_eq!(response.terminated_mask.len(), 1);
        }
        let elapsed = started.elapsed();
        assert!(
            elapsed < delay * 2,
            "two lane steps should overlap: took {elapsed:?} for delay {delay:?}"
        );
        // Every op of a lane ran on that lane's own thread, and the two lanes
        // never shared one.
        let threads: Vec<Vec<std::thread::ThreadId>> = lane_threads
            .iter()
            .map(|t| t.lock().unwrap().clone())
            .collect();
        for lane in &threads {
            assert_eq!(lane.len(), 2, "reset + step");
            assert_eq!(lane[0], lane[1], "a lane stays on its thread");
        }
        assert_ne!(threads[0][0], threads[1][0], "lanes have their own threads");
        server.abort();
    }
}
