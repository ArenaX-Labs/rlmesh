//! A lockstep robot in a session: the host is involved only at attach,
//! declare, arm and one lease grant (§12.1); every step after that is the
//! lockstep exchange alone.
//!
//! A robot or a driver names the robot by its participant address
//! (`[host/][session/]robot`, [`rlmesh_session::address`]); the session is
//! placed by the hosting rule, so the first participant may host it in its
//! own process. [`LockstepBinding::address`] is the robot's address with the
//! host's real port, what drivers in other processes connect to.
//!
//! The lockstep class (§10.2) is its own Zenoh session per participant
//! (RealTime, reliable, BLOCK, express), opened only when lockstep is
//! used: it listens on any port and meets its peers through the host's
//! control endpoint. A driver and an env in one process skip it: the
//! driver calls the env's handler directly.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};
use std::time::Duration;

use rlmesh_session::address::{Address, LEASE_GRACE, Placed, SessionAddress, place};
use rlmesh_session::lease::LeaseGroup;
use rlmesh_session::transport::{Endpoints, TransportTuning, zenoh_config};
use rlmesh_session::wire::{
    AuthoritySnapshot, ChannelId, CtlStatus, Epoch, Fingerprint, LeaseToken, Mode, PartSpec,
    ParticipantId, PriorityClass, Role, SessionKeys,
};
use rlmesh_session::{Participant, SessionError, Transport};
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;
use tracing::{info, warn};

use crate::lockstep::client::{Keep, LockstepClient};
use crate::lockstep::error::GymError;
use crate::lockstep::exchange::{
    DirectLink, Gate, Gated, LockstepHandler, LockstepLink, ZenohLink, serve_zenoh,
};
use crate::lockstep::map::{self, GymEnvInfo, GymMap};
use crate::lockstep::record::Tee;
use crate::lockstep::served::{GymEnv, Served};
use crate::lockstep::view::{self, GymView};

/// The notional rate of a Gym robot's parts: lockstep never paces, and
/// `tick_ns` defaults to 1 ms (§11.2).
pub const GYM_HZ: f64 = 1000.0;

/// A lockstep robot's map with what lockstep lets it leave out filled in: a
/// part without `hz` runs once per step (at the notional [`GYM_HZ`]), and a
/// part without `kind` is a custom part of its own, kind
/// `local/{its last path segment}`, declared by its `fields`. A game or a
/// classic-control env so needs no robot vocabulary.
pub fn lockstep_robot_map(robot_json: &str) -> Result<String, GymError> {
    use serde_json::value::RawValue;
    type Decl = indexmap::IndexMap<String, Box<RawValue>>;
    let bad = |e: serde_json::Error| GymError::Config(format!("robot map: {e}"));
    let parts: indexmap::IndexMap<String, Decl> = serde_json::from_str(robot_json).map_err(bad)?;
    let mut out = indexmap::IndexMap::with_capacity(parts.len());
    for (path, mut decl) in parts {
        if !decl.contains_key("kind") {
            let last = path.rsplit('/').next().unwrap_or(&path);
            let kind = serde_json::to_string(&format!("local/{last}")).map_err(bad)?;
            decl.insert("kind".into(), RawValue::from_string(kind).map_err(bad)?);
        }
        if !decl.contains_key("hz") {
            decl.insert(
                "hz".into(),
                RawValue::from_string(GYM_HZ.to_string()).map_err(bad)?,
            );
        }
        out.insert(path, decl);
    }
    serde_json::to_string(&out).map_err(bad)
}

/// Parses a participant address; a malformed one (or an old form, with its
/// upgrade) is a configuration error.
fn participant_address(text: &str) -> Result<Address, GymError> {
    Address::participant(text).map_err(|e| match e {
        SessionError::Config(why) => GymError::Config(why),
        other => GymError::Config(other.to_string()),
    })
}

/// A robot's participant address, its session placed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LockstepAddress {
    /// The session.
    pub session: SessionAddress,
    /// The robot.
    pub robot: ParticipantId,
}

impl LockstepAddress {
    /// Parses the participant address `text` and places its session by the
    /// hosting rule.
    async fn place(text: &str) -> Result<(Self, Placed), GymError> {
        let address = participant_address(text)?;
        let robot = ParticipantId::new(address.participant.clone().unwrap_or_default())
            .map_err(|e| GymError::Config(e.to_string()))?;
        let placed = place(&address, LEASE_GRACE).await?;
        Ok((
            Self {
                session: placed.address.clone(),
                robot,
            },
            placed,
        ))
    }

    /// The robot's key in this process's registry: named sessions recur
    /// across hosts, so the host's endpoint is part of it.
    fn key(&self) -> String {
        format!(
            "{}/{}/{}/{}",
            self.session.base, self.session.tenant, self.session.session, self.robot
        )
    }
}

impl LockstepAddress {
    /// The address with its session's key: the invite drivers connect by
    /// ([`SessionAddress::invite`]).
    pub fn invite(&self) -> String {
        self.session.participant_invite(self.robot.as_str())
    }
}

/// Without the key, for logs and errors.
impl std::fmt::Display for LockstepAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.session, self.robot)
    }
}

/// Lockstep envs served in this process, for the direct link.
fn local() -> &'static Mutex<HashMap<String, Weak<dyn LockstepHandler>>> {
    static LOCAL: OnceLock<Mutex<HashMap<String, Weak<dyn LockstepHandler>>>> = OnceLock::new();
    LOCAL.get_or_init(Mutex::default)
}

fn local_handler(key: &str) -> Option<Weak<dyn LockstepHandler>> {
    local()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(key)
        .filter(|handler| handler.strong_count() > 0)
        .cloned()
}

/// A fresh session on this process's shared in-process host, for a run's
/// local env. The host starts once, on OS-assigned local ports, and lives
/// for the process; each run's session stops when its participants leave,
/// so many runs cost one host.
pub fn local_run_session() -> Result<String, GymError> {
    static HOST: OnceLock<std::result::Result<SocketAddr, String>> = OnceLock::new();
    let base = HOST
        .get_or_init(|| {
            // Its own thread and runtime: a caller inside an async context
            // cannot block on it, and the host outlives every run.
            std::thread::spawn(|| {
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(1)
                    .thread_name("rlmesh-local-host")
                    .enable_all()
                    .build()
                    .map_err(|e| e.to_string())?;
                let placed = runtime
                    .block_on(rlmesh_session::address::host_at(
                        SocketAddr::from(([127, 0, 0, 1], 0)),
                        rlmesh_session::address::DEFAULT_SESSION,
                        LEASE_GRACE,
                        false,
                    ))
                    .map_err(|e| e.to_string())?;
                let base = placed.address.base;
                // Both live for the process.
                std::mem::forget(placed);
                std::mem::forget(runtime);
                Ok(base)
            })
            .join()
            .unwrap_or_else(|_| Err("the in-process host panicked".into()))
        })
        .clone()
        .map_err(GymError::Config)?;
    Ok(format!("{base}/run-{}", uuid_suffix()))
}

/// Joins `session` as `id`.
async fn join(
    session: &SessionAddress,
    id: ParticipantId,
    roles: Vec<Role>,
    priority: PriorityClass,
    timeout: Duration,
) -> Result<Participant, GymError> {
    Ok(session.join(id.as_str(), roles, priority, timeout).await?)
}

fn listen_ip(base: SocketAddr) -> IpAddr {
    if base.ip().is_loopback() {
        base.ip()
    } else {
        IpAddr::from([0, 0, 0, 0])
    }
}

/// Whether `transport` opened links of its own, which its participant
/// closes: not a shared session, nor an in-process host's it rides.
pub(crate) fn owns_links(transport: &Transport) -> bool {
    !(transport.is_shared() || transport.is_in_process())
}

/// The lockstep class session of a participant of the session at
/// `address` (§10.2): the transport's own when it opens no links of its own
/// (close it only when it does), else a link of its own to the host,
/// presenting what the participant's links did (design §5).
pub async fn lockstep_session(
    transport: &Transport,
    address: &SessionAddress,
) -> Result<zenoh::Session, GymError> {
    if !owns_links(transport) {
        return Ok(transport.control().clone());
    }
    let base = address.base;
    let endpoints = Endpoints {
        listen: vec![format!("tcp/{}", SocketAddr::new(listen_ip(base), 0))],
        connect: vec![format!("tcp/{base}")],
    };
    // A client participant's lockstep session is a client too.
    let tuning = TransportTuning {
        client: transport.is_client(),
        link: transport.link().cloned(),
        ..TransportTuning::default()
    };
    let mut config = zenoh_config(&endpoints, &tuning)?;
    // Return once the host is connected rather than waiting out the scouting
    // delay for gossiped peers: a driver probes the queryable it calls
    // (`wait_ready`) before its first request anyway.
    config
        .insert_json5("open/return_conditions/connect_scouted", "false")
        .map_err(|e| GymError::Config(e.to_string()))?;
    zenoh::open(config)
        .await
        .map_err(|e| GymError::Closed(e.to_string()))
}

/// Env-side options.
#[derive(Clone, Debug, Default)]
pub struct ServeOptions {
    /// The workflow edition the env declares (`None`: this build's current).
    pub workflow_edition: Option<String>,
    /// Record the session to this MCAP file (the lockstep tee, §13).
    pub record: Option<PathBuf>,
}

/// Checks that an env with `contract` maps to a robot (§12.5) before
/// anything is served, so a refusal (a Text action, a space with no part
/// layout) reaches the caller at once instead of surfacing later from the
/// serve task.
pub fn check(contract: rlmesh_spaces::EnvContract) -> Result<(), GymError> {
    let info = GymEnvInfo {
        contract,
        workflow_edition: String::new(),
        supported_workflow_editions: Vec::new(),
        capabilities: Vec::new(),
    };
    GymMap::new(info, GYM_HZ).map(|_| ())
}

/// What a served env tells a handshake, for its `gym_env`.
fn env_info(env: &dyn Served, declared: Option<&str>) -> GymEnvInfo {
    let (workflow_edition, supported_workflow_editions) = crate::editions::offer(declared);
    GymEnvInfo {
        contract: env.contract(),
        workflow_edition,
        supported_workflow_editions,
        capabilities: if env.subset_step() {
            vec![map::SUBSET_STEP.to_string()]
        } else {
            Vec::new()
        },
    }
}

/// A lockstep address bound for serving: the session joined or hosted,
/// and the robot attached to it under its name, which no other live
/// participant holds.
pub struct LockstepBinding {
    address: LockstepAddress,
    placed: Placed,
    participant: Participant,
}

impl std::fmt::Debug for LockstepBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LockstepBinding")
            .field("address", &self.address)
            .field("hosted", &self.placed.hosts())
            .finish()
    }
}

/// Binds a robot's participant address for serving: places its session by
/// the hosting rule and attaches the robot, so a name another participant
/// holds is refused here.
pub async fn bind(address: &str) -> Result<LockstepBinding, GymError> {
    let (address, mut placed) = LockstepAddress::place(address).await?;
    let attached = join(
        &address.session,
        address.robot.clone(),
        vec![Role::Embodiment],
        PriorityClass::Scripted,
        Duration::from_secs(10),
    )
    .await;
    match attached {
        Ok(participant) => Ok(LockstepBinding {
            address,
            placed,
            participant,
        }),
        Err(err) => {
            let _ = placed.close().await;
            Err(err)
        }
    }
}

impl LockstepBinding {
    /// The full address drivers connect to: an invite, key included
    /// ([`LockstepAddress::invite`]).
    pub fn address(&self) -> String {
        self.address.invite()
    }

    /// Leaves the session and stops its host, for a binding that never
    /// served (dropping it would leave a host it runs listening).
    pub async fn close(self) {
        let LockstepBinding {
            mut placed,
            participant,
            ..
        } = self;
        participant.close().await;
        let _ = placed.close().await;
    }

    /// Attaches as the robot, declares the env's robot map (§12.5), arms
    /// and serves the env over the lockstep exchange until `shutdown`
    /// fires.
    pub async fn serve(
        self,
        env: Arc<dyn Served>,
        options: ServeOptions,
        shutdown: CancellationToken,
    ) -> Result<(), GymError> {
        let info = env_info(env.as_ref(), options.workflow_edition.as_deref());
        let map = match GymMap::new(info, GYM_HZ) {
            Ok(map) => map,
            Err(err) => {
                self.close().await;
                return Err(err);
            }
        };
        let robot_json = map.robot_json.clone();
        self.serve_robot(&robot_json, options, shutdown, move |declared| {
            let mut map = map;
            map.bind_channels(&declared.parts)?;
            let lanes = map.info.contract.num_envs.max(1) as usize;
            let mut served = GymEnv::new(env, map);
            if let Some(tee) = declared.tee {
                served = served.with_tee(Tee::new(tee, declared.parts, lanes, TICK_NS));
            }
            Ok(Arc::new(served) as Arc<dyn LockstepHandler>)
        })
        .await
    }

    /// Attaches as the robot, declares `robot_json` (a v2 robot map, §3),
    /// arms, and serves the handler `build` makes from the declared parts
    /// over the lockstep exchange until `shutdown` fires. Every step is
    /// checked against the robot's own authority view first.
    pub async fn serve_robot(
        self,
        robot_json: &str,
        options: ServeOptions,
        shutdown: CancellationToken,
        build: impl FnOnce(Declared) -> Result<Arc<dyn LockstepHandler>, GymError>,
    ) -> Result<(), GymError> {
        let LockstepBinding {
            address,
            mut placed,
            participant,
        } = self;
        let result = serve_robot(&address, participant, robot_json, options, shutdown, build).await;
        let _ = placed.close().await;
        result
    }
}

async fn serve_robot(
    address: &LockstepAddress,
    participant: Participant,
    robot_json: &str,
    options: ServeOptions,
    shutdown: CancellationToken,
    build: impl FnOnce(Declared) -> Result<Arc<dyn LockstepHandler>, GymError>,
) -> Result<(), GymError> {
    let participant = Arc::new(participant);
    let snapshot = participant.declare_parts(robot_json).await?;
    let parts = own_parts(&snapshot, &address.robot);
    let commands: Vec<ChannelId> = parts
        .iter()
        .filter(|p| p.command.is_some())
        .map(|p| p.channel)
        .collect();
    let snapshot = match participant.arm().await {
        Ok(snapshot) => snapshot,
        // Already armed (a session another participant armed).
        Err(err) => {
            warn!(%err, "lockstep robot: arm refused; serving the current mode");
            participant.fetch_authority().await?
        }
    };
    let gate = Gate::closed();
    gate.observe(&snapshot, &address.robot, &commands);
    // Aborted however this ends, so it never outlives the serve holding
    // the participant.
    let watcher = {
        let (gate, participant, robot) = (gate.clone(), participant.clone(), address.robot.clone());
        let commands = commands.clone();
        AbortOnDropHandle::new(tokio::spawn(async move {
            let mut watch = participant.watch();
            let mut rearmed = None;
            while watch.changed().await.is_ok() {
                let snapshot = watch.borrow_and_update().clone();
                let Some(snapshot) = snapshot else { continue };
                gate.observe(&snapshot, &robot, &commands);
                // A driver that left (its lease lapsed) puts the session
                // in SAFE. A lockstep robot then re-arms once, so the next
                // driver can attach, as a v1 env server serves one Join
                // session after another. An e-stop stays latched.
                let held = snapshot
                    .leases
                    .iter()
                    .any(|l| commands.contains(&l.channel));
                if snapshot.mode == Mode::Safe
                    && !snapshot.estop_latched
                    && !held
                    && rearmed != Some(snapshot.epoch)
                {
                    rearmed = Some(snapshot.epoch);
                    if let Err(err) = participant.arm().await {
                        warn!(%err, "lockstep robot: re-arm after SAFE refused");
                    }
                }
            }
        }))
    };

    let mut recorder = None;
    let local_key = address.key();
    let started = async {
        let mut tee = None;
        if let Some(path) = &options.record {
            let observer = join(
                &address.session,
                ParticipantId::new(format!("{}.recorder", address.robot))
                    .map_err(|e| GymError::Config(e.to_string()))?,
                vec![Role::Observer],
                PriorityClass::Scripted,
                Duration::from_secs(10),
            )
            .await?;
            let started = rlmesh_session::record::Recorder::start(Arc::new(observer), path)
                .await
                .map_err(|e| GymError::Config(e.to_string()))?;
            tee = Some(started.tee());
            recorder = Some(started);
        }
        let handler: Arc<dyn LockstepHandler> = Gated::new(gate, build(Declared { parts, tee })?);
        local()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(local_key.clone(), Arc::downgrade(&handler));
        let lockstep = lockstep_session(participant.transport(), &address.session).await?;
        Ok((handler, lockstep))
    }
    .await;
    let (handler, lockstep) = match started {
        Ok(started) => started,
        // A robot that never served leaves as one that did, so its name
        // is free to bind again.
        Err(err) => {
            local()
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .remove(&local_key);
            watcher.abort();
            let _ = watcher.await;
            if let Some(recorder) = recorder {
                let _ = recorder.finish().await;
            }
            if let Ok(participant) = Arc::try_unwrap(participant) {
                participant.close().await;
            }
            return Err(err);
        }
    };
    let key = participant.keys().lockstep(&address.robot);
    info!(robot = %address.robot, %key, "lockstep robot serving");
    let served = serve_zenoh(
        lockstep.clone(),
        key,
        handler.clone(),
        shutdown,
        participant.sealer(),
    )
    .await;

    local()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .remove(&local_key);
    // Its clone of the participant goes first, so the robot leaves before
    // its binding stops a host it runs.
    watcher.abort();
    let _ = watcher.await;
    if owns_links(participant.transport()) {
        let _ = lockstep.close().await;
    }
    // The tee writes off the critical path: what it holds goes in first.
    handler.flush().await;
    if let Some(recorder) = recorder
        && let Err(err) = recorder.finish().await
    {
        warn!(%err, "lockstep recording did not finish");
    }
    if let Ok(participant) = Arc::try_unwrap(participant) {
        participant.close().await;
    }
    served
}

/// What a robot's handler is built from: its parts as the host declared
/// them (channel ids assigned) and, when recording, the recorder's tee.
pub struct Declared {
    /// The robot's parts.
    pub parts: Vec<PartSpec>,
    /// The in-process recorder (§13), when recording.
    pub tee: Option<rlmesh_session::record::SampleTee>,
}

/// The notional lockstep tick (§11.2: 1 ms unless the env declares one).
pub const TICK_NS: u64 = 1_000_000;

fn own_parts(snapshot: &AuthoritySnapshot, robot: &ParticipantId) -> Vec<PartSpec> {
    snapshot
        .parts
        .iter()
        .filter(|p| &p.owner == robot)
        .cloned()
        .collect()
}

/// What a driver keeps alive for its run: its participant, its lockstep
/// session, and the session's host when the driver hosts it.
struct Held {
    participant: Participant,
    lockstep: Option<zenoh::Session>,
    placed: Placed,
}

#[async_trait::async_trait]
impl Keep for Held {
    async fn release(self: Box<Self>) {
        let Held {
            participant,
            lockstep,
            mut placed,
        } = *self;
        let released = off_caller(async move {
            if let Err(err) = participant.detach().await {
                tracing::debug!(%err, "lockstep driver: detach");
            }
            if let Some(lockstep) = lockstep {
                let _ = lockstep.close().await;
            }
            let _ = placed.close().await;
        });
        if let Err(err) = released.await {
            tracing::debug!(%err, "lockstep driver: release");
        }
    }
}

/// Driver-side options.
#[derive(Clone, Debug)]
pub struct ConnectOptions {
    /// The driver's participant id (default: a fresh `driver-*` id).
    pub id: Option<String>,
    /// How long to wait for the host and the robot.
    pub timeout: Duration,
    /// Call an env served in this process directly (no serialisation, §12.3);
    /// `false` takes the lockstep session even then.
    pub direct: bool,
}

impl Default for ConnectOptions {
    fn default() -> Self {
        Self {
            id: None,
            timeout: Duration::from_secs(10),
            direct: true,
        }
    }
}

/// A driver attached to a lockstep robot's session.
struct Attached {
    participant: Participant,
    address: LockstepAddress,
    parts: Vec<PartSpec>,
    placed: Placed,
}

/// Attaches a driver and waits for the robot's parts (`ready` decides).
async fn attach_driver(
    address: &str,
    options: &ConnectOptions,
    ready: impl FnMut(&AuthoritySnapshot) -> bool,
) -> Result<Attached, GymError> {
    let (address, placed) = LockstepAddress::place(address).await?;
    let id = match &options.id {
        Some(id) => id.clone(),
        None => format!("driver-{}", uuid_suffix()),
    };
    let participant = join(
        &address.session,
        ParticipantId::new(id).map_err(|e| GymError::Config(e.to_string()))?,
        vec![Role::Controller],
        PriorityClass::Model,
        options.timeout,
    )
    .await?;
    let snapshot = participant
        .wait_for("the lockstep robot's parts", options.timeout, ready)
        .await?;
    let parts = own_parts(&snapshot, &address.robot);
    Ok(Attached {
        participant,
        address,
        parts,
        placed,
    })
}

impl Attached {
    /// Declares a binding over `channels` (identity, no delivery contracts:
    /// lockstep publishes no streams), takes one lease on them and opens
    /// the link.
    async fn lease(
        self,
        table: String,
        fingerprint: Option<Fingerprint>,
        channels: Vec<ChannelId>,
        direct: bool,
        timeout: Duration,
    ) -> Result<(Arc<dyn LockstepLink>, Epoch, LeaseToken, Held), GymError> {
        let fingerprint = fingerprint.unwrap_or_else(|| {
            let hash = blake3::hash(table.as_bytes());
            let mut bytes = [0_u8; 8];
            bytes.copy_from_slice(&hash.as_bytes()[..8]);
            Fingerprint(bytes)
        });
        let participant = self.participant;
        // Requested through the holder's lease state machine; the lease ends
        // with the driver's detach.
        let mut group = LeaseGroup::new(participant.id(), &channels, 0);
        // The robot may be re-arming after the previous driver left (SAFE,
        // then a fresh epoch): wait for an armed session and retry a verb
        // that raced the new epoch.
        let deadline = tokio::time::Instant::now() + timeout;
        let leases = loop {
            participant
                .wait_for("an armed lockstep session", timeout, |s| {
                    matches!(s.mode, Mode::Armed | Mode::Active)
                })
                .await?;
            let granted = async {
                participant
                    .declare_binding(fingerprint, &table, channels.clone(), 0.0)
                    .await?;
                let request = group.request(
                    participant.binding_fingerprint(),
                    None,
                    participant.clock().now_ns(),
                );
                let reply = participant.command(request).await?;
                group.on_reply(
                    &reply,
                    participant.clock_estimate().as_ref(),
                    participant.clock().now_ns(),
                );
                reply.result?;
                Ok::<_, SessionError>(reply.leases)
            }
            .await;
            match granted {
                Ok(leases) => break leases,
                Err(SessionError::Rejected(err))
                    if matches!(
                        err.status,
                        CtlStatus::StaleEpoch | CtlStatus::InvalidTransition
                    ) && tokio::time::Instant::now() < deadline =>
                {
                    participant.fetch_authority().await?;
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                Err(err) => return Err(err.into()),
            }
        };
        let token = leases.first().map(|l| l.token).unwrap_or_default();
        let epoch = participant.authority().map(|s| s.epoch).unwrap_or_default();
        let local_key = self.address.key();
        let local = direct.then(|| local_handler(&local_key)).flatten();
        let (link, lockstep): (Arc<dyn LockstepLink>, _) = match local {
            Some(handler) => (Arc::new(DirectLink(handler)), None),
            None => {
                let session =
                    lockstep_session(participant.transport(), &self.address.session).await?;
                let keys: &SessionKeys = participant.keys();
                let link = ZenohLink::new(
                    session.clone(),
                    keys.lockstep(&self.address.robot),
                    participant.sealer(),
                    self.address.robot.clone(),
                    token,
                );
                link.wait_ready(timeout).await?;
                let owned = owns_links(participant.transport()).then_some(session);
                (Arc::new(link), owned)
            }
        };
        Ok((
            link,
            epoch,
            token,
            Held {
                participant,
                lockstep,
                placed: self.placed,
            },
        ))
    }
}

/// What a driver drives: a Gym env's `LockstepClient`, or a v2-native
/// robot's [`GymView`].
pub enum Driven {
    /// A Gym env (its robot declares `gym_env`).
    Gym(LockstepClient),
    /// Any other robot, keyed by part path.
    View(GymView),
}

/// Attaches a driver to the robot at the participant address `address`
/// and waits for its parts.
async fn attach_robot(
    address: &str,
    options: &ConnectOptions,
) -> Result<(Attached, ParticipantId), GymError> {
    let robot = ParticipantId::new(
        participant_address(address)?
            .participant
            .unwrap_or_default(),
    )
    .map_err(|e| GymError::Config(e.to_string()))?;
    let attached = attach_driver(address, options, |s| {
        s.parts.iter().any(|p| p.owner == robot)
    })
    .await?;
    Ok((attached, robot))
}

fn is_gym_env(parts: &[PartSpec]) -> bool {
    parts.iter().any(|p| {
        p.attrs
            .contains_key(rlmesh_session::spec::normalise::attrs::GYM_ENV)
    })
}

/// Connects a driver to the robot at the participant address `address`: a
/// Gym env, rebuilt from its declaration,
/// as a `LockstepClient`; any other robot as a [`GymView`] keyed by part
/// path.
pub async fn connect_driver(address: &str, options: ConnectOptions) -> Result<Driven, GymError> {
    let address = address.to_string();
    off_caller(async move {
        let (attached, robot) = attach_robot(&address, &options).await?;
        if is_gym_env(&attached.parts) {
            return gym_client(attached, robot, options).await.map(Driven::Gym);
        }
        view_of(attached, robot, None, options)
            .await
            .map(Driven::View)
    })
    .await?
}

/// Connects a driver to the lockstep Gym env at `address`: attaches as a
/// controller, rebuilds the env's map from its declaration, takes one
/// lease on its commandable parts and opens the link (direct when the env
/// is in this process).
pub async fn connect(address: &str, options: ConnectOptions) -> Result<LockstepClient, GymError> {
    let address = address.to_string();
    off_caller(async move {
        let (attached, robot) = attach_robot(&address, &options).await?;
        gym_client(attached, robot, options).await
    })
    .await?
}

/// Runs `work` on the library's multi-thread runtime: zenoh opens and
/// closes a session with `block_in_place`, which a caller on a
/// current-thread runtime (`#[tokio::test]`,
/// `#[tokio::main(flavor = "current_thread")]`) cannot run.
async fn off_caller<T: Send + 'static>(
    work: impl Future<Output = T> + Send + 'static,
) -> Result<T, GymError> {
    match crate::run::drive::runtime().spawn(work).await {
        Ok(result) => Ok(result),
        Err(err) if err.is_panic() => std::panic::resume_unwind(err.into_panic()),
        Err(err) => Err(GymError::Closed(err.to_string())),
    }
}

async fn gym_client(
    attached: Attached,
    robot: ParticipantId,
    options: ConnectOptions,
) -> Result<LockstepClient, GymError> {
    let info = attached
        .parts
        .iter()
        .find_map(|p| p.attrs.get(rlmesh_session::spec::normalise::attrs::GYM_ENV))
        .ok_or_else(|| GymError::Map(format!("robot {robot} is not a Gym env")))
        .and_then(|text| GymEnvInfo::from_attr(text))?;
    let mut map = GymMap::new(info, GYM_HZ)?;
    map.bind_channels(&attached.parts)?;
    let channels: Vec<ChannelId> = map.action.parts.iter().map(|p| p.spec.channel).collect();
    let table = format!("{{\"lockstep\":{:?}}}", robot.to_string());
    // Models the client invites join by it: the key goes with it.
    let session = attached.address.session.invite();
    let (link, epoch, token, held) = attached
        .lease(table, None, channels, options.direct, options.timeout)
        .await?;
    Ok(
        LockstepClient::with_keep(link, map, epoch, token, Some(Box::new(held)))
            .in_session(session),
    )
}

/// Connects a Gym view (§12.6) to the v2-native lockstep robot at
/// `address`, through `policy` (a v2 policy map, JSON; `None`: every part
/// by path).
pub async fn connect_view(
    address: &str,
    policy: Option<(&rlmesh_session::spec::Policy, &str)>,
    options: ConnectOptions,
) -> Result<GymView, GymError> {
    let (attached, robot) = attach_robot(address, &options).await?;
    view_of(attached, robot, policy, options).await
}

async fn view_of(
    attached: Attached,
    robot: ParticipantId,
    policy: Option<(&rlmesh_session::spec::Policy, &str)>,
    options: ConnectOptions,
) -> Result<GymView, GymError> {
    let parts = attached.parts.clone();
    if policy.is_none() && is_gym_env(&parts) {
        return Err(GymError::Config(format!(
            "robot {robot} serves a Gym env: drive it with RemoteEnv(\"{}\"), which keeps its \
             original spaces, or give the view a policy",
            attached.address
        )));
    }
    let (table, fingerprint, channels) = match policy {
        Some((policy, name)) => {
            let binding =
                rlmesh_session::spec::bind(policy, &parts, &view::run_options(name, &robot))
                    .map_err(|e| GymError::Map(e.to_string()))?;
            (
                binding.table_text(),
                Some(binding.fingerprint),
                binding.channels(),
            )
        }
        None => (
            format!("{{\"lockstep_view\":{:?}}}", robot.to_string()),
            None,
            parts
                .iter()
                .filter(|p| p.command.is_some())
                .map(|p| p.channel)
                .collect(),
        ),
    };
    let (link, epoch, token, held) = attached
        .lease(
            table,
            fingerprint,
            channels,
            options.direct,
            options.timeout,
        )
        .await?;
    GymView::new(
        link,
        &parts,
        policy.map(|(p, name)| (p, view::run_options(name, &robot))),
        epoch,
        token,
        Some(Box::new(held)),
    )
}

pub(crate) fn uuid_suffix() -> String {
    let id = rlmesh_session::wire::SessionId::new().to_string();
    id.chars()
        .filter(char::is_ascii_alphanumeric)
        .rev()
        .take(12)
        .collect()
}
