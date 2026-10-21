//! A lockstep robot in a session: the host is involved only at attach,
//! declare, arm and one lease grant (§12.1); every step after that is the
//! lockstep exchange alone.
//!
//! Addresses:
//!
//! - `lockstep://{ip}:{port}/{tenant}/{session}/{robot}`: the robot `robot`
//!   in the session whose host listens on `ip:port` (the address a v2
//!   `Host` prints, plus the robot).
//! - `lockstep://{ip}:{port}/{robot}` (env side only): host a new session
//!   here, on `ip:port` (port 0: any); [`LockstepBinding::address`] is
//!   then the full address drivers connect to.
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

use rlmesh_grpc::env::Environment;
use rlmesh_session::transport::{Endpoints, TransportTuning, zenoh_config};
use rlmesh_session::wire::{
    AuthoritySnapshot, ChannelId, CtlStatus, Epoch, Fingerprint, LeaseToken, Mode, PartSpec,
    ParticipantId, PriorityClass, Role, SessionId, SessionKeys, TenantId,
};
use rlmesh_session::{
    HostConfig, Participant, ParticipantConfig, SessionError, SessionHost, SystemMonotonicClock,
    Transport, TransportConfig,
};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::compat::{GymEnv, LockstepEnv, env_info};
use crate::error::GymError;
use crate::lockstep::{
    DirectLink, Gate, Gated, LockstepHandler, LockstepLink, ZenohLink, serve_zenoh,
};
use crate::map::{GymEnvInfo, GymMap};
use crate::record::Tee;
use crate::view::{self, GymView};

/// The address scheme.
pub const SCHEME: &str = "lockstep://";

/// Whether `address` names a lockstep robot.
pub fn is_lockstep_address(address: &str) -> bool {
    address.starts_with(SCHEME)
}

/// The notional rate of a Gym robot's parts: lockstep never paces, and
/// `tick_ns` defaults to 1 ms (§11.2).
pub const GYM_HZ: f64 = 1000.0;

/// A parsed lockstep address.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LockstepAddress {
    /// The host's base (control) endpoint.
    pub base: SocketAddr,
    /// The session, or `None` to host one.
    pub session: Option<(TenantId, SessionId)>,
    /// The robot.
    pub robot: ParticipantId,
}

impl LockstepAddress {
    /// Parses `lockstep://{ip}:{port}[/{tenant}/{session}]/{robot}`.
    pub fn parse(text: &str) -> Result<Self, GymError> {
        let bad = |why: &str| {
            GymError::Config(format!(
                "lockstep address {text:?}: {why} (expected \
                 lockstep://{{ip}}:{{port}}/{{tenant}}/{{session}}/{{robot}})"
            ))
        };
        let rest = text
            .strip_prefix(SCHEME)
            .ok_or_else(|| bad("no lockstep:// scheme"))?;
        let parts: Vec<&str> = rest.split('/').collect();
        let (base, session, robot) = match parts.as_slice() {
            [base, robot] => (*base, None, *robot),
            [base, tenant, session, robot] => (*base, Some((*tenant, *session)), *robot),
            _ => return Err(bad("two or four parts")),
        };
        Ok(Self {
            base: base.parse().map_err(|_| bad("not ip:port"))?,
            session: session
                .map(|(t, s)| -> Result<_, GymError> {
                    Ok((
                        TenantId::new(t).map_err(|e| bad(&e.to_string()))?,
                        s.parse().map_err(|_| bad("not a session id"))?,
                    ))
                })
                .transpose()?,
            robot: ParticipantId::new(robot).map_err(|e| bad(&e.to_string()))?,
        })
    }

    fn key(&self) -> Option<String> {
        self.session
            .as_ref()
            .map(|(tenant, session)| format!("{tenant}/{session}/{}", self.robot))
    }
}

impl std::fmt::Display for LockstepAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.session {
            Some((tenant, session)) => {
                write!(f, "{SCHEME}{}/{tenant}/{session}/{}", self.base, self.robot)
            }
            None => write!(f, "{SCHEME}{}/{}", self.base, self.robot),
        }
    }
}

/// Lockstep envs served in this process, for the direct link.
fn local() -> &'static Mutex<HashMap<String, Weak<dyn LockstepHandler>>> {
    static LOCAL: OnceLock<Mutex<HashMap<String, Weak<dyn LockstepHandler>>>> = OnceLock::new();
    LOCAL.get_or_init(Mutex::default)
}

fn local_handler(key: &str) -> Option<Arc<dyn LockstepHandler>> {
    local()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(key)
        .and_then(Weak::upgrade)
}

/// Opens a participant's transport (listening on the host's interface, any
/// port) and attaches.
async fn join(
    base: SocketAddr,
    tenant: &TenantId,
    session: SessionId,
    id: ParticipantId,
    roles: Vec<Role>,
    priority: PriorityClass,
    timeout: Duration,
) -> Result<Participant, GymError> {
    let listen_ip = listen_ip(base);
    let transport = Transport::open(&TransportConfig::per_class_tcp(
        Some(SocketAddr::new(listen_ip, 0)),
        Some(base),
    )?)
    .await?;
    let mut config = ParticipantConfig::new(tenant.clone(), session, id, roles, priority);
    config.clock = SystemMonotonicClock.shared();
    config.host_timeout = timeout;
    Ok(Participant::attach(transport, config).await?)
}

fn listen_ip(base: SocketAddr) -> IpAddr {
    if base.ip().is_loopback() {
        base.ip()
    } else {
        IpAddr::from([0, 0, 0, 0])
    }
}

/// The lockstep class session of a participant (§10.2).
pub async fn lockstep_session(
    transport: &Transport,
    base: SocketAddr,
) -> Result<zenoh::Session, GymError> {
    if transport.is_shared() {
        return Ok(transport.control().clone());
    }
    let endpoints = Endpoints {
        listen: vec![format!("tcp/{}", SocketAddr::new(listen_ip(base), 0))],
        connect: vec![format!("tcp/{base}")],
    };
    let config = zenoh_config(&endpoints, &TransportTuning::default())?;
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
    /// How long to wait for the host.
    pub timeout: Option<Duration>,
}

/// A lockstep address bound for serving: the session joined or hosted.
pub struct LockstepBinding {
    address: LockstepAddress,
    hosted: Option<(SessionHost, Transport)>,
}

impl std::fmt::Debug for LockstepBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LockstepBinding")
            .field("address", &self.address)
            .field("hosted", &self.hosted.is_some())
            .finish()
    }
}

/// Binds an env-side lockstep address: hosts the session when the address
/// names none.
pub async fn bind(address: &str) -> Result<LockstepBinding, GymError> {
    let mut address = LockstepAddress::parse(address)?;
    let mut hosted = None;
    if address.session.is_none() {
        let (transport, base) = if address.base.port() == 0 {
            rlmesh_session::transport::open_local_listeners(address.base.ip()).await?
        } else {
            let transport =
                Transport::open(&TransportConfig::per_class_tcp(Some(address.base), None)?).await?;
            (transport, address.base)
        };
        let tenant = TenantId::new("local").map_err(|e| GymError::Config(e.to_string()))?;
        let session = SessionId::new();
        let mut config = HostConfig::new(tenant.clone(), session);
        config.clock = SystemMonotonicClock.shared();
        config.health_log = None;
        let host = SessionHost::start(transport.clone(), config).await?;
        address.base = base;
        address.session = Some((tenant, session));
        hosted = Some((host, transport));
    }
    Ok(LockstepBinding { address, hosted })
}

impl LockstepBinding {
    /// The full address drivers connect to.
    pub fn address(&self) -> String {
        self.address.to_string()
    }

    /// Attaches as the robot, declares the env's robot map (§12.5), arms
    /// and serves the env over the lockstep exchange until `shutdown`
    /// fires.
    pub async fn serve<E: Environment + 'static>(
        self,
        env: Arc<E>,
        options: ServeOptions,
        shutdown: CancellationToken,
    ) -> Result<(), GymError> {
        let info = env_info(env.as_ref(), options.workflow_edition.as_deref());
        let map = GymMap::new(info, GYM_HZ)?;
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
        let result = self.serve_inner(robot_json, options, shutdown, build).await;
        if let Some((host, transport)) = self.hosted {
            host.shutdown().await;
            let _ = transport.close().await;
        }
        result
    }

    async fn serve_inner(
        &self,
        robot_json: &str,
        options: ServeOptions,
        shutdown: CancellationToken,
        build: impl FnOnce(Declared) -> Result<Arc<dyn LockstepHandler>, GymError>,
    ) -> Result<(), GymError> {
        let address = &self.address;
        let (tenant, session) = address
            .session
            .clone()
            .ok_or_else(|| GymError::Config("no session".into()))?;
        let participant = Arc::new(
            join(
                address.base,
                &tenant,
                session,
                address.robot.clone(),
                vec![Role::Embodiment],
                PriorityClass::Scripted,
                options.timeout.unwrap_or(Duration::from_secs(10)),
            )
            .await?,
        );
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
        let watcher = {
            let (gate, participant, robot) =
                (gate.clone(), participant.clone(), address.robot.clone());
            let commands = commands.clone();
            tokio::spawn(async move {
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
            })
        };

        let mut recorder = None;
        let mut tee = None;
        if let Some(path) = &options.record {
            let observer = join(
                address.base,
                &tenant,
                session,
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
        let local_key = address.key().unwrap_or_default();
        local()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(local_key.clone(), Arc::downgrade(&handler));
        let lockstep = lockstep_session(participant.transport(), address.base).await?;
        let key = participant.keys().lockstep(&address.robot);
        info!(robot = %address.robot, %key, "lockstep robot serving");
        let served = serve_zenoh(lockstep.clone(), key, handler.clone(), shutdown).await;

        local()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&local_key);
        watcher.abort();
        if !participant.transport().is_shared() {
            let _ = lockstep.close().await;
        }
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

/// What a driver keeps alive for its run.
struct Held {
    _participant: Participant,
    _lockstep: Option<zenoh::Session>,
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
}

/// Attaches a driver and waits for the robot's parts (`ready` decides).
async fn attach_driver(
    address: &str,
    options: &ConnectOptions,
    ready: impl FnMut(&AuthoritySnapshot) -> bool,
) -> Result<Attached, GymError> {
    let address = LockstepAddress::parse(address)?;
    let (tenant, session) = address.session.clone().ok_or_else(|| {
        GymError::Config(format!(
            "lockstep address {address}: a driver needs the session (the robot's server \
             prints its full address)"
        ))
    })?;
    let id = match &options.id {
        Some(id) => id.clone(),
        None => format!("driver-{}", uuid_suffix()),
    };
    let participant = join(
        address.base,
        &tenant,
        session,
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
                participant
                    .grant_leases(channels.clone(), Vec::new(), None)
                    .await
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
        let local_key = self.address.key().unwrap_or_default();
        let local = direct.then(|| local_handler(&local_key)).flatten();
        let (link, lockstep): (Arc<dyn LockstepLink>, _) = match local {
            Some(handler) => (Arc::new(DirectLink(handler)), None),
            None => {
                let session = lockstep_session(participant.transport(), self.address.base).await?;
                let keys: &SessionKeys = participant.keys();
                let link = ZenohLink::new(session.clone(), keys.lockstep(&self.address.robot));
                link.wait_ready(timeout).await?;
                (Arc::new(link), Some(session))
            }
        };
        Ok((
            link,
            epoch,
            token,
            Held {
                _participant: participant,
                _lockstep: lockstep,
            },
        ))
    }
}

/// Connects a driver to the lockstep Gym env at `address`: attaches as a
/// controller, rebuilds the env's map from its declaration, takes one
/// lease on its commandable parts and opens the link (direct when the env
/// is in this process).
pub async fn connect(address: &str, options: ConnectOptions) -> Result<LockstepEnv, GymError> {
    let robot = LockstepAddress::parse(address)?.robot;
    let attached = attach_driver(address, &options, |s| {
        s.parts.iter().any(|p| {
            p.owner == robot
                && p.attrs
                    .contains_key(rlmesh_session::spec::normalise::attrs::GYM_ENV)
        })
    })
    .await?;
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
    let (link, epoch, token, held) = attached
        .lease(table, None, channels, options.direct, options.timeout)
        .await?;
    Ok(LockstepEnv::with_keep(
        link,
        map,
        epoch,
        token,
        Some(Box::new(held)),
    ))
}

/// Connects a Gym view (§12.6) to the v2-native lockstep robot at
/// `address`, through `policy` (a v2 policy map, JSON; `None`: every part
/// by path).
pub async fn connect_view(
    address: &str,
    policy: Option<(&rlmesh_session::spec::Policy, &str)>,
    options: ConnectOptions,
) -> Result<GymView, GymError> {
    let robot = LockstepAddress::parse(address)?.robot;
    let attached = attach_driver(address, &options, |s| {
        s.parts.iter().any(|p| p.owner == robot)
    })
    .await?;
    view_of(attached, robot, policy, options).await
}

/// [`connect_view`] by session address (`{ip}:{port}/{tenant}/{session}`,
/// a v2 host's address) and robot; `None` takes the session's only robot.
pub async fn connect_view_in(
    session: &str,
    robot: Option<&str>,
    policy: Option<(&rlmesh_session::spec::Policy, &str)>,
    options: ConnectOptions,
) -> Result<GymView, GymError> {
    if let Some(robot) = robot {
        return connect_view(&format!("{SCHEME}{session}/{robot}"), policy, options).await;
    }
    // Any robot: attach under a placeholder, then take the one owner.
    let placeholder = format!("{SCHEME}{session}/any");
    let mut attached = attach_driver(&placeholder, &options, |s| !s.parts.is_empty()).await?;
    let snapshot = attached
        .participant
        .authority()
        .ok_or_else(|| GymError::Config("no authority view".into()))?;
    let mut owners: Vec<ParticipantId> = snapshot.parts.iter().map(|p| p.owner.clone()).collect();
    owners.sort();
    owners.dedup();
    let robot = match owners.as_slice() {
        [one] => one.clone(),
        many => {
            return Err(GymError::Config(format!(
                "the session has {} robots ({}): name one",
                many.len(),
                many.iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
    };
    attached.address.robot = robot.clone();
    attached.parts = own_parts(&snapshot, &robot);
    view_of(attached, robot, policy, options).await
}

async fn view_of(
    attached: Attached,
    robot: ParticipantId,
    policy: Option<(&rlmesh_session::spec::Policy, &str)>,
    options: ConnectOptions,
) -> Result<GymView, GymError> {
    let parts = attached.parts.clone();
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

fn uuid_suffix() -> String {
    let id = SessionId::new().to_string();
    id.chars()
        .filter(char::is_ascii_alphanumeric)
        .rev()
        .take(12)
        .collect()
}
