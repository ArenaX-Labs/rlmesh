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
use rlmesh_session::{
    HostConfig, Participant, ParticipantConfig, SessionHost, SystemMonotonicClock, Transport,
    TransportConfig,
};
use rlmesh_wire::{
    AuthoritySnapshot, ChannelId, Fingerprint, PartSpec, ParticipantId, PriorityClass, Role,
    SessionId, SessionKeys, TenantId,
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

    /// Attaches as the robot, declares the env's robot map, arms and
    /// serves the lockstep exchange until `shutdown` fires.
    pub async fn serve<E: Environment + 'static>(
        self,
        env: Arc<E>,
        options: ServeOptions,
        shutdown: CancellationToken,
    ) -> Result<(), GymError> {
        let result = self.serve_inner(env, options, shutdown).await;
        if let Some((host, transport)) = self.hosted {
            host.shutdown().await;
            let _ = transport.close().await;
        }
        result
    }

    async fn serve_inner<E: Environment + 'static>(
        &self,
        env: Arc<E>,
        options: ServeOptions,
        shutdown: CancellationToken,
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
        let info = env_info(env.as_ref(), options.workflow_edition.as_deref());
        let mut map = GymMap::new(info, GYM_HZ)?;
        let snapshot = participant.declare_parts(&map.robot_json).await?;
        map.bind_channels(&own_parts(&snapshot, &address.robot))?;
        let commands: Vec<ChannelId> = map.action.parts.iter().map(|p| p.spec.channel).collect();
        let snapshot = match participant.arm().await {
            Ok(snapshot) => snapshot,
            // Already armed (a session another participant armed).
            Err(err) => {
                warn!(%err, "lockstep env: arm refused; serving the current mode");
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
                while watch.changed().await.is_ok() {
                    let snapshot = watch.borrow_and_update().clone();
                    if let Some(snapshot) = snapshot {
                        gate.observe(&snapshot, &robot, &commands);
                    }
                }
            })
        };

        let mut recorder = None;
        let mut served = GymEnv::new(env, map.clone());
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
            let started = rlmesh_record::Recorder::start(Arc::new(observer), path)
                .await
                .map_err(|e| GymError::Config(e.to_string()))?;
            let parts: Vec<PartSpec> = map.parts().map(|p| p.spec.clone()).collect();
            let lanes = map.info.contract.num_envs.max(1) as usize;
            served = served.with_tee(Tee::new(started.tee(), parts, lanes, 1_000_000));
            recorder = Some(started);
        }

        let handler: Arc<dyn LockstepHandler> = Gated::new(gate, Arc::new(served));
        let local_key = address.key().unwrap_or_default();
        local()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(local_key.clone(), Arc::downgrade(&handler));
        let lockstep = lockstep_session(participant.transport(), address.base).await?;
        let key = participant.keys().lockstep(&address.robot);
        info!(robot = %address.robot, %key, "lockstep env serving");
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

/// Connects a driver to the lockstep robot at `address`: attaches as a
/// controller, rebuilds the env's map from its declaration, takes one
/// lease on its commandable parts and opens the link (direct when the env
/// is in this process).
pub async fn connect(address: &str, options: ConnectOptions) -> Result<LockstepEnv, GymError> {
    let address = LockstepAddress::parse(address)?;
    let (tenant, session) = address.session.clone().ok_or_else(|| {
        GymError::Config(format!(
            "lockstep address {address}: a driver needs the session (the env server prints \
             its full address)"
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
    let robot = address.robot.clone();
    let snapshot = participant
        .wait_for("the lockstep robot's parts", options.timeout, |s| {
            s.parts.iter().any(|p| {
                p.owner == robot && p.attrs.contains_key(rlmesh_spec::normalise::attrs::GYM_ENV)
            })
        })
        .await?;
    let parts = own_parts(&snapshot, &robot);
    let info = parts
        .iter()
        .find_map(|p| p.attrs.get(rlmesh_spec::normalise::attrs::GYM_ENV))
        .ok_or_else(|| GymError::Map(format!("robot {robot} is not a Gym env")))
        .and_then(|text| GymEnvInfo::from_attr(text))?;
    let mut map = GymMap::new(info, GYM_HZ)?;
    map.bind_channels(&parts)?;
    let channels: Vec<ChannelId> = map.action.parts.iter().map(|p| p.spec.channel).collect();
    let table = format!("{{\"lockstep\":{:?}}}", robot.to_string());
    let hash = blake3::hash(table.as_bytes());
    let mut fingerprint = [0_u8; 8];
    fingerprint.copy_from_slice(&hash.as_bytes()[..8]);
    participant
        .declare_binding(Fingerprint(fingerprint), &table, channels.clone(), 0.0)
        .await?;
    let leases = participant.grant_leases(channels, Vec::new(), None).await?;
    let token = leases.first().map(|l| l.token).unwrap_or_default();
    let epoch = participant.authority().map(|s| s.epoch).unwrap_or_default();

    let local_key = address.key().unwrap_or_default();
    let local = options.direct.then(|| local_handler(&local_key)).flatten();
    let (link, lockstep): (Arc<dyn LockstepLink>, _) = match local {
        Some(handler) => (Arc::new(DirectLink(handler)), None),
        None => {
            let session = lockstep_session(participant.transport(), address.base).await?;
            let keys: &SessionKeys = participant.keys();
            let link = ZenohLink::new(session.clone(), keys.lockstep(&robot));
            (Arc::new(link), Some(session))
        }
    };
    let held = Held {
        _participant: participant,
        _lockstep: lockstep,
    };
    Ok(LockstepEnv::with_keep(
        link,
        map,
        epoch,
        token,
        Some(Box::new(held)),
    ))
}

fn uuid_suffix() -> String {
    let id = SessionId::new().to_string();
    id.chars()
        .filter(char::is_ascii_alphanumeric)
        .rev()
        .take(12)
        .collect()
}
