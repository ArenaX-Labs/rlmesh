//! How a host runs: [`HostConfig`] and the history store's [`HistoryConfig`].

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use crate::wire::{ParticipantId, SessionId, TenantId};
use rlmesh_session_core::auth::{Device, HostIdentity, Pin};

#[cfg(feature = "fleet")]
use crate::authority::SceneConfig;
use crate::authority::SealPolicy;
use crate::clock::{Clock, MonotonicClock, OffsetEstimatorConfig};
use crate::error::SessionError;
use crate::host::OsRng;
#[cfg(feature = "fleet")]
use crate::lobby::LobbyService;
use crate::virtual_clock::VirtualConfig;
#[cfg(doc)]
use crate::{authority::AuthorityConfig, host::SessionHost};

/// What the history store keeps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryConfig {
    /// Keep histories at all.
    pub enabled: bool,
    /// Keep bulk streams (cameras) too. Off by default: a raw camera ring is
    /// large, and windows on bulk streams are rare.
    pub bulk: bool,
    /// Samples kept per stream: the policy-side history cap.
    pub depth: usize,
}

impl Default for HistoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            bulk: false,
            depth: 1024,
        }
    }
}

/// How a host runs.
#[derive(Clone)]
pub struct HostConfig {
    /// The tenant namespace.
    pub tenant: TenantId,
    /// The session.
    pub session: SessionId,
    /// The session clock (wall mode: the host's monotonic clock).
    pub clock: Arc<dyn Clock>,
    /// How long a lease holder's heartbeats may stop before its lease lapses.
    pub lease_grace: Duration,
    /// How long any participant's heartbeats may stop before it is lost (at
    /// least `lease_grace`; a liveliness delete shortens it to that).
    pub presence_grace: Duration,
    /// How often lease expiry is checked: the lapse latency on top of grace.
    pub tick: Duration,
    /// Republish the snapshot this often, so clock mappings stay fresh.
    /// While any lease is held the host also republishes every quarter of
    /// `lease_grace`, so edges see renewed expiries in time.
    pub republish: Duration,
    /// Log every participant's link health this often (`None`: never).
    pub health_log: Option<Duration>,
    /// How participants' clock offsets are filtered.
    pub offset: OffsetEstimatorConfig,
    /// Features this session implements.
    pub supports: Vec<String>,
    /// The history store.
    pub history: HistoryConfig,
    /// Virtual clock mode; `None` is wall mode.
    pub virtual_clock: Option<VirtualConfig>,
    /// The parts whose streams, chunks and lockstep traffic are encrypted
    /// rather than signed: none by default.
    pub seal: SealPolicy,
    /// The host's keys, shared by every session a multi-session host runs:
    /// participants handshake with its static key, and it signs its
    /// broadcasts with its Ed25519 key. Holders of its local secret (the
    /// address's key) attach as device `local`.
    pub identity: Arc<HostIdentity>,
    /// The devices besides `local` it accepts, each with its entry; none by default.
    /// One listed as `local` is another key of that
    /// device, its entry unused: the home machine's on a satellite's run. Change them
    /// at runtime with
    /// [`SessionHost::set_access`], [`SessionHost::admit`] and
    /// [`SessionHost::revoke`].
    pub devices: Vec<Device>,
    /// Robot name → its pin (rlmeshd's `[robots.*]`): the
    /// device, checked at the handshake, and a tethered board, checked at
    /// `DeclareParts`. None by default, so a name binds to its first
    /// embodiment's device for the host instance. Change one at runtime with
    /// [`SessionHost::pin`].
    pub pins: BTreeMap<ParticipantId, Pin>,
    /// Whether a robot that went safe, restarted or came back from a loss
    /// may be leased again at once ([`AuthorityConfig::rearm_robots`]).
    /// Off by default: it waits to be re-armed by name. Other robots never
    /// stop for it either way.
    pub rearm_robots: bool,
    /// Where each placed robot's parts appear in the session's view: one binding drives
    /// every placed robot. Empty by default:
    /// no view.
    pub places: Vec<crate::wire::Place>,
    /// The scene this session runs: its member table and
    /// its identity. `None` by default: an ordinary session.
    /// `crate::scene::Scene::host` builds one from a scene file.
    #[cfg(feature = "fleet")]
    pub scene: Option<SceneConfig>,
    /// The home daemon's lobby, on its `_lobby` session
    /// only: the lobby verbs go to it. `None` by default: the session
    /// refuses them.
    #[cfg(feature = "fleet")]
    pub lobby: Option<Arc<dyn LobbyService>>,
    /// Answers `Keep` (the cockpit's Keep): keeps the session's recent
    /// recording, and says where. `None` by default: refused UNSUPPORTED. A
    /// recording [`crate::server::SessionServer`] sets it.
    pub keep: Option<KeepRecording>,
    /// The host's datagram path: robots that take it get their realtime
    /// parts over UDP. `None` by default: every part rides the session's
    /// links, and the handshake offers nothing. Wall-clock sessions only.
    pub datagram: Option<Arc<crate::host::DatagramRelay>>,
    /// Keep every robot on the datagram path whatever it loses: a setting
    /// for measuring the path (`rlmesh host --board-data udp-only`), where
    /// loss is loss. Off by default: the host moves a robot's realtime parts
    /// back to the session's links when the path is shown broken, and onto
    /// it again once its probes arrive.
    pub datagram_strict: bool,
}

/// Keeps `session`'s recent recording as an incident named `label`: where
/// it went, or why not.
pub type KeepRecording = Arc<dyn Fn(SessionId, &str) -> Result<String, String> + Send + Sync>;

impl HostConfig {
    /// Defaults for `session` in `tenant`: wall mode, a 250 ms lease grace,
    /// a 1 s presence grace, a 10 ms expiry tick, a 500 ms republish, a
    /// 1 s health log, and fresh keys.
    ///
    /// # Panics
    /// If the operating system's random number generator fails.
    pub fn new(tenant: TenantId, session: SessionId) -> Self {
        Self {
            identity: Arc::new(
                HostIdentity::generate(&mut OsRng)
                    .expect("the operating system's random number generator failed"),
            ),
            tenant,
            session,
            clock: MonotonicClock::new().shared(),
            lease_grace: Duration::from_millis(250),
            presence_grace: Duration::from_secs(1),
            tick: Duration::from_millis(10),
            republish: Duration::from_millis(500),
            health_log: Some(Duration::from_secs(1)),
            offset: OffsetEstimatorConfig::default(),
            supports: crate::wire::version::FEATURES
                .iter()
                .map(ToString::to_string)
                .collect(),
            history: HistoryConfig::default(),
            virtual_clock: None,
            seal: SealPolicy::None,
            devices: Vec::new(),
            pins: BTreeMap::new(),
            rearm_robots: false,
            places: Vec::new(),
            #[cfg(feature = "fleet")]
            scene: None,
            #[cfg(feature = "fleet")]
            lobby: None,
            keep: None,
            datagram: None,
            datagram_strict: false,
        }
    }

    pub(super) fn validate(&self) -> Result<(), SessionError> {
        if self.lease_grace.is_zero() || self.presence_grace.is_zero() {
            return Err(SessionError::Config(
                "lease_grace and presence_grace must be positive".into(),
            ));
        }
        if self.tick.is_zero() || self.republish.is_zero() {
            return Err(SessionError::Config(
                "tick and republish must be positive".into(),
            ));
        }
        if self.virtual_clock.is_some_and(|v| v.tick_ns == 0) {
            return Err(SessionError::Config("tick_ns must be positive".into()));
        }
        for (i, device) in self.devices.iter().enumerate() {
            if self.devices[..i]
                .iter()
                .any(|d| d.id == device.id || d.public == device.public)
            {
                return Err(SessionError::Config(format!(
                    "device {} is listed twice",
                    device.id
                )));
            }
        }
        Ok(())
    }
}

impl std::fmt::Debug for HostConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut f = f.debug_struct("HostConfig");
        f.field("tenant", &self.tenant)
            .field("session", &self.session)
            .field("lease_grace", &self.lease_grace)
            .field("presence_grace", &self.presence_grace)
            .field("tick", &self.tick)
            .field("republish", &self.republish)
            .field("health_log", &self.health_log)
            .field("offset", &self.offset)
            .field("supports", &self.supports)
            .field("history", &self.history)
            .field("virtual_clock", &self.virtual_clock)
            .field("seal", &self.seal)
            .field("devices", &self.devices)
            .field("pins", &self.pins)
            .field("rearm_robots", &self.rearm_robots)
            .field("places", &self.places)
            .field("datagram", &self.datagram.as_ref().map(|d| d.port()));
        #[cfg(feature = "fleet")]
        f.field("scene", &self.scene)
            .field("lobby", &self.lobby.is_some());
        f.finish_non_exhaustive()
    }
}
