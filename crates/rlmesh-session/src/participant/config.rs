//! How a participant attaches: [`ParticipantConfig`].

use std::sync::Arc;
use std::time::Duration;

use crate::wire::{ParticipantId, PriorityClass, Reason, Role, SessionId, TenantId};
use rlmesh_session_core::auth::Credential;

use crate::clock::{Clock, MonotonicClock};
#[cfg(doc)]
use crate::participant::Participant;
use crate::participant::edge::LocalLatch;

/// How a participant attaches.
#[derive(Clone)]
pub struct ParticipantConfig {
    /// The tenant namespace.
    pub tenant: TenantId,
    /// The session.
    pub session: SessionId,
    /// This participant's id.
    pub id: ParticipantId,
    /// A display name.
    pub name: String,
    /// Its roles.
    pub roles: Vec<Role>,
    /// The class lease requests are arbitrated at.
    pub priority: PriorityClass,
    /// This participant's clock domain.
    pub clock: Arc<dyn Clock>,
    /// How often to heartbeat.
    pub heartbeat_period: Duration,
    /// How long to wait for a `ctl` reply.
    pub ctl_timeout: Duration,
    /// How long to wait for the host to appear.
    pub host_timeout: Duration,
    /// Features this participant implements.
    pub supports: Vec<String>,
    /// Features it cannot run without (the attach is refused when the
    /// session lacks one).
    pub requires: Vec<String>,
    /// The workflow edition it declares (empty: none).
    pub workflow_edition: String,
    /// Every workflow edition it runs.
    pub workflow_editions: Vec<String>,
    /// An embodiment's e-stop latch carried from its previous session
    /// ([`Participant::local_estop`]): it attaches latched, and the session
    /// starts in ESTOP.
    pub estop: Option<Reason>,
    /// The robot process's latch, when it outlives this participant (a
    /// robot from the lobby, one run after another): it attaches latched
    /// while it is, and its edges read the same latch.
    pub(crate) latch: Option<Arc<LocalLatch>>,
    /// What it attaches with: its device's key and the host's static key
    /// (`SessionHost::credential` in-process; a session address's key
    /// otherwise). Required.
    pub credential: Option<Credential>,
}

impl ParticipantConfig {
    /// Defaults: a 50 ms heartbeat, a 2 s `ctl` timeout, a 10 s wait for the
    /// host, this process's monotonic clock.
    pub fn new(
        tenant: TenantId,
        session: SessionId,
        id: ParticipantId,
        roles: Vec<Role>,
        priority: PriorityClass,
    ) -> Self {
        Self {
            tenant,
            session,
            name: id.to_string(),
            id,
            roles,
            priority,
            clock: MonotonicClock::new().shared(),
            heartbeat_period: Duration::from_millis(50),
            ctl_timeout: Duration::from_secs(2),
            host_timeout: Duration::from_secs(10),
            supports: Vec::new(),
            requires: Vec::new(),
            workflow_edition: String::new(),
            workflow_editions: Vec::new(),
            estop: None,
            latch: None,
            credential: None,
        }
    }
}

impl std::fmt::Debug for ParticipantConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParticipantConfig")
            .field("tenant", &self.tenant)
            .field("session", &self.session)
            .field("id", &self.id)
            .field("roles", &self.roles)
            .field("priority", &self.priority)
            .field("heartbeat_period", &self.heartbeat_period)
            .field("ctl_timeout", &self.ctl_timeout)
            .finish_non_exhaustive()
    }
}
