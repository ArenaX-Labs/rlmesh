//! A participant: presence, heartbeats, the authority view and `ctl` verbs.
//!
//! [`Participant::attach`] announces presence (a liveliness token), waits
//! for the host, subscribes to authority updates, runs the handshake with
//! its [`Credential`], attaches through `ctl` and starts
//! heartbeating on the control class. Every verb and heartbeat it sends is
//! sealed under its `k_up`, every reply and echo it takes is opened under
//! its `k_down`, and every `host/state` and `host/tick` it takes is
//! verified against the host's signing key: anything else is dropped.
//! Verbs carry rising request ids, so a retry is answered once.
//!
//! **Keys**: the host delivers the participant's other
//! keys on `p/{id}/keys` (subscribed before the handshake), and a grant's
//! reply carries the holder's. The participant installs them in its link
//! and acknowledges each delivery (`KeysAck`); a message under a key it
//! doesn't hold makes it ask for every key of its generation (`GetKeys`).
//! Its chunk publishers seal under the lease's `k_cmd`, its sample
//! publishers under the stream's `k_stream`, and its subscribers, edges and
//! history queries open under them ([`Sealer`]).
//!
//! From then on it keeps the newest authority snapshot (from updates and
//! from every `ctl` reply) and hands out endpoints addressed by robot and
//! path: sample publishers and subscribers, chunk publishers, history
//! queries, the virtual-mode tick stream, and the hooks an edge runtime
//! needs ([`AuthorityView`], [`EdgeReport`], [`Participant::report_safe`],
//! the local e-stop latch).
//!
//! An embodiment's publishers follow their streams' publication plans: a background
//! task installs each `Demand` the host publishes on
//! the stream's publishers and acknowledges its revision (`DemandAck`), so
//! controllers bound to the stream can be granted leases.

mod attach;
mod config;
mod edge;
mod endpoints;
mod sealer;
mod shared;
mod tasks;
mod verbs;

pub use config::ParticipantConfig;
pub(crate) use edge::LocalLatch;
pub use edge::{AuthorityView, EdgeReport, ManualView, SafeReporter};
pub use endpoints::{HealthSubscription, TickStream};
pub use sealer::Sealer;
pub use shared::Pending;

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

#[cfg(feature = "gateway")]
use crate::wire::HeartbeatEcho;
use crate::wire::codec;
use crate::wire::{
    AuthoritySnapshot, ChannelId, PartSpec, ParticipantId, Reason, ReasonCode, SessionKeys,
    TimingSample,
};
use rlmesh_session_core::crypto::SignedReceiver;
use rlmesh_session_core::crypto::signed;
use tokio::sync::{Notify, mpsc, watch};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use zenoh::sample::SampleKind;

use crate::clock::{Clock, OffsetEstimate};
use crate::error::SessionError;
use crate::participant::attach::ask;
use crate::participant::edge::Report;
use crate::participant::shared::{KeyWork, SharedLink};
use crate::report::RunSource;
use crate::sample::Outlet;
use crate::transport::Transport;
#[cfg(doc)]
use rlmesh_session_core::auth::Credential;

/// A lock that a panicked holder left behind is still the state to use.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

struct Shared {
    transport: Transport,
    /// A robot that asked for its device view (`view.device`): the host
    /// answers its verbs with the view, not the state.
    takes_view: bool,
    /// This process's link to the host (its keys, request ids and the
    /// host's broadcast floor), set once the handshake completes. One lock:
    /// a seq is allocated and its message sealed under it.
    link: SharedLink,
    /// The key task's queue.
    key_work: mpsc::UnboundedSender<KeyWork>,
    /// Bumped on every key delivery, for waiters on a key.
    keys_changed: watch::Sender<u64>,
    /// The heartbeat publisher, once attached.
    heartbeat: std::sync::OnceLock<zenoh::pubsub::Publisher<'static>>,
    /// Where the host's echoes go while this participant relays another
    /// clock's heartbeats ([`Participant::relay_heartbeats`]).
    #[cfg(feature = "gateway")]
    relay: Mutex<Option<mpsc::UnboundedSender<HeartbeatEcho>>>,
    keys: SessionKeys,
    id: ParticipantId,
    ctl_timeout: Duration,
    history_timeout: Duration,
    clock: Arc<dyn Clock>,
    view: watch::Sender<Option<AuthoritySnapshot>>,
    /// The view's epoch, for publishers stamping samples off the async path.
    epoch: Arc<AtomicU64>,
    /// This participant's clock when it last took a newer snapshot.
    view_local_ns: Arc<AtomicI64>,
    heartbeats_enabled: AtomicBool,
    health: AtomicU8,
    host_present: watch::Sender<bool>,
    /// This participant's clock when the host's liveliness went away
    /// (`i64::MIN` while it is present).
    host_lost_ns: AtomicI64,
    /// This participant's clock when the host last authenticated itself
    /// (its handshake, then each heartbeat echo).
    #[cfg(feature = "fleet")]
    host_heard_ns: AtomicI64,
    /// The last completed exchange, sent with the next heartbeat.
    completed: Mutex<Option<TimingSample>>,
    /// The newest seq received per subscribed channel.
    seen: Mutex<BTreeMap<ChannelId, Arc<AtomicU64>>>,
    edge: Arc<EdgeReport>,
    /// This robot process's stops, sent in order by the reporter task.
    reports: mpsc::UnboundedSender<Report>,
    /// The local e-stop latch every edge of this process reads each tick.
    estop: Arc<LocalLatch>,
    /// This controller's binding revisions.
    binding_revision: AtomicU32,
    /// This embodiment's publishers by stream, for installing demands.
    outlets: Mutex<BTreeMap<ChannelId, Vec<Weak<Outlet>>>>,
    /// Wakes the demand installer: a publisher was declared.
    demand_wake: Notify,
    /// A bound controller's run counters, for its reports.
    run: Arc<Mutex<Option<RunSource>>>,
}

/// An attached participant. Dropping it stops heartbeats and withdraws its
/// liveliness token, which the host treats as the participant leaving; prefer
/// [`Participant::detach`] or [`Participant::close`].
pub struct Participant {
    shared: Arc<Shared>,
    cancel: CancellationToken,
    tasks: Vec<JoinHandle<()>>,
    // Dropped in order: the watch on the host goes before this token, whose
    // loss can stop the session (and with it the host's token).
    _host: zenoh::pubsub::Subscriber<()>,
    /// A robot process's e-stop signal subscription.
    _estops: Option<zenoh::pubsub::Subscriber<()>>,
    /// Its presence; [`Participant::withdraw`] takes it back.
    liveliness: std::sync::Mutex<Option<zenoh::liveliness::LivelinessToken>>,
    _state: zenoh::pubsub::Subscriber<()>,
    _echo: zenoh::pubsub::Subscriber<()>,
    _keys: zenoh::pubsub::Subscriber<()>,
}

impl Participant {
    /// This participant's id.
    pub fn id(&self) -> &ParticipantId {
        &self.shared.id
    }

    /// The generation the host issued this process at its handshake.
    pub fn generation(&self) -> Option<u32> {
        self.shared
            .link
            .get()
            .map(|link| lock(link).accept().generation)
    }

    /// The host instance this process's handshake authenticated (a restarted host is
    /// another instance).
    #[cfg(feature = "fleet")]
    pub fn host_instance(&self) -> Option<u64> {
        self.shared
            .link
            .get()
            .map(|link| lock(link).accept().host_instance)
    }

    /// How long since the host last authenticated itself: the handshake,
    /// then each heartbeat echo. Its liveliness token is unsigned; this is
    /// not.
    #[cfg(feature = "fleet")]
    pub fn host_unheard(&self) -> Duration {
        let heard = self.shared.host_heard_ns.load(Ordering::Relaxed);
        let ago = self.shared.clock.now_ns().saturating_sub(heard);
        Duration::from_nanos(u64::try_from(ago).unwrap_or(0))
    }

    /// The session's keys.
    pub fn keys(&self) -> &SessionKeys {
        &self.shared.keys
    }

    /// The transport this participant rides.
    pub fn transport(&self) -> &Transport {
        &self.shared.transport
    }

    /// The newest authority snapshot seen (from updates or `ctl` replies).
    pub fn authority(&self) -> Option<AuthoritySnapshot> {
        self.shared.view.borrow().clone()
    }

    /// Every new snapshot.
    pub fn watch(&self) -> watch::Receiver<Option<AuthoritySnapshot>> {
        self.shared.view.subscribe()
    }

    /// The live view, for code off the async path.
    pub fn authority_view(&self) -> AuthorityView {
        AuthorityView {
            rx: self.shared.view.subscribe(),
            observed_local_ns: self.shared.view_local_ns.clone(),
            epoch: self.shared.epoch.clone(),
        }
    }

    /// Whether the host's liveliness token is visible.
    pub fn host_present(&self) -> watch::Receiver<bool> {
        self.shared.host_present.subscribe()
    }

    /// `HOST_LOST` (at this participant's clock) once the host's liveliness
    /// went away after attaching; `None` while it is visible. The edge keeps
    /// running on its own clock meanwhile.
    pub fn host_lost(&self) -> Option<Reason> {
        let at_ns = self.shared.host_lost_ns.load(Ordering::Relaxed);
        (at_ns != i64::MIN).then(|| Reason {
            at_ns,
            ..Reason::new(ReasonCode::HostLost)
        })
    }

    /// Waits until the authority view satisfies `predicate`.
    pub async fn wait_for(
        &self,
        what: &str,
        timeout: Duration,
        mut predicate: impl FnMut(&AuthoritySnapshot) -> bool,
    ) -> Result<AuthoritySnapshot, SessionError> {
        let mut watch = self.watch();
        let found = tokio::time::timeout(
            timeout,
            watch.wait_for(|view| view.as_ref().is_some_and(&mut predicate)),
        )
        .await
        .map_err(|_| SessionError::Timeout(timeout, what.to_string()))?
        .map_err(|_| SessionError::Closed)?;
        found.clone().ok_or(SessionError::Closed)
    }

    /// Queries the host for its current snapshot, verified. A reply that
    /// a newer publication overtook leaves the newer view.
    pub async fn fetch_authority(&self) -> Result<AuthoritySnapshot, SessionError> {
        let key = self.shared.keys.host_state();
        let timeout = self.shared.ctl_timeout;
        let signed = ask(self.shared.transport.control(), &key, None, timeout).await?;
        if let Some(snapshot) = self.shared.verify_state(&key, &signed) {
            self.shared.observe(snapshot);
        }
        self.authority().ok_or(SessionError::NoReply(timeout))
    }

    /// A verifier of the host's broadcasts (`host/state`, `host/tick`), for
    /// a subscriber of this process's own (a recorder, the gateway's relay):
    /// the host's key and instance from this participant's handshake, and
    /// the floor of its own state updates.
    pub fn host_verifier(&self) -> Option<SignedReceiver> {
        self.shared
            .link
            .get()
            .map(|link| lock(link).host_verifier())
    }

    /// The host's estimate of this participant's clock offset, from the
    /// current view.
    pub fn clock_estimate(&self) -> Option<OffsetEstimate> {
        self.shared.view.borrow().as_ref().and_then(|view| {
            view.clock(&self.shared.id)
                .map(OffsetEstimate::from_mapping)
        })
    }

    /// A part of `robot`, from the current view.
    pub fn part(&self, robot: &ParticipantId, path: &str) -> Result<PartSpec, SessionError> {
        self.authority()
            .and_then(|view| view.part_at(robot, path).cloned())
            .ok_or_else(|| SessionError::UnknownPart(format!("{robot}:{path}")))
    }

    /// One of this embodiment's own parts.
    pub fn own_part(&self, path: &str) -> Result<PartSpec, SessionError> {
        self.part(&self.shared.id, path)
    }

    /// A part by its stream id, from the current view.
    pub fn part_by_id(&self, channel: ChannelId) -> Result<PartSpec, SessionError> {
        self.authority()
            .and_then(|view| view.part(channel).cloned())
            .ok_or_else(|| SessionError::UnknownPart(channel.to_string()))
    }

    /// This participant's clock.
    pub fn clock(&self) -> &Arc<dyn Clock> {
        &self.shared.clock
    }

    /// This participant's clock converted to session time, through its
    /// clock mapping (unconverted until the host has one).
    pub fn session_now_ns(&self) -> i64 {
        self.to_session_ns(self.shared.clock.now_ns())
    }

    /// One of this participant's timestamps in session time.
    pub fn to_session_ns(&self, local_ns: i64) -> i64 {
        local_ns.saturating_add(self.offset_ns())
    }

    /// A session time in this participant's clock.
    pub fn to_local_ns(&self, session_ns: i64) -> i64 {
        session_ns.saturating_sub(self.offset_ns())
    }

    fn offset_ns(&self) -> i64 {
        self.shared
            .view
            .borrow()
            .as_ref()
            .and_then(|v| v.clock(&self.shared.id).map(|c| c.offset_ns))
            .unwrap_or(0)
    }
}

impl Drop for Participant {
    fn drop(&mut self) {
        self.cancel.cancel();
        for task in &self.tasks {
            task.abort();
        }
    }
}

/// Waits until the session host's liveliness token is visible.
pub async fn wait_for_host(
    transport: &Transport,
    keys: &SessionKeys,
    timeout: Duration,
) -> Result<(), SessionError> {
    let host = transport
        .control()
        .liveliness()
        .declare_subscriber(keys.host_liveliness())
        .history(true)
        .await?;
    let present = async {
        while let Ok(sample) = host.recv_async().await {
            if sample.kind() == SampleKind::Put {
                return Ok(());
            }
        }
        Err(SessionError::Closed)
    };
    tokio::time::timeout(timeout, present)
        .await
        .map_err(|_| SessionError::Timeout(timeout, "the session host".into()))?
}

/// Queries a session host for its authority snapshot, without attaching:
/// how anyone gets state at once. **Unverified**: without a handshake there
/// is no host key to check its signature against (authority state is
/// public), so act only on what [`Participant::fetch_authority`]
/// returns.
pub async fn query_authority(
    transport: &Transport,
    keys: &SessionKeys,
    timeout: Duration,
) -> Result<AuthoritySnapshot, SessionError> {
    let signed = ask(transport.control(), &keys.host_state(), None, timeout).await?;
    let body = signed::peek(&signed)
        .map_err(|err| {
            SessionError::Wire(crate::wire::WireError::Malformed {
                message: "host state",
                reason: err.to_string(),
            })
        })?
        .body;
    Ok(codec::decode_state(body)?)
}
