//! Attaching: the handshake, waiting for a host, and leaving again.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::wire::codec;
use crate::wire::version::{PROTOCOL_MINOR, VIEW_DEVICE};
use crate::wire::{Command, CtlError, CtlStatus, Health, Role, SessionId, SessionKeys, TenantId};
use rlmesh_session_core::auth;
use rlmesh_session_core::auth::Link;
use rlmesh_session_core::crypto::signed;
use tokio::sync::{Notify, mpsc, watch};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use zenoh::query::ConsolidationMode;
use zenoh::sample::SampleKind;

use crate::error::SessionError;
use crate::host::OsRng;
use crate::participant::config::ParticipantConfig;
use crate::participant::edge::{EdgeReport, report_stops};
use crate::participant::shared::KeyWork;
use crate::participant::tasks::{
    health_code, heartbeat_loop, install_demands, report_streams, work_keys,
};
use crate::participant::{Participant, Shared, lock};
use crate::transport::{CONTROL_CONGESTION, CONTROL_PRIORITY, HEARTBEAT_CONGESTION, Transport};

/// What a participant without the address's key is told.
const NO_CREDENTIAL: &str = "no credential for this session: join with the address its host \
                             printed (it ends in #key), or set ParticipantConfig.credential";

/// How long a refused handshake waits for the host's state to name another
/// instance: an on-demand session restarting as this participant joins.
const RESTART_WAIT: Duration = Duration::from_secs(1);

/// Runs the handshake: reads the host instance from the public snapshot (a
/// hint: a wrong one only fails the handshake), sends msg 1 on `hs` and
/// reads msg 2. A session a host starts on demand stops when its last
/// participant leaves and starts again, under a new instance, for the next
/// one; a participant joining just then can read the old instance. So a
/// refusal is retried once the state names another instance, within
/// [`RESTART_WAIT`]; the same instance refusing again is final.
///
/// A host too busy to answer one of them within `ctl_timeout` (a loaded
/// machine, a session just starting) is asked again while the join's own
/// `host_timeout` lasts: nothing is attached yet, so asking twice is safe.
async fn handshake(
    control: &zenoh::Session,
    keys: &SessionKeys,
    config: &ParticipantConfig,
) -> Result<auth::Attached, SessionError> {
    let deadline = tokio::time::Instant::now() + config.host_timeout;
    loop {
        match handshake_once(control, keys, config).await {
            Err(SessionError::NoReply(waited)) if tokio::time::Instant::now() < deadline => {
                debug!(participant = %config.id, ?waited, "no handshake reply; asking again");
            }
            result => return result,
        }
    }
}

async fn handshake_once(
    control: &zenoh::Session,
    keys: &SessionKeys,
    config: &ParticipantConfig,
) -> Result<auth::Attached, SessionError> {
    let instance = host_instance(control, keys, config.ctl_timeout).await?;
    match handshake_with(control, keys, config, instance).await {
        Err(SessionError::Rejected(refusal)) if refusal.status == CtlStatus::Unauthenticated => {
            let deadline = tokio::time::Instant::now() + RESTART_WAIT;
            loop {
                tokio::time::sleep(Duration::from_millis(50)).await;
                match host_instance(control, keys, config.ctl_timeout).await {
                    Ok(next) if next != instance => {
                        debug!(participant = %config.id, "the session restarted; joining it again");
                        return handshake_with(control, keys, config, next).await;
                    }
                    _ if tokio::time::Instant::now() >= deadline => {
                        return Err(SessionError::Rejected(refusal));
                    }
                    _ => {}
                }
            }
        }
        result => result,
    }
}

/// The host instance the session's public state names.
async fn host_instance(
    control: &zenoh::Session,
    keys: &SessionKeys,
    timeout: Duration,
) -> Result<u64, SessionError> {
    let state = ask(control, &keys.host_state(), None, timeout).await?;
    let header = signed::peek(&state).map_err(|err| {
        SessionError::Rejected(CtlError::new(
            CtlStatus::Unauthenticated,
            format!("the host's state is not signed ({err})"),
        ))
    })?;
    Ok(header.header.host_instance)
}

async fn handshake_with(
    control: &zenoh::Session,
    keys: &SessionKeys,
    config: &ParticipantConfig,
    host_instance: u64,
) -> Result<auth::Attached, SessionError> {
    let credential = config.credential.as_ref().ok_or_else(|| {
        SessionError::Rejected(CtlError::new(CtlStatus::Unauthenticated, NO_CREDENTIAL))
    })?;
    let unauthenticated =
        |why: String| SessionError::Rejected(CtlError::new(CtlStatus::Unauthenticated, why));
    let (initiator, hello) = credential
        .start(
            &mut OsRng,
            config.tenant.as_str(),
            &config.session.to_string(),
            host_instance,
            &config.id,
        )
        .map_err(|err| unauthenticated(err.to_string()))?;
    let reply = ask(
        control,
        &keys.handshake(),
        Some(codec::encode_handshake_request(&hello)),
        config.ctl_timeout,
    )
    .await?;
    let msg2 = codec::decode_handshake_reply(&reply)??;
    auth::finish(initiator, &msg2).map_err(|err| {
        unauthenticated(format!(
            "the host's handshake reply failed ({err}): the address's key is not this host's"
        ))
    })
}

/// The first reply to a query on `key`.
pub(super) async fn ask(
    control: &zenoh::Session,
    key: &str,
    payload: Option<Vec<u8>>,
    timeout: Duration,
) -> Result<Vec<u8>, SessionError> {
    let mut get = control
        .get(key)
        .consolidation(ConsolidationMode::None)
        .priority(CONTROL_PRIORITY)
        .congestion_control(CONTROL_CONGESTION)
        .express(true)
        .timeout(timeout);
    if let Some(payload) = payload {
        get = get.payload(payload);
    }
    let replies = get.await?;
    while let Ok(reply) = replies.recv_async().await {
        if let Ok(sample) = reply.result() {
            return Ok(sample.payload().to_bytes().into_owned());
        }
    }
    Err(SessionError::NoReply(timeout))
}

/// Whether `transport` reaches another Zenoh session within `timeout`: at
/// once for one sharing a session, or riding a host of this process.
async fn linked(transport: &Transport, timeout: Duration) -> bool {
    let control = transport.control();
    if transport.is_shared() || transport.is_in_process() {
        return true;
    }
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        // Zenoh answers these in a nested `block_on`: off this thread.
        let session = control.clone();
        let found = crate::transport::off_thread(move || {
            use zenoh::Wait;
            session.info().peers_zid().wait().next().is_some()
                || session.info().routers_zid().wait().next().is_some()
        })
        .await
        .unwrap_or(false);
        if found {
            return true;
        }
        if tokio::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// How often a wait for the host asks for its token again, so that the
/// wait never rests on one event (the history reply, a later declaration, or
/// the wakeup either sends) for the whole of its timeout.
pub(crate) const HOST_RECHECK: Duration = Duration::from_millis(250);

/// Waits until `present` says the host is there, asking for its liveliness
/// token `key` every [`HOST_RECHECK`] meanwhile (setting `present` when it
/// answers).
pub(crate) async fn host_seen(
    control: &zenoh::Session,
    key: String,
    present: &watch::Sender<bool>,
) {
    let mut seen = present.subscribe();
    loop {
        if tokio::time::timeout(HOST_RECHECK, seen.wait_for(|p| *p))
            .await
            .is_ok_and(|seen| seen.is_ok())
        {
            return;
        }
        if live(control, key.clone()).await {
            present.send_replace(true);
            return;
        }
    }
}

/// Whether a live participant holds the liveliness token `key`.
async fn live(control: &zenoh::Session, key: String) -> bool {
    match control
        .liveliness()
        .get(key)
        .timeout(Duration::from_millis(500))
        .await
    {
        Ok(replies) => {
            let mut any = false;
            while let Ok(reply) = replies.recv_async().await {
                any |= reply.result().is_ok();
            }
            any
        }
        Err(_) => false,
    }
}

/// Resolves when the host this transport reaches can never serve
/// `session`: it hosts other sessions, and is no multi-session host of
/// `tenant` (which would start `session` for this participant). Pending
/// while no host answers at all, so a host still starting is waited for.
async fn not_served(
    control: &zenoh::Session,
    tenant: &TenantId,
    session: SessionId,
) -> SessionError {
    use crate::wire::keys::{session_hosts, session_server};
    // Whether any queryable answers `key` (queries reach every matching
    // queryable, where liveliness tokens reach only declared interests).
    let answers = |key: String| async move {
        match control.get(key).timeout(Duration::from_millis(500)).await {
            Ok(replies) => {
                let mut any = false;
                while let Ok(reply) = replies.recv_async().await {
                    any |= reply.result().is_ok();
                }
                any
            }
            Err(_) => false,
        }
    };
    let own = SessionKeys::new(tenant, session).host_state();
    loop {
        // Give a multi-session host time to see this participant first.
        tokio::time::sleep(Duration::from_millis(300)).await;
        if answers(session_server(tenant)).await || answers(own.clone()).await {
            return std::future::pending().await;
        }
        if answers(session_hosts()).await {
            return SessionError::NotServed(format!(
                "it hosts other sessions and starts none for tenant {tenant}"
            ));
        }
    }
}

impl Participant {
    /// Announces presence, waits for the host, subscribes to authority
    /// updates, attaches through `ctl` and starts heartbeating.
    pub async fn attach(
        transport: Transport,
        config: ParticipantConfig,
    ) -> Result<Self, SessionError> {
        if config.heartbeat_period.is_zero() {
            return Err(SessionError::Config(
                "heartbeat_period must be positive".into(),
            ));
        }
        let keys = SessionKeys::new(&config.tenant, config.session);
        let control = transport.control().clone();
        let (view, _) = watch::channel(None);
        let (host_present, _) = watch::channel(false);
        let (reports, report_rx) = mpsc::unbounded_channel();
        let (key_work, key_rx) = mpsc::unbounded_channel();
        let (keys_changed, _) = watch::channel(0);
        let shared = Arc::new(Shared {
            transport: transport.clone(),
            takes_view: config.roles.contains(&Role::Embodiment)
                && config.supports.iter().any(|f| f == VIEW_DEVICE),
            link: Arc::default(),
            key_work,
            keys_changed,
            heartbeat: std::sync::OnceLock::new(),
            #[cfg(feature = "gateway")]
            relay: Mutex::new(None),
            keys: keys.clone(),
            id: config.id.clone(),
            ctl_timeout: config.ctl_timeout,
            history_timeout: config.ctl_timeout,
            clock: config.clock.clone(),
            view,
            epoch: Arc::new(AtomicU64::new(0)),
            view_local_ns: Arc::new(AtomicI64::new(0)),
            heartbeats_enabled: AtomicBool::new(true),
            health: AtomicU8::new(health_code(Health::Ok)),
            host_present,
            host_lost_ns: AtomicI64::new(i64::MIN),
            #[cfg(feature = "fleet")]
            host_heard_ns: AtomicI64::new(i64::MIN),
            completed: Mutex::new(None),
            seen: Mutex::new(BTreeMap::new()),
            edge: Arc::new(EdgeReport::default()),
            reports,
            estop: config.latch.clone().unwrap_or_default(),
            binding_revision: AtomicU32::new(0),
            outlets: Mutex::new(BTreeMap::new()),
            demand_wake: Notify::new(),
            run: Arc::default(),
        });

        // Subscriptions first: nothing the host sends after the
        // handshake can be missed. Until it completes, nothing verifies.
        let state = control
            .declare_subscriber(keys.host_state())
            .callback({
                let (shared, key) = (shared.clone(), keys.host_state());
                move |sample| {
                    if let Some(snapshot) = shared.verify_state(&key, &sample.payload().to_bytes())
                    {
                        shared.observe(snapshot);
                    }
                }
            })
            .await?;
        let echo = control
            .declare_subscriber(keys.heartbeat_echo(&config.id))
            .callback({
                let (shared, clock) = (shared.clone(), config.clock.clone());
                let key = keys.heartbeat_echo(&config.id);
                move |sample| {
                    let t4 = clock.now_ns();
                    let Some(link) = shared.link.get() else {
                        return;
                    };
                    let mut bytes = sample.payload().to_bytes().into_owned();
                    let echo = lock(link).echo(&key, &mut bytes);
                    #[cfg(feature = "fleet")]
                    if echo.is_ok() {
                        shared.host_heard_ns.store(t4, Ordering::Relaxed);
                    }
                    match echo {
                        #[cfg(feature = "gateway")]
                        Ok(echo) => match &*lock(&shared.relay) {
                            Some(relay) => {
                                let _ = relay.send(echo);
                            }
                            None => *lock(&shared.completed) = Some(echo.complete(t4)),
                        },
                        #[cfg(not(feature = "gateway"))]
                        Ok(echo) => *lock(&shared.completed) = Some(echo.complete(t4)),
                        Err(err) => debug!(%err, "heartbeat echo dropped"),
                    }
                }
            })
            .await?;
        let deliveries = control
            .declare_subscriber(keys.participant_keys(&config.id))
            .callback({
                let shared = shared.clone();
                let key = keys.participant_keys(&config.id);
                move |sample| {
                    let Some(link) = shared.link.get() else {
                        return;
                    };
                    let mut bytes = sample.payload().to_bytes().into_owned();
                    let now = shared.clock.now_ns();
                    let ack = lock(link).keys(&key, &mut bytes, now);
                    match ack {
                        Ok(ack) => {
                            shared.keys_changed.send_modify(|n| *n += 1);
                            let _ = shared.key_work.send(KeyWork::Ack(ack));
                        }
                        Err(err) => debug!(%err, "key delivery dropped"),
                    }
                }
            })
            .await?;
        let host = control
            .liveliness()
            .declare_subscriber(keys.host_liveliness())
            .history(true)
            .callback({
                let shared = shared.clone();
                move |sample| {
                    let present = sample.kind() == SampleKind::Put;
                    let lost_ns = if present {
                        i64::MIN
                    } else {
                        warn!(participant = %shared.id, "session host liveliness lost");
                        shared.clock.now_ns()
                    };
                    shared.host_lost_ns.store(lost_ns, Ordering::Relaxed);
                    shared.host_present.send_replace(present);
                }
            })
            .await?;
        // Once linked, this transport sees every live participant's token:
        // another one under this id is refused, never shared.
        let key = keys.participant(&config.id);
        if linked(&transport, config.host_timeout).await && live(&control, key.clone()).await {
            return Err(SessionError::Config(format!(
                "participant {} is already in this session: serve under another name, or stop \
                 the one there first",
                config.id
            )));
        }
        // A multi-session host starts the session when it sees this token.
        let liveliness = control.liveliness().declare_token(key).await?;
        // The host declares its queryables before its liveliness token, and
        // Zenoh keeps declaration order per link, so once the token is seen
        // the first `ctl` query cannot miss the host.
        let seen = host_seen(&control, keys.host_liveliness(), &shared.host_present);
        tokio::select! {
            found = tokio::time::timeout(config.host_timeout, seen) => {
                found.map_err(|_| {
                    SessionError::Timeout(config.host_timeout, "the session host".into())
                })?;
            }
            refusal = not_served(&control, &config.tenant, config.session) => return Err(refusal),
        }

        let attached = handshake(&control, &keys, &config).await?;
        let generation = attached.accept.generation;
        let _ = shared
            .link
            .set(Mutex::new(Link::new(config.id.clone(), attached)));
        #[cfg(feature = "fleet")]
        shared
            .host_heard_ns
            .store(config.clock.now_ns(), Ordering::Relaxed);

        let cancel = CancellationToken::new();
        let mut participant = Self {
            shared: shared.clone(),
            tasks: vec![
                tokio::spawn(report_stops(shared.clone(), report_rx, cancel.clone())),
                tokio::spawn(install_demands(shared.clone(), cancel.clone())),
                tokio::spawn(report_streams(shared.clone(), cancel.clone())),
                tokio::spawn(work_keys(shared.clone(), key_rx, cancel.clone())),
            ],
            cancel,
            liveliness: std::sync::Mutex::new(Some(liveliness)),
            _state: state,
            _echo: echo,
            _keys: deliveries,
            _host: host,
            _estops: None,
            health_watch: tokio::sync::OnceCell::new(),
        };
        shared
            .ctl(Command::Attach {
                name: config.name.clone(),
                roles: config.roles.clone(),
                priority: config.priority,
                protocol_minor: PROTOCOL_MINOR,
                supports: config.supports.clone(),
                requires: config.requires.clone(),
                workflow_edition: config.workflow_edition.clone(),
                workflow_editions: config.workflow_editions.clone(),
                estop: config.estop.clone().or_else(|| shared.estop.reason()),
            })
            .await?;
        // A robot process latches every member's e-stop signal at once,
        // with or without a host.
        if config.roles.contains(&Role::Embodiment) {
            if let Some(reason) = config.estop.clone() {
                shared.latch(reason);
            }
            let subscriber = control
                .declare_subscriber(keys.estops())
                .callback({
                    let shared = shared.clone();
                    move |sample| {
                        shared.on_estop_signal(sample.key_expr().as_str(), sample.payload())
                    }
                })
                .await?;
            participant._estops = Some(subscriber);
        }

        let publisher = control
            .declare_publisher(keys.heartbeat(&config.id))
            .priority(CONTROL_PRIORITY)
            .congestion_control(HEARTBEAT_CONGESTION)
            .express(true)
            .await?;
        let _ = shared.heartbeat.set(publisher);
        participant.tasks.push(tokio::spawn(heartbeat_loop(
            shared,
            config.heartbeat_period,
            participant.cancel.clone(),
        )));
        info!(participant = %config.id, session = %config.session, generation, "attached");
        Ok(participant)
    }

    /// Withdraws its liveliness token: the host and everyone else see it
    /// leave, and a new process may join under its id (a robot rejoining
    /// after its host dropped it). It stops heartbeating too.
    pub fn withdraw(&self) {
        self.set_heartbeats_enabled(false);
        let token = self
            .liveliness
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(token) = token {
            let _ = zenoh::Wait::wait(token.undeclare());
        }
    }

    /// Detaches through `ctl` (releasing leases and removing owned
    /// channels), then closes.
    pub async fn detach(self) -> Result<(), SessionError> {
        let result = self.shared.ctl(Command::Detach).await.map(|_| ());
        self.close().await;
        result
    }

    /// Stops heartbeats and withdraws presence without detaching: the host
    /// sees liveliness loss (leases lapse, channels stay for a reconnect).
    pub async fn close(mut self) {
        self.cancel.cancel();
        for task in self.tasks.drain(..) {
            if let Err(err) = task.await
                && !err.is_cancelled()
            {
                warn!(participant = %self.shared.id, %err, "participant task failed");
            }
        }
    }
}
