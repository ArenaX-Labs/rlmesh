//! A multi-session host: one [`SessionHost`] per session of a tenant,
//! started when its first participant appears and stopped when its last
//! one leaves.
//!
//! The server watches every participant's liveliness token in its tenant
//! (`{tenant}/s/*/p/*`). A participant declares its token before it waits
//! for its session's host, so the server sees it, starts that session's
//! host on the same transport, and the participant then finds the host as
//! it would a host started ahead of it. The server's queryable
//! (`{tenant}/server`) tells a participant that a host here will serve its
//! session, so one that finds only other sessions' hosts fails at once.
//!
//! A session started explicitly ([`SessionServer::start_session`]: the
//! home daemon's lobby, a scene's run) is kept however its participants
//! come and go, until [`SessionServer::stop`]; once stopped, it is never
//! started again for a straggler.
//!
//! With recording on ([`SessionServer::start_recording`]), each session
//! also gets an always-on recorder that joins as the observer
//! [`RECORDER`]. It never counts as a member: the session still stops when
//! its last other participant leaves, after the recorder has finished.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use tokio::sync::Mutex;
use tracing::{info, warn};
use zenoh::Wait;
use zenoh::sample::SampleKind;

use crate::auth::{Device, Pin};
use crate::error::SessionError;
use crate::host::{HostConfig, SessionHost, token_declared};
use crate::participant::{Participant, ParticipantConfig};
#[cfg(feature = "record")]
use crate::record::{Recorder, Rolling};
use crate::transport::Transport;
use crate::wire::keys::{session_of, session_server, tenant_participants};
use crate::wire::{AuthoritySnapshot, ParticipantId, SessionId};
#[cfg(feature = "record")]
use crate::wire::{PriorityClass, Role};

type Hosts = Arc<Mutex<HashMap<SessionId, SessionHost>>>;

/// The sessions started explicitly: running (`true`, never stopped for
/// want of members) or stopped (`false`, never started again).
type Kept = Arc<std::sync::Mutex<HashMap<SessionId, bool>>>;

/// The devices and pins every session starts with ([`HostConfig::devices`],
/// [`HostConfig::pins`]); changed only while the hosts are locked, so a
/// session starts with the table every other one has.
type Table = Arc<std::sync::Mutex<(Vec<Device>, BTreeMap<ParticipantId, Pin>)>>;

/// The id the server's own recorder joins each session as.
pub const RECORDER: &str = "_recorder";

#[cfg(feature = "record")]
type Recorders = Arc<Mutex<HashMap<SessionId, tokio::task::JoinHandle<Option<Recorder>>>>>;

/// A participant's liveliness token appeared (`true`) or left.
type Presence = (bool, SessionId, String);

/// A running multi-session host. Dropping it stops its sessions; prefer
/// [`SessionServer::shutdown`], which waits for them.
pub struct SessionServer {
    hosts: Hosts,
    kept: Kept,
    table: Table,
    /// The sessions started with their own devices and pins
    /// ([`Self::host_session`]), which [`Self::set_devices`] leaves alone.
    own_tables: std::sync::Mutex<HashSet<SessionId>>,
    transport: Transport,
    /// What its own participants attach with.
    own: (
        crate::wire::TenantId,
        Arc<dyn crate::clock::Clock>,
        crate::auth::Credential,
    ),
    #[cfg(feature = "record")]
    recorders: Recorders,
    _server: zenoh::query::Queryable<()>,
    _watch: zenoh::pubsub::Subscriber<()>,
}

/// Removes `key` from `members` when its token left: not when its
/// participant rejoined and its new token is declared ahead of the old one's
/// late delete, however many other members the session has.
async fn member_left(members: &mut HashSet<String>, key: &str, control: &zenoh::Session) -> bool {
    if members.contains(key)
        && let Ok(expr) = zenoh::key_expr::KeyExpr::try_from(key)
        && token_declared(control, &expr).await
    {
        return false;
    }
    members.remove(key);
    true
}

impl SessionServer {
    /// Serves every session of `config.tenant` on `transport`, each with
    /// `config` (its `session` replaced), starting `config.session` at once.
    pub async fn start(transport: Transport, config: HostConfig) -> Result<Self, SessionError> {
        Self::serve(
            transport,
            config,
            #[cfg(feature = "record")]
            None,
        )
        .await
    }

    /// [`Self::start`], recording every session it hosts into `rolling`
    /// ([`Recorder::start_rolling`]).
    #[cfg(feature = "record")]
    pub async fn start_recording(
        transport: Transport,
        config: HostConfig,
        rolling: Rolling,
    ) -> Result<Self, SessionError> {
        Self::serve(transport, config, Some(rolling)).await
    }

    async fn serve(
        transport: Transport,
        config: HostConfig,
        #[cfg(feature = "record")] rolling: Option<Rolling>,
    ) -> Result<Self, SessionError> {
        let hosts: Hosts = Arc::default();
        let kept: Kept = Arc::default();
        let own = (
            config.tenant.clone(),
            config.clock.clone(),
            config.identity.local_credential(),
        );
        let table: Table = Arc::new(std::sync::Mutex::new((
            config.devices.clone(),
            config.pins.clone(),
        )));
        #[cfg(feature = "record")]
        let recorders: Recorders = Arc::default();
        // Only a recording server's own recorder is not a member.
        #[cfg(feature = "record")]
        let recording = rolling.is_some();
        #[cfg(not(feature = "record"))]
        let recording = false;
        #[cfg(feature = "record")]
        let record = {
            let (transport, config) = (transport.clone(), config.clone());
            move |session: SessionId| {
                let rolling = rolling.clone()?;
                let (transport, config) = (transport.ridden(), config.clone());
                Some(tokio::spawn(async move {
                    let id = ParticipantId::new(RECORDER).ok()?;
                    let mut participant = ParticipantConfig::new(
                        config.tenant.clone(),
                        session,
                        id,
                        vec![Role::Observer],
                        PriorityClass::Scripted,
                    );
                    participant.clock = config.clock.clone();
                    participant.credential = Some(config.identity.local_credential());
                    let recorded = async {
                        let participant = Participant::attach(transport, participant).await?;
                        Recorder::start_rolling(Arc::new(participant), rolling, session).await
                    };
                    match recorded.await {
                        Ok(recorder) => Some(recorder),
                        Err(err) => {
                            warn!(%session, %err, "session not recorded");
                            None
                        }
                    }
                }))
            }
        };
        hosts.lock().await.insert(
            config.session,
            SessionHost::start(transport.clone(), config.clone()).await?,
        );
        #[cfg(feature = "record")]
        if let Some(started) = record(config.session) {
            recorders.lock().await.insert(config.session, started);
        }
        let control = transport.control().clone();
        let tenant_id = config.tenant.clone();
        let tenant = config.tenant.to_string();
        // Presence changes are applied in order, by one task.
        let (presence, mut changes) = tokio::sync::mpsc::unbounded_channel::<Presence>();
        tokio::spawn({
            let hosts = hosts.clone();
            let kept = kept.clone();
            let table = table.clone();
            let transport = transport.clone();
            #[cfg(feature = "record")]
            let recorders = recorders.clone();
            let control = control.clone();
            async move {
                let mut present: HashMap<SessionId, HashSet<String>> = HashMap::new();
                while let Some((joined, session, key)) = changes.recv().await {
                    if recording && key.rsplit('/').next() == Some(RECORDER) {
                        continue;
                    }
                    let members = present.entry(session).or_default();
                    if joined {
                        members.insert(key);
                    } else if !member_left(members, &key, &control).await {
                        continue;
                    }
                    let empty = members.is_empty();
                    if empty {
                        present.remove(&session);
                    }
                    let mut hosts = hosts.lock().await;
                    let explicit = kept
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .contains_key(&session);
                    if !empty && !explicit && !hosts.contains_key(&session) {
                        let (devices, pins) = table
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .clone();
                        let config = HostConfig {
                            session,
                            devices,
                            pins,
                            ..config.clone()
                        };
                        match SessionHost::start(transport.clone(), config).await {
                            Ok(host) => {
                                info!(%session, "session started for its first participant");
                                hosts.insert(session, host);
                                #[cfg(feature = "record")]
                                if let Some(started) = record(session) {
                                    recorders.lock().await.insert(session, started);
                                }
                            }
                            Err(err) => warn!(%session, %err, "session did not start"),
                        }
                    } else if empty
                        && !joined
                        && !explicit
                        && let Some(host) = hosts.remove(&session)
                    {
                        info!(%session, "session stopped: its last participant left");
                        #[cfg(feature = "record")]
                        finish_recorder(&recorders, session).await;
                        host.shutdown().await;
                    }
                }
            }
        });
        let watch = control
            .liveliness()
            .declare_subscriber(tenant_participants(&tenant_id))
            .history(true)
            .callback(move |sample| {
                let key = sample.key_expr().as_str();
                let Some((key_tenant, session)) = session_of(key) else {
                    return;
                };
                if key_tenant != tenant {
                    return;
                }
                let joined = sample.kind() == SampleKind::Put;
                let _ = presence.send((joined, session, key.to_string()));
            })
            .await?;
        let server = control
            .declare_queryable(session_server(&tenant_id))
            .callback(|query| {
                let key = query.key_expr().clone();
                let _ = query.reply(key, Vec::<u8>::new()).wait();
            })
            .await?;
        Ok(Self {
            hosts,
            kept,
            table,
            own_tables: std::sync::Mutex::default(),
            own,
            transport,
            #[cfg(feature = "record")]
            recorders,
            _server: server,
            _watch: watch,
        })
    }

    /// The sessions it hosts.
    pub async fn sessions(&self) -> Vec<SessionId> {
        self.hosts.lock().await.keys().copied().collect()
    }

    /// Every session's latest authority snapshot.
    pub async fn snapshots(&self) -> Vec<AuthoritySnapshot> {
        self.hosts
            .lock()
            .await
            .values()
            .map(SessionHost::snapshot)
            .collect()
    }

    /// Replaces the devices and pins every session accepts, now and from
    /// now on (rlmeshd's `devices.toml`, design §4): a device gone is
    /// revoked, a new one admitted, a changed entry or pin applied (see
    /// [`SessionHost`]'s `set_access`, `admit`, `revoke` and `pin`).
    pub async fn set_devices(&self, devices: Vec<Device>, pins: BTreeMap<ParticipantId, Pin>) {
        let hosts = self.hosts.lock().await;
        let (old, old_pins) = std::mem::replace(
            &mut *self
                .table
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            (devices.clone(), pins.clone()),
        );
        let own_tables = self
            .own_tables
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        for (session, host) in hosts.iter().filter(|(s, _)| !own_tables.contains(s)) {
            if let Err(err) = apply(host, (&old, &old_pins), (&devices, &pins)).await {
                warn!(%session, %err, "device table not applied");
            }
        }
    }

    /// Starts `config.session` now, with `config` (its devices and pins
    /// replaced by the server's): kept until [`Self::stop`], whoever
    /// attaches or leaves. Its authority snapshots, as they change.
    ///
    /// # Errors
    /// [`SessionError::Config`] when the session is already hosted here, or
    /// what starting it failed with.
    pub async fn start_session(
        &self,
        config: HostConfig,
    ) -> Result<tokio::sync::watch::Receiver<AuthoritySnapshot>, SessionError> {
        self.start_kept(config, false).await
    }

    /// [`Self::start_session`] with `config`'s own devices and pins, which
    /// the server's table never changes: a satellite's run, whose table is
    /// its home daemon's (scene design §4.4).
    ///
    /// # Errors
    /// As [`Self::start_session`].
    pub async fn host_session(
        &self,
        config: HostConfig,
    ) -> Result<tokio::sync::watch::Receiver<AuthoritySnapshot>, SessionError> {
        self.start_kept(config, true).await
    }

    async fn start_kept(
        &self,
        config: HostConfig,
        own_table: bool,
    ) -> Result<tokio::sync::watch::Receiver<AuthoritySnapshot>, SessionError> {
        let mut hosts = self.hosts.lock().await;
        let known = self
            .kept
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(&config.session);
        if known || hosts.contains_key(&config.session) {
            return Err(SessionError::Config(format!(
                "session {} is already hosted here",
                config.session
            )));
        }
        let config = if own_table {
            self.own_tables
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(config.session);
            config
        } else {
            let (devices, pins) = self
                .table
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            HostConfig {
                devices,
                pins,
                ..config
            }
        };
        let session = config.session;
        let host = SessionHost::start(self.transport.clone(), config).await?;
        let watch = host.watch();
        self.kept
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(session, true);
        hosts.insert(session, host);
        info!(%session, "session started");
        Ok(watch)
    }

    /// Attaches `id` to `session` from this process, riding the server's
    /// transport, as device `local` (the home daemon watching and arming
    /// its runs).
    ///
    /// # Errors
    /// What attaching failed with.
    pub async fn attach(
        &self,
        session: SessionId,
        id: ParticipantId,
        roles: Vec<crate::wire::Role>,
        priority: crate::wire::PriorityClass,
    ) -> Result<Participant, SessionError> {
        let (tenant, clock, credential) = &self.own;
        let mut config = ParticipantConfig::new(tenant.clone(), session, id, roles, priority);
        config.clock = clock.clone();
        config.credential = Some(credential.clone());
        Participant::attach(self.transport.ridden(), config).await
    }

    /// Closes the scene `session` runs (see [`SessionHost::close_scene`]).
    ///
    /// # Errors
    /// [`SessionError::Closed`] when it isn't hosted here, or what closing
    /// failed with.
    pub async fn close_scene(
        &self,
        session: SessionId,
        code: crate::wire::ReasonCode,
        by: &str,
        detail: &str,
    ) -> Result<(), SessionError> {
        let hosts = self.hosts.lock().await;
        let host = hosts.get(&session).ok_or(SessionError::Closed)?;
        host.close_scene(code, by, detail).await
    }

    /// Stops `session` (participants see its host leave).
    pub async fn stop(&self, session: SessionId) {
        self.kept
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(session, false);
        let host = self.hosts.lock().await.remove(&session);
        if let Some(host) = host {
            #[cfg(feature = "record")]
            finish_recorder(&self.recorders, session).await;
            host.shutdown().await;
            info!(%session, "session stopped");
        }
    }

    /// Stops every session (participants see their hosts leave).
    pub async fn shutdown(&self) {
        #[cfg(feature = "record")]
        {
            let sessions: Vec<SessionId> = self.recorders.lock().await.keys().copied().collect();
            for session in sessions {
                finish_recorder(&self.recorders, session).await;
            }
        }
        let hosts: Vec<SessionHost> = self.hosts.lock().await.drain().map(|(_, h)| h).collect();
        for host in hosts {
            host.shutdown().await;
        }
    }
}

/// Changes `host`'s device table from `old` to `new`: revocations first, so a
/// key moved to another id is free when it is admitted.
async fn apply(
    host: &SessionHost,
    (old, old_pins): (&[Device], &BTreeMap<ParticipantId, Pin>),
    (new, pins): (&[Device], &BTreeMap<ParticipantId, Pin>),
) -> Result<(), SessionError> {
    for was in old {
        match new.iter().find(|d| d.id == was.id) {
            Some(now) if now.public == was.public => {
                if now.access != was.access {
                    host.set_access(&now.id, now.access.clone()).await?;
                }
            }
            _ => host.revoke(&was.id).await?,
        }
    }
    for now in new {
        if !old.iter().any(|d| d.id == now.id && d.public == now.public) {
            host.admit(now.clone()).await?;
        }
    }
    let robots: std::collections::BTreeSet<&ParticipantId> =
        old_pins.keys().chain(pins.keys()).collect();
    for robot in robots {
        if old_pins.get(robot) != pins.get(robot) {
            host.pin(robot, pins.get(robot).cloned()).await?;
        }
    }
    Ok(())
}

/// Finishes `session`'s recorder, if it has one, so its last segment is
/// complete before the host stops.
#[cfg(feature = "record")]
async fn finish_recorder(recorders: &Recorders, session: SessionId) {
    let started = recorders.lock().await.remove(&session);
    if let Some(started) = started
        && let Ok(Some(recorder)) = started.await
        && let Err(err) = recorder.finish().await
    {
        warn!(%session, %err, "recording did not finish cleanly");
    }
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr};

    use super::*;
    use crate::transport::open_local_listeners;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_late_delete_of_a_rejoined_members_old_token_keeps_it_beside_others() {
        let (transport, _) = open_local_listeners(IpAddr::from(Ipv4Addr::LOCALHOST))
            .await
            .expect("transport");
        let control = transport.control();
        let (a, b) = ("rlmesh/test/members/a", "rlmesh/test/members/b");
        // A rejoined: its new token is declared when its old one's delete lands.
        let token = control.liveliness().declare_token(a).await.expect("token");
        let mut members: HashSet<String> = [a.to_string(), b.to_string()].into();
        assert!(!member_left(&mut members, a, control).await);
        assert!(members.contains(a), "a rejoined member stays tracked");
        // B left for good: no token of its own.
        assert!(member_left(&mut members, b, control).await);
        assert_eq!(members, HashSet::from([a.to_string()]));
        // Then A leaves for good: its replacement token's delete removes it.
        token.undeclare().await.expect("undeclare");
        assert!(member_left(&mut members, a, control).await);
        assert!(members.is_empty(), "{members:?}");
        transport.close().await.expect("close");
    }
}
