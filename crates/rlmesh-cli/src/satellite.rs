//! rlmeshd as a satellite (scene design §1, §4.4): on a machine paired with
//! another rlmeshd, it waits in that home daemon's lobby as
//! `rlmeshd.<device>` and takes its orders, one at a time:
//! - `HostOrder`: host the run's session here, under host keys made here
//!   for the run (private keys never travel), with the order's member
//!   table, device records and pins, and answer `HostStarted` with the
//!   run host's key and this daemon's addresses;
//! - `StopOrder`: close the run's scene with the order's reason, give its
//!   members 2 s to leave, and stop its host.
//!
//! It serves every host it has a host file for, so it takes orders only
//! from the daemons it paired with, found at their stored addresses or
//! over mDNS ([`discovery::locate`]), retried from 1 s to 30 s. A home lost
//! for 10 s (its liveliness gone, or no authenticated echo for
//! [`rlmesh_session::lobby::HOME_SILENT`]) has the runs hosted here for it
//! closed (`SCENE_FAILED`), so no run outlives the daemon that can stop it;
//! a home back as another lobby instance (it restarted) has them closed at
//! once. Runs are kept per home: two homes may pick the same run id.

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use rlmesh_session::address::{LEASE_GRACE, SessionAddress, TENANT};
use rlmesh_session::auth::{Device, Pin};
use rlmesh_session::authority::{DeviceAccess, Verb};
use rlmesh_session::discovery;
use rlmesh_session::lobby::{LOBBY_SESSION, SatelliteClient};
use rlmesh_session::pairing::HostFile;
use rlmesh_session::scene::Scene;
use rlmesh_session::server::SessionServer;
use rlmesh_session::wire::{
    Assignment, AuthoritySnapshot, CtlError, CtlStatus, HostOrder, HostStarted, ParticipantId,
    Reason, ReasonCode, SessionId, StopOrder, TenantId,
};
use rlmesh_session::{HostConfig, SessionError, SystemMonotonicClock};
use tokio::sync::watch;
use tokio::task::JoinHandle;

/// How long a home may be gone before the runs hosted for it close.
const HOME_LOST: Duration = Duration::from_secs(10);
/// How long a closing run's members get to leave.
const STRAGGLERS: Duration = Duration::from_secs(2);
/// How long a satellite waits for its home's lobby to answer an attach.
const ATTACH: Duration = Duration::from_secs(10);

/// A run hosted here.
struct Hosted {
    /// The lobby instance of its home that ordered it.
    instance: u64,
    session: SessionId,
    members: Vec<ParticipantId>,
    snapshots: watch::Receiver<AuthoritySnapshot>,
}

/// This daemon's satellite side.
pub struct Satellite {
    server: Arc<SessionServer>,
    /// The base addresses this daemon's runs are joined at.
    addresses: Vec<String>,
    /// By home (host id) and run.
    runs: Mutex<HashMap<(String, String), Hosted>>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn verbs(names: &[String]) -> Vec<Verb> {
    names
        .iter()
        .filter_map(|v| match v.as_str() {
            "arm" => Some(Verb::Arm),
            "disarm" => Some(Verb::Disarm),
            _ => None,
        })
        .collect()
}

impl Satellite {
    /// Serves every home in `hosts` (host files, [`rlmesh_session::pairing`])
    /// but this daemon's own (`own_id`), hosting their runs on `server`,
    /// joined at `addresses`, until the returned tasks are aborted.
    pub fn start(
        server: Arc<SessionServer>,
        hosts: Option<PathBuf>,
        own_id: &str,
        addresses: Vec<String>,
    ) -> Vec<JoinHandle<()>> {
        let satellite = Arc::new(Self {
            server,
            addresses,
            runs: Mutex::default(),
        });
        // TODO: the host files are read once, at start; a machine paired
        // while rlmeshd runs serves its new home after a restart.
        let mut paths: Vec<PathBuf> = hosts
            .and_then(|dir| std::fs::read_dir(dir).ok())
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "toml"))
            .filter(|path| path.file_stem().is_some_and(|stem| stem != own_id))
            .collect();
        paths.sort();
        paths
            .into_iter()
            .map(|path| tokio::spawn(satellite.clone().serve(path)))
            .collect()
    }

    /// Serves the home of the host file at `path`: attached to its lobby,
    /// taking its orders, and reattaching from 1 s to 30 s apart (back to
    /// 1 s once an order came).
    async fn serve(self: Arc<Self>, path: PathBuf) {
        let mut backoff = Duration::from_secs(1);
        let mut lost: Option<JoinHandle<()>> = None;
        let mut reported = false;
        loop {
            let attached = async {
                let file = HostFile::read(&path)?;
                let (credential, device) = (file.credential(), file.device.clone());
                let (file, client) = discovery::locate(&path, |base| {
                    let (credential, device) = (credential.clone(), device.clone());
                    async move {
                        let lobby = SessionAddress {
                            credential: Some(credential),
                            ..SessionAddress::new(base, LOBBY_SESSION)?
                        };
                        SatelliteClient::connect(&lobby, &device, ATTACH).await
                    }
                })
                .await?;
                Ok::<_, SessionError>((file.host_id, client))
            };
            match attached.await {
                Ok((home, client)) => {
                    // Back in the same lobby, its runs stay; a restarted
                    // home has none of them any more.
                    if let Some(lost) = lost.take() {
                        lost.abort();
                    }
                    let instance = client.home_instance();
                    self.home_lost(&home, Some(instance), "home daemon restarted")
                        .await;
                    reported = false;
                    let (err, took) = self.take_orders(&home, instance, &client).await;
                    let _ = writeln!(std::io::stderr(), "rlmeshd: out of {home}'s lobby: {err}");
                    let _ = client.close().await;
                    if took {
                        backoff = Duration::from_secs(1);
                    }
                    let me = self.clone();
                    lost = Some(tokio::spawn(async move {
                        tokio::time::sleep(HOME_LOST).await;
                        me.home_lost(&home, None, "home daemon lost").await;
                    }));
                }
                // Once until it is reached again.
                Err(err) if !reported => {
                    reported = true;
                    let _ = writeln!(
                        std::io::stderr(),
                        "rlmeshd: the home of {} not reached: {err}",
                        path.display()
                    );
                }
                Err(_) => {}
            }
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(Duration::from_secs(30));
        }
    }

    /// Takes the home `home`'s orders until the lobby fails: why, and
    /// whether any order came.
    async fn take_orders(
        &self,
        home: &str,
        instance: u64,
        client: &SatelliteClient,
    ) -> (SessionError, bool) {
        let mut took = false;
        loop {
            let order = match client.next_order().await {
                Ok(order) => order,
                Err(err) => return (err, took),
            };
            took = true;
            match order {
                Assignment::Host(order) => {
                    let started = HostStarted {
                        run: order.run.clone(),
                        result: self.host(home, instance, order).await,
                    };
                    if let Err(err) = client.host_started(started).await {
                        return (err, took);
                    }
                }
                Assignment::Stop(order) => self.stop(home, order).await,
                // TODO: serve members are refused at home until a catalog
                // model can bind as a scene controller; start `serve` here
                // then.
                Assignment::Serve(_) | Assignment::Join(_) => {}
            }
        }
    }

    /// Hosts `order`'s run here: the run host's key and this daemon's
    /// addresses.
    async fn host(
        &self,
        home: &str,
        instance: u64,
        order: HostOrder,
    ) -> Result<([u8; 32], Vec<String>), CtlError> {
        let refused = |status, message: String| CtlError::new(status, message);
        let key = (home.to_string(), order.run.clone());
        if lock(&self.runs).contains_key(&key) {
            return Err(refused(
                CtlStatus::InvalidRequest,
                format!("run {} is hosted here already", order.run),
            ));
        }
        let scene = Scene::parse(&order.scene_toml)
            .map_err(|refusal| refused(CtlStatus::InvalidRequest, refusal.to_string()))?;
        let devices: Vec<Device> = order
            .devices
            .iter()
            .map(|record| Device {
                id: record.id.clone(),
                public: record.public_key,
                access: DeviceAccess {
                    max_priority: record.max_priority,
                    parts: record.parts.clone(),
                    verbs: verbs(&record.verbs),
                },
            })
            .collect();
        let pins: BTreeMap<ParticipantId, Pin> = order
            .pins
            .iter()
            .map(|pin| {
                (
                    pin.robot.clone(),
                    Pin {
                        device: pin.device.clone(),
                        board_id: pin.board_id,
                    },
                )
            })
            .collect();
        let tenant = TenantId::new(TENANT)
            .map_err(|err| refused(CtlStatus::InvalidRequest, err.to_string()))?;
        // Fresh keys: the run host's own, made here.
        let mut config = HostConfig::new(tenant, order.session);
        config.clock = SystemMonotonicClock.shared();
        config.lease_grace = LEASE_GRACE;
        config.health_log = None;
        config.scene = Some(scene.host(&order.run, &pins));
        config.seal = scene.seal();
        config.devices = devices;
        config.pins = pins;
        let host_key = *config.identity.static_public();
        let snapshots = self
            .server
            .host_session(config)
            .await
            .map_err(|err| refused(CtlStatus::NotReady, format!("the run's host: {err}")))?;
        lock(&self.runs).insert(
            key,
            Hosted {
                instance,
                session: order.session,
                members: scene.members.iter().map(|m| m.name.clone()).collect(),
                snapshots,
            },
        );
        Ok((host_key, self.addresses.clone()))
    }

    /// Closes home `home`'s run `order.run` with its reason, waits up to
    /// 2 s for its members to leave, and stops its host.
    async fn stop(&self, home: &str, order: StopOrder) {
        let key = (home.to_string(), order.run);
        let Some(mut hosted) = lock(&self.runs).remove(&key) else {
            return;
        };
        let reason = order.reason;
        if self
            .server
            .close_scene(hosted.session, reason.code, &reason.by, &reason.detail)
            .await
            .is_ok()
        {
            let members = hosted.members.clone();
            let _ = tokio::time::timeout(
                STRAGGLERS,
                hosted.snapshots.wait_for(|s| {
                    !members
                        .iter()
                        .any(|m| s.participant(m).is_some_and(|p| p.present))
                }),
            )
            .await;
        }
        self.server.stop(hosted.session).await;
    }

    /// Closes every run hosted for home `home` (`SCENE_FAILED`, `why`) but
    /// those of its lobby instance `keep`: it has been gone for
    /// [`HOME_LOST`] (`None`), or is back as another instance.
    async fn home_lost(&self, home: &str, keep: Option<u64>, why: &str) {
        let runs: Vec<String> = lock(&self.runs)
            .iter()
            .filter(|((of, _), hosted)| of == home && Some(hosted.instance) != keep)
            .map(|((_, run), _)| run.clone())
            .collect();
        for run in runs {
            let mut reason = Reason::new(ReasonCode::SceneFailed);
            reason.detail = why.into();
            self.stop(home, StopOrder { run, reason }).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::net::{Ipv4Addr, SocketAddr};

    use rlmesh_session::address::{self, DEFAULT_SESSION, session_id};

    use super::*;

    fn order(run: &str) -> HostOrder {
        HostOrder {
            run: run.into(),
            session: SessionId::new(),
            scene_toml: "[scene]\nformat = 1\nname = \"bench\"\n\n\
                         [members.arm]\nrole = \"robot\"\ndevice = \"local\"\n"
                .into(),
            devices: Vec::new(),
            pins: Vec::new(),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn runs_are_kept_per_home_and_a_restarted_homes_close_at_once() {
        let (transport, _) =
            address::open_listeners(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)), None)
                .await
                .expect("listeners");
        let mut config = HostConfig::new(
            TenantId::new(TENANT).expect("tenant"),
            session_id(DEFAULT_SESSION),
        );
        config.health_log = None;
        let server = Arc::new(
            SessionServer::start(transport, config)
                .await
                .expect("server"),
        );
        let satellite = Satellite {
            server: server.clone(),
            addresses: Vec::new(),
            runs: Mutex::default(),
        };
        // Two homes pick the same run id; a third run of the first home.
        let (a, b, a2) = (order("r-7k2m"), order("r-7k2m"), order("r-9p3q"));
        let sessions = (a.session, b.session, a2.session);
        satellite.host("rlm-aaaaaa", 1, a).await.expect("a's");
        satellite.host("rlm-bbbbbb", 1, b).await.expect("b's");
        let again = satellite.host("rlm-aaaaaa", 1, order("r-7k2m")).await;
        assert_eq!(again.map_err(|e| e.status), Err(CtlStatus::InvalidRequest));
        satellite
            .host("rlm-aaaaaa", 1, a2)
            .await
            .expect("a's second");
        let hosted = || async { server.sessions().await };

        // A's stop closes a's run only.
        let mut reason = Reason::new(ReasonCode::SceneEnded);
        reason.by = "cli.1a2b3c4d".into();
        let stop = StopOrder {
            run: "r-7k2m".into(),
            reason,
        };
        satellite.stop("rlm-aaaaaa", stop).await;
        let now = hosted().await;
        assert!(!now.contains(&sessions.0) && now.contains(&sessions.1));

        // A back as the same lobby instance keeps its run; as another (it
        // restarted), its run closes, and b's stays.
        satellite
            .home_lost("rlm-aaaaaa", Some(1), "home daemon restarted")
            .await;
        assert!(hosted().await.contains(&sessions.2));
        satellite
            .home_lost("rlm-aaaaaa", Some(2), "home daemon restarted")
            .await;
        let now = hosted().await;
        assert!(!now.contains(&sessions.2) && now.contains(&sessions.1));
        server.shutdown().await;
    }
}
