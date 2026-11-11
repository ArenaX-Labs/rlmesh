//! Tests of the lobby, its runs and their drivers.

use std::net::{Ipv4Addr, SocketAddr};

use rlmesh_session::SystemMonotonicClock;
use rlmesh_session::address::TENANT;
use rlmesh_session::address::{self, DEFAULT_SESSION, SessionAddress};
use rlmesh_session::auth::Credential;
use rlmesh_session::auth::address::{AddressKey, base32};
use rlmesh_session::crypto::StaticKeypair;
use rlmesh_session::edge::{EdgeError, Robot};
use rlmesh_session::lobby::{
    Lobby, LobbyClient, SatelliteClient, home_lobby, run_address, wait_for_join,
};
use rlmesh_session::pairing::{HostFile, Paired};
use rlmesh_session::sdk;
use rlmesh_session::spec::RunOptions;
use rlmesh_session::spec::apply::{Tensor, Value};
use rlmesh_session::wire::{DType, PairInfo, TenantId};
use rlmesh_session::wire::{Mode, ReasonCode, Role};
use rlmesh_session::{Controller, Observation, Predictor, RunConfig, RunError};

use super::*;

const WAIT: Duration = Duration::from_secs(10);
const ARM: &str = r#"{"arm": {"kind": "joints", "count": 2, "fields": ["pos"], "hz": 100,
    "command": "position", "safe": "stop", "hold_max_ms": 100}}"#;
const PILOT: &str = r#"{"hz": 20, "inputs": {"q": {"kind": "joints", "at": "arm", "fields": ["pos"]}},
    "actions": {"arm": {"kind": "joints", "at": "arm", "command": "position", "hz": 20, "chunk": 2}}}"#;

fn scene(wait: &str) -> String {
    format!(
        "[scene]\nformat = 1\nname = \"bench\"\nwait = \"{wait}\"\n\n\
         [members.arm]\nrole = \"robot\"\n\n\
         [members.pilot]\nrole = \"controller\"\ndevice = \"local\"\n"
    )
}

/// A home daemon in `dir` on a loopback port: its lobby's address.
async fn home(dir: &std::path::Path) -> (Arc<SessionServer>, SessionAddress) {
    let (server, lobby, _) = home_gated(dir, true).await;
    (server, lobby)
}

/// [`home`], its links gated by its key's password or (`gated` false)
/// by none, as rlmeshd's: paired machines link to it too. Its devices
/// are `devices.toml`'s.
async fn home_gated(
    dir: &std::path::Path,
    gated: bool,
) -> (Arc<SessionServer>, SessionAddress, Arc<Home>) {
    let identity = Arc::new(crate::host::identity(dir).expect("identity"));
    let key = AddressKey::of(&identity);
    let password = gated.then(|| key.link_password());
    let (transport, base) = address::open_listeners(
        SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
        password.as_ref(),
    )
    .await
    .expect("listeners");
    let mut config = HostConfig::new(
        TenantId::new(TENANT).expect("tenant"),
        session_id(DEFAULT_SESSION),
    );
    config.identity = identity;
    config.clock = SystemMonotonicClock.shared();
    config.health_log = None;
    (config.devices, config.pins) = Table::read(dir)
        .and_then(|table| table.host())
        .expect("devices.toml");
    let server = Arc::new(
        SessionServer::start(transport, config.clone())
            .await
            .expect("server"),
    );
    let home = Home::start(&server, dir.into(), config, vec![base.to_string()])
        .await
        .expect("lobby");
    let lobby = SessionAddress {
        key: Some(key),
        ..home_lobby(Some(&base.to_string())).expect("lobby address")
    };
    (server, lobby, home)
}

/// A robot process: waits in the lobby, joins its run, stays until the
/// run closes, then comes back to the lobby.
async fn robot(lobby: SessionAddress) {
    let mut me = Lobby::new(ParticipantId::new("arm").expect("id"));
    let (_, address) = wait_for_join(&lobby, &mut me, "arm", Some(WAIT * 3))
        .await
        .expect("a join");
    let robot = sdk::connect_robot(&address, "arm", ARM, WAIT)
        .await
        .expect("joined");
    robot
        .participant()
        .wait_for("closing", WAIT * 3, |s| {
            s.scene.as_ref().is_some_and(|s| s.closing.is_some())
        })
        .await
        .expect("closing");
    robot.detach().await.expect("detach");
    let back = wait_for_join(&lobby, &mut me, "arm", Some(Duration::from_secs(1))).await;
    assert!(back.is_err(), "no second run assigns it");
}

async fn until(
    client: &LobbyClient,
    mut state: RunState,
    done: impl Fn(&RunState) -> bool,
) -> RunState {
    while !done(&state) {
        assert!(
            !matches!(state.stage, RunStage::Ended | RunStage::Failed) || done(&state),
            "run ended early: {state:?}"
        );
        state = client
            .status(&state.run, state.version, Some(WAIT * 2))
            .await
            .expect("status");
    }
    state
}

fn stage(state: &RunState, name: &str) -> MemberStage {
    state
        .members
        .iter()
        .find(|m| m.name.as_str() == name)
        .map(|m| m.stage)
        .expect("member")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lobby_robot_is_assigned_armed_once_its_policy_bound_claimed_and_returned() {
    let dir = tempfile::tempdir().expect("dir");
    let (server, lobby) = home(dir.path()).await;
    let robot = tokio::spawn(robot(lobby.clone()));
    let client = LobbyClient::connect(&lobby, WAIT).await.expect("client");
    let state = client.run_scene(&scene("20s"), false).await.expect("run");
    assert_eq!(state.stage, RunStage::Hosting);

    // Declared, but its controller hasn't bound: no arm yet.
    let state = until(&client, state, |s| s.stage == RunStage::Binding).await;
    assert_eq!(stage(&state, "arm"), MemberStage::Declared);
    let address = run_address(&lobby, &state)
        .expect("credential")
        .expect("hosted");
    let busy = client
        .run_scene(&scene("20s"), false)
        .await
        .expect_err("arm is claimed");
    match busy {
        SessionError::Rejected(err) => {
            assert_eq!(err.status, CtlStatus::RobotBusy);
            assert!(err.message.contains(&state.run), "{}", err.message);
        }
        other => panic!("{other}"),
    }

    let pilot = sdk::bind_policy(
        &address,
        "pilot",
        PILOT,
        &RunOptions::named("pilot"),
        PriorityClass::Model,
        WAIT,
    )
    .await
    .expect("bound");
    let state = until(&client, state, |s| s.stage == RunStage::Running).await;
    assert_eq!(stage(&state, "pilot"), MemberStage::Bound);
    pilot
        .participant()
        .wait_for("armed", WAIT, |s| {
            matches!(s.mode, Mode::Armed | Mode::Active)
        })
        .await
        .expect("armed");

    client.stop(&state.run).await.expect("stop");
    let state = until(&client, state, |s| {
        s.stage == RunStage::Ended && stage(s, "arm") == MemberStage::Returned
    })
    .await;
    assert_eq!(state.error, None);
    drop(pilot);
    robot.await.expect("robot");
    // Its claim went with its return: the robot may run again.
    let again = client.run_scene(&scene("1s"), false).await.expect("free");
    assert_eq!(again.stage, RunStage::Hosting);
    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_robot_back_in_the_lobby_as_its_run_ends_is_returned_and_free_for_the_next() {
    const RUNS: usize = 8;
    let dir = tempfile::tempdir().expect("dir");
    let (server, lobby, service) = home_gated(dir.path(), true).await;
    // A robot that is back in the lobby within milliseconds of seeing
    // its run close, as a board is.
    let robot = tokio::spawn({
        let lobby = lobby.clone();
        async move {
            let mut me = Lobby::new(ParticipantId::new("arm").expect("id"));
            for _ in 0..RUNS {
                let (_, address) = wait_for_join(&lobby, &mut me, "arm", Some(WAIT * 3))
                    .await
                    .expect("a join");
                let robot = sdk::connect_robot(&address, "arm", ARM, WAIT)
                    .await
                    .expect("joined");
                robot
                    .participant()
                    .wait_for("closing", WAIT * 3, |s| {
                        s.scene.as_ref().is_some_and(|s| s.closing.is_some())
                    })
                    .await
                    .expect("closing");
                robot.detach().await.expect("detach");
            }
            let back = wait_for_join(&lobby, &mut me, "arm", Some(Duration::from_secs(1))).await;
            assert!(back.is_err(), "no other run assigns it");
        }
    });
    let client = LobbyClient::connect(&lobby, WAIT).await.expect("client");
    // The robot alone: its run stops as soon as it left.
    let alone = scene("20s").replace(
        "\n[members.pilot]\nrole = \"controller\"\ndevice = \"local\"\n",
        "",
    );
    for _ in 0..RUNS {
        let state = client.run_scene(&alone, false).await.expect("free");
        let state = until(&client, state, |s| s.stage == RunStage::Running).await;
        client.stop(&state.run).await.expect("stop");
        let state = until(&client, state, |s| s.stage == RunStage::Ended).await;
        assert_eq!(stage(&state, "arm"), MemberStage::Returned, "{state:?}");
    }
    robot.await.expect("robot");

    // Back in the lobby once its run is stopping, before it is ending:
    // returned at once, not sent into the closing run.
    let arm = ParticipantId::new("arm").expect("id");
    let wait = || match service.handle(&arm, "local", Command::LobbyWait(LobbyWait::default())) {
        LobbyAnswer::Later(waited) => waited,
        LobbyAnswer::Now(answer) => panic!("not a wait: {answer:?}"),
    };
    let waited = wait();
    let state = client.run_scene(&alone, false).await.expect("free");
    let assigned = waited.await.expect("answered");
    assert!(
        matches!(assigned, Ok(LobbyReply::Assignment(Assignment::Join(_)))),
        "{assigned:?}"
    );
    let stopped = std::time::Instant::now();
    // As a stop does before the run hears of it.
    let run = service.run(&state.run).expect("run");
    run.stopping.store(true, Ordering::SeqCst);
    let back = wait();
    let stage_then = run.state.borrow().stage;
    assert!(!over(stage_then), "{stage_then:?}");
    client.stop(&state.run).await.expect("stop");
    let state = until(&client, state, |s| s.stage == RunStage::Ended).await;
    assert_eq!(stage(&state, "arm"), MemberStage::Returned, "{state:?}");
    assert!(
        stopped.elapsed() < STRAGGLERS / 2,
        "{:?}",
        stopped.elapsed()
    );
    client.run_scene(&alone, false).await.expect("free at once");
    let next = back.await.expect("answered");
    assert!(
        matches!(next, Ok(LobbyReply::Assignment(Assignment::Join(_)))),
        "{next:?}"
    );
    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_robot_process_carries_its_uncleared_estop_into_its_next_run() {
    let dir = tempfile::tempdir().expect("dir");
    let (server, lobby) = home(dir.path()).await;
    let client = LobbyClient::connect(&lobby, WAIT).await.expect("client");
    let alone = scene("20s").replace(
        "\n[members.pilot]\nrole = \"controller\"\ndevice = \"local\"\n",
        "",
    );
    let from_lobby = || sdk::robot_from_lobby_at(&lobby, "arm", ARM, Some(WAIT * 3));
    let state = client.run_scene(&alone, false).await.expect("run");
    let (robot, _) = from_lobby().await.expect("joined");
    let state = until(&client, state, |s| s.stage == RunStage::Running).await;
    // Its button, then the run ends: the documented loop drops the
    // robot and waits again, never clearing.
    robot.estop("button");
    client.stop(&state.run).await.expect("stop");
    robot
        .participant()
        .wait_for("closing", WAIT, |s| {
            s.scene.as_ref().is_some_and(|s| s.closing.is_some())
        })
        .await
        .expect("closing");
    robot.detach().await.expect("detach");
    let back = tokio::spawn({
        let lobby = lobby.clone();
        async move { sdk::robot_from_lobby_at(&lobby, "arm", ARM, Some(WAIT * 3)).await }
    });
    until(&client, state, |s| s.stage == RunStage::Ended).await;

    let state = client.run_scene(&alone, false).await.expect("free");
    let (robot, _) = back.await.expect("task").expect("joined again");
    assert!(robot.participant().estop_latched(), "the same latch");
    let state = until(&client, state, |s| s.stage == RunStage::Failed).await;
    let error = state.error.expect("why");
    assert_eq!(error.status, CtlStatus::EstopLatched, "{}", error.message);
    let mode = robot.participant().fetch_authority().await.map(|s| s.mode);
    assert!(matches!(mode, Ok(Mode::Estop) | Err(_)), "{mode:?}");
    drop(robot);
    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_run_fails_at_its_stage_with_its_code() {
    let dir = tempfile::tempdir().expect("dir");
    let (server, lobby) = home(dir.path()).await;
    let client = LobbyClient::connect(&lobby, WAIT).await.expect("client");
    let refused = client
        .run_scene("[scene]\nformat = 2\n", false)
        .await
        .expect_err("a bad file");
    assert!(
        matches!(&refused, SessionError::Rejected(e) if e.status == CtlStatus::InvalidRequest),
        "{refused}"
    );
    // No robot waits: it fails at assignment once `wait` passes.
    let state = client.run_scene(&scene("1s"), false).await.expect("run");
    let state = until(&client, state, |s| s.stage == RunStage::Failed).await;
    let error = state.error.expect("why");
    assert_eq!(error.status, CtlStatus::NotPresent);
    assert!(
        error.message.contains("arm is not in the lobby"),
        "{}",
        error.message
    );
    // A robot that declared, and a controller that never bound.
    let robot = tokio::spawn(robot(lobby.clone()));
    let state = client.run_scene(&scene("3s"), false).await.expect("run");
    let state = until(&client, state, |s| s.stage == RunStage::Failed).await;
    let error = state.error.expect("why");
    assert_eq!(error.status, CtlStatus::NotReady);
    assert!(
        error.message.contains("pilot has not bound"),
        "{}",
        error.message
    );
    robot.await.expect("robot");
    server.shutdown().await;
}

const ARM_CHANGED: &str = r#"{"arm": {"kind": "joints", "count": 3, "fields": ["pos"], "hz": 100,
    "command": "position", "safe": "stop", "hold_max_ms": 100}}"#;

struct Hold;

impl Predictor for Hold {
    type Error = String;

    fn predict(&mut self, _: Observation, _: i64) -> Result<Vec<(String, Value)>, String> {
        let hold = Tensor::from_f64(DType::F32, vec![2, 2], &[0.0; 4]);
        Ok(vec![("arm".into(), Value::Tensor(hold))])
    }
}

/// Robot `arm` from the lobby, declaring `map` and publishing its
/// joints; a refused join is reported in its next wait.
async fn joined(lobby: &SessionAddress, me: &mut Lobby, map: &str) -> Option<Arc<Robot>> {
    let (join, address) = wait_for_join(lobby, me, "arm", Some(WAIT * 3))
        .await
        .expect("a join");
    let robot = match sdk::connect_robot(&address, "arm", map, WAIT).await {
        Ok(robot) => Arc::new(robot),
        Err(EdgeError::Session(SessionError::Rejected(err))) => {
            me.failed(&join.run, err);
            return None;
        }
        Err(err) => panic!("{err}"),
    };
    let weak = Arc::downgrade(&robot);
    tokio::spawn(async move {
        while let Some(robot) = weak.upgrade() {
            let now = robot.participant().clock().now_ns();
            let _ = robot.publish("arm", now, &[&[0_u8; 8]]);
            drop(robot);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });
    Some(robot)
}

/// Stops `robot`'s heartbeats and drops it: the host loses it.
fn lose(robot: Arc<Robot>) {
    robot.participant().set_heartbeats_enabled(false);
    drop(robot);
}

async fn holding(pilot: &Controller) {
    let deadline = tokio::time::Instant::now() + WAIT;
    while !pilot.holds() {
        assert!(tokio::time::Instant::now() < deadline, "no lease");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

fn rejoin_scene(rejoin: &str) -> String {
    scene("20s").replace(
        "wait = \"20s\"\n",
        &format!("wait = \"20s\"\nrejoin = \"{rejoin}\"\n"),
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lost_robot_rejoins_with_its_map_and_the_run_rearms_only_when_asked() {
    let dir = tempfile::tempdir().expect("dir");
    let (server, lobby) = home(dir.path()).await;
    let client = LobbyClient::connect(&lobby, WAIT).await.expect("client");
    let mut me = Lobby::new(ParticipantId::new("arm").expect("id"));
    let state = client
        .run_scene(&rejoin_scene("20s"), false)
        .await
        .expect("run");
    let robot = joined(&lobby, &mut me, ARM).await.expect("declared");
    let state = until(&client, state, |s| s.stage == RunStage::Binding).await;
    let address = run_address(&lobby, &state)
        .expect("credential")
        .expect("hosted");
    let pilot = Arc::new(
        sdk::bind_policy(
            &address,
            "pilot",
            PILOT,
            &RunOptions::named("pilot"),
            PriorityClass::Model,
            WAIT,
        )
        .await
        .expect("bound"),
    );
    // Lost while ARMED, before its policy leased: the session goes SAFE
    // all the same, and is never ACTIVE again until the explicit arm.
    let state = until(&client, state, |s| s.stage == RunStage::Running).await;
    lose(robot);
    let state = client
        .wait_until(&state.run, RunStage::Rejoining, WAIT)
        .await
        .expect("rejoining");
    assert_eq!(stage(&state, "arm"), MemberStage::Lost);
    let active = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut modes = pilot.participant().watch();
    let watching = tokio::spawn({
        let active = active.clone();
        async move {
            while modes.changed().await.is_ok() {
                if modes
                    .borrow()
                    .as_ref()
                    .is_some_and(|s| s.mode == Mode::Active)
                {
                    active.store(true, std::sync::atomic::Ordering::SeqCst);
                }
            }
        }
    });
    let running = std::thread::spawn({
        let pilot = pilot.clone();
        move || pilot.run(&mut Hold, RunConfig::default())
    });
    // Attached under its name but not declared: still rejoining.
    let (_, back) = wait_for_join(&lobby, &mut me, "arm", Some(WAIT * 3))
        .await
        .expect("its join again");
    let back = back
        .join("arm", vec![Role::Observer], PriorityClass::Scripted, WAIT)
        .await
        .expect("attached");
    tokio::time::sleep(Duration::from_millis(300)).await;
    let status = client.status(&state.run, 0, Some(WAIT)).await;
    assert_eq!(status.expect("status").stage, RunStage::Rejoining);
    back.set_heartbeats_enabled(false);
    drop(back);
    assert!(
        !running.is_finished(),
        "ROBOT_LOST in a scene is not the end"
    );
    // Re-flashed with another map: refused, and the run waits on.
    assert!(joined(&lobby, &mut me, ARM_CHANGED).await.is_none());
    let robot = joined(&lobby, &mut me, ARM).await.expect("the same map");
    let state = until(&client, state, |s| s.stage == RunStage::Arming).await;
    let refusal = state
        .members
        .iter()
        .find(|m| m.name.as_str() == "arm")
        .and_then(|m| m.error.clone());
    assert_eq!(refusal.map(|e| e.status), Some(CtlStatus::Incompatible));
    // `arm = "auto"`, and still it waits for an explicit arm.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let state = client
        .status(&state.run, 0, Some(WAIT))
        .await
        .expect("status");
    assert_eq!(state.stage, RunStage::Arming);
    assert!(
        !active.load(std::sync::atomic::Ordering::SeqCst),
        "ACTIVE before the explicit arm"
    );
    watching.abort();
    let state = client.arm(&state.run).await.expect("arm");
    assert_eq!(state.stage, RunStage::Running);
    holding(&pilot).await;
    // Reset and back at once, likely before its loss was seen: a new
    // generation whose rejoin window the run may never see. SAFE all the
    // same, and armed again on request.
    lose(robot);
    let restarted = joined(&lobby, &mut me, ARM).await.expect("restarted");
    let state = until(&client, state, |s| s.stage == RunStage::Arming).await;
    let state = client
        .arm(&state.run)
        .await
        .expect("armed after the restart");
    assert_eq!(state.stage, RunStage::Running);
    holding(&pilot).await;
    let robot = restarted;
    // Armed again by a routine script: nothing happens.
    let state = client.arm(&state.run).await.expect("a no-op");
    assert_eq!(state.stage, RunStage::Running);
    // Re-arming goes to the run's own host: a robot not held is left as it
    // is, and only a robot in the run can be named.
    let id = |name| ParticipantId::new(name).expect("id");
    client
        .rearm(&state.run, &id("arm"))
        .await
        .expect("not held: a no-op");
    let err = client
        .rearm(&state.run, &id("ghost"))
        .await
        .expect_err("unknown");
    assert!(err.to_string().contains("ghost"), "{err}");

    client.stop(&state.run).await.expect("stop");
    // This process stopped the run: the policy's run returns.
    running.join().expect("thread").expect("returned");
    let state = until(&client, state, |s| s.stage == RunStage::Ended).await;
    assert_eq!(state.error, None);
    drop(robot);
    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lost_robot_that_misses_the_rejoin_window_fails_the_run() {
    let dir = tempfile::tempdir().expect("dir");
    let (server, lobby) = home(dir.path()).await;
    let client = LobbyClient::connect(&lobby, WAIT).await.expect("client");
    let mut me = Lobby::new(ParticipantId::new("arm").expect("id"));
    let state = client
        .run_scene(&rejoin_scene("1s"), false)
        .await
        .expect("run");
    let robot = joined(&lobby, &mut me, ARM).await.expect("declared");
    let state = until(&client, state, |s| s.stage == RunStage::Binding).await;
    let address = run_address(&lobby, &state)
        .expect("credential")
        .expect("hosted");
    let pilot = Arc::new(
        sdk::bind_policy(
            &address,
            "pilot",
            PILOT,
            &RunOptions::named("pilot"),
            PriorityClass::Model,
            WAIT,
        )
        .await
        .expect("bound"),
    );
    let running = std::thread::spawn({
        let pilot = pilot.clone();
        move || pilot.run(&mut Hold, RunConfig::default())
    });
    let state = until(&client, state, |s| s.stage == RunStage::Running).await;
    lose(robot);
    let state = until(&client, state, |s| s.stage == RunStage::Failed).await;
    let error = state.error.expect("why");
    assert!(
        error.message.contains("arm did not rejoin within 1 s"),
        "{}",
        error.message
    );
    match running.join().expect("thread") {
        Err(RunError::Lost(reason)) => assert_eq!(reason.code, ReasonCode::SceneFailed),
        other => panic!("{other:?}"),
    }
    server.shutdown().await;
}

/// A `devices.toml` entry for device `id` with `key`.
fn device(id: &str, key: &StaticKeypair, kind: &str, max: &str) -> String {
    format!(
        "[devices.{id}]\npublic_key = \"{}\"\nkind = \"{kind}\"\nname = \"{id}\"\n\
         max_priority = \"{max}\"\n\n",
        base32(key.public())
    )
}

/// `scene("20s")` hosted on `host`.
fn hosted_on(host: &str) -> String {
    scene("20s").replace("wait = ", &format!("host = \"{host}\"\nwait = "))
}

/// rlmeshd on another machine paired with the home daemon in `home` as
/// device `device` (`key`): its server on its own loopback port, its
/// host file in `hosts`, and its satellite's tasks.
async fn satellite(
    home: &std::path::Path,
    lobby: &SessionAddress,
    device: &str,
    key: &StaticKeypair,
    hosts: &std::path::Path,
) -> (
    Arc<SessionServer>,
    SocketAddr,
    Vec<tokio::task::JoinHandle<()>>,
) {
    let identity = crate::host::identity(home).expect("identity");
    let paired = Paired {
        info: PairInfo {
            device: device.into(),
            host_key: *identity.static_public(),
            host_id: identity.host_id(),
            addresses: Vec::new(),
        },
        device: key.clone(),
    };
    HostFile::new(&paired, &lobby.base.to_string())
        .write(hosts)
        .expect("host file");
    let (transport, base) =
        address::open_listeners(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)), None)
            .await
            .expect("listeners");
    let mut config = HostConfig::new(
        TenantId::new(TENANT).expect("tenant"),
        session_id(DEFAULT_SESSION),
    );
    config.health_log = None;
    let own = config.identity.host_id();
    let server = Arc::new(
        SessionServer::start(transport, config)
            .await
            .expect("server"),
    );
    let tasks = crate::satellite::Satellite::start(
        server.clone(),
        Some(hosts.into()),
        &own,
        vec![base.to_string()],
    );
    (server, base, tasks)
}

/// `client` runs `text`, retried while its satellite isn't in the lobby
/// yet.
async fn run_when_attached(client: &LobbyClient, text: &str) -> RunState {
    let deadline = tokio::time::Instant::now() + WAIT;
    loop {
        match client.run_scene(text, false).await {
            Ok(state) => return state,
            Err(SessionError::Rejected(err))
                if err.status == CtlStatus::NotPresent
                    && tokio::time::Instant::now() < deadline =>
            {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            Err(err) => panic!("{err}"),
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_scene_hosted_on_a_satellite_runs_there_until_a_members_device_changes_at_home() {
    let dir = tempfile::tempdir().expect("dir");
    let hosts = tempfile::tempdir().expect("hosts");
    let gpu = StaticKeypair::from_private(&mut [11; 32]);
    let arm_pc = StaticKeypair::from_private(&mut [12; 32]);
    let arm_pc_key = arm_pc.clone();
    std::fs::write(
        dir.path().join(crate::devices::DEVICES),
        device("gpu-desktop", &gpu, "machine", "operator")
            + &device("arm-pc", &arm_pc, "machine", "model")
            + "[robots.arm]\ndevice = \"arm-pc\"\n",
    )
    .expect("devices.toml");
    let (server, lobby, home) = home_gated(dir.path(), false).await;
    let (remote, remote_base, tasks) =
        satellite(dir.path(), &lobby, "gpu-desktop", &gpu, hosts.path()).await;
    let client = LobbyClient::connect(&lobby, WAIT).await.expect("client");

    // The arm, a robot process on arm-pc: its join takes it to the
    // satellite, where it attaches with its own device key.
    let home_key = *lobby.credential().expect("credential").host_static();
    let robot_lobby = SessionAddress {
        key: None,
        credential: Some(Credential::new(arm_pc, home_key)),
        ..lobby.clone()
    };
    let robot = tokio::spawn(async move {
        let mut me = Lobby::new(ParticipantId::new("arm").expect("id"));
        let (join, address) = wait_for_join(&robot_lobby, &mut me, "arm", Some(WAIT * 3))
            .await
            .expect("a join");
        assert_eq!(
            (address.base, join.addresses),
            (remote_base, vec![remote_base.to_string()])
        );
        let robot = sdk::connect_robot(&address, "arm", ARM, WAIT)
            .await
            .expect("joined on the satellite");
        let arm = ParticipantId::new("arm").expect("id");
        let attached = robot
            .participant()
            .fetch_authority()
            .await
            .expect("state")
            .participant(&arm)
            .map(|p| p.device.clone());
        assert_eq!(attached.as_deref(), Some("arm-pc"));
        robot
            .participant()
            .wait_for("closing", WAIT * 3, |s| {
                s.scene.as_ref().is_some_and(|s| s.closing.is_some())
            })
            .await
            .expect("closing");
        robot.detach().await.expect("detach");
    });

    let state = run_when_attached(&client, &hosted_on("gpu-desktop")).await;
    let state = until(&client, state, |s| s.stage == RunStage::Binding).await;
    let session = state.session.expect("hosted");
    assert_ne!(state.host_key, Some(home_key), "the satellite's own keys");
    assert!(remote.sessions().await.contains(&session));
    assert!(!server.sessions().await.contains(&session));
    let address = run_address(&lobby, &state)
        .expect("credential")
        .expect("hosted");
    let pilot = sdk::bind_policy(
        &address,
        "pilot",
        PILOT,
        &RunOptions::named("pilot"),
        PriorityClass::Model,
        WAIT,
    )
    .await
    .expect("bound on the satellite");
    let state = until(&client, state, |s| s.stage == RunStage::Running).await;

    // arm-pc narrowed at home: the satellite holds its old record, so
    // the run fails, and its StopOrder stops the satellite's host.
    std::fs::write(
        dir.path().join(crate::devices::DEVICES),
        device("gpu-desktop", &gpu, "machine", "operator")
            + &device("arm-pc", &arm_pc_key, "machine", "scripted")
            + "[robots.arm]\ndevice = \"arm-pc\"\n",
    )
    .expect("devices.toml");
    home.devices_changed(&Table::read(dir.path()).expect("table"));
    let state = until(&client, state, |s| s.stage == RunStage::Failed).await;
    let error = state.error.expect("why");
    assert_eq!(error.status, CtlStatus::NotAuthorized, "{}", error.message);
    drop(pilot);
    robot.await.expect("robot");
    let deadline = tokio::time::Instant::now() + WAIT;
    while remote.sessions().await.contains(&session) {
        assert!(tokio::time::Instant::now() < deadline, "still hosted");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    for task in tasks {
        task.abort();
    }
    remote.shutdown().await;
    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn only_a_paired_machine_waits_as_a_satellite_and_hosts_a_run() {
    let dir = tempfile::tempdir().expect("dir");
    let gpu = StaticKeypair::from_private(&mut [11; 32]);
    let eye = StaticKeypair::from_private(&mut [13; 32]);
    let guest = StaticKeypair::from_private(&mut [14; 32]);
    std::fs::write(
        dir.path().join(crate::devices::DEVICES),
        device("gpu-desktop", &gpu, "machine", "operator")
            + &device("eye-l", &eye, "board", "operator")
            + &device("guest", &guest, "machine", "model")
            + "[robots.arm]\ndevice = \"local\"\n",
    )
    .expect("devices.toml");
    let (server, lobby, _) = home_gated(dir.path(), false).await;
    let home_key = *lobby.credential().expect("credential").host_static();
    let as_device = |key: &StaticKeypair| SessionAddress {
        key: None,
        credential: Some(Credential::new(key.clone(), home_key)),
        ..lobby.clone()
    };
    // A board, and a machine under another's name: refused at the wait.
    for (key, name) in [(&eye, "eye-l"), (&gpu, "laptop-josh")] {
        let refused = SatelliteClient::connect(&as_device(key), name, WAIT)
            .await
            .expect("attached as an observer")
            .next_order()
            .await
            .expect_err("not a satellite");
        assert!(
            matches!(&refused, SessionError::Rejected(e) if e.status == CtlStatus::NotAuthorized),
            "{name}: {refused}"
        );
    }
    // A robot waits only from a device that may embody it: a pinned
    // name from its pin, an unpinned one from a device that may pin it
    // (not a guest without `arm`, nor a board under another name).
    for (key, robot) in [(&eye, "arm"), (&guest, "skull"), (&eye, "skull")] {
        let mut me = Lobby::new(ParticipantId::new(robot).expect("id"));
        let refused = wait_for_join(&as_device(key), &mut me, robot, Some(WAIT))
            .await
            .expect_err("may not wait as it");
        assert!(
            matches!(&refused, SessionError::Rejected(e) if e.status == CtlStatus::NotAuthorized),
            "{robot}: {refused}"
        );
    }
    // A satellite whose home's authenticated echoes stop, its liveliness
    // still up (an unsigned token anyone on the link can keep), counts
    // the home gone.
    let silenced = SatelliteClient::connect(&as_device(&gpu), "gpu-desktop", WAIT)
        .await
        .expect("a satellite");
    silenced.participant().set_heartbeats_enabled(false);
    let gone = tokio::time::timeout(
        rlmesh_session::lobby::HOME_SILENT + Duration::from_secs(2),
        silenced.next_order(),
    )
    .await
    .expect("bounded");
    assert!(matches!(gone, Err(SessionError::Closed)), "{gone:?}");
    assert!(*silenced.participant().host_present().borrow());
    // Hosted on the board: refused at check; on a machine not in the
    // lobby: not present.
    let client = LobbyClient::connect(&lobby, WAIT).await.expect("client");
    for (host, status) in [
        ("eye-l", CtlStatus::NotAuthorized),
        ("gpu-desktop", CtlStatus::NotPresent),
    ] {
        let refused = client
            .run_scene(&hosted_on(host), false)
            .await
            .expect_err("refused");
        assert!(
            matches!(&refused, SessionError::Rejected(e) if e.status == status),
            "{host}: {refused}"
        );
    }
    server.shutdown().await;
}
