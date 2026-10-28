//! `Robot::parts` is in authored order, even after a re-declaration gives a
//! part a new, higher channel id than the parts authored after it.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use rlmesh_session::edge::{EdgeConfig, Robot};
use rlmesh_session::transport::open_local_listeners;
use rlmesh_session::wire::{ParticipantId, PriorityClass, Role, SessionId, TenantId};
use rlmesh_session::{
    HostConfig, Participant, ParticipantConfig, SessionHost, SystemMonotonicClock, Transport,
    TransportConfig,
};

const FIRST: &str = r#"{"z": {"kind": "joints", "count": 2, "fields": ["pos"], "hz": 50},
                        "a": {"kind": "joints", "count": 1, "fields": ["pos"], "hz": 50}}"#;
// `z` changes schema (its rate), so the host retires its channel and assigns
// a new one, after `a`'s.
const SECOND: &str = r#"{"z": {"kind": "joints", "count": 2, "fields": ["pos"], "hz": 100},
                         "a": {"kind": "joints", "count": 1, "fields": ["pos"], "hz": 50}}"#;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn parts_stay_in_authored_order_after_a_redeclaration() {
    let (transport, base) = open_local_listeners("127.0.0.1".parse().unwrap())
        .await
        .unwrap();
    let tenant = TenantId::new("order").unwrap();
    let session = SessionId::new();
    let mut config = HostConfig::new(tenant.clone(), session);
    config.clock = SystemMonotonicClock.shared();
    config.health_log = None;
    let host = SessionHost::start(transport, config).await.unwrap();

    let any = "127.0.0.1:0".parse().unwrap();
    let peer = Transport::open(&TransportConfig::per_class_tcp(Some(any), Some(base)).unwrap())
        .await
        .unwrap();
    let mut pc = ParticipantConfig::new(
        tenant,
        session,
        ParticipantId::new("bot").unwrap(),
        vec![Role::Embodiment],
        PriorityClass::Scripted,
    );
    pc.clock = SystemMonotonicClock.shared();
    pc.credential = Some(host.credential());
    let bot = Participant::attach(peer, pc).await.unwrap();

    let first = bot.declare_parts(FIRST).await.unwrap();
    let id = |view: &rlmesh_session::wire::AuthoritySnapshot, path: &str| {
        view.parts
            .iter()
            .find(|p| p.path.as_str() == path)
            .unwrap()
            .channel
    };
    assert!(id(&first, "z") < id(&first, "a"));

    let robot = Robot::declare(bot, SECOND, EdgeConfig::default())
        .await
        .unwrap();
    let paths: Vec<&str> = robot.parts().iter().map(|p| p.path.as_str()).collect();
    assert_eq!(paths, ["z", "a"]);
    let z = robot.part("z").unwrap().channel;
    let a = robot.part("a").unwrap().channel;
    assert!(z > a, "z was re-declared after a: {z:?} vs {a:?}");
    robot.detach().await.unwrap();
}
