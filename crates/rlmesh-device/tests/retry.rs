//! The robot's retries with no host at all: what it asks of the transport
//! when nothing answers, or something other than a host does.

use rlmesh_device::robot::{Action, Config, Fault, Phase, Robot};
use rlmesh_session::auth::Credential;
use rlmesh_session::crypto::{Rng, RngError, StaticKeypair};
use rlmesh_session::wire::{ParticipantId, SessionId, TenantId};

const MAP: &str = r#"{"eyes": {"kind": "joints", "names": ["pan", "tilt"], "fields": ["pos"],
  "hz": 50, "command": "position", "safe": "hold"}}"#;
const S: u64 = 1_000_000;

struct Zeros;

impl Rng for Zeros {
    fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), RngError> {
        dest.fill(0);
        Ok(())
    }
}

fn robot(homes: &[(&str, u16)]) -> Robot {
    let mut key = [0x42_u8; 32];
    Robot::new(Config {
        tenant: TenantId::new("local").expect("tenant"),
        session: SessionId::new(),
        name: ParticipantId::new("eyes").expect("id"),
        credential: Credential::new(StaticKeypair::from_private(&mut key), [7; 32]),
        robot_json: MAP.to_owned(),
        heartbeat_ms: 50,
        lobby: false,
        homes: homes.iter().map(|(h, p)| ((*h).to_owned(), *p)).collect(),
        home_id: "rlm-test".to_owned(),
    })
    .expect("a valid map")
}

/// The query among `actions`.
fn query(actions: &[Action]) -> Option<u32> {
    actions.iter().find_map(|a| match a {
        Action::Query { id, .. } => Some(*id),
        _ => None,
    })
}

fn declares(actions: &[Action]) -> (usize, usize) {
    let subs = actions
        .iter()
        .filter(|a| matches!(a, Action::Subscribe { .. }))
        .count();
    let tokens = actions
        .iter()
        .filter(|a| matches!(a, Action::Token { .. }))
        .count();
    (subs, tokens)
}

/// Asks the host, and lets the query end unanswered.
fn unanswered(robot: &mut Robot, now: u64) {
    let id = query(&robot.poll(now)).expect("it asks for the host");
    robot.query_done(id, now);
}

#[test]
fn a_retry_declares_its_subscriptions_again() {
    let mut robot = robot(&[("10.0.0.1", 7447)]);
    let first = robot.poll(0);
    assert_eq!(declares(&first), (5, 1));
    robot.query_done(query(&first).expect("a query"), 0);
    assert_eq!(robot.phase(), Phase::Retry);
    assert_eq!(robot.fault(), Some(&Fault::NoReply));

    // The transport may not have reached the host to declare anything: it
    // drops what it holds and declares it all again, then waits its pause.
    let retry = robot.poll(0);
    assert!(matches!(retry.first(), Some(Action::Reset)), "{retry:?}");
    assert_eq!(declares(&retry), (5, 1));
    assert_eq!(query(&retry), None, "it waits before asking again");
    assert!(query(&robot.poll(S)).is_some(), "it asks after a second");
}

#[test]
fn a_reply_that_isnt_the_hosts_state_does_not_end_the_attempt() {
    let mut robot = robot(&[("10.0.0.1", 7447)]);
    let id = query(&robot.poll(0)).expect("a query");
    robot.reply(id, b"not a signed state", 0, &mut Zeros);
    assert_eq!(
        robot.phase(),
        Phase::Host,
        "the host's own reply may follow"
    );
    robot.query_done(id, 0);
    assert_eq!(robot.fault(), Some(&Fault::NoReply));
}

#[test]
fn the_home_keeps_one_found_address_besides_its_stored_ones() {
    let mut robot = robot(&[("10.0.0.1", 7447)]);
    unanswered(&mut robot, 0);
    assert!(robot.home_unreachable());

    robot.add_home("10.0.0.2", 7447, 0);
    assert_eq!(robot.host(), ("10.0.0.2", 7447));
    unanswered(&mut robot, 0);
    // A whole round failed: back off before the next.
    assert_eq!(robot.host(), ("10.0.0.1", 7447));
    unanswered(&mut robot, 2 * S);
    assert_eq!(robot.host(), ("10.0.0.2", 7447));

    // Another answer replaces the one found before.
    robot.add_home("10.0.0.3", 7447, 2 * S);
    assert_eq!(robot.host(), ("10.0.0.3", 7447));
    unanswered(&mut robot, 2 * S);
    assert_eq!(robot.host(), ("10.0.0.1", 7447));
    unanswered(&mut robot, 10 * S);
    assert_eq!(robot.host(), ("10.0.0.3", 7447), "10.0.0.2 is gone");
}
