//! The robot's retries with no host at all: what it asks of the transport
//! when nothing answers, or something other than a host does.

use rlmesh_device::robot::{
    Action, Config, Fault, LINK_LOST_MS, Phase, QUERY_MARGIN_MS, QUERY_MS, Robot,
};
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

fn robot(homes: &[(&str, u16)]) -> Box<Robot> {
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
    assert_eq!(declares(&first), (6, 1));
    robot.query_done(query(&first).expect("a query"), 0);
    assert_eq!(robot.phase(), Phase::Retry);
    assert_eq!(robot.fault(), Some(&Fault::NoReply));

    // The transport may not have reached the host to declare anything: it
    // drops what it holds and declares it all again, then waits its pause.
    let retry = robot.poll(0);
    assert!(matches!(retry.first(), Some(Action::Reset)), "{retry:?}");
    assert_eq!(declares(&retry), (6, 1));
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

/// When a query the robot asked at `now` is over without a word.
fn deadline(now: u64) -> u64 {
    now + u64::from(QUERY_MS + QUERY_MARGIN_MS) * 1000
}

#[test]
fn a_query_nobody_reports_on_is_over_at_its_deadline() {
    let mut robot = robot(&[("10.0.0.1", 7447)]);
    let id = query(&robot.poll(0)).expect("a query");

    // No reply and no end from the transport: the robot still waits, up to
    // its own deadline.
    robot.poll(deadline(0) - 1);
    assert_eq!(robot.phase(), Phase::Host);
    assert_eq!(robot.fault(), None);
    robot.poll(deadline(0));
    assert_eq!(robot.phase(), Phase::Retry);
    assert_eq!(robot.fault(), Some(&Fault::NoReply));

    // Whatever comes for it later changes nothing, and the normal retry
    // follows.
    let at = deadline(0);
    robot.reply(id, b"late", at, &mut Zeros);
    robot.query_done(id, at);
    assert_eq!(robot.phase(), Phase::Retry);
    assert_eq!(query(&robot.poll(at)), None, "it waits its pause");
    let retried = query(&robot.poll(at + S)).expect("it asks again");
    assert_ne!(retried, id);
}

/// `Action::Reconnect`s over ten minutes of silence.
fn reconnects_in_silence(robot: &mut Robot) -> Vec<u64> {
    let mut at = Vec::new();
    let mut now = 0;
    while now < 600 * S {
        if robot
            .poll(now)
            .iter()
            .any(|a| matches!(a, Action::Reconnect))
        {
            at.push(now);
        }
        now += S / 10;
    }
    at
}

#[test]
fn silence_reopens_the_links_with_backoff_for_ever() {
    let mut robot = robot(&[("10.0.0.1", 7447)]);
    let at = reconnects_in_silence(&mut robot);
    let first = *at.first().expect("it reopens its links");
    assert!(
        first >= u64::from(LINK_LOST_MS) * 1000,
        "not before the link counts as lost: {at:?}"
    );
    assert!(first <= 15 * S, "soon after: {at:?}");
    // Each pause is at least the last, up to the backoff's cap plus a query.
    let gaps: Vec<u64> = at.windows(2).map(|w| w[1] - w[0]).collect();
    assert!(gaps.windows(2).all(|g| g[1] + S / 10 >= g[0]), "{gaps:?}");
    assert!(gaps.iter().all(|g| *g <= 33 * S), "{gaps:?}");
    assert!(
        *at.last().expect("some") >= 560 * S,
        "and never stops: {at:?}"
    );
}
