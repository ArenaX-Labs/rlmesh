//! Hand-backs: a release that gives the parts to the policy its lease
//! preempted, and every refusal of one.

use super::*;
use crate::wire::version::LEASE_HANDBACK;

/// The fixture, armed, with `model` (a model-class controller that resumes
/// a part handed back to it) bound to every commandable part.
fn armed() -> Fixture {
    let mut f = Fixture::new();
    let mut command = attach("model", vec![Role::Controller], PriorityClass::Model);
    if let Command::Attach { supports, .. } = &mut command {
        supports.push(LEASE_HANDBACK.into());
    }
    f.run("model", command, 0).expect("attach");
    f.bind(
        "model",
        9,
        &["cmd1", "cmd2", "cmd3", "cmd4", "arm", "arm/tcp"],
    )
    .expect("bind");
    f.run("operator", Command::Arm, 0).expect("arm");
    f
}

fn hand_back(f: &mut Fixture, lease: &Lease, to: &str) -> Result<Applied, CtlError> {
    f.run(
        "operator",
        Command::ReleaseLease {
            channels: vec![lease.channel],
            token: lease.token,
            to: Some(pid(to)),
        },
        20,
    )
}

fn refusal(result: Result<Applied, CtlError>) -> (CtlStatus, String) {
    let err = result.expect_err("refused");
    (err.status, err.message)
}

#[test]
fn a_release_hands_a_part_back_only_to_the_policy_its_lease_took_it_from() {
    let mut f = armed();
    // The operator takes a member of the model's command group.
    f.grant("model", "arm", 1).expect("model");
    let tcp = f
        .grant("operator", "arm/tcp", 2)
        .expect("operator preempts");
    assert_eq!(
        refusal(hand_back(&mut f, &tcp, "script")),
        (
            CtlStatus::InvalidRequest,
            "cannot hand robot:arm/tcp back to script: this lease took it from model, and only \
             that policy is parked on it"
                .into()
        )
    );
    assert_eq!(
        f.auth.lease(f.ch("arm/tcp")),
        Some(&tcp),
        "a refusal changes nothing"
    );
    hand_back(&mut f, &tcp, "model").expect("handed back");
    assert!(f.auth.lease(f.ch("arm/tcp")).is_none(), "released");
    let end = f.auth.snapshot(Vec::new(), 20).ended.pop().expect("end");
    assert_eq!(
        (end.lease.token, end.reason.code, end.to),
        (tcp.token, ReasonCode::Released, Some(pid("model")))
    );
    assert_eq!(end.reason.detail, "handed back to model");

    // A plain release names nobody.
    f.grant("model", "cmd1", 3).expect("model");
    let cmd1 = f.grant("operator", "cmd1", 4).expect("operator preempts");
    f.release("operator", &cmd1, 5).expect("release");
    let end = f.auth.snapshot(Vec::new(), 5).ended.pop().expect("end");
    assert_eq!((end.lease.token, end.to), (cmd1.token, None));
    // ...and its lease is gone: the caller holds nothing to hand back.
    assert_eq!(
        refusal(hand_back(&mut f, &cmd1, "model")).0,
        CtlStatus::StaleToken
    );
}

#[test]
fn a_hand_back_is_refused_with_nothing_parked_an_old_peer_or_the_robot_held() {
    let mut f = armed();
    // Nobody held cmd2 before the operator: nothing is parked on it.
    let free = f.grant("operator", "cmd2", 1).expect("free part");
    assert_eq!(
        refusal(hand_back(&mut f, &free, "model")),
        (
            CtlStatus::InvalidRequest,
            "cannot hand robot:cmd2 back to model: this lease took it from nobody, so no policy \
             is parked on it to hand back to"
                .into()
        )
    );
    // A policy that does not resume a part handed back would stay parked.
    f.grant("policy", "cmd3", 2).expect("old peer");
    let cmd3 = f.grant("operator", "cmd3", 3).expect("operator preempts");
    assert_eq!(
        refusal(hand_back(&mut f, &cmd3, "policy")),
        (
            CtlStatus::Incompatible,
            "cannot hand robot:cmd3 back to policy: it does not support lease.handback and \
             would stay parked; release plainly and call resume() in its process"
                .into()
        )
    );
    // A robot held (new commands while armed) keeps its hold.
    f.grant("model", "cmd1", 4).expect("model");
    let cmd1 = f.grant("operator", "cmd1", 5).expect("operator preempts");
    f.run(
        "robot",
        declare(&robot_map(
            r#", "cmd5": {"kind": "joints", "hz": 500, "count": 6, "fields": ["pos"],
                 "command": "position", "safe": "stop"}"#,
        )),
        6,
    )
    .expect("declare more");
    assert_eq!(
        f.auth.lease(f.ch("cmd1")),
        Some(&cmd1),
        "still held by the operator"
    );
    let (status, message) = refusal(hand_back(&mut f, &cmd1, "model"));
    assert_eq!(status, CtlStatus::NotReady);
    assert!(
        message.starts_with("cannot hand robot:cmd1 back to model: robot stopped on its own")
            && message.ends_with("; a hand-back never clears that"),
        "{message}"
    );
    f.release("operator", &cmd1, 7)
        .expect("a plain release still goes");
    assert!(
        f.auth
            .snapshot(Vec::new(), 7)
            .held
            .iter()
            .any(|h| h.robot == pid("robot")),
        "the hold stays"
    );
}
