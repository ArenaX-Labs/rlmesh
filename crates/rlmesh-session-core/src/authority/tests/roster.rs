//! Attaching, presence and loss.

use super::*;

#[test]
fn verbs_need_an_attached_participant() {
    let mut f = Fixture::new();
    assert_eq!(
        Fixture::status(f.run("stranger", Command::Arm, 0)),
        CtlStatus::NotAttached
    );
}

#[test]
fn a_device_attaches_only_with_the_roles_its_entry_allows() {
    let mut access = Access::default();
    access.set(
        "eye-1",
        DeviceAccess {
            roles: vec![Role::Embodiment, Role::Observer],
            ..DeviceAccess::new(Some(PriorityClass::Model))
        },
    );
    let status = |r: Result<(), CtlError>| r.err().map(|e| e.status);
    assert_eq!(
        status(access.check_attach("eye-1", &[Role::Embodiment], PriorityClass::Scripted)),
        None
    );
    assert_eq!(
        status(access.check_attach("eye-1", &[Role::Observer], PriorityClass::Scripted)),
        None
    );
    assert_eq!(
        status(access.check_attach("eye-1", &[Role::Controller], PriorityClass::Model)),
        Some(CtlStatus::NotAuthorized),
        "a stolen board never drives other robots"
    );
}

#[test]
fn an_attach_requiring_a_missing_feature_is_incompatible() {
    let mut f = Fixture::new();
    let future = |requires: Vec<String>| Command::Attach {
        name: "future".into(),
        roles: vec![Role::Controller],
        priority: PriorityClass::Model,
        protocol_minor: PROTOCOL_MINOR + 1,
        supports: vec!["merge.blend".into()],
        requires,
        workflow_edition: String::new(),
        workflow_editions: Vec::new(),
        estop: None,
    };
    assert_eq!(
        Fixture::status(f.run("future", future(vec!["merge.blend".into()]), 0)),
        CtlStatus::Incompatible
    );
    assert!(!f.auth.is_attached(&pid("future")));
    f.run("future", future(Vec::new()), 0)
        .expect("a newer minor with no requirement attaches");
    let snap = f.auth.snapshot(Vec::new(), 0);
    assert_eq!(
        snap.participant(&pid("future")).map(|p| p.supports.clone()),
        Some(vec!["merge.blend".to_string()])
    );
}

#[test]
fn heartbeats_renew_and_their_absence_lapses_to_safe() {
    let mut f = Fixture::new();
    f.run("operator", Command::Arm, 0).expect("arm");
    let a = f.grant("policy", "cmd1", 0).expect("lease");
    f.grant("script", "cmd2", 0).expect("lease");
    let epoch = f.auth.epoch();
    for now in [50, 100, 150] {
        f.auth.heartbeat(&pid("policy"), epoch, now);
        f.auth.heartbeat(&pid("script"), epoch, now);
        assert!(f.auth.expire(now).is_empty());
    }
    f.auth.heartbeat(&pid("policy"), Epoch(1), 200);
    f.auth.heartbeat(&pid("script"), epoch, 200);
    let events = f.auth.expire(251);
    assert_eq!(f.auth.mode(), Mode::Safe);
    assert!(events.iter().any(|e| matches!(
        e,
        AuthorityEvent::LeaseEnded { lease, reason }
            if lease.token == a.token && reason.code == ReasonCode::HeartbeatLapse
    )));
    let reason = f.auth.mode_reason();
    assert_eq!(
        (reason.code, reason.by.as_str()),
        (ReasonCode::HeartbeatLapse, "policy")
    );
    assert!(f.auth.lease(f.ch("cmd2")).is_none());
    assert_eq!(
        Fixture::status(f.grant("policy", "cmd1", 260)),
        CtlStatus::InvalidTransition
    );
    f.run("operator", Command::Arm, 270)
        .expect("re-arm from SAFE");
    assert_eq!(f.auth.epoch(), epoch.next());
}

#[test]
fn losing_the_holder_goes_safe_and_losing_an_observer_does_not() {
    let mut f = Fixture::new();
    f.run("operator", Command::Arm, 0).expect("arm");
    f.grant("policy", "cmd1", 0).expect("lease");
    f.auth.participant_lost(&pid("viewer"), 0);
    assert_eq!(f.auth.mode(), Mode::Active);
    f.auth.participant_lost(&pid("policy"), 0);
    assert_eq!(f.auth.mode(), Mode::Safe);
    assert_eq!(f.auth.mode_reason().code, ReasonCode::HolderLost);
    assert!(!f.auth.is_attached(&pid("policy")));
    assert!(
        f.auth.binding(&pid("policy")).is_some(),
        "kept for a reconnect"
    );
}

#[test]
fn presence_is_heartbeats_and_a_lost_robot_stays_in_the_roster() {
    let mut f = Fixture::new();
    f.run("operator", Command::Arm, 0).expect("arm");
    let lease = f.grant("policy", "cmd1", 0).expect("lease");
    let epoch = f.auth.epoch();
    let beat = |f: &mut Fixture, ids: &[&str], now| {
        for id in ids {
            f.auth.heartbeat(&pid(id), epoch, now);
        }
    };
    let everyone = ["robot", "policy", "operator", "script", "viewer"];
    // A liveliness delete is only a hint: a heartbeat after it keeps the
    // participant present.
    f.auth.presence_hint(&pid("robot"), 10);
    beat(&mut f, &everyone, 50);
    assert!(f.auth.expire(140).is_empty());
    beat(&mut f, &everyone, 140);
    // Without one, a robot is lost a lease grace after the hint.
    f.auth.presence_hint(&pid("robot"), 150);
    beat(&mut f, &everyone[1..], 200);
    f.auth.expire(251);
    // Only the robot stops: its lease ends and it is held; the session
    // has no lease left, so it is ARMED.
    assert_eq!(f.auth.mode(), Mode::Armed);
    assert_eq!(
        f.auth.held(&pid("robot")).map(|r| (r.code, r.by.as_str())),
        Some((ReasonCode::RobotLost, "robot"))
    );
    let view = f.auth.snapshot(Vec::new(), 251);
    let robot = view.participant(&pid("robot")).expect("kept");
    assert!(!robot.present);
    assert_eq!(
        robot.left.as_ref().map(|r| r.code),
        Some(ReasonCode::RobotLost)
    );
    assert!(view.part(f.ch("cmd1")).is_some(), "its parts remain");
    assert!(
        view.ended
            .iter()
            .any(|e| { e.lease.token == lease.token && e.reason.code == ReasonCode::RobotLost })
    );
    assert_eq!(
        Fixture::status(f.run("robot", Command::Clear, 252)),
        CtlStatus::NotAttached
    );
    // Re-attaching makes it present again.
    let back = attach("robot", vec![Role::Embodiment], PriorityClass::Scripted);
    f.run("robot", back, 260).expect("re-attach");
    assert!(f.auth.is_attached(&pid("robot")));
    // Anyone silent past the presence grace is lost; a non-robot leaves.
    f.auth.expire(260 + 1_000_000_001);
    let view = f.auth.snapshot(Vec::new(), 0);
    assert!(view.participant(&pid("viewer")).is_none());
    assert!(view.participant(&pid("robot")).is_some_and(|r| !r.present));
}
