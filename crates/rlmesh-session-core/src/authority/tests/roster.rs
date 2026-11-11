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
fn heartbeats_renew_and_a_lapse_ends_only_that_holders_leases() {
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
    assert_eq!((f.auth.mode(), f.auth.epoch()), (Mode::Active, epoch));
    assert!(events.iter().any(|e| matches!(
        e,
        AuthorityEvent::LeaseEnded { lease, reason, .. }
            if lease.token == a.token && reason.code == ReasonCode::HeartbeatLapse
    )));
    assert!(f.auth.lease(f.ch("cmd1")).is_none());
    assert!(f.auth.lease(f.ch("cmd2")).is_some());
    f.auth.heartbeat(&pid("policy"), epoch, 260);
    f.grant("policy", "cmd1", 260)
        .expect("no arm needed to lease again");
}

#[test]
fn a_lost_holder_stops_only_its_parts_of_one_robot() {
    let mut f = Fixture::new();
    f.run("operator", Command::Arm, 0).expect("arm");
    f.grant_with("policy", &["cmd1", "cmd3"], &[], 0)
        .expect("lease");
    f.grant("script", "cmd2", 0).expect("lease");
    let epoch = f.auth.epoch();
    assert!(f.auth.participant_lost(&pid("viewer"), 0).len() == 1);
    let events = f.auth.participant_lost(&pid("policy"), 1);
    assert_eq!((f.auth.mode(), f.auth.epoch()), (Mode::Active, epoch));
    assert!(f.auth.lease(f.ch("cmd1")).is_none());
    assert!(f.auth.lease(f.ch("cmd2")).is_some());
    assert!(f.auth.held(&pid("robot")).is_none());
    assert!(!f.auth.is_attached(&pid("policy")));
    assert!(
        f.auth.binding(&pid("policy")).is_some(),
        "kept for a reconnect"
    );
    let told = events
        .iter()
        .find(|e| matches!(e, AuthorityEvent::HolderLost { .. }))
        .expect("the loss is reported");
    assert_eq!(
        told.to_string(),
        "holder policy lost (HOLDER_LOST): released robot:cmd1, robot:cmd3"
    );
    // The last holder gone leaves an armed session with no lease.
    f.auth.participant_lost(&pid("script"), 2);
    assert_eq!((f.auth.mode(), f.auth.epoch()), (Mode::Armed, epoch));
    assert_eq!(f.auth.mode_reason().code, ReasonCode::LastRelease);
}

#[test]
fn a_lost_holder_on_one_robot_leaves_another_robot_driving() {
    let mut f = Fixture::new();
    f.run(
        "other",
        attach("other", vec![Role::Embodiment], PriorityClass::Scripted),
        0,
    )
    .expect("attach");
    let map = r#"{"wheel": {"kind": "joints", "hz": 500, "count": 6, "fields": ["pos"],
                  "command": "position", "safe": "stop"}}"#;
    f.run("other", declare(map), 0).expect("declare");
    f.bind("policy", 11, &["cmd1"]).expect("bind");
    f.bind("script", 12, &["wheel"]).expect("bind");
    f.run("operator", Command::Arm, 0).expect("arm");
    f.grant("policy", "cmd1", 0).expect("lease");
    f.grant("script", "wheel", 0).expect("lease");
    let epoch = f.auth.epoch();
    f.auth.participant_lost(&pid("policy"), 1);
    assert_eq!((f.auth.mode(), f.auth.epoch()), (Mode::Active, epoch));
    assert!(f.auth.lease(f.ch("wheel")).is_some());
    assert!(f.auth.held(&pid("robot")).is_none());
    assert!(f.auth.held(&pid("other")).is_none());
}

#[test]
fn a_replaced_holder_ends_only_its_leases() {
    let mut f = Fixture::new();
    f.run("operator", Command::Arm, 0).expect("arm");
    f.grant("policy", "cmd1", 0).expect("lease");
    f.grant("script", "cmd2", 0).expect("lease");
    let epoch = f.auth.epoch();
    let restarted = Identity {
        device: crate::auth::LOCAL_DEVICE.into(),
        generation: 1,
    };
    let applied = f
        .auth
        .apply_from(
            Request {
                from: pid("policy"),
                epoch: None,
                request_id: 0,
                command: attach("policy", vec![Role::Controller], PriorityClass::Model),
            },
            &restarted,
            1,
        )
        .expect("re-attach");
    assert_eq!((f.auth.mode(), f.auth.epoch()), (Mode::Active, epoch));
    assert!(f.auth.lease(f.ch("cmd1")).is_none());
    assert!(f.auth.lease(f.ch("cmd2")).is_some());
    assert!(applied.events.iter().any(|e| matches!(
        e,
        AuthorityEvent::HolderLost { holder, code: ReasonCode::HolderDetached, parts }
            if holder == &pid("policy") && parts == &["robot:cmd1"]
    )));
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
