//! The mode: arming, fallbacks and the e-stop latch.

use super::*;

#[test]
fn arm_bumps_the_epoch_only_from_disarmed_or_safe() {
    let mut f = Fixture::new();
    assert_eq!(f.auth.epoch(), Epoch(1));
    f.run("operator", Command::Arm, 1).expect("arm");
    assert_eq!((f.auth.mode(), f.auth.epoch()), (Mode::Armed, Epoch(2)));
    let revision = f.auth.revision();
    let again = f.run("operator", Command::Arm, 2).expect("a no-op");
    assert!(again.events.is_empty());
    assert_eq!((f.auth.epoch(), f.auth.revision()), (Epoch(2), revision));
    f.grant("policy", "cmd1", 2).expect("lease");
    f.run("operator", Command::Arm, 2).expect("ACTIVE too");
    assert_eq!(f.auth.mode(), Mode::Active);
    f.run("operator", Command::Disarm, 3).expect("disarm");
    f.run("operator", Command::Arm, 4).expect("re-arm");
    assert_eq!(f.auth.epoch(), Epoch(3));
}

#[test]
fn an_embodiments_fallback_stops_its_robot_alone_until_rearmed_and_estop_records_it() {
    let mut f = Fixture::new();
    let safe = |code| Command::EnterSafe {
        reason: "board link".into(),
        code,
    };
    let stopped_now = |f: &Fixture| {
        assert_eq!(
            f.auth.last_stop(),
            LastStop {
                epoch: f.auth.epoch(),
                revision: f.auth.revision()
            }
        );
    };
    // Between episodes: the robot is held, the session's mode stays,
    // and its next drive needs a re-arm by name.
    f.run("robot", safe(ReasonCode::DeviceWatchdog), 1)
        .expect("from DISARMED");
    assert_eq!(f.auth.mode(), Mode::Disarmed);
    let held = f.auth.held(&pid("robot")).expect("held");
    assert_eq!(
        (held.code, held.by.as_str()),
        (ReasonCode::DeviceWatchdog, "robot")
    );
    f.run("operator", Command::Arm, 2).expect("arm");
    assert!(
        f.auth.held(&pid("robot")).is_some(),
        "an arm clears no hold"
    );
    f.run("operator", rearm("robot"), 2).expect("re-arm");
    assert!(f.auth.held(&pid("robot")).is_none());
    f.grant("policy", "cmd1", 2).expect("lease");
    f.grant("script", "cmd2", 2).expect("lease");
    assert_eq!(
        Fixture::status(f.run("policy", safe(ReasonCode::QueueDry), 3)),
        CtlStatus::Forbidden
    );
    // From ACTIVE: only this robot's leases end; the session goes on.
    let epoch = f.auth.epoch();
    f.run("robot", safe(ReasonCode::QueueDry), 4)
        .expect("from ACTIVE");
    assert_eq!((f.auth.mode(), f.auth.epoch()), (Mode::Armed, epoch));
    assert!(f.auth.lease(f.ch("cmd1")).is_none(), "its leases end");
    let ended = f.auth.snapshot(Vec::new(), 4).ended;
    assert_eq!(
        ended.last().map(|e| (e.reason.code, e.reason.by.as_str())),
        Some((ReasonCode::VoidedBySafe, "robot"))
    );
    assert_eq!(
        Fixture::status(f.grant("policy", "cmd1", 5)),
        CtlStatus::NotReady,
        "held until re-armed"
    );
    f.run("operator", Command::Arm, 6).expect("a no-op");
    assert_eq!(
        Fixture::status(f.grant("policy", "cmd1", 6)),
        CtlStatus::NotReady,
        "an arm is not a re-arm"
    );
    // A re-arm frees the robot alone, in its epoch.
    f.run("operator", rearm("robot"), 6)
        .expect("re-arm the robot");
    assert_eq!(f.auth.epoch(), epoch);
    f.grant("policy", "cmd1", 6).expect("leased again");
    // ESTOP stays ESTOP and records the cause; recording is no new stop.
    let estop = Command::Estop {
        reason: "button".into(),
        code: ReasonCode::Unspecified,
        origin: None,
        at_ns: 0,
    };
    f.run("viewer", estop, 7).expect("estop");
    assert_eq!(f.auth.mode_reason().code, ReasonCode::EstopRemote);
    stopped_now(&f);
    let stop = f.auth.last_stop();
    f.run("robot", safe(ReasonCode::DeviceWatchdog), 8)
        .expect("recorded");
    assert_eq!(
        (f.auth.mode(), f.auth.mode_reason().code, f.auth.last_stop()),
        (Mode::Estop, ReasonCode::DeviceWatchdog, stop)
    );
    assert!(f.auth.estop_latched());
}

fn rearm(robot: &str) -> Command {
    Command::Rearm { robot: pid(robot) }
}

/// A fixture with a second robot, `other`, whose `wheel` every controller
/// may lease; `robot` and `other` are both held by their own faults.
fn two_held_robots() -> Fixture {
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
    f.bind("policy", 11, &["cmd1", "wheel"]).expect("bind");
    for robot in ["robot", "other"] {
        let fault = Command::EnterSafe {
            reason: "board link".into(),
            code: ReasonCode::DeviceWatchdog,
        };
        f.run(robot, fault, 0).expect("fault");
    }
    f
}

#[test]
fn a_rearm_clears_only_the_named_robot() {
    let mut f = two_held_robots();
    f.run("operator", Command::Arm, 1).expect("arm");
    f.run("operator", Command::Arm, 2).expect("a routine arm");
    assert!(f.auth.held(&pid("robot")).is_some());
    assert!(f.auth.held(&pid("other")).is_some());
    let applied = f.run("operator", rearm("other"), 3).expect("re-arm");
    assert!(f.auth.held(&pid("robot")).is_some());
    assert!(f.auth.held(&pid("other")).is_none());
    assert!(applied.events.iter().any(|e| matches!(
        e,
        AuthorityEvent::RobotRearmed { robot, by }
            if robot == &pid("other") && by == &pid("operator")
    )));
    f.grant("policy", "wheel", 3).expect("leased");
    // Not held any more: a no-op that changes nothing.
    let revision = f.auth.revision();
    let again = f.run("operator", rearm("other"), 4).expect("a no-op");
    assert!(again.events.is_empty());
    assert_eq!(f.auth.revision(), revision);
    assert!(f.auth.lease(f.ch("wheel")).is_some());
}

#[test]
fn a_refused_lease_names_the_exact_rearm() {
    let mut f = two_held_robots();
    f.run("operator", Command::Arm, 1).expect("arm");
    let refused = f.grant("policy", "cmd1", 1).expect_err("held");
    assert_eq!(refused.status, CtlStatus::NotReady);
    assert_eq!(
        refused.message,
        "robot stopped on its own (DEVICE_WATCHDOG): call robot.rearm(\"robot\") to drive it again"
    );
}

#[test]
fn a_rearm_needs_an_armed_session_a_known_robot_and_the_verb() {
    let mut f = two_held_robots();
    for mode in ["DISARMED", "SAFE"] {
        if mode == "SAFE" {
            f.run("operator", Command::Arm, 1).expect("arm");
            f.run("operator", rearm("robot"), 1).expect("re-arm");
            f.grant("policy", "cmd1", 1).expect("lease");
            let why = f.auth.reason(ReasonCode::Unspecified, "test", "");
            f.auth.enter_safe(why, &mut Vec::new());
        }
        let refused = f.run("operator", rearm("other"), 2).expect_err("not armed");
        assert_eq!(refused.status, CtlStatus::InvalidTransition);
        assert_eq!(
            refused.message,
            format!("session is {mode}; arm it first, then re-arm other")
        );
    }
    f.run("operator", Command::Arm, 3).expect("arm");
    for stranger in ["ghost", "policy"] {
        let refused = f.run("operator", rearm(stranger), 3).expect_err("unknown");
        assert_eq!(refused.status, CtlStatus::InvalidRequest);
        assert!(refused.message.contains(stranger), "{}", refused.message);
    }
    assert_eq!(
        Fixture::status(f.run("stranger", rearm("other"), 3)),
        CtlStatus::NotAttached
    );
    let mut access = Access::default();
    access.set(
        "guest",
        DeviceAccess {
            verbs: Vec::new(),
            ..DeviceAccess::new(Some(PriorityClass::Model))
        },
    );
    f.auth.config.access = access;
    let guest = Identity {
        device: "guest".into(),
        generation: 1,
    };
    let request = |command| Request {
        from: pid("guest"),
        epoch: None,
        request_id: 0,
        command,
    };
    f.auth
        .apply_from(
            request(attach(
                "guest",
                vec![Role::Observer],
                PriorityClass::Scripted,
            )),
            &guest,
            3,
        )
        .expect("attach");
    let refused = f
        .auth
        .apply_from(request(rearm("other")), &guest, 3)
        .expect_err("no verb");
    assert_eq!(refused.status, CtlStatus::NotAuthorized);
    assert!(f.auth.held(&pid("other")).is_some());
}

#[test]
fn a_hold_outlasts_a_session_wide_safe_and_a_disarm() {
    let mut f = two_held_robots();
    f.run("operator", Command::Arm, 1).expect("arm");
    f.run("operator", rearm("robot"), 1).expect("re-arm");
    f.grant("policy", "cmd1", 1).expect("lease");
    let why = f.auth.reason(ReasonCode::Unspecified, "test", "");
    f.auth.enter_safe(why, &mut Vec::new());
    assert_eq!(f.auth.mode(), Mode::Safe);
    f.run("operator", Command::Arm, 2).expect("arm from SAFE");
    assert!(f.auth.held(&pid("other")).is_some());
    f.run("operator", Command::Disarm, 3).expect("disarm");
    f.run("operator", Command::Arm, 4).expect("arm");
    assert!(f.auth.held(&pid("other")).is_some());
    assert_eq!(
        Fixture::status(f.grant("policy", "wheel", 4)),
        CtlStatus::NotReady
    );
    f.run("operator", rearm("other"), 5).expect("re-arm");
    f.grant("policy", "wheel", 5).expect("leased");
}

fn held_in(f: &Fixture, now: i64) -> Vec<(String, ReasonCode, String, i64)> {
    f.auth
        .snapshot(Vec::new(), now)
        .held
        .iter()
        .map(|h| {
            (
                h.robot.to_string(),
                h.reason.code,
                h.reason.detail.clone(),
                h.reason.at_ns,
            )
        })
        .collect()
}

#[test]
fn the_snapshot_lists_a_held_robot_from_its_fault_until_its_rearm() {
    let mut f = Fixture::new();
    f.run("operator", Command::Arm, 1).expect("arm");
    assert!(held_in(&f, 1).is_empty());
    let revision = f.auth.revision();
    let fault = Command::EnterSafe {
        reason: "board link".into(),
        code: ReasonCode::DeviceWatchdog,
    };
    let applied = f.run("robot", fault, 2).expect("fault");
    assert!(
        f.auth.revision() > revision,
        "a hold is published even with no lease to end"
    );
    assert!(matches!(
        applied.events.as_slice(),
        [AuthorityEvent::RobotHeld { robot, .. }] if *robot == pid("robot")
    ));
    assert_eq!(
        held_in(&f, 3),
        [(
            "robot".to_string(),
            ReasonCode::DeviceWatchdog,
            "board link".to_string(),
            2
        )]
    );
    let refused = f.grant("policy", "cmd1", 3).expect_err("held");
    assert_eq!(
        f.auth.snapshot(Vec::new(), 3).held_refusal(&pid("robot")),
        Some(refused.message),
        "the snapshot says what the host refuses with"
    );
    let revision = f.auth.revision();
    f.run("operator", rearm("robot"), 4).expect("re-arm");
    assert!(f.auth.revision() > revision);
    assert!(held_in(&f, 4).is_empty());
    assert_eq!(
        f.auth.snapshot(Vec::new(), 4).held_refusal(&pid("robot")),
        None
    );
}

#[test]
fn the_snapshot_keeps_a_hold_through_a_session_wide_safe_and_a_new_arm() {
    let mut f = two_held_robots();
    f.run("operator", Command::Arm, 1).expect("arm");
    f.run("operator", rearm("robot"), 1).expect("re-arm");
    f.grant("policy", "cmd1", 1).expect("lease");
    let why = f.auth.reason(ReasonCode::Unspecified, "test", "");
    f.auth.enter_safe(why, &mut Vec::new());
    assert_eq!(f.auth.mode(), Mode::Safe);
    f.run("operator", Command::Arm, 2).expect("arm from SAFE");
    assert_eq!(
        held_in(&f, 2),
        [(
            "other".to_string(),
            ReasonCode::DeviceWatchdog,
            "board link".to_string(),
            0
        )]
    );
}

#[test]
fn a_robot_lost_or_restarted_while_armed_is_held_and_one_while_disarmed_is_not() {
    let restart = |f: &mut Fixture, generation: u32, now: i64| {
        let sender = Identity {
            device: crate::auth::LOCAL_DEVICE.into(),
            generation,
        };
        let request = Request {
            from: pid("robot"),
            epoch: None,
            request_id: 0,
            command: attach("robot", vec![Role::Embodiment], PriorityClass::Scripted),
        };
        f.auth.apply_from(request, &sender, now).expect("restart");
    };
    let mut f = Fixture::new();
    restart(&mut f, 1, 1);
    f.auth.participant_lost(&pid("robot"), 2);
    assert!(f.auth.held(&pid("robot")).is_none(), "nothing was driving");
    restart(&mut f, 2, 3);
    f.run("operator", Command::Arm, 4).expect("arm");
    f.grant("policy", "cmd1", 4).expect("driven with no re-arm");
    restart(&mut f, 3, 5);
    let held = f.auth.held(&pid("robot")).map(|r| r.code);
    assert_eq!(held, Some(ReasonCode::VoidedByEpoch));
    f.run("operator", rearm("robot"), 6).expect("re-arm");
    f.auth.participant_lost(&pid("robot"), 7);
    let held = f.auth.held(&pid("robot")).map(|r| r.code);
    assert_eq!(held, Some(ReasonCode::RobotLost));
    // Gone, it can't be acknowledged yet.
    assert_eq!(
        Fixture::status(f.run("operator", rearm("robot"), 8)),
        CtlStatus::NotReady
    );
    assert!(f.auth.held(&pid("robot")).is_some());
}

#[test]
fn only_the_robot_that_latched_an_estop_or_local_clears_it() {
    let mut f = Fixture::new();
    let mut access = Access::default();
    for board in ["eye-1", "eye-2"] {
        access.set(board, DeviceAccess::new(Some(PriorityClass::Model)));
    }
    f.auth.config.access = access;
    let as_board = |f: &mut Fixture, board: &str, command: Command, now: i64| {
        let sender = Identity {
            device: board.into(),
            generation: 1,
        };
        f.auth.apply_from(
            Request {
                from: pid(board),
                epoch: None,
                request_id: 0,
                command,
            },
            &sender,
            now,
        )
    };
    for (board, now) in [("eye-1", 1), ("eye-2", 2)] {
        as_board(
            &mut f,
            board,
            attach(board, vec![Role::Embodiment], PriorityClass::Scripted),
            now,
        )
        .expect("attach");
    }
    let estop = Command::Estop {
        reason: "button".into(),
        code: ReasonCode::EstopLocal,
        origin: None,
        at_ns: 0,
    };
    as_board(&mut f, "eye-1", estop, 3).expect("eye 1 latches");
    assert_eq!(
        Fixture::status(as_board(&mut f, "eye-2", Command::Clear, 4)),
        CtlStatus::Forbidden,
        "another robot never clears it"
    );
    assert!(f.auth.estop_latched());
    as_board(&mut f, "eye-1", Command::Clear, 5).expect("the robot that latched it");
    assert!(!f.auth.estop_latched());

    // A local robot process clears any latch.
    let estop = Command::Estop {
        reason: "button".into(),
        code: ReasonCode::EstopLocal,
        origin: None,
        at_ns: 0,
    };
    as_board(&mut f, "eye-2", estop, 6).expect("eye 2 latches");
    f.run("robot", Command::Clear, 7).expect("local clears");
}

#[test]
fn estop_latches_and_needs_clear_disarm_arm() {
    let mut f = Fixture::new();
    f.run("operator", Command::Arm, 0).expect("arm");
    let lease = f.grant("policy", "cmd1", 0).expect("lease");
    f.run(
        "stranger",
        Command::Estop {
            reason: "red button".into(),
            code: ReasonCode::EstopRemote,
            origin: None,
            at_ns: 0,
        },
        1,
    )
    .expect("estop");
    assert_eq!(f.auth.mode(), Mode::Estop);
    assert!(f.auth.estop_latched());
    let epoch = f.auth.epoch();
    assert!(
        f.auth
            .check_fence(f.ch("cmd1"), lease.token, epoch, 2)
            .is_err()
    );
    let revision = f.auth.revision();
    f.run(
        "operator",
        Command::Estop {
            reason: "again".into(),
            code: ReasonCode::EstopLocal,
            origin: None,
            at_ns: 0,
        },
        2,
    )
    .expect("estop");
    assert_eq!(f.auth.revision(), revision, "a second estop is a no-op");
    assert_eq!(
        Fixture::status(f.run("operator", Command::Arm, 3)),
        CtlStatus::EstopLatched
    );
    assert_eq!(
        Fixture::status(f.grant("operator", "cmd1", 3)),
        CtlStatus::EstopLatched
    );
    assert_eq!(
        Fixture::status(f.run("operator", Command::Clear, 3)),
        CtlStatus::Forbidden
    );
    f.run("robot", Command::Clear, 4).expect("local clear");
    assert_eq!(f.auth.mode_reason().code, ReasonCode::EstopCleared);
    // A local latch reported while cleared latches again, as the edge's.
    let local = Command::Estop {
        reason: "button".into(),
        code: ReasonCode::EstopLocal,
        origin: None,
        at_ns: 0,
    };
    f.run("robot", local, 4).expect("relatch");
    assert!(f.auth.estop_latched());
    assert_eq!(f.auth.mode_reason().code, ReasonCode::EstopLocal);
    f.run("robot", Command::Clear, 4).expect("local clear");
    assert_eq!(
        Fixture::status(f.run("operator", Command::Arm, 5)),
        CtlStatus::InvalidTransition
    );
    f.run("operator", Command::Disarm, 6).expect("disarm");
    f.run("operator", Command::Arm, 7).expect("arm");
    assert_eq!(f.auth.epoch(), epoch.next());
}
