//! The mode: arming, fallbacks and the e-stop latch.

use super::*;

#[test]
fn arm_bumps_the_epoch_only_from_disarmed_or_safe() {
    let mut f = Fixture::new();
    assert_eq!(f.auth.epoch(), Epoch(1));
    f.run("operator", Command::Arm, 1).expect("arm");
    assert_eq!((f.auth.mode(), f.auth.epoch()), (Mode::Armed, Epoch(2)));
    assert_eq!(
        Fixture::status(f.run("operator", Command::Arm, 2)),
        CtlStatus::InvalidTransition
    );
    f.run("operator", Command::Disarm, 3).expect("disarm");
    f.run("operator", Command::Arm, 4).expect("re-arm");
    assert_eq!(f.auth.epoch(), Epoch(3));
}

#[test]
fn an_embodiments_fallback_stops_its_robot_alone_until_an_arm_and_estop_records_it() {
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
    // and its next drive needs an arm.
    f.run("robot", safe(ReasonCode::DeviceWatchdog), 1)
        .expect("from DISARMED");
    assert_eq!(f.auth.mode(), Mode::Disarmed);
    let held = f.auth.held(&pid("robot")).expect("held");
    assert_eq!(
        (held.code, held.by.as_str()),
        (ReasonCode::DeviceWatchdog, "robot")
    );
    f.run("operator", Command::Arm, 2).expect("arm");
    assert!(f.auth.held(&pid("robot")).is_none(), "an arm re-arms it");
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
        "held until an arm"
    );
    // An arm in an armed session re-arms the robot alone, in its epoch.
    f.run("operator", Command::Arm, 6)
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
