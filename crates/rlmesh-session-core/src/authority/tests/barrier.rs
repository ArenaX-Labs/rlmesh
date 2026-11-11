//! The virtual clock's tick barrier.

use super::*;

#[test]
fn a_virtual_session_ticks_through_the_barrier() {
    let mut auth = Authority::new(
        SessionId::new(),
        AuthorityConfig {
            virtual_clock: Some(VirtualConfig::default()),
            ..AuthorityConfig::default()
        },
    );
    let mut f = {
        for (id, roles, priority) in [
            ("robot", vec![Role::Embodiment], PriorityClass::Scripted),
            ("policy", vec![Role::Controller], PriorityClass::Model),
        ] {
            auth.apply(
                Request {
                    from: pid(id),
                    epoch: None,
                    request_id: 0,
                    command: attach(id, roles, priority),
                },
                0,
            )
            .expect("attach");
        }
        Fixture { auth }
    };
    f.run("robot", declare(&robot_map("")), 0).expect("declare");
    let channels = vec![f.ch("cmd1")];
    f.run(
        "policy",
        Command::DeclareBinding {
            fingerprint: Fingerprint::from(5),
            revision: 1,
            table_json: "{}".into(),
            channels,
            replan_hz: 500.0,
            contracts: Vec::new(),
            grids: Vec::new(),
        },
        0,
    )
    .expect("bind at 500 Hz: every second 1 ms tick");
    f.run("policy", Command::Arm, 0).expect("arm");
    assert_eq!(
        f.auth.snapshot(Vec::new(), 0).clock_mode,
        ClockMode::Virtual
    );
    let ready = |tick| Command::Ready {
        tick,
        watermarks: Vec::new(),
    };
    f.run("robot", ready(0), 1).expect("ready");
    let ticks = f.auth.take_ticks();
    assert!(matches!(ticks.as_slice(), [VirtualTick::Observe(s)] if s.due == [pid("policy")]));
    let revision = f.auth.revision();
    f.run(
        "policy",
        Command::Decided {
            tick: 0,
            chunks: Vec::new(),
        },
        2,
    )
    .expect("decided");
    assert!(matches!(f.auth.take_ticks().as_slice(), [VirtualTick::Step(s)] if s.tick == 0));
    assert_eq!(f.auth.revision(), revision, "ticks are not structural");
    f.run("robot", ready(1), 3).expect("ready");
    assert_eq!(f.auth.take_ticks().len(), 2, "not due: observe and step");
    let stale = f.auth.apply(
        Request {
            from: pid("robot"),
            epoch: Some(Epoch(1)),
            request_id: 0,
            command: ready(2),
        },
        4,
    );
    assert_eq!(Fixture::status(stale), CtlStatus::StaleEpoch);

    // A wall-clock session refuses tick verbs.
    let mut wall = Fixture::new();
    assert_eq!(
        Fixture::status(wall.run("robot", ready(0), 0)),
        CtlStatus::Unsupported
    );
}

/// A virtual session, armed: `robot`, and `policy` bound to `cmd1` at
/// `replan_hz`.
fn armed_at(replan_hz: f64) -> Fixture {
    let mut auth = Authority::new(
        SessionId::new(),
        AuthorityConfig {
            virtual_clock: Some(VirtualConfig::default()),
            ..AuthorityConfig::default()
        },
    );
    for (id, roles, priority) in [
        ("robot", vec![Role::Embodiment], PriorityClass::Scripted),
        ("policy", vec![Role::Controller], PriorityClass::Model),
    ] {
        auth.apply(
            Request {
                from: pid(id),
                epoch: None,
                request_id: 0,
                command: attach(id, roles, priority),
            },
            0,
        )
        .expect("attach");
    }
    let mut f = Fixture { auth };
    f.run("robot", declare(&robot_map("")), 0).expect("declare");
    let channels = vec![f.ch("cmd1")];
    f.run(
        "policy",
        Command::DeclareBinding {
            fingerprint: Fingerprint::from(5),
            revision: 1,
            table_json: "{}".into(),
            channels,
            replan_hz,
            contracts: Vec::new(),
            grids: Vec::new(),
        },
        0,
    )
    .expect("bind");
    f.run("policy", Command::Arm, 0).expect("arm");
    f
}

fn ready_first_tick(f: &mut Fixture) {
    let ready = Command::Ready {
        tick: 0,
        watermarks: Vec::new(),
    };
    f.run("robot", ready, 1).expect("ready");
}

#[test]
fn a_controller_that_never_replans_is_not_due() {
    let mut f = armed_at(0.0);
    ready_first_tick(&mut f);
    let ticks = f.auth.take_ticks();
    assert!(matches!(ticks.first(), Some(VirtualTick::Observe(s)) if s.due.is_empty()));
}

#[test]
fn a_controller_that_leaves_stops_holding_up_the_barrier() {
    let mut f = armed_at(500.0);
    ready_first_tick(&mut f);
    let ticks = f.auth.take_ticks();
    assert!(matches!(ticks.as_slice(), [VirtualTick::Observe(s)] if s.due == [pid("policy")]));
    f.auth.participant_lost(&pid("policy"), 2);
    assert!(matches!(f.auth.take_ticks().as_slice(), [VirtualTick::Step(s)] if s.tick == 0));
}

#[test]
fn a_controller_is_due_once_every_period_of_ticks() {
    // 500 Hz on 1 ms ticks: every second tick.
    let mut f = armed_at(500.0);
    let ready = |tick| Command::Ready {
        tick,
        watermarks: Vec::new(),
    };
    let due = |ticks: &[VirtualTick]| match ticks.first() {
        Some(VirtualTick::Observe(s)) => s.due.clone(),
        other => panic!("{other:?}"),
    };
    f.run("robot", ready(0), 1).expect("ready");
    assert_eq!(due(&f.auth.take_ticks()), [pid("policy")]);
    let decided = Command::Decided {
        tick: 0,
        chunks: Vec::new(),
    };
    f.run("policy", decided, 2).expect("decided");
    f.auth.take_ticks();
    f.run("robot", ready(1), 3).expect("ready");
    assert!(due(&f.auth.take_ticks()).is_empty());
    f.run("robot", ready(2), 4).expect("ready");
    assert_eq!(due(&f.auth.take_ticks()), [pid("policy")]);
}
