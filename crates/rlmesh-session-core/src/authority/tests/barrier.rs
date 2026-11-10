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
