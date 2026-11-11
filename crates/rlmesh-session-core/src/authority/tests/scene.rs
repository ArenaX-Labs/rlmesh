//! A scene's run (feature `fleet`).

use super::*;

#[cfg(feature = "fleet")]
#[test]
fn closing_a_scene_disarms_ends_every_lease_and_reaches_the_device_view() {
    let mut f = Fixture::new();
    f.auth.config.scene = Some(SceneConfig {
        info: crate::wire::SceneInfo {
            run: "r-7k2m".into(),
            name: "eyeball-rig".into(),
            fingerprint: Fingerprint::from(7),
            closing: None,
            rejoin_until_ns: None,
        },
        members: BTreeMap::new(),
        rejoin_ns: 0,
    });
    f.run("operator", Command::Arm, 0).expect("arm");
    f.grant("policy", "cmd1", 0).expect("lease");
    let revision = f.auth.revision();
    f.auth
        .close_scene(ReasonCode::SceneEnded, "cli.1a2b3c4d", "", 5);
    let state = f.auth.snapshot(Vec::new(), 5);
    assert_eq!(state.revision, revision + 1, "one revision");
    assert_eq!(
        (state.mode, state.mode_reason.code),
        (Mode::Disarmed, ReasonCode::SceneEnded)
    );
    assert!(state.leases.is_empty());
    assert_eq!(
        state.ended.last().map(|e| e.reason.code),
        Some(ReasonCode::SceneEnded)
    );
    let closing = |r: Option<&Reason>| r.map(|r| (r.code, r.by.clone()));
    let want = Some((ReasonCode::SceneEnded, "cli.1a2b3c4d".to_string()));
    assert_eq!(
        closing(state.scene.as_ref().and_then(|s| s.closing.as_ref())),
        want
    );
    let view = crate::wire::DeviceView::of(&state, &pid("robot"));
    assert_eq!(closing(view.closing.as_ref()), want);
    // Nothing arms it or leases in it again.
    assert_eq!(
        Fixture::status(f.run("operator", Command::Arm, 6)),
        CtlStatus::InvalidTransition
    );
    let refused = f.grant("policy", "cmd1", 6).expect_err("closing");
    assert!(refused.message.contains("closing"), "{}", refused.message);
    assert_eq!(f.auth.mode(), Mode::Disarmed);
    // Closing is once.
    assert!(
        f.auth
            .close_scene(ReasonCode::SceneFailed, "", "", 6)
            .is_empty()
    );
}

#[cfg(feature = "fleet")]
#[test]
fn a_lost_scene_member_is_waited_for_and_back_only_with_its_map_and_an_explicit_arm() {
    let mut f = Fixture::new();
    let member = |device: &str| crate::authority::Member::Controller {
        device: device.into(),
        max_priority: PriorityClass::Operator,
    };
    f.auth.config.scene = Some(SceneConfig {
        info: crate::wire::SceneInfo {
            run: "r-7k2m".into(),
            name: "rig".into(),
            fingerprint: Fingerprint::from(7),
            closing: None,
            rejoin_until_ns: None,
        },
        members: [
            (
                pid("robot"),
                crate::authority::Member::Robot {
                    device: None,
                    may_pin: vec![crate::auth::LOCAL_DEVICE.into()],
                },
            ),
            (pid("policy"), member("local")),
        ]
        .into(),
        rejoin_ns: 60,
    });
    f.run("operator", Command::Arm, 0).expect("arm");
    f.grant("policy", "cmd1", 0).expect("lease");
    f.auth.participant_lost(&pid("robot"), 10);
    let window = |f: &Fixture| {
        f.auth
            .snapshot(Vec::new(), 20)
            .scene
            .and_then(|s| s.rejoin_until_ns)
    };
    assert_eq!((f.auth.mode(), window(&f)), (Mode::Safe, Some(70)));
    // Back, but not declared: still waited for, and no arm.
    f.run(
        "robot",
        attach("robot", vec![Role::Embodiment], PriorityClass::Scripted),
        20,
    )
    .expect("rejoin");
    assert_eq!(window(&f), Some(70));
    assert_eq!(
        Fixture::status(f.run("operator", Command::Arm, 21)),
        CtlStatus::NotReady
    );
    let changed = declare(&robot_map(
        r#","extra": {"kind": "joints", "hz": 500, "count": 1, "fields": ["pos"]}"#,
    ));
    assert_eq!(
        Fixture::status(f.run("robot", changed, 22)),
        CtlStatus::Incompatible
    );
    f.run("robot", declare(&robot_map("")), 23)
        .expect("the same map");
    assert_eq!(window(&f), None);
    assert_eq!(f.auth.mode(), Mode::Safe, "never armed on its own");
    f.run("operator", Command::Arm, 24).expect("explicit arm");
    // Lost again while only ARMED: SAFE all the same, and an attach
    // under its name that isn't an embodiment doesn't bring it back.
    f.auth.participant_lost(&pid("robot"), 30);
    assert_eq!(f.auth.mode(), Mode::Safe);
    f.run(
        "robot",
        attach("robot", vec![Role::Observer], PriorityClass::Scripted),
        31,
    )
    .expect("an observer");
    assert!(window(&f).is_some());
    // A restart that heartbeat expiry never saw, from ACTIVE and from
    // ARMED: the same loss, and the same explicit arm.
    f.run(
        "robot",
        attach("robot", vec![Role::Embodiment], PriorityClass::Scripted),
        32,
    )
    .expect("an embodiment again");
    f.run("robot", declare(&robot_map("")), 32).expect("back");
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
        f.auth.apply_from(request, &sender, now).expect("restarted");
        assert_eq!(f.auth.mode(), Mode::Safe, "generation {generation}");
        assert!(window(f).is_some());
        f.run("robot", declare(&robot_map("")), now + 1)
            .expect("the same map");
        assert_eq!(f.auth.mode(), Mode::Safe, "never armed on its own");
        f.run("operator", Command::Arm, now + 2)
            .expect("explicit arm");
    };
    f.run("operator", Command::Arm, 33).expect("arm");
    f.grant("policy", "cmd1", 34).expect("lease");
    assert_eq!(f.auth.mode(), Mode::Active);
    restart(&mut f, 1, 40);
    assert_eq!(f.auth.mode(), Mode::Armed);
    restart(&mut f, 2, 50);
}

#[cfg(feature = "fleet")]
#[test]
fn a_scene_stops_for_a_lost_holder_it_counts_as_a_member() {
    let mut f = Fixture::new();
    f.auth.config.scene = Some(SceneConfig {
        info: crate::wire::SceneInfo {
            run: "r-7k2m".into(),
            name: "rig".into(),
            fingerprint: Fingerprint::from(7),
            closing: None,
            rejoin_until_ns: None,
        },
        members: [(
            pid("policy"),
            crate::authority::Member::Controller {
                device: crate::auth::LOCAL_DEVICE.into(),
                max_priority: PriorityClass::Operator,
            },
        )]
        .into(),
        rejoin_ns: 60,
    });
    f.run("operator", Command::Arm, 0).expect("arm");
    f.grant("policy", "cmd1", 0).expect("lease");
    f.grant("script", "cmd2", 0).expect("lease");
    f.auth.participant_lost(&pid("policy"), 10);
    assert_eq!(f.auth.mode(), Mode::Safe);
    assert_eq!(f.auth.lease_count(), 0);
}
