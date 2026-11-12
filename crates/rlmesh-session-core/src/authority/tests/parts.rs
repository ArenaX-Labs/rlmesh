//! Declaring parts: channel ids, re-declaration and removal.

use super::*;

#[test]
fn the_host_assigns_part_ids_and_never_reuses_them() {
    let mut f = Fixture::new();
    let parts: Vec<&PartSpec> = f.auth.parts().collect();
    assert_eq!(parts.len(), 7);
    let arm = f.auth.part(f.ch("arm")).expect("arm").clone();
    let applied = arm
        .applied_channel
        .expect("commandable parts get an applied id");
    assert_ne!(applied, arm.channel);
    assert!(
        f.auth
            .part(f.ch("ft"))
            .expect("ft")
            .applied_channel
            .is_none()
    );
    let mut ids: Vec<u32> = f
        .auth
        .parts()
        .flat_map(|p| [Some(p.channel), p.applied_channel])
        .flatten()
        .map(|c| c.0)
        .collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), 7 + 6, "every stream id is distinct");

    // The snapshot carries the declaration fingerprint.
    let contract = |f: &Fixture| {
        f.auth
            .snapshot(Vec::new(), 0)
            .contract
            .get(&pid("robot"))
            .copied()
    };
    let declared = contract(&f).expect("a declaration fingerprint");

    // An identical re-declaration is a no-op.
    let revision = f.auth.revision();
    f.run("robot", declare(&robot_map("")), 1)
        .expect("redeclare");
    assert_eq!(f.auth.revision(), revision);
    assert_eq!(f.ch("arm"), arm.channel);
    assert_eq!(contract(&f), Some(declared));

    // A changed schema retires the id; armed, only sensors may change.
    f.run("operator", Command::Arm, 1).expect("arm");
    let ft = f.ch("ft");
    let hot = robot_map(r#", "wrist/cam": {"kind": "camera", "hz": 30, "width": 4, "height": 4}"#)
        .replace(r#""hz": 1000"#, r#""hz": 500"#);
    let applied = f.run("robot", declare(&hot), 2).expect("hot-plug");
    assert_ne!(
        contract(&f),
        Some(declared),
        "a new schema, a new fingerprint"
    );
    assert!(matches!(
        applied.events.as_slice(),
        [AuthorityEvent::PartsRemoved { channels, .. }, AuthorityEvent::PartsDeclared { .. }]
            if channels == &[ft]
    ));
    assert!(
        f.ch("ft") > ft && f.ch("wrist/cam") > ft,
        "fresh ids, never reused"
    );
    let changed = robot_map("").replace(
        r#""cmd1": {"kind": "joints", "hz": 500"#,
        r#""cmd1": {"kind": "joints", "hz": 250"#,
    );
    // A sensor hot-plug holds nothing: the robot is leased at once.
    let lease = f.grant("policy", "cmd1", 3).expect("lease");
    assert_eq!(
        Fixture::status(f.run("robot", declare(&changed), 3)),
        CtlStatus::InvalidTransition,
        "a commandable part is not re-declared while it is leased"
    );
    f.run(
        "policy",
        Command::ReleaseLease {
            channels: vec![lease.channel],
            token: lease.token,
            to: None,
        },
        4,
    )
    .expect("release");
    let cmd1 = f.ch("cmd1");
    f.run("robot", declare(&changed), 5)
        .expect("no longer leased");
    assert!(f.ch("cmd1") > cmd1);
    assert!(f.auth.part(cmd1).is_none());

    // Only embodiments declare parts; a bad map changes nothing.
    assert_eq!(
        Fixture::status(f.run("viewer", declare(&robot_map("")), 6)),
        CtlStatus::Forbidden
    );
    let before = f.auth.parts().count();
    assert_eq!(
        Fixture::status(f.run("robot", declare(r#"{"Bad": {}}"#), 6)),
        CtlStatus::InvalidRequest
    );
    assert_eq!(f.auth.parts().count(), before);
}

#[test]
fn a_detaching_owner_takes_its_parts_and_their_leases_along() {
    let mut f = Fixture::new();
    f.run("operator", Command::Arm, 0).expect("arm");
    f.grant("policy", "cmd1", 0).expect("lease");
    let applied = f.run("robot", Command::Detach, 1).expect("detach");
    assert!(applied.events.iter().any(|e| matches!(
        e,
        AuthorityEvent::LeaseEnded { reason, .. } if reason.code == ReasonCode::PartRemoved
    )));
    assert_eq!(f.auth.parts().count(), 0);
    assert!(f.auth.snapshot(Vec::new(), 1).contract.is_empty());
    assert_eq!(f.auth.mode(), Mode::Armed);
    f.run("policy", Command::Detach, 2).expect("detach");
    assert!(f.auth.binding(&pid("policy")).is_none());
}

#[test]
fn a_seal_policy_seals_exactly_the_parts_it_names() {
    let (arm, gripper, other) = (pid("arm"), pid("gripper"), pid("other"));
    assert!(!SealPolicy::None.seals(&arm, "ft"));
    assert!(SealPolicy::All.seals(&arm, "ft"));
    let policy = SealPolicy::Parts(vec![
        "arm/ft".into(),
        "arm/wrist/cam".into(),
        "gripper/*".into(),
        "malformed".into(),
    ]);
    for (robot, path, sealed) in [
        (&arm, "ft", true),
        (&arm, "wrist/cam", true),
        (&arm, "cmd", false),
        (&gripper, "anything", true),
        (&gripper, "ft", true),
        (&other, "ft", false),
        (&other, "anything", false),
    ] {
        assert_eq!(policy.seals(robot, path), sealed, "{robot}/{path}");
    }
}
