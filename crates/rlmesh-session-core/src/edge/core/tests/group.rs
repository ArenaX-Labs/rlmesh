//! Command groups: one leased member dispatches, and which parts get an edge.

use super::*;

#[test]
fn a_group_dispatches_only_its_leased_member_and_hands_over() {
    let mut f = Fixture::with(EdgeConfig::default(), true, &["arm", "arm/tcp"], None);
    assert_eq!(f.edge.group(), Some("arm"));
    let joint = f.grant_with("policy", "arm", &[], 0);
    f.publish(0);
    f.accept(f.chunk_for("arm", "policy", &joint, 1, 0, 50, 3.0), 0)
        .expect("accept");
    let tick = f.tick(0);
    assert_eq!(tick.channel, Some(f.ch("arm")));
    assert_eq!(
        &tick.action.expect("due").value[..4],
        &1.0_f32.to_le_bytes(),
        "clamped to the arm's limit"
    );
    assert_eq!(tick.applied.len(), 2);
    assert_eq!(tick.applied[1].applied, Applied::absent(EdgeMode::Idle));

    // The model switches to the Cartesian member in one revision.
    let tcp = f.grant_with("policy", "arm/tcp", &["arm"], MS);
    f.publish(MS);
    f.accept(
        f.chunk_for("arm/tcp", "policy", &tcp, 1, 2 * MS, 5, 0.01),
        MS,
    )
    .expect("accept");
    assert_eq!(
        f.accept(f.chunk_for("arm", "policy", &joint, 2, MS, 5, 0.0), MS),
        Err(Rejection::NoLease)
    );
    let tick = f.tick(MS);
    let handover = tick.handover.expect("a handover");
    assert_eq!((handover.from, handover.to), (f.ch("arm"), f.ch("arm/tcp")));
    assert!(tick.action.is_none() && tick.safe_behavior.is_none());
    f.edge.handover_done();
    let tick = f.tick(2 * MS);
    assert_eq!(
        (tick.mode, tick.channel),
        (EdgeMode::Driving, Some(f.ch("arm/tcp")))
    );
    assert!(tick.action.is_some(), "a delta's first tick in its slot");
    assert!(f.tick(2 * MS + MS / 2).action.is_none(), "dispatched once");
    assert_eq!(f.edge.stats().handovers, 1);
}

#[test]
fn only_commandable_parts_of_one_group_and_owner_get_an_edge() {
    let f = Fixture::new();
    let part = |path: &str| {
        f.auth
            .parts()
            .find(|p| p.path.as_str() == path)
            .cloned()
            .expect(path)
    };
    let mut sense = part("cmd");
    sense.command = None;
    assert!(matches!(
        EdgeCore::new(vec![sense], pid("robot"), EdgeConfig::default()),
        Err(GroupError::NotCommandable(_))
    ));
    assert!(matches!(
        EdgeCore::new(vec![part("cmd")], pid("other"), EdgeConfig::default()),
        Err(GroupError::NotOwner { .. })
    ));
    assert!(matches!(
        EdgeCore::new(
            vec![part("cmd"), part("grip")],
            pid("robot"),
            EdgeConfig::default()
        ),
        Err(GroupError::NotOneGroup(_))
    ));
    assert!(matches!(
        EdgeCore::new(Vec::new(), pid("robot"), EdgeConfig::default()),
        Err(GroupError::NoParts)
    ));
}
