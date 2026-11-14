//! Leases: grants, bindings, preemption, command groups and fencing.

use super::*;

#[test]
fn first_lease_activates_and_releasing_the_last_rearms() {
    let mut f = Fixture::new();
    assert_eq!(
        Fixture::status(f.grant("policy", "cmd4", 0)),
        CtlStatus::InvalidTransition,
        "grant before arm"
    );
    f.run("operator", Command::Arm, 0).expect("arm");
    assert_eq!(
        Fixture::status(f.grant("viewer", "cmd4", 0)),
        CtlStatus::Forbidden
    );
    let lease = f.grant("policy", "cmd4", 10).expect("grant");
    assert_eq!(f.auth.mode(), Mode::Active);
    assert_eq!(lease.expires_ns, 10 + GRACE as i64);
    assert_eq!(lease.binding, f.fingerprint("policy"));
    f.auth
        .check_fence(f.ch("cmd4"), lease.token, f.auth.epoch(), 20)
        .expect("fence");

    let revision = f.auth.revision();
    let renewed = f.grant("policy", "cmd4", 50).expect("re-grant");
    assert_eq!(renewed.token, lease.token, "renewal keeps the token");
    assert_eq!(f.auth.revision(), revision, "renewal is not structural");

    f.release("policy", &lease, 60).expect("release");
    assert_eq!(f.auth.mode(), Mode::Armed);
    assert!(f.auth.lease(f.ch("cmd4")).is_none());
}

#[test]
fn a_multi_part_grant_is_all_or_none() {
    let mut f = Fixture::new();
    f.run("operator", Command::Arm, 0).expect("arm");
    let script = f.grant("script", "cmd3", 0).expect("script lease");
    f.grant("operator", "cmd2", 0).expect("operator lease");
    // cmd2 is held by a higher class: nothing of the request is granted,
    // not even the free cmd1, and the preemptible cmd3 is untouched.
    assert_eq!(
        Fixture::status(f.grant_with("policy", &["cmd1", "cmd3", "cmd2"], &[], 1)),
        CtlStatus::LeaseBusy
    );
    assert!(f.auth.lease(f.ch("cmd1")).is_none());
    assert_eq!(
        f.auth.lease(f.ch("cmd3")).map(|l| l.token),
        Some(script.token)
    );

    let leases = f
        .grant_with("policy", &["cmd1", "cmd3"], &[], 2)
        .expect("all of it");
    assert_eq!(leases.len(), 2);
    assert_eq!(leases[0].token, leases[1].token, "one token per request");
    assert!(leases[0].token > script.token);

    let cmd1 = f.ch("cmd1");
    for (paths, release) in [
        (vec!["cmd4", "cmd4"], vec![]),
        (vec!["cmd4"], vec!["cmd4"]),
        (vec!["cmd4"], vec!["cmd2"]),
        (vec!["ft"], vec![]),
        (vec![], vec![]),
    ] {
        assert_eq!(
            Fixture::status(f.grant_with("policy", &paths, &release, 3)),
            CtlStatus::InvalidRequest,
            "{paths:?} releasing {release:?}"
        );
    }
    assert_eq!(
        f.auth.lease(cmd1).map(|l| l.holder.clone()),
        Some(pid("policy"))
    );
}

#[test]
fn a_lease_needs_the_requesters_current_binding() {
    let mut f = Fixture::new();
    f.run("operator", Command::Arm, 0).expect("arm");
    f.bind("script", 30, &["cmd1"]).expect("narrow binding");
    assert_eq!(
        Fixture::status(f.grant("script", "cmd2", 0)),
        CtlStatus::NotBound,
        "outside the binding"
    );
    let stale = Command::GrantLease {
        channels: vec![f.ch("cmd1")],
        release: Vec::new(),
        binding: Fingerprint::from(3),
        ttl_ns: None,
    };
    assert_eq!(
        Fixture::status(f.run("script", stale, 0)),
        CtlStatus::NotBound,
        "a binding that is not current"
    );
    f.run(
        "robot",
        attach("x", vec![Role::Controller], PriorityClass::Model),
        0,
    )
    .expect("re-attach as a controller");
    let unbound = Command::GrantLease {
        channels: vec![f.ch("cmd1")],
        release: Vec::new(),
        binding: Fingerprint::from(1),
        ttl_ns: None,
    };
    assert_eq!(
        Fixture::status(f.run("robot", unbound, 0)),
        CtlStatus::NotBound
    );
    assert_eq!(
        Fixture::status(f.bind("script", 31, &["ft"])),
        CtlStatus::UnknownPath
    );
    assert_eq!(
        Fixture::status(f.bind("script", 31, &["cmd1", "cmd1"])),
        CtlStatus::InvalidRequest
    );
    assert_eq!(
        Fixture::status(f.bind("script", 30, &["cmd2"])),
        CtlStatus::InvalidRequest,
        "one fingerprint, one binding"
    );
    assert_eq!(
        Fixture::status(f.bind("viewer", 40, &[])),
        CtlStatus::Forbidden
    );
}

#[test]
fn rebinding_reissues_the_holders_leases_under_a_fresh_token() {
    let mut f = Fixture::new();
    f.run("operator", Command::Arm, 0).expect("arm");
    let old = f
        .grant_with("policy", &["cmd1", "cmd2"], &[], 0)
        .expect("leases");
    let epoch = f.auth.epoch();
    // The same binding again changes nothing.
    let applied = f
        .bind(
            "policy",
            1,
            &["cmd1", "cmd2", "cmd3", "cmd4", "arm", "arm/tcp"],
        )
        .expect("same binding");
    assert!(applied.events.is_empty());
    // A new binding keeps cmd1 under a new token and drops cmd2.
    let applied = f.bind("policy", 11, &["cmd1"]).expect("rebind");
    let rebound = f.auth.lease(f.ch("cmd1")).expect("kept").clone();
    assert!(rebound.token > old[0].token, "a fresh token");
    assert_eq!(rebound.binding, Fingerprint::from(11));
    assert!(
        f.auth.lease(f.ch("cmd2")).is_none(),
        "outside the new binding"
    );
    assert!(applied.events.iter().any(|e| matches!(
        e,
        AuthorityEvent::LeaseRebound { previous, .. } if previous.token == old[0].token
    )));
    assert!(matches!(
        f.auth.check_fence(f.ch("cmd1"), old[0].token, epoch, 1),
        Err(FenceError::StaleToken { .. })
    ));
    assert_eq!(f.auth.mode(), Mode::Active);
    // Dropping every part settles back to ARMED.
    f.bind("policy", 12, &["cmd4"]).expect("rebind");
    assert_eq!(f.auth.mode(), Mode::Armed);
}

#[test]
fn command_groups_hold_one_member_and_switch_atomically() {
    let mut f = Fixture::new();
    f.run("operator", Command::Arm, 0).expect("arm");
    assert_eq!(
        Fixture::status(f.grant_with("policy", &["arm", "arm/tcp"], &[], 0)),
        CtlStatus::InvalidRequest,
        "two members of one group in one request"
    );
    let joint = f.grant("policy", "arm", 0).expect("arm");
    assert_eq!(
        Fixture::status(f.grant("policy", "arm/tcp", 1)),
        CtlStatus::GroupBusy,
        "holding a member without releasing it"
    );
    assert_eq!(
        Fixture::status(f.grant("script", "arm/tcp", 1)),
        CtlStatus::GroupBusy,
        "a member held by a higher class"
    );
    let applied = f
        .run(
            "policy",
            Command::GrantLease {
                channels: vec![f.ch("arm/tcp")],
                release: vec![f.ch("arm")],
                binding: f.fingerprint("policy"),
                ttl_ns: None,
            },
            2,
        )
        .expect("switch");
    let tcp = applied.leases[0].clone();
    assert!(tcp.token > joint.token);
    assert!(f.auth.lease(f.ch("arm")).is_none());
    let revision = f.auth.revision();

    // The operator takes the other member: the model's is preempted in
    // the same revision.
    let human = f
        .grant("operator", "arm", 3)
        .expect("preempt across the group");
    assert_eq!(f.auth.revision(), revision + 1);
    assert!(f.auth.lease(f.ch("arm/tcp")).is_none());
    assert_eq!(
        f.auth.lease(f.ch("arm")).map(|l| l.token),
        Some(human.token)
    );
    assert_eq!(
        Fixture::status(f.grant("policy", "arm/tcp", 4)),
        CtlStatus::GroupBusy
    );
    // Other parts are independent of the group.
    f.grant("policy", "cmd1", 4).expect("ungrouped");
}

#[test]
fn a_grant_from_a_stale_epoch_is_refused() {
    let mut f = Fixture::new();
    f.run("operator", Command::Arm, 0).expect("arm");
    let result = f.auth.apply(
        Request {
            from: pid("policy"),
            epoch: Some(Epoch(1)),
            request_id: 0,
            command: Command::GrantLease {
                channels: vec![f.ch("cmd1")],
                release: Vec::new(),
                binding: f.fingerprint("policy"),
                ttl_ns: None,
            },
        },
        0,
    );
    assert_eq!(Fixture::status(result), CtlStatus::StaleEpoch);
}

#[test]
fn operator_preempts_model_and_the_old_token_is_fenced() {
    let mut f = Fixture::new();
    f.run("operator", Command::Arm, 0).expect("arm");
    let model = f.grant("policy", "cmd1", 0).expect("model lease");
    let human = f.grant("operator", "cmd1", 5).expect("preempt");
    assert!(human.token > model.token, "preemption bumps the token");
    assert_eq!(f.auth.mode(), Mode::Active);

    let epoch = f.auth.epoch();
    let cmd1 = f.ch("cmd1");
    assert_eq!(
        f.auth.check_fence(cmd1, model.token, epoch, 6),
        Err(FenceError::StaleToken {
            channel: cmd1,
            current: human.token,
            got: model.token
        })
    );
    f.auth
        .check_fence(cmd1, human.token, epoch, 6)
        .expect("new token");
    assert_eq!(
        Fixture::status(f.release("policy", &model, 7)),
        CtlStatus::StaleToken
    );
    assert_eq!(
        Fixture::status(f.grant("policy", "cmd1", 8)),
        CtlStatus::LeaseBusy
    );
    f.release("operator", &human, 9).expect("release");
    let again = f.grant("policy", "cmd1", 10).expect("model again");
    assert!(again.token > human.token);
    // The snapshot says why each lease ended, and who took it.
    let ended = f.auth.snapshot(Vec::new(), 10).ended;
    let why: Vec<_> = ended
        .iter()
        .map(|e| (e.lease.token, e.reason.code, e.by.as_ref().map(|b| b.token)))
        .collect();
    assert_eq!(
        why,
        [
            (model.token, ReasonCode::Preempted, Some(human.token)),
            (human.token, ReasonCode::Released, None),
        ]
    );
    assert_eq!(f.auth.mode_reason().code, ReasonCode::FirstLease);
}

#[test]
fn equal_or_lower_classes_cannot_preempt() {
    let mut f = Fixture::new();
    f.run("operator", Command::Arm, 0).expect("arm");
    f.grant("policy", "cmd1", 0).expect("lease");
    assert_eq!(
        Fixture::status(f.grant("script", "cmd1", 1)),
        CtlStatus::LeaseBusy
    );
    f.grant("script", "cmd2", 1).expect("other part");
}

#[test]
fn the_snapshot_fence_matches_the_host() {
    let mut f = Fixture::new();
    f.run("operator", Command::Arm, 0).expect("arm");
    let lease = f.grant("policy", "cmd3", 0).expect("lease");
    let snap = f.auth.snapshot(Vec::new(), 1);
    snap.check_fence(f.ch("cmd3"), lease.token, snap.epoch)
        .expect("fence");
    assert_eq!(snap.protocol_minor, PROTOCOL_MINOR);
    assert_eq!(snap.clock_mode, ClockMode::Wall);
    assert_eq!(
        snap.part_at(&pid("robot"), "cmd3").map(|p| p.channel),
        Some(f.ch("cmd3"))
    );
    assert_eq!(snap.group_members(&pid("robot"), "arm").count(), 2);
    let applied = snap
        .part(f.ch("cmd3"))
        .and_then(|p| p.applied_channel)
        .expect("applied");
    assert_eq!(
        snap.stream(applied).map(|s| (s.applied, s.path)),
        Some((true, "cmd3/_applied".to_string()))
    );
    assert!(snap.binding(&pid("policy")).is_some());
    let later = f.auth.snapshot(Vec::new(), 2);
    assert!(later.is_newer_than(&snap));
    assert!(!snap.is_newer_than(&snap) && !snap.is_newer_than(&later));
}

#[test]
fn refusals_and_events_name_the_part_as_robot_and_path() {
    let mut f = Fixture::new();
    f.run("operator", Command::Arm, 0).expect("arm");
    let lease = f.grant("operator", "cmd2", 0).expect("operator lease");
    let busy = f.grant("policy", "cmd2", 1).expect_err("held higher");
    assert_eq!(
        busy.message,
        "robot:cmd2 is held by operator at Operator; policy at Model cannot preempt it"
    );
    f.grant("operator", "arm", 2).expect("group lease");
    let group = f.grant("policy", "arm/tcp", 3).expect_err("group held");
    assert_eq!(
        group.message,
        "robot:arm, in the command group of robot:arm/tcp, is held by operator at Operator"
    );
    let release = f.release("script", &lease, 4).expect_err("not the holder");
    assert!(
        release
            .message
            .starts_with("robot:cmd2 is held by operator"),
        "{}",
        release.message
    );
    let granted = f
        .run(
            "policy",
            Command::GrantLease {
                channels: vec![f.ch("cmd1")],
                release: Vec::new(),
                binding: f.fingerprint("policy"),
                ttl_ns: None,
            },
            5,
        )
        .expect("free part");
    let line = granted.events[0].to_string();
    assert!(
        line.starts_with("lease robot:cmd1 granted to policy"),
        "{line}"
    );
}

#[test]
fn the_host_fence_holds_through_the_expiry_instant_and_no_later() {
    let mut f = Fixture::new();
    f.run("operator", Command::Arm, 0).expect("arm");
    let lease = f.grant("policy", "cmd4", 10).expect("grant");
    let (channel, epoch) = (lease.channel, f.auth.epoch());
    f.auth
        .check_fence(channel, lease.token, epoch, lease.expires_ns)
        .expect("at the instant");
    assert!(matches!(
        f.auth
            .check_fence(channel, lease.token, epoch, lease.expires_ns + 1),
        Err(FenceError::Expired { .. })
    ));
}

#[test]
fn lease_count_counts_the_leases_held() {
    let mut f = Fixture::new();
    f.run("operator", Command::Arm, 0).expect("arm");
    assert_eq!(f.auth.lease_count(), 0);
    f.grant("policy", "cmd1", 0).expect("grant");
    f.grant("policy", "cmd2", 0).expect("grant");
    assert_eq!(f.auth.lease_count(), 2);
}

#[test]
fn an_authority_reports_the_session_it_was_made_for() {
    let session = SessionId::named("demo");
    let auth = Authority::new(session, AuthorityConfig::default());
    assert_eq!(auth.session(), session);
}

#[test]
fn a_lease_of_the_same_class_is_not_preempted() {
    let mut f = Fixture::new();
    f.auth
        .apply(
            Request {
                from: pid("rival"),
                epoch: None,
                request_id: 0,
                command: attach("rival", vec![Role::Controller], PriorityClass::Model),
            },
            0,
        )
        .expect("attach");
    f.bind("rival", 9, &["cmd1"]).expect("bind");
    f.run("operator", Command::Arm, 0).expect("arm");
    f.grant("policy", "cmd1", 0).expect("grant");
    assert_eq!(
        Fixture::status(f.grant("rival", "cmd1", 0)),
        CtlStatus::LeaseBusy
    );
}

#[test]
fn a_holder_asking_for_a_peer_of_its_own_part_is_told_to_release_it() {
    let mut f = Fixture::new();
    f.run("operator", Command::Arm, 0).expect("arm");
    f.grant("policy", "arm", 0).expect("grant");
    let err = f.grant("policy", "arm/tcp", 0).expect_err("refused");
    assert_eq!(err.status, CtlStatus::GroupBusy);
    assert!(
        err.message.contains("release it in the same request"),
        "{}",
        err.message
    );
}

#[test]
fn a_peer_held_at_the_same_class_is_not_preempted() {
    let mut f = Fixture::new();
    f.auth
        .apply(
            Request {
                from: pid("rival"),
                epoch: None,
                request_id: 0,
                command: attach("rival", vec![Role::Controller], PriorityClass::Model),
            },
            0,
        )
        .expect("attach");
    f.bind("rival", 9, &["arm/tcp"]).expect("bind");
    f.run("operator", Command::Arm, 0).expect("arm");
    let tcp = f.grant("rival", "arm/tcp", 0).expect("grant");
    assert_eq!(
        Fixture::status(f.grant("policy", "arm", 0)),
        CtlStatus::GroupBusy
    );
    assert_eq!(f.auth.lease(tcp.channel), Some(&tcp));
}

#[test]
fn a_zero_ttl_takes_the_grace_and_any_other_is_taken_as_asked() {
    let mut f = Fixture::new();
    f.run("operator", Command::Arm, 0).expect("arm");
    let mut grant = |path: &str, ttl_ns| {
        let command = Command::GrantLease {
            channels: vec![f.ch(path)],
            release: Vec::new(),
            binding: f.fingerprint("policy"),
            ttl_ns: Some(ttl_ns),
        };
        f.run("policy", command, 10).expect("grant").leases[0].expires_ns
    };
    assert_eq!(grant("cmd1", 0), 10 + GRACE as i64);
    assert_eq!(grant("cmd2", 5), 15);
}

#[test]
fn an_estop_ends_every_lease_as_voided_by_it() {
    let mut f = Fixture::new();
    f.run("operator", Command::Arm, 0).expect("arm");
    f.grant("policy", "cmd1", 0).expect("grant");
    let estop = Command::Estop {
        reason: "button".into(),
        code: ReasonCode::Unspecified,
        origin: None,
        at_ns: 0,
    };
    f.run("viewer", estop, 1).expect("estop");
    let ended = f.auth.snapshot(Vec::new(), 1).ended;
    assert_eq!(
        ended.last().map(|e| e.reason.code),
        Some(ReasonCode::VoidedByEstop)
    );
}
