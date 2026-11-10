//! Delivery contracts: admission, demand and install before authority.

use super::*;

fn contract(
    channel: ChannelId,
    mode: crate::wire::ContractMode,
    delivery_hz: f64,
    history: u32,
    max_age_ms: Option<u64>,
) -> crate::wire::StreamContract {
    crate::wire::StreamContract {
        channel,
        mode,
        delivery_hz,
        history,
        max_age: Some(core::time::Duration::from_millis(max_age_ms.unwrap_or(150))),
        max_age_explicit: max_age_ms.is_some(),
    }
}

impl Fixture {
    fn bind_with(
        &mut self,
        from: &str,
        fp: u64,
        paths: &[&str],
        contracts: Vec<crate::wire::StreamContract>,
    ) -> Result<Applied, CtlError> {
        let channels: Vec<ChannelId> = paths.iter().map(|p| self.ch(p)).collect();
        let grids = channels
            .iter()
            .map(|c| crate::wire::ActionGrid {
                channel: *c,
                dt: core::time::Duration::from_millis(10),
            })
            .collect();
        self.run(
            from,
            Command::DeclareBinding {
                fingerprint: Fingerprint::from(fp),
                revision: 2,
                table_json: "{}".into(),
                channels,
                replan_hz: 20.0,
                contracts,
                grids,
            },
            0,
        )
    }

    fn demand(&self, path: &str) -> Option<Demand> {
        self.auth.demands.get(&self.ch(path)).cloned()
    }

    fn ack(&mut self, path: &str, revision: u64) -> Result<Applied, CtlError> {
        let channel = self.ch(path);
        self.run("robot", Command::DemandAck { channel, revision }, 0)
    }
}

#[test]
fn a_lease_waits_until_every_bound_stream_installed_its_plan() {
    use crate::wire::{ContractMode, Plan};
    let mut f = Fixture::new();
    f.run("operator", Command::Arm, 0).expect("arm");
    let ft = f.ch("ft");
    let window = contract(ft, ContractMode::History, 20.0, 50, None);
    f.bind_with("policy", 20, &["cmd1"], vec![window])
        .expect("bind");
    let demand = f.demand("ft").expect("a demand on ft");
    assert_eq!((demand.revision, demand.acked), (1, 0));
    assert_eq!(demand.plan, Plan::Every, "a window needs every sample");
    let err = f.grant("policy", "cmd1", 0).expect_err("not installed");
    assert_eq!(err.status, CtlStatus::NotReady);
    assert!(
        err.message.contains("ft has not installed"),
        "{}",
        err.message
    );
    assert_eq!(
        Fixture::status(f.run(
            "policy",
            Command::DemandAck {
                channel: ft,
                revision: 1
            },
            0
        )),
        CtlStatus::Forbidden,
        "only the stream's publisher acknowledges"
    );
    assert_eq!(Fixture::status(f.ack("ft", 2)), CtlStatus::InvalidRequest);
    let applied = f.ack("ft", 1).expect("ack");
    assert!(matches!(
        applied.events.as_slice(),
        [AuthorityEvent::DemandInstalled { revision: 1, .. }]
    ));
    let lease = f.grant("policy", "cmd1", 0).expect("installed: granted");
    assert_eq!(
        f.auth.snapshot(Vec::new(), 0).demand(ft).map(|d| d.acked),
        Some(1)
    );

    // A second controller wants a deeper window: the demand (and its
    // revision) changes, but the first binding stays served by
    // revision 1.
    let deeper = contract(ft, ContractMode::History, 20.0, 100, None);
    f.bind_with("operator", 21, &["cmd2"], vec![deeper])
        .expect("bind");
    let demand = f.demand("ft").expect("demand");
    assert_eq!((demand.revision, demand.acked), (2, 1));
    assert_eq!(demand.plan, Plan::Every);
    assert_eq!(
        f.grant("policy", "cmd1", 0).expect("renewal").token,
        lease.token
    );
    assert_eq!(
        Fixture::status(f.grant("operator", "cmd2", 0)),
        CtlStatus::NotReady
    );
    f.ack("ft", 2).expect("ack");
    f.grant("operator", "cmd2", 0).expect("granted");

    // A controller that goes away takes its contract along; the last
    // one withdraws the demand.
    f.auth.participant_lost(&pid("operator"), 1);
    assert_eq!(f.demand("ft").map(|d| d.revision), Some(3));
    f.run("policy", Command::Detach, 2).expect("detach");
    assert_eq!(f.demand("ft"), None);
}

#[test]
fn contracts_a_stream_cannot_meet_are_refused_at_admission() {
    use crate::wire::ContractMode;
    let mut f = Fixture::new();
    let (ft, cmd1) = (f.ch("ft"), f.ch("cmd1"));
    for (contracts, status, needle) in [
        (
            vec![contract(ft, ContractMode::Latest, 20.0, 1, Some(5))],
            CtlStatus::InvalidRequest,
            "below the 50 ms",
        ),
        (
            vec![contract(
                ChannelId(999),
                ContractMode::Latest,
                20.0,
                1,
                None,
            )],
            CtlStatus::UnknownPath,
            "not a declared stream",
        ),
        (
            vec![
                contract(ft, ContractMode::Latest, 20.0, 1, None),
                contract(ft, ContractMode::Latest, 30.0, 1, None),
            ],
            CtlStatus::InvalidRequest,
            "two contracts",
        ),
        (
            vec![contract(ft, ContractMode::Latest, 20.0, 5, None)],
            CtlStatus::InvalidRequest,
            "no window",
        ),
    ] {
        let err = f
            .bind_with("policy", 30, &["cmd1"], contracts)
            .expect_err("refused");
        assert_eq!(err.status, status, "{}", err.message);
        assert!(err.message.contains(needle), "{}", err.message);
    }
    // A window on a commandable part's applied stream is a stream too.
    let applied = f
        .auth
        .part(cmd1)
        .and_then(|p| p.applied_channel)
        .expect("applied");
    f.bind_with(
        "policy",
        31,
        &["cmd1"],
        vec![contract(applied, ContractMode::Latest, 20.0, 1, None)],
    )
    .expect("an applied stream");
    assert!(f.auth.demands.contains_key(&applied));
    // An action grid on a part the binding does not command.
    let off_grid = Command::DeclareBinding {
        fingerprint: Fingerprint::from(33),
        revision: 3,
        table_json: "{}".into(),
        channels: vec![cmd1],
        replan_hz: 20.0,
        contracts: Vec::new(),
        grids: vec![crate::wire::ActionGrid {
            channel: f.ch("cmd2"),
            dt: core::time::Duration::from_millis(10),
        }],
    };
    assert_eq!(
        Fixture::status(f.run("policy", off_grid, 0)),
        CtlStatus::InvalidRequest
    );
    assert_eq!(
        f.auth.binding(&pid("policy")).and_then(|b| b.grid(cmd1)),
        Some(core::time::Duration::from_millis(10))
    );
}

#[test]
fn removing_a_stream_withdraws_its_demand() {
    use crate::wire::ContractMode;
    let mut f = Fixture::new();
    let ft = f.ch("ft");
    f.bind_with(
        "policy",
        40,
        &["cmd1"],
        vec![contract(ft, ContractMode::Every, 1000.0, 1, Some(3))],
    )
    .expect("bind");
    assert_eq!(
        f.demand("ft").map(|d| d.plan),
        Some(crate::wire::Plan::Every)
    );
    let applied = f.run("robot", Command::Detach, 1).expect("detach");
    assert!(applied.events.iter().any(|e| matches!(
        e,
        AuthorityEvent::DemandChanged { channel, demand: None } if *channel == ft
    )));
    assert!(f.auth.demands.is_empty());
}
