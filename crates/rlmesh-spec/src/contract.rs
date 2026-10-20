//! Delivery contracts (§8.2), freshness (§8.7) and publication plans: the
//! rules the binder applies when it states a contract and the host applies
//! again when it admits one, so neither trusts the other.
//!
//! - **Mode.** An input without `history` is `latest@D`; with `history`,
//!   `history@D`; when D is at or above the source rate S (or the stream is
//!   an event), `every`.
//! - **Freshness.** The stricter of an explicit policy `max_age_ms` and an
//!   explicit robot `max_age_ms`; otherwise `max(3000 / min(S, D), 5)` ms.
//! - **Admission.** A window needs every sample, so `history` is refused on a
//!   part whose `send` ceiling decimates. An explicit bound below the
//!   contract's sample interval `1 / min(S, D)` cannot be met and is refused.
//!   A window spans at most 10 s; a controller's rings hold at most 512 MiB.
//! - **Plan.** One per stream, from every admitted contract on it: any
//!   `every` publishes each sample; else any `history` publishes increments
//!   flushed at `1 / max D` (earlier at 256 records or 4 MiB); else the
//!   newest sample at `max D`. The plan never exceeds the `send` ceiling.

use std::time::Duration;

use rlmesh_wire::part::{MAX_BATCH, MAX_CHUNK_BYTES, PartSpec, SendPolicy};
use rlmesh_wire::{ContractMode, Plan, StreamContract};

/// The longest window, in samples (§3.4).
pub const MAX_HISTORY: u32 = 1024;
/// The longest window, in time.
pub const MAX_HISTORY_SPAN: Duration = Duration::from_secs(10);
/// The most ring memory one controller's windows hold (§3.4).
pub const MAX_RING_BYTES: u64 = 512 << 20;
/// How long beyond its window a publisher retains samples, for recovery.
pub const RECOVERY: Duration = Duration::from_secs(1);
/// The derived freshness floor.
const MIN_FRESHNESS: Duration = Duration::from_millis(5);

/// The source rate of `part`'s stream: its `hz`, or for its applied stream
/// the edge control rate (one applied sample per control tick).
pub fn source_hz(part: &PartSpec, applied: bool) -> f64 {
    if applied { part.control_hz } else { part.hz }
}

/// The contract mode for an input on a stream at source rate `source_hz`.
pub fn mode(part: &PartSpec, source_hz: f64, delivery_hz: f64, history: u32) -> ContractMode {
    if part.event || delivery_hz >= source_hz {
        ContractMode::Every
    } else if history > 1 {
        ContractMode::History
    } else {
        ContractMode::Latest
    }
}

/// The interval between new samples a contract can deliver: `1 / min(S, D)`.
pub fn sample_interval(source_hz: f64, delivery_hz: f64) -> Duration {
    let hz = source_hz.min(delivery_hz);
    if hz > 0.0 {
        Duration::from_secs_f64(1.0 / hz)
    } else {
        Duration::ZERO
    }
}

/// An input's freshness bound and whether it was declared (§8.7). Events
/// have none.
pub fn freshness(
    part: &PartSpec,
    applied: bool,
    delivery_hz: f64,
    policy_max_age: Option<Duration>,
) -> (Option<Duration>, bool) {
    if part.event {
        return (None, false);
    }
    let robot = if applied { None } else { part.max_age };
    match (policy_max_age, robot) {
        (Some(a), Some(b)) => (Some(a.min(b)), true),
        (Some(a), None) | (None, Some(a)) => (Some(a), true),
        (None, None) => {
            let derived = sample_interval(source_hz(part, applied), delivery_hz) * 3;
            (Some(derived.max(MIN_FRESHNESS)), false)
        }
    }
}

/// The sample size a window holds, raw.
pub fn sample_bytes(part: &PartSpec, applied: bool) -> u64 {
    let fields = match (&part.command, applied) {
        (Some(command), true) => &command.fields,
        _ => &part.fields,
    };
    fields
        .iter()
        .map(|f| f.byte_len().unwrap_or(0).next_multiple_of(8))
        .sum()
}

/// Why `contract` cannot be admitted on `part`'s stream (`applied`: its
/// `_applied` stream), or `Ok`.
pub fn admit(contract: &StreamContract, part: &PartSpec, applied: bool) -> Result<(), String> {
    let path = if applied {
        part.path.applied()
    } else {
        part.path.to_string()
    };
    let d = contract.delivery_hz;
    if !(d.is_finite() && d > 0.0 && d <= 10_000.0) {
        return Err(format!("{path}: delivery rate {d} is not in (0, 10000]"));
    }
    if contract.history == 0 || contract.history > MAX_HISTORY {
        return Err(format!(
            "{path}: a window of {} samples (1 to {MAX_HISTORY})",
            contract.history
        ));
    }
    if contract.mode == ContractMode::Latest && contract.history > 1 {
        return Err(format!("{path}: a latest contract has no window"));
    }
    let s = source_hz(part, applied);
    if contract.history > 1 {
        if !applied && part.send_policy() != SendPolicy::default() {
            return Err(format!(
                "{path}: a window of {} needs every sample, but it publishes {}",
                contract.history, part.send
            ));
        }
        let span = Duration::from_secs_f64(f64::from(contract.history) / s.max(f64::MIN_POSITIVE));
        if span > MAX_HISTORY_SPAN {
            return Err(format!(
                "{path}: a window of {} at {} Hz spans {} s (at most 10 s)",
                contract.history,
                crate::explain::hz(s),
                crate::explain::seconds(span)
            ));
        }
    }
    if contract.max_age_explicit
        && let Some(bound) = contract.max_age
    {
        let interval = sample_interval(s, d);
        if bound < interval && !part.event {
            return Err(format!(
                "{path}: max_age_ms {} is below the {} ms between samples of {}@{}",
                crate::explain::ms(bound),
                crate::explain::ms(interval),
                mode_name(contract.mode),
                crate::explain::hz(d)
            ));
        }
    }
    Ok(())
}

/// The ring memory a set of contracts holds, for the 512 MiB cap.
pub fn ring_bytes<'a>(
    contracts: impl IntoIterator<Item = (&'a StreamContract, &'a PartSpec, bool)>,
) -> u64 {
    contracts
        .into_iter()
        .map(|(c, part, applied)| u64::from(c.history).saturating_mul(sample_bytes(part, applied)))
        .fold(0, u64::saturating_add)
}

/// A contract mode's name (`latest`, `history`, `every`).
pub fn mode_name(mode: ContractMode) -> &'static str {
    match mode {
        ContractMode::Latest => "latest",
        ContractMode::History => "history",
        ContractMode::Every => "every",
    }
}

/// The publication plan for a stream from its admitted contracts, and the
/// ring retention it needs; `None` without contracts.
pub fn plan<'a>(
    contracts: impl IntoIterator<Item = &'a StreamContract>,
    part: &PartSpec,
    applied: bool,
) -> Option<(Plan, Duration)> {
    let contracts: Vec<&StreamContract> = contracts.into_iter().collect();
    let max_d = contracts
        .iter()
        .map(|c| c.delivery_hz)
        .fold(None, |m: Option<f64>, d| Some(m.map_or(d, |m| m.max(d))))?;
    let s = source_hz(part, applied);
    let window = contracts
        .iter()
        .filter(|c| c.history > 1)
        .map(|c| Duration::from_secs_f64(f64::from(c.history) / s.max(f64::MIN_POSITIVE)))
        .max();
    let flush = Duration::from_secs_f64(1.0 / max_d);
    let plan = if contracts.iter().any(|c| c.mode == ContractMode::Every) {
        Plan::Every
    } else if contracts.iter().any(|c| c.mode == ContractMode::History) {
        Plan::Increments {
            flush,
            max_records: MAX_BATCH as u32,
            max_bytes: MAX_CHUNK_BYTES as u64,
        }
    } else {
        let ceiling = if applied { None } else { part.send_policy().hz };
        Plan::Latest {
            hz: ceiling.map_or(max_d, |c| c.min(max_d)),
        }
    };
    let retain = window.map_or(Duration::ZERO, |w| w + flush + RECOVERY);
    Some((plan, retain))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalise::normalise;
    use rlmesh_wire::{ChannelId, ParticipantId, version::SPEC_VERSION};

    fn part(json: &str) -> PartSpec {
        normalise(&ParticipantId::new("r").expect("id"), SPEC_VERSION, json)
            .expect("normalise")
            .remove(0)
    }

    fn contract(
        mode: ContractMode,
        d: f64,
        history: u32,
        max_age_ms: Option<u64>,
    ) -> StreamContract {
        StreamContract {
            channel: ChannelId(1),
            mode,
            delivery_hz: d,
            history,
            max_age: max_age_ms.map(Duration::from_millis),
            max_age_explicit: max_age_ms.is_some(),
        }
    }

    #[test]
    fn healthy_decimated_delivery_is_never_stale() {
        let joints = part(r#"{"arm": {"kind": "joints", "count": 6, "hz": 1000}}"#);
        assert_eq!(
            freshness(&joints, false, 20.0, None),
            (Some(Duration::from_millis(150)), false),
            "1 kHz delivered at 20 Hz"
        );
        assert_eq!(
            freshness(&joints, false, 50.0, Some(Duration::from_millis(30))),
            (Some(Duration::from_millis(30)), true)
        );
        assert_eq!(
            freshness(&joints, false, 2000.0, None).0,
            Some(MIN_FRESHNESS)
        );
        assert_eq!(mode(&joints, 1000.0, 20.0, 50), ContractMode::History);
        assert_eq!(mode(&joints, 1000.0, 1000.0, 50), ContractMode::Every);
    }

    #[test]
    fn admission_refuses_what_a_contract_cannot_meet() {
        let ft = part(r#"{"ft": {"kind": "wrench", "frame": "tool0", "hz": 1000}}"#);
        assert!(
            admit(
                &contract(ContractMode::History, 20.0, 50, Some(100)),
                &ft,
                false
            )
            .is_ok()
        );
        let err = admit(
            &contract(ContractMode::History, 20.0, 50, Some(5)),
            &ft,
            false,
        )
        .expect_err("5 ms against a 50 ms flush");
        assert_eq!(
            err,
            "ft: max_age_ms 5 is below the 50 ms between samples of history@20"
        );
        assert!(
            admit(
                &contract(ContractMode::Every, 1000.0, 1, Some(3)),
                &ft,
                false
            )
            .is_ok()
        );
        assert!(
            admit(
                &contract(ContractMode::History, 20.0, 20_000, None),
                &ft,
                false
            )
            .is_err()
        );
        let decimated = part(
            r#"{"ft": {"kind": "wrench", "frame": "tool0", "hz": 1000, "send": "latest(hz=100)"}}"#,
        );
        assert_eq!(
            admit(
                &contract(ContractMode::History, 20.0, 50, None),
                &decimated,
                false
            ),
            Err("ft: a window of 50 needs every sample, but it publishes latest(hz=100)".into())
        );
        let slow = part(r#"{"ft": {"kind": "wrench", "frame": "tool0", "hz": 10}}"#);
        assert!(
            admit(
                &contract(ContractMode::History, 5.0, 200, None),
                &slow,
                false
            )
            .expect_err("20 s window")
            .contains("at most 10 s")
        );
    }

    #[test]
    fn one_plan_per_stream_from_every_contract() {
        let ft = part(r#"{"ft": {"kind": "wrench", "frame": "tool0", "hz": 1000}}"#);
        let history = contract(ContractMode::History, 20.0, 50, None);
        let latest = contract(ContractMode::Latest, 50.0, 1, None);
        let every = contract(ContractMode::Every, 1000.0, 1, Some(3));
        assert_eq!(plan([], &ft, false), None);
        assert_eq!(
            plan([&latest], &ft, false),
            Some((Plan::Latest { hz: 50.0 }, Duration::ZERO))
        );
        let (increments, retain) = plan([&history, &latest], &ft, false).expect("plan");
        assert_eq!(
            increments,
            Plan::Increments {
                flush: Duration::from_millis(20),
                max_records: 256,
                max_bytes: 4 << 20
            },
            "flushed at 1 / max D"
        );
        assert_eq!(retain, Duration::from_millis(50 + 20) + RECOVERY);
        assert_eq!(
            plan([&history, &every], &ft, false).map(|p| p.0),
            Some(Plan::Every)
        );
        let camera = part(
            r#"{"cam": {"kind": "camera", "width": 4, "height": 4, "hz": 30, "send": "latest(hz=10)"}}"#,
        );
        assert_eq!(
            plan(
                [&contract(ContractMode::Latest, 20.0, 1, None)],
                &camera,
                false
            )
            .map(|p| p.0),
            Some(Plan::Latest { hz: 10.0 }),
            "within the send ceiling"
        );
    }
}
