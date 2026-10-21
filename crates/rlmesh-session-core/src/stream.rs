//! One stream's receive rules, sans-IO: which samples a subscriber accepts
//! (epoch, seq order, gaps as losses), whether a sample fits its declared
//! stream, and when a value is stale against the stream's `max_age`.
//!
//! `rlmesh-session`'s `SampleSubscriber` applies these to every sample it
//! receives; the browser binding applies the same ones in wasm. Missing
//! samples are explicit: a gap in `seq` is a loss, never a silent freeze.

use std::time::Duration;

use crate::wire::{Epoch, Sample, StreamSpec};

/// Loss and freshness counters for one subscription.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SampleStats {
    /// Samples accepted.
    pub received: u64,
    /// Samples skipped over by `seq` gaps.
    pub lost: u64,
    /// Arrived with a seq at or below one already seen (reordered,
    /// duplicated, or already served from history); dropped.
    pub out_of_order: u64,
    /// From an older epoch than the subscriber's view; dropped.
    pub stale_epoch: u64,
    /// Not a decodable sample.
    pub undecodable: u64,
    /// A sample that does not fit the stream (`malformed`: the layout
    /// check, a wrong channel).
    pub malformed: u64,
    /// Reads of the latest value that found it stale (`SampleSubscriber::latest`
    /// in `rlmesh-session`).
    pub stale_reads: u64,
    /// Samples served from the host's history store.
    pub from_history: u64,
    /// The newest seq accepted.
    pub last_seq: u64,
}

impl SampleStats {
    /// Counts one decoded sample of the stream and says whether to keep it:
    /// one from an epoch older than the subscriber's view (`view_epoch`) is
    /// dropped (`stale_epoch`), one at or below the newest seq is dropped
    /// (`out_of_order`), and a jump past the next seq counts the skipped
    /// seqs as `lost`.
    pub fn accept(
        &mut self,
        seq: u64,
        epoch: Epoch,
        view_epoch: Epoch,
        from_history: bool,
    ) -> bool {
        if epoch < view_epoch {
            self.stale_epoch += 1;
            false
        } else if self.last_seq > 0 && seq <= self.last_seq {
            self.out_of_order += 1;
            false
        } else {
            if self.last_seq > 0 {
                self.lost += seq - self.last_seq - 1;
            }
            self.last_seq = seq;
            self.received += 1;
            if from_history {
                self.from_history += 1;
            }
            true
        }
    }
}

/// Whether `sample` fits `spec`: the key's stream, and the payload against
/// the declared layout (O(fields), before any view of the value). An applied
/// tick that dispatched nothing carries no payload.
pub fn check_sample(spec: &StreamSpec, sample: &Sample) -> Result<(), String> {
    if sample.channel != spec.channel {
        return Err(format!("{} on {}'s key", sample.channel, spec.channel));
    }
    match (spec.applied, sample.applied) {
        (true, Some(a)) if !a.present => {
            if sample.payload.is_empty() {
                Ok(())
            } else {
                Err("an absent applied value with a payload".into())
            }
        }
        (true, None) => Err("an applied sample without metadata".into()),
        _ => spec.check_payload(&sample.payload),
    }
}

/// Whether a value `age_ns` old is stale against `max_age` (never, without
/// one).
pub fn is_stale(age_ns: i64, max_age: Option<Duration>) -> bool {
    max_age.is_some_and(|max| u128::try_from(age_ns).is_ok_and(|age| age > max.as_nanos()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaps_are_losses_and_old_or_repeated_seqs_are_dropped() {
        let mut stats = SampleStats::default();
        assert!(stats.accept(1, Epoch(1), Epoch(1), false));
        assert!(stats.accept(2, Epoch(1), Epoch(1), true));
        assert!(stats.accept(5, Epoch(1), Epoch(1), false));
        assert!(!stats.accept(5, Epoch(1), Epoch(1), false));
        assert!(!stats.accept(3, Epoch(1), Epoch(1), false));
        assert!(!stats.accept(9, Epoch(1), Epoch(2), false));
        assert_eq!(
            stats,
            SampleStats {
                received: 3,
                lost: 2,
                out_of_order: 2,
                stale_epoch: 1,
                from_history: 1,
                last_seq: 5,
                ..SampleStats::default()
            }
        );
    }

    #[test]
    fn staleness_needs_a_bound_and_a_non_negative_age() {
        let bound = Some(Duration::from_millis(10));
        assert!(!is_stale(10_000_000, bound));
        assert!(is_stale(10_000_001, bound));
        assert!(!is_stale(-1, bound));
        assert!(!is_stale(i64::MAX, None));
    }
}
