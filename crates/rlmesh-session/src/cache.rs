//! The controller's input cache and history windows, built into the
//! controller runtime so users write no cache code.
//!
//! [`InputCache`] is sans-IO: one slot per bound stream plus a ring as deep as
//! the deepest window on it, fed sample by sample and snapshotted at each
//! policy start.
//!
//! - **Snapshot.** Every input's newest sample, its seq and its conservative
//!   age `now − capture + ε_pub + ε_self`, each capture time exposed.
//! - **Fresh** means a conservative age within the entry's bound. A sample with
//!   no clock mapping, or stamped in the future beyond the error bound
//!   (`future_timestamp`), is never fresh.
//! - **Stale required input:** with `allow: hold` it is delivered and flagged;
//!   without, the start is skipped (`stale_input`).
//! - **Never received** in this epoch: the start waits (`waiting_for`). Blank
//!   inputs never wait.
//! - **Windows** are the last `history` samples by seq, oldest first. A seq
//!   gap inside the window skips starts (`window_gap`) until it is filled
//!   ([`Inputs::recover`]) or a complete new window exists. An incomplete
//!   window is never delivered: at episode start the start waits for a full
//!   one, unless `allow: warmup` pads it by repeating the first sample.
//! - **Epochs.** When the session's epoch advances (an `arm`), every state
//!   stream is cleared ([`InputCache::advance`]) and samples from an older
//!   epoch are dropped. Event streams keep their last value until a newer one
//!   replaces it.
//! - **Provenance.** Each snapshot records, per stream, the seq range used
//!   (`based_on` for the chunks computed from it).
//!
//! [`Inputs`] is the network end: one subscription per bound stream, feeding
//! the cache from the transport callback (which keeps draining while `predict`
//! runs; windows and event streams warm from the host's history store on
//! attach). [`Inputs::observe`] is a policy start: the snapshot, each sample it
//! uses decoded and each entry's plan applied, so the values are what
//! `predict` receives. [`Inputs::stats`] counts each input's samples received
//! and lost, and a snapshot warns, at most once per input every
//! [`LOSS_WARNING_GAP`], when an input that takes every sample lost some.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::schedule::InputStats;
use crate::spec::Binding;
use crate::spec::apply::{ApplyError, Value};
use crate::wire::{ChannelId, ContractMode, Epoch, HistoryQuery, StreamSpec};
use bytes::Bytes;
use tracing::warn;

use crate::error::SessionError;
use crate::media::decode_payload;
use crate::participant::{AuthorityView, Participant};
use crate::sample::{Received, SampleSubscriber, SubscribeOptions};

pub use rlmesh_session_core::cache::{Cached, InputCache, InputSnapshot, Skip, Snapshot};

/// The least time between two warnings of one input's lost samples.
pub const LOSS_WARNING_GAP: Duration = Duration::from_secs(5);

#[derive(Clone, Copy, Debug, Default)]
struct LossWarning {
    warned: u64,
    at: Option<i64>,
}

impl LossWarning {
    fn due(&mut self, lost: u64, now_ns: i64) -> Option<u64> {
        let gap = i64::try_from(LOSS_WARNING_GAP.as_nanos()).unwrap_or(i64::MAX);
        if lost <= self.warned || self.at.is_some_and(|at| now_ns.saturating_sub(at) < gap) {
            return None;
        }
        let since = lost - self.warned;
        self.warned = lost;
        self.at = Some(now_ns);
        Some(since)
    }
}

/// A policy start: each input as `predict` receives it, and the snapshot it
/// came from (ages, staleness, padding, provenance: `obs.meta`).
#[derive(Clone, Debug, PartialEq)]
pub struct Observation {
    /// The values, by entry, in policy order.
    pub values: Vec<(String, Value)>,
    /// The snapshot.
    pub meta: Snapshot,
}

impl Observation {
    /// The value of `entry`.
    pub fn value(&self, entry: &str) -> Option<&Value> {
        self.values.iter().find(|(e, _)| e == entry).map(|(_, v)| v)
    }
}

/// Why a start produced no observation.
#[derive(Debug, thiserror::Error)]
pub enum ObserveError {
    /// The start is skipped.
    #[error("start skipped: {0:?}")]
    Skipped(Skip),
    /// A sample did not decode (`decode_mismatch`).
    #[error("{entry}: {source}")]
    Decode {
        /// The entry.
        entry: String,
        /// Why.
        source: SessionError,
    },
    /// A sample does not fit its plan.
    #[error("{entry}: {source}")]
    Apply {
        /// The entry.
        entry: String,
        /// Why.
        source: ApplyError,
    },
}

/// A binding's inputs, fed from the network: the cache plus one
/// subscription per bound stream.
pub struct Inputs {
    binding: Binding,
    cache: Arc<Mutex<InputCache>>,
    view: AuthorityView,
    me: crate::wire::ParticipantId,
    streams: Vec<SampleSubscriber>,
    losses: Mutex<Vec<Option<LossWarning>>>,
    decoded: Mutex<Vec<Option<(Decoded, Value)>>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Decoded {
    epoch: Epoch,
    seqs: Vec<u64>,
    padded: u32,
}

impl Inputs {
    /// Subscribes to every stream `binding` reads, feeding a fresh cache.
    pub async fn open(participant: &Participant, binding: &Binding) -> Result<Self, SessionError> {
        let cache = Arc::new(Mutex::new(InputCache::new(binding)));
        let view = participant.authority_view();
        let mut streams = Vec::new();
        let bound: Vec<(ChannelId, usize)> = cache
            .lock()
            .map(|c| c.channels().map(|ch| (ch, c.depth(ch))).collect())
            .unwrap_or_default();
        for (channel, depth) in bound {
            let stream: StreamSpec = view
                .with(|v| v.and_then(|v| v.stream(channel)))
                .ok_or_else(|| SessionError::UnknownPart(channel.to_string()))?;
            let feed = {
                let (cache, view, owner) = (cache.clone(), view.clone(), stream.owner.clone());
                move |received: &Received| {
                    let sample = &received.sample;
                    let mapping = view.with(|v| {
                        v.and_then(|v| match v.clock_mode {
                            crate::wire::ClockMode::Wall => {
                                v.clock(&owner).map(|c| (c.offset_ns, c.error_ns))
                            }
                            _ => Some((0, 0)),
                        })
                    });
                    let cached = Cached {
                        seq: sample.seq,
                        capture_ns: mapping
                            .map(|(offset, _)| sample.capture_ns.saturating_add(offset)),
                        error_ns: mapping.map_or(0, |(_, e)| e),
                        payload: sample.payload.clone(),
                        applied: sample.applied,
                    };
                    if let Ok(mut cache) = cache.lock() {
                        cache.push(sample.channel, sample.epoch, cached);
                    }
                }
            };
            // An event stream warms too: it is published once per episode, so a
            // controller attaching later would otherwise wait for it forever.
            let options = SubscribeOptions {
                on_sample: Some(Arc::new(feed)),
                window: depth,
                warm_from_history: depth > 1 || stream.event,
            };
            streams.push(participant.subscribe_stream(stream, options).await?);
        }
        let losses = streams
            .iter()
            .map(|s| {
                let channel = s.spec().channel;
                binding
                    .contracts
                    .iter()
                    .any(|c| {
                        c.channel == channel
                            && matches!(c.mode, ContractMode::Every | ContractMode::History)
                    })
                    .then(LossWarning::default)
            })
            .collect();
        Ok(Self {
            decoded: Mutex::new(vec![None; binding.inputs.len()]),
            losses: Mutex::new(losses),
            binding: binding.clone(),
            cache,
            view,
            me: participant.id().clone(),
            streams,
        })
    }

    /// A policy start now: the snapshot, decoded and through each entry's
    /// plan.
    pub fn observe(&self, now_session_ns: i64) -> Result<Observation, ObserveError> {
        let meta = self
            .snapshot(now_session_ns)
            .map_err(ObserveError::Skipped)?;
        let epoch = self.view.epoch();
        let mut memo = self
            .decoded
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut values = Vec::with_capacity(meta.inputs.len());
        for (i, (row, input)) in self.binding.inputs.iter().zip(&meta.inputs).enumerate() {
            let stream = row
                .channel
                .and_then(|c| self.streams.iter().find(|s| s.spec().channel == c))
                .map(SampleSubscriber::spec);
            let encoded = stream.filter(|spec| spec.encoding != crate::wire::Encoding::Raw);
            let key = (encoded.is_some() && !input.samples.is_empty()).then(|| Decoded {
                epoch,
                seqs: input.samples.iter().map(|s| s.seq).collect(),
                padded: input.padded,
            });
            if let (Some(key), Some(Some((seen, value)))) = (&key, memo.get(i))
                && key == seen
            {
                values.push((row.entry.clone(), value.clone()));
                continue;
            }
            let decoded: Vec<Bytes> = match encoded {
                Some(spec) => input
                    .samples
                    .iter()
                    .map(|s| decode_payload(spec, &s.payload).map(Bytes::from))
                    .collect::<Result<_, _>>()
                    .map_err(|source| ObserveError::Decode {
                        entry: row.entry.clone(),
                        source,
                    })?,
                None => input.samples.iter().map(|s| s.payload.clone()).collect(),
            };
            let refs: Vec<&[u8]> = decoded.iter().map(|b| b.as_ref()).collect();
            let value = row.apply(&refs).map_err(|source| ObserveError::Apply {
                entry: row.entry.clone(),
                source,
            })?;
            if let (Some(key), Some(slot)) = (key, memo.get_mut(i)) {
                *slot = Some((key, value.clone()));
            }
            values.push((row.entry.clone(), value));
        }
        drop(memo);
        Ok(Observation { values, meta })
    }

    /// Each input bound to a part, in policy order: the samples its
    /// subscription received and lost.
    pub fn stats(&self) -> Vec<InputStats> {
        self.binding
            .inputs
            .iter()
            .filter_map(|row| {
                let stream = self
                    .streams
                    .iter()
                    .find(|s| Some(s.spec().channel) == row.channel)?;
                let stats = stream.stats();
                Some(InputStats {
                    entry: row.entry.clone(),
                    part: stream.spec().qualified(),
                    received: stats.received,
                    lost: stats.lost,
                })
            })
            .collect()
    }

    fn warn_losses(&self, now_session_ns: i64) {
        let Ok(mut losses) = self.losses.lock() else {
            return;
        };
        for (stream, loss) in self.streams.iter().zip(losses.iter_mut()) {
            let Some(loss) = loss else {
                continue;
            };
            if let Some(lost) = loss.due(stream.stats().lost, now_session_ns) {
                let part = stream.spec().qualified();
                warn!(
                    policy = %self.me,
                    %part,
                    lost,
                    "{part}: {lost} samples never arrived (seq gaps); \
                     a congested link drops best-effort samples"
                );
            }
        }
    }

    /// The snapshot at a policy start now (session time, this controller's
    /// clock mapping).
    pub fn snapshot(&self, now_session_ns: i64) -> Result<Snapshot, Skip> {
        self.warn_losses(now_session_ns);
        let self_error = self
            .view
            .with(|v| v.and_then(|v| v.clock(&self.me)).map_or(0, |c| c.error_ns));
        let epoch = self.view.epoch();
        match self.cache.lock() {
            Ok(mut cache) => {
                cache.advance(epoch);
                cache.snapshot(now_session_ns, self_error)
            }
            Err(_) => Err(Skip::default()),
        }
    }

    /// Fills each window's first gap from the host's history store (attach, swap and
    /// recovery only). Returns how many samples arrived.
    pub async fn recover(&self, participant: &Participant) -> Result<usize, SessionError> {
        let gaps: Vec<(StreamSpec, Epoch, u64, u64)> = {
            let Ok(cache) = self.cache.lock() else {
                return Ok(0);
            };
            self.streams
                .iter()
                .filter_map(|s| {
                    cache
                        .gap(s.spec().channel)
                        .map(|(epoch, after, until)| (s.spec().clone(), epoch, after, until))
                })
                .collect()
        };
        let mut filled = 0;
        for (stream, epoch, after_seq, until_seq) in gaps {
            let pages = participant
                .query_history(
                    &stream,
                    HistoryQuery {
                        epoch,
                        after_seq,
                        until_seq: Some(until_seq),
                        limit: None,
                        request_id: 0,
                    },
                )
                .await?;
            let mapping = self.view.with(|v| {
                v.and_then(|v| v.clock(&stream.owner))
                    .map(|c| (c.offset_ns, c.error_ns))
            });
            let stride = stream.layout.stride().filter(|_| stream.batchable());
            let Ok(mut cache) = self.cache.lock() else {
                break;
            };
            for page in pages.iter().filter(|p| p.epoch == epoch) {
                let Ok(samples) = page.split(stride) else {
                    continue;
                };
                for sample in samples {
                    filled += 1;
                    cache.push(
                        stream.channel,
                        sample.epoch,
                        Cached {
                            seq: sample.seq,
                            capture_ns: mapping
                                .map(|(offset, _)| sample.capture_ns.saturating_add(offset)),
                            error_ns: mapping.map_or(0, |(_, e)| e),
                            payload: sample.payload,
                            applied: sample.applied,
                        },
                    );
                }
            }
        }
        Ok(filled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: i64 = 1_000_000_000;

    #[test]
    fn losses_are_warned_of_at_most_once_per_gap_with_what_was_lost_since() {
        let mut loss = LossWarning::default();
        assert_eq!(loss.due(0, 0), None, "nothing lost");
        assert_eq!(loss.due(3, S), Some(3), "the first loss at once");
        assert_eq!(loss.due(5, 2 * S), None, "within the gap");
        assert_eq!(loss.due(9, 5 * S), None, "still within it");
        assert_eq!(loss.due(9, 6 * S), Some(6), "what was lost since, once due");
        assert_eq!(loss.due(9, 20 * S), None, "nothing new");
        assert_eq!(
            loss.due(10, 20 * S),
            Some(1),
            "a new loss after a quiet spell"
        );
    }
}
