//! The controller's input cache and history windows (§8.8), built into the
//! controller runtime so users write no cache code.
//!
//! [`InputCache`] is sans-IO (it lives in `rlmesh-session-core` and is
//! re-exported here): one slot per bound stream (shared by the
//! entries bound to it) plus a ring as deep as the deepest window on it,
//! fed sample by sample and snapshotted at each policy start:
//!
//! - **Snapshot.** Every input's newest sample, its seq and its
//!   **conservative age** `now − capture + ε_pub + ε_self` (§9.1), taken
//!   together but captured independently: the snapshot exposes each capture
//!   time and the skew between them.
//! - **Fresh** means a conservative age within the entry's bound. A sample
//!   with no clock mapping, or stamped in the future beyond the error bound
//!   (`future_timestamp`), is never fresh.
//! - **Stale required input:** with `allow: hold` it is delivered and
//!   flagged; without, the start is skipped (`stale_input`).
//! - **Never received** in this epoch: the start waits (`waiting_for`).
//!   Blank inputs never wait.
//! - **Windows** are the last `history` samples by seq, oldest first,
//!   assembled from live samples. A seq gap inside the window skips
//!   starts (`window_gap`) until the gap is filled (from the host store,
//!   [`Inputs::recover`]) or a complete new window exists. An incomplete
//!   window is never delivered: at episode start the start waits for a full
//!   one, unless `allow: warmup` pads it by repeating the first sample.
//! - **Epochs** (§11.1). When the session's epoch advances (an `arm`),
//!   every state stream is cleared ([`InputCache::advance`]) and a sample
//!   from an older epoch is dropped, so nothing seeds the new epoch from the
//!   old one. Episodic text and the other event streams keep their last
//!   value (the session has no episodes on the wire yet, so an event value
//!   lasts until a newer one replaces it).
//! - **Provenance.** Each snapshot records, per stream, the seq range used
//!   (`based_on` for the chunks computed from it).
//!
//! On an applied stream (§9.4) a tick that dispatched nothing carries no
//! value: the input keeps the previous dispatched value.
//!
//! [`Inputs`] is the network end: one subscription per bound stream,
//! feeding the cache from the transport callback (which keeps draining
//! while `predict` runs; windows and event streams warm from the host's
//! history store on attach), converting capture times to session time through
//! the publishers' clock mappings. [`Inputs::observe`] is a policy start:
//! the snapshot, each sample it uses decoded (JPEG) and each entry's plan
//! applied, so the values are what `predict` receives and the snapshot is
//! `obs.meta`.

use std::sync::{Arc, Mutex};

use crate::spec::Binding;
use crate::spec::apply::{ApplyError, Value};
use bytes::Bytes;
use rlmesh_wire::{ChannelId, Epoch, HistoryQuery, StreamSpec};

use crate::error::SessionError;
use crate::media::decode_payload;
use crate::participant::{AuthorityView, Participant};
use crate::sample::{Received, SampleSubscriber, SubscribeOptions};

pub use rlmesh_session_core::cache::{Cached, InputCache, InputSnapshot, Skip, Snapshot};

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
    /// The start is skipped (§8.8).
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
    me: rlmesh_wire::ParticipantId,
    streams: Vec<SampleSubscriber>,
    /// Per input, the last decoded value of an encoded stream and the
    /// samples it came from: a 30 fps JPEG camera read by a 100 Hz policy
    /// is decoded and converted once per frame, not once per start.
    decoded: Mutex<Vec<Option<(Decoded, Value)>>>,
}

/// Which samples a decoded value came from.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Decoded {
    epoch: Epoch,
    seqs: Vec<u64>,
    padded: u32,
}

impl Inputs {
    /// Subscribes to every stream `binding` reads, feeding a fresh cache.
    /// Windowed and event streams warm from the host's history store.
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
                            rlmesh_wire::ClockMode::Wall => {
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
            // The subscription's own window sizes the history warm-up. An
            // event stream (the task text) warms too: it is published once
            // per episode, so a controller attaching later would wait for
            // it forever.
            let options = SubscribeOptions {
                on_sample: Some(Arc::new(feed)),
                window: depth,
                warm_from_history: depth > 1 || stream.event,
            };
            streams.push(participant.subscribe_stream(stream, options).await?);
        }
        Ok(Self {
            decoded: Mutex::new(vec![None; binding.inputs.len()]),
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
            let encoded = stream.filter(|spec| spec.encoding != rlmesh_wire::Encoding::Raw);
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

    /// The snapshot at a policy start now (session time, this controller's
    /// clock mapping).
    pub fn snapshot(&self, now_session_ns: i64) -> Result<Snapshot, Skip> {
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

    /// Fills each window's first gap from the host's history store (§8.8:
    /// attach, swap and recovery only). Returns how many samples arrived.
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
