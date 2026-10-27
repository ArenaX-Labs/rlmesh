//! Samples, batches and history queries (§9.4, §10.6, §8.8).

#[allow(unused_imports)]
use crate::prelude::*;

use core::fmt;

use bytes::Bytes;

use crate::wire::ids::{ChannelId, Epoch, LeaseToken};

/// What the edge did on one control tick, per command group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EdgeMode {
    /// No lease, or nothing applied under it yet.
    Idle,
    /// Dispatching the action due.
    Driving,
    /// Waiting (at most `hold_max`) for a valid action.
    Hold,
    /// Running the declared safe behaviour.
    Safe,
    /// The session is in ESTOP.
    Estop,
}

impl EdgeMode {
    /// Every mode, in stats order.
    pub const ALL: [Self; 5] = [
        Self::Idle,
        Self::Driving,
        Self::Hold,
        Self::Safe,
        Self::Estop,
    ];
}

impl fmt::Display for EdgeMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Idle => "idle",
            Self::Driving => "driving",
            Self::Hold => "hold",
            Self::Safe => "safe",
            Self::Estop => "estop",
        })
    }
}

/// The envelope metadata of an applied sample (§9.4): what the edge
/// dispatched on one tick, after clamping, and on whose authority. It is the
/// dispatched command, not confirmed physical motion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Applied {
    /// The edge mode on that tick.
    pub mode: EdgeMode,
    /// Whether a value was dispatched (Driving, or a position-type Hold).
    /// Without one the sample has no payload.
    pub present: bool,
    /// The lease token of the dispatched action's chunk.
    pub lease_token: LeaseToken,
    /// That chunk's seq.
    pub chunk_seq: u64,
    /// The action's index in it.
    pub action_index: u32,
}

impl Applied {
    /// A tick that dispatched nothing.
    pub fn absent(mode: EdgeMode) -> Self {
        Self {
            mode,
            present: false,
            lease_token: LeaseToken(0),
            chunk_seq: 0,
            action_index: 0,
        }
    }
}

/// One sample of one stream.
///
/// Missing samples are explicit: a gap in `seq` is a loss, and a consumer
/// judges age from `capture_ns`, so it never silently freezes on an old
/// value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sample {
    /// The stream.
    pub channel: ChannelId,
    /// The publisher's view of the epoch.
    pub epoch: Epoch,
    /// Per stream, from 1; a gap is a loss.
    pub seq: u64,
    /// Publisher clock (wall mode) or session time at capture.
    pub capture_ns: i64,
    /// Publisher clock at publish.
    pub publish_ns: i64,
    /// The virtual or lockstep tick; 0 in wall mode.
    pub tick: u64,
    /// The packed value (or one encoded image, or UTF-8). Decoded samples
    /// slice the received buffer rather than copy it.
    pub payload: Bytes,
    /// Applied streams only.
    pub applied: Option<Applied>,
}

/// One sample of a batch, without its value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleMeta {
    /// Its seq.
    pub seq: u64,
    /// Its capture time.
    pub capture_ns: i64,
    /// Its tick.
    pub tick: u64,
    /// Applied streams only.
    pub applied: Option<Applied>,
}

/// An inclusive run of seqs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SeqRange {
    /// The first seq.
    pub first: u64,
    /// The last seq.
    pub last: u64,
}

/// Several samples of one stream in one epoch: a history-store page. The
/// payload holds `samples.len()` values at the
/// stream's stride (one value of an unbatchable stream).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SampleBatch {
    /// The stream.
    pub channel: ChannelId,
    /// The epoch every sample belongs to; batches never span epochs.
    pub epoch: Epoch,
    /// Sender clock at publish.
    pub publish_ns: i64,
    /// One per sample, in seq order.
    pub samples: Vec<SampleMeta>,
    /// The values.
    pub payload: Bytes,
    /// History replies: retention ran out before the requested range.
    pub truncated: bool,
    /// History replies: seqs known missing.
    pub gaps: Vec<SeqRange>,
}

impl SampleBatch {
    /// Splits the batch into samples, slicing the payload (no copy). `stride`
    /// is the stream's stride; `None` for an unbatchable stream, whose batch
    /// holds at most one sample. Fails when the payload does not match.
    pub fn split(&self, stride: Option<usize>) -> Result<Vec<Sample>, String> {
        let n = self.samples.len();
        let expected = match stride {
            Some(stride) => n.checked_mul(stride).ok_or("overflow")?,
            None if n <= 1 => self.payload.len(),
            None => return Err(format!("{n} samples of an unbatchable stream")),
        };
        if self.payload.len() != expected {
            return Err(format!(
                "{} payload bytes for {n} samples, expected {expected}",
                self.payload.len()
            ));
        }
        let range = |i: usize| match stride {
            Some(stride) => i * stride..(i + 1) * stride,
            None => 0..self.payload.len(),
        };
        Ok(self
            .samples
            .iter()
            .enumerate()
            .map(|(i, meta)| Sample {
                channel: self.channel,
                epoch: self.epoch,
                seq: meta.seq,
                capture_ns: meta.capture_ns,
                publish_ns: self.publish_ns,
                tick: meta.tick,
                payload: self.payload.slice(range(i)),
                applied: meta.applied,
            })
            .collect())
    }
}

/// A query of the host's history store (§8.8): samples with
/// `after_seq < seq <= until_seq` in `epoch`, at most `limit` of them (the
/// newest of the range).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HistoryQuery {
    /// The epoch the caller is in; another epoch's samples are never served.
    pub epoch: Epoch,
    /// Exclusive lower bound.
    pub after_seq: u64,
    /// Inclusive upper bound (a watermark from the live stream); `None`:
    /// unbounded.
    pub until_seq: Option<u64>,
    /// At most this many samples; `None` takes the store's cap.
    pub limit: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_batch_splits_into_samples_at_the_stride() {
        let meta = |seq: u64| SampleMeta {
            seq,
            capture_ns: i64::try_from(seq).expect("seq") * 10,
            tick: 0,
            applied: None,
        };
        let batch = SampleBatch {
            channel: ChannelId(4),
            epoch: Epoch(2),
            publish_ns: 99,
            samples: vec![meta(3), meta(5)],
            payload: Bytes::from((0..16_u8).collect::<Vec<_>>()),
            truncated: false,
            gaps: vec![SeqRange { first: 4, last: 4 }],
        };
        let samples = batch.split(Some(8)).expect("split");
        assert_eq!(samples[1].seq, 5);
        assert_eq!(&samples[1].payload[..], &[8, 9, 10, 11, 12, 13, 14, 15]);
        assert!(batch.split(Some(16)).is_err());
        assert!(batch.split(None).is_err(), "unbatchable streams hold one");
        let one = SampleBatch {
            samples: vec![meta(7)],
            ..batch
        };
        assert_eq!(one.split(None).expect("one").len(), 1);
    }
}
