//! Samples: one value of one channel, stamped by its publisher.

use bytes::Bytes;

use crate::ids::{ChannelId, Epoch};

/// One sample of one channel.
///
/// Missing samples are explicit: a gap in `seq` is a loss, and a consumer
/// judges age from `capture_ns`, so it never silently freezes on an old
/// value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sample {
    /// The channel.
    pub channel: ChannelId,
    /// Per channel per publisher, from 1; a gap is a loss.
    pub seq: u64,
    /// The publisher's view of the epoch.
    pub epoch: Epoch,
    /// Publisher's monotonic clock at capture.
    pub capture_ns: i64,
    /// Publisher's monotonic clock at publish.
    pub publish_ns: i64,
    /// The virtual or lockstep tick; 0 in wall mode.
    pub tick: u64,
    /// The value, one buffer per leaf of the channel's layout (one JPEG for
    /// a JPEG channel). Decoded frames slice the received buffer, so these
    /// share it rather than copy it.
    pub leaves: Vec<Bytes>,
}

impl Sample {
    /// Total payload bytes.
    pub fn payload_len(&self) -> usize {
        self.leaves.iter().map(Bytes::len).sum()
    }
}
