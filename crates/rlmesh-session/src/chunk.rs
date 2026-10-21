//! Sending action chunks: the controller side of a commandable part.
//!
//! A controller holding a lease sends timed chunks through a
//! [`ChunkPublisher`], which stamps its lease token, the lease's binding,
//! the epoch, its own id as the source and a seq (increasing per token), and
//! checks the actions against the part's command layout (§10.6). The
//! embodiment's edge (`rlmesh-edge`) fences, validates, splices and applies
//! them.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::wire::codec;
use crate::wire::part::MAX_CHUNK_ACTIONS;
use crate::wire::{
    ActionChunk, ClockMode, Epoch, Fingerprint, InputRange, Lease, LeaseToken, PartSpec,
    ParticipantId,
};
use bytes::Bytes;
use zenoh::Wait;

use crate::clock::Clock;
use crate::error::SessionError;
use crate::transport::{TrafficClass, Transport, declare_publisher, frame_zbytes, stream_qos};

/// What a controller fills in; the [`ChunkPublisher`] adds the rest. Times
/// are session time.
#[derive(Clone, Debug, Default)]
pub struct ChunkDraft {
    /// The inputs it was computed from (`based_on`).
    pub based_on: Vec<InputRange>,
    /// Session time of the first action's slot.
    pub t0_ns: i64,
    /// The action period; zero takes the action grid of the lease's
    /// binding (else the part's control period).
    pub dt_ns: u64,
    /// Keep committed actions before this; replace from here.
    pub replace_from_ns: i64,
    /// The whole chunk is void at and after this.
    pub valid_until_ns: i64,
    /// The actions, packed back to back at the command stride.
    pub actions: Bytes,
    /// The virtual or lockstep tick it was decided on; 0 in wall mode.
    pub tick: u64,
}

/// Sends chunks for one commandable part under one lease.
pub struct ChunkPublisher {
    publisher: zenoh::pubsub::Publisher<'static>,
    spec: PartSpec,
    stride: usize,
    source: ParticipantId,
    token: LeaseToken,
    binding: Fingerprint,
    epoch: Epoch,
    seq: AtomicU64,
    clock: Arc<dyn Clock>,
}

impl ChunkPublisher {
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn declare(
        transport: &Transport,
        key: String,
        spec: PartSpec,
        source: ParticipantId,
        lease: &Lease,
        epoch: Epoch,
        clock_mode: ClockMode,
        clock: Arc<dyn Clock>,
    ) -> Result<Self, SessionError> {
        let wrong = |reason: String| SessionError::WrongChannel {
            channel: spec.channel,
            reason,
        };
        let stride = spec
            .command
            .as_ref()
            .ok_or_else(|| wrong("not a commandable part".into()))?
            .layout()
            .map_err(wrong)?
            .stride()
            .ok_or_else(|| wrong("a text command".into()))?;
        let stream = spec.stream().map_err(wrong)?;
        let publisher = declare_publisher(
            transport,
            TrafficClass::of(spec.priority),
            key,
            stream_qos(&stream, clock_mode),
        )
        .await?;
        Ok(Self {
            publisher,
            spec,
            stride,
            source,
            token: lease.token,
            binding: lease.binding,
            epoch,
            seq: AtomicU64::new(0),
            clock,
        })
    }

    /// The lease token every chunk carries.
    pub fn token(&self) -> LeaseToken {
        self.token
    }

    /// The commandable part.
    pub fn spec(&self) -> &PartSpec {
        &self.spec
    }

    /// One action's packed size.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Checks the actions against the command layout, then stamps and sends
    /// the chunk; returns its seq.
    pub fn send(&self, draft: ChunkDraft) -> Result<u64, SessionError> {
        let invalid = |reason: String| SessionError::InvalidValue {
            channel: self.spec.channel,
            reason,
        };
        let len = draft.actions.len();
        if len == 0 || !len.is_multiple_of(self.stride) {
            return Err(invalid(format!(
                "{len} action bytes is not a whole number of {}-byte actions",
                self.stride
            )));
        }
        let count = len / self.stride;
        if count > MAX_CHUNK_ACTIONS {
            return Err(invalid(format!(
                "{count} actions (at most {MAX_CHUNK_ACTIONS})"
            )));
        }
        let seq = self.seq.fetch_add(1, Ordering::Relaxed) + 1;
        let chunk = ActionChunk {
            channel: self.spec.channel,
            epoch: self.epoch,
            lease_token: self.token,
            seq,
            source: self.source.clone(),
            binding: self.binding,
            based_on: draft.based_on,
            t0_ns: draft.t0_ns,
            dt_ns: draft.dt_ns,
            replace_from_ns: draft.replace_from_ns,
            valid_until_ns: draft.valid_until_ns,
            count: u32::try_from(count).unwrap_or(u32::MAX),
            actions: draft.actions,
            publish_ns: self.clock.now_ns(),
            tick: draft.tick,
        };
        let encoded = codec::encode_chunk(&chunk).map_err(|err| invalid(err.to_string()))?;
        self.publisher.put(frame_zbytes(encoded)).wait()?;
        Ok(seq)
    }
}
