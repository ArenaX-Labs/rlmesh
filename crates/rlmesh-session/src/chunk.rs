//! Sending action chunks: the controller side of an actuation channel.
//!
//! A controller holding a lease sends timed chunks through a
//! [`ChunkPublisher`], which stamps its lease token, the epoch and a seq
//! (increasing per token), and checks every action against the channel's
//! layout. The embodiment's edge (`rlmesh-edge`) fences, splices and applies
//! them.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use rlmesh_wire::codec;
use rlmesh_wire::{
    Action, ActionChunk, ChannelSpec, Epoch, Lease, LeaseToken, ObservationRef, ParticipantId,
};
use zenoh::Wait;

use crate::clock::Clock;
use crate::error::SessionError;
use crate::transport::{Transport, frame_zbytes, zenoh_congestion, zenoh_priority};

/// What a controller fills in; the [`ChunkPublisher`] adds the rest. Times
/// are session time.
#[derive(Clone, Debug, Default)]
pub struct ChunkDraft {
    /// The observations it was computed from.
    pub based_on: Vec<ObservationRef>,
    /// Session time of the first action.
    pub t0_ns: i64,
    /// The action period; zero takes the channel's declared period.
    pub dt_ns: u64,
    /// Keep committed actions before this; replace from here.
    pub replace_from_ns: i64,
    /// The whole chunk is void at and after this.
    pub valid_until_ns: i64,
    /// The actions, one buffer per leaf of the channel's layout.
    pub actions: Vec<Action>,
}

/// Sends chunks for one actuation channel under one lease.
pub struct ChunkPublisher {
    publisher: zenoh::pubsub::Publisher<'static>,
    spec: ChannelSpec,
    source: ParticipantId,
    token: LeaseToken,
    epoch: Epoch,
    seq: AtomicU64,
    clock: Arc<dyn Clock>,
}

impl ChunkPublisher {
    pub(crate) async fn declare(
        transport: &Transport,
        key: String,
        spec: ChannelSpec,
        source: ParticipantId,
        lease: &Lease,
        epoch: Epoch,
        clock: Arc<dyn Clock>,
    ) -> Result<Self, SessionError> {
        let qos = spec.qos;
        let publisher = transport
            .for_priority(qos.priority)
            .declare_publisher(key)
            .priority(zenoh_priority(qos.priority))
            .congestion_control(zenoh_congestion(&qos))
            .express(qos.express())
            .await?;
        Ok(Self {
            publisher,
            spec,
            source,
            token: lease.token,
            epoch,
            seq: AtomicU64::new(0),
            clock,
        })
    }

    /// The lease token every chunk carries.
    pub fn token(&self) -> LeaseToken {
        self.token
    }

    /// The channel.
    pub fn spec(&self) -> &ChannelSpec {
        &self.spec
    }

    /// Checks every action against the channel's layout, then stamps and
    /// sends the chunk; returns its seq.
    pub fn send(&self, draft: ChunkDraft) -> Result<u64, SessionError> {
        for (i, action) in draft.actions.iter().enumerate() {
            self.spec
                .check_leaf_lens(action.iter().map(Bytes::len))
                .map_err(|reason| SessionError::InvalidValue {
                    channel: self.spec.id,
                    reason: format!("action {i}: {reason}"),
                })?;
        }
        let seq = self.seq.fetch_add(1, Ordering::Relaxed) + 1;
        let chunk = ActionChunk {
            channel: self.spec.id,
            source: self.source.clone(),
            lease_token: self.token,
            epoch: self.epoch,
            seq,
            publish_ns: self.clock.now_ns(),
            tick: 0,
            based_on: draft.based_on,
            t0_ns: draft.t0_ns,
            dt_ns: draft.dt_ns,
            replace_from_ns: draft.replace_from_ns,
            valid_until_ns: draft.valid_until_ns,
            actions: draft.actions,
        };
        let encoded = codec::encode_chunk(&chunk).map_err(|err| SessionError::InvalidValue {
            channel: self.spec.id,
            reason: err.to_string(),
        })?;
        self.publisher.put(frame_zbytes(encoded)).wait()?;
        Ok(seq)
    }
}
