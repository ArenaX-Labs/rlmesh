//! The byte encoding: protobuf envelopes with raw payloads.
//!
//! This is the only module that knows how messages look on the wire (see the
//! crate docs for why). Everything else in the crate, and every crate above
//! it, speaks the plain types this module encodes and decodes.
//!
//! Every message is one protobuf message of `rlmesh.session.v2` (see
//! `proto/rlmesh/session/v2/session.proto`) per payload; the key says which.
//!
//! - Control messages ([`Request`], [`CtlReply`], [`AuthoritySnapshot`],
//!   [`DeviceView`], [`Heartbeat`], [`HeartbeatEcho`], [`ParticipantReport`],
//!   [`VirtualTick`], [`HistoryQuery`])
//!   encode to one buffer and decode from any byte slice.
//! - [`Sample`]s, [`SampleBatch`]es and [`ActionChunk`]s carry their values
//!   as one packed `bytes` payload. [`encode_sample`], [`encode_batch`] and
//!   [`encode_chunk`] emit the message as [`Encoded`] segments: small
//!   buffers for the fields and the payload's tag and length, and the
//!   payload as its own [`Bytes`] segment, so a transport that gathers
//!   segments (Zenoh's `ZBytes`) sends it without a copy. The concatenation
//!   is ordinary protobuf. The decoders take the received buffer as
//!   [`Bytes`], and the payload slices it.
//! - The lockstep and served-model messages encode to one buffer (one copy
//!   of the slabs).
//!
//! Payload contents are not validated here; the layout check against the
//! declaration is [`StreamSpec::check_payload`] (samples) and the edge's
//! intake (chunks). Lengths are bounded here: a chunk's action count and
//! payload size, a batch's sample count.
//!
//! [`StreamSpec::check_payload`]: crate::wire::part::StreamSpec::check_payload

#[allow(unused_imports)]
use crate::prelude::*;

#[cfg(doc)]
use crate::wire::{
    action::ActionChunk,
    authority::{AuthoritySnapshot, CtlReply, Request},
    heartbeat::{Heartbeat, HeartbeatEcho},
    report::ParticipantReport,
    sample::{HistoryQuery, Sample, SampleBatch},
    tick::VirtualTick,
    view::DeviceView,
};
#[cfg(doc)]
use bytes::Bytes;

/// Generated `rlmesh.session.v2` messages: private, so the schema can move
/// without touching any public type.
#[allow(clippy::all, clippy::pedantic)]
mod pb {
    include!(concat!(env!("OUT_DIR"), "/rlmesh.session.v2.rs"));
}

/// The `rlmesh.session.v2` protobuf descriptors (a serialized
/// `google.protobuf.FileDescriptorSet`): the schema a recording names for
/// each envelope it stores.
pub const FILE_DESCRIPTOR_SET: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/session_descriptor.bin"));

/// The full protobuf names of the envelopes a session carries, by role.
pub mod schema {
    /// A data sample (`d/...` keys, one per MCAP message).
    pub const SAMPLE: &str = "rlmesh.session.v2.Sample";
    /// An action chunk (`c/...` keys).
    pub const ACTION_CHUNK: &str = "rlmesh.session.v2.ActionChunk";
    /// The authority snapshot (`host/state`).
    pub const AUTHORITY_STATE: &str = "rlmesh.session.v2.AuthorityState";
    /// One controller's binding.
    pub const BINDING_RECORD: &str = "rlmesh.session.v2.BindingRecord";
    /// A participant's heartbeat (`p/{id}/hb`).
    pub const HEARTBEAT: &str = "rlmesh.session.v2.Heartbeat";
    /// A participant's observer telemetry (`p/{id}/report`).
    pub const PARTICIPANT_REPORT: &str = "rlmesh.session.v2.ParticipantReport";
    /// The host's link and edge health (`host/health`).
    pub const HOST_HEALTH: &str = "rlmesh.session.v2.HostHealth";
}

mod bound;
mod convert;
mod ctl;
mod handshake;
mod health;
mod keys;
mod lockstep;
mod model;
mod pair;
mod part;
mod reader;
mod scene;
mod state;
mod stream;
mod tick;
mod view;
pub use ctl::{
    decode_estop_signal, decode_reply, decode_request, decode_unauthenticated, decode_view_reply,
    decode_view_reply_into, encode_estop_signal, encode_reply, encode_request, encode_view_reply,
};
pub use handshake::{
    decode_handshake_accept, decode_handshake_reply, decode_handshake_request,
    encode_handshake_accept, encode_handshake_reply, encode_handshake_request,
};
pub use health::{
    decode_echo, decode_health, decode_heartbeat, decode_report, encode_echo, encode_health,
    encode_heartbeat, encode_report,
};
pub use keys::{decode_key_delivery, encode_key_delivery};
pub use lockstep::{
    decode_lockstep_reply, decode_lockstep_request, decode_lockstep_response, decode_lockstep_step,
    encode_lockstep_reply, encode_lockstep_request, encode_lockstep_response, encode_lockstep_step,
};
pub use model::{
    decode_model_reply, decode_model_request, encode_model_reply, encode_model_request,
};
pub use pair::{
    decode_pair_commit, decode_pair_nonce, decode_pair_reply, decode_pair_request,
    decode_pair_result, decode_pair_reveal, encode_pair_commit, encode_pair_nonce,
    encode_pair_reply, encode_pair_request, encode_pair_result, encode_pair_reveal,
    encode_pair_wait,
};
pub use state::{decode_binding, decode_state, encode_binding, encode_state};
pub use stream::{
    Encoded, decode_batch, decode_chunk, decode_sample, encode_batch, encode_chunk, encode_sample,
};
pub use tick::{decode_history_query, decode_tick, encode_history_query, encode_tick};
pub use view::{
    MAX_VIEW_CLOCKS, MAX_VIEW_DEMANDS, MAX_VIEW_GRIDS, MAX_VIEW_LEASES, MAX_VIEW_MEMBERS,
    MAX_VIEW_PARTS, MAX_VIEW_TEXT, decode_view, decode_view_into, encode_view,
};

#[cfg(test)]
mod tests;
