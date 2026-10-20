//! The RLMesh real-time session wire (protocol generation 2).
//!
//! This crate owns what two session peers must agree on byte for byte, and
//! nothing about how they talk:
//!
//! - [`ids`]: session, tenant, participant and channel ids, epochs and lease
//!   tokens;
//! - [`keys`]: the Zenoh key layout of one session;
//! - [`version`]: the protocol generation, minor, frame version and the
//!   compatibility rule;
//! - the message types: authority state and control verbs ([`authority`]),
//!   heartbeats and clock pings ([`heartbeat`]), channel declarations
//!   ([`channel`]), samples ([`sample`]) and timed action chunks
//!   ([`action`]);
//! - [`codec`]: the one module that knows the byte encoding.
//!
//! Every type outside [`codec`] is plain Rust: no generated code, no transport
//! types. The schema can therefore be revised (the v0.2 wire spec is still
//! being written) by changing `proto/` and `codec.rs` alone.
//!
//! # Encoding
//!
//! **Protobuf for every message, with tensors as raw `bytes` leaves** in a
//! layout declared once per channel ([`TensorLayout`]):
//!
//! - Protobuf is the one schema language every binding RLMesh targets
//!   (Python, C#, C++, Go on the platform) generates code for, the workspace
//!   already builds it, and its field rules (unknown fields skipped, numbers
//!   never reused) are exactly the additive evolution the protocol needs.
//! - Tensors are not serialized: a leaf is the tensor's little-endian
//!   row-major bytes. The encoder emits each leaf as its own [`bytes::Bytes`]
//!   segment between small field headers ([`codec::Encoded`]), which Zenoh
//!   gathers without a copy; the decoder takes the received buffer as
//!   [`bytes::Bytes`] and the leaves slice it. A 900 kB camera frame is never
//!   copied by this crate, and the concatenated segments are ordinary
//!   protobuf that any implementation reads.
//! - The cost of protobuf metadata (a varint parse for tens of bytes) is not
//!   measurable next to the transport: the spike's protobuf samples held a
//!   1 kHz loop with 0.07 / 0.18 ms (p50 / p99) transport latency on
//!   loopback.
//!
//! Rejected: a fixed binary data frame (the v0.2 spec draft's §10.5): O(1)
//! field offsets and alignment for shared memory, but a second codec with its
//! own grammar to specify (batch strides, variable-length fields, header
//! evolution) and no measured bottleneck to justify it yet; FlatBuffers or
//! Cap'n Proto (zero-copy for metadata that does not need it, and a heavier
//! codegen story per binding); serde formats such as bincode or postcard (a
//! Rust-only schema with no cross-language evolution rules). The choice is
//! confined to [`codec`] so it can still change.
//!
//! # Versioning
//!
//! See [`version`]: the generation is in the key prefix (`rlmesh/v2/...`) and
//! the protobuf package (`rlmesh.session.v2`), so generations never meet; any
//! protocol minor talks to any other, and features added in a minor are
//! capability strings checked at attach.

pub mod action;
pub mod authority;
pub mod channel;
pub mod codec;
pub mod error;
pub mod heartbeat;
pub mod ids;
pub mod keys;
pub mod sample;
pub mod version;

pub use action::{Action, ActionChunk, ObservationRef};
pub use authority::{
    AuthoritySnapshot, ClockMapping, Command, CtlError, CtlReply, CtlStatus, FenceError, Lease,
    Mode, ParticipantInfo, PriorityClass, Request, Role,
};
pub use channel::{
    Actuation, ChannelPriority, ChannelQos, ChannelSpec, Congestion, DType, Delivery, Direction,
    Encoding, History, MergePolicy, Rate, TensorLayout,
};
pub use codec::Encoded;
pub use error::WireError;
pub use heartbeat::{Health, Heartbeat, HeartbeatEcho, TimingSample};
pub use ids::{ChannelId, Epoch, IdError, LeaseToken, ParticipantId, SessionId, TenantId};
pub use keys::SessionKeys;
pub use sample::Sample;
