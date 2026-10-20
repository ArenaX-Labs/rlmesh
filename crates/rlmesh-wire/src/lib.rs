//! The RLMesh real-time session wire (protocol generation 2, the v0.2 spec
//! §10).
//!
//! This crate owns what two session peers must agree on byte for byte, and
//! nothing about how they talk:
//!
//! - [`ids`]: session, tenant, participant and channel ids, epochs, lease
//!   tokens and binding fingerprints;
//! - [`path`]: part paths (§2), the only address of a part;
//! - [`keys`]: the Zenoh key layout of one session (§10.3);
//! - [`version`]: the protocol generation, minor, kind table version and
//!   the compatibility rule (§10.7);
//! - the message types: authority state and control verbs ([`authority`]),
//!   heartbeats and clock pings ([`heartbeat`]), observer telemetry
//!   ([`report`]), parts, streams and payload
//!   packing ([`part`]), delivery contracts, action grids and publication
//!   plans ([`contract`]), samples, batches and history queries ([`sample`]),
//!   timed action chunks ([`action`]), the virtual-mode tick protocol
//!   ([`tick`]) and the lockstep fast path ([`lockstep`]);
//! - [`codec`]: the one module that knows the byte encoding.
//!
//! Every type outside [`codec`] is plain Rust: no generated code, no
//! transport types.
//!
//! # Encoding (decided Q8)
//!
//! **Protobuf envelopes with raw payloads**, for control and data:
//!
//! - Protobuf is the one schema language every binding RLMesh targets
//!   (Python, C#, C++, Go on the platform) generates code for, and its field
//!   rules (unknown fields skipped, numbers never reused) are exactly the
//!   additive evolution the protocol needs.
//! - Values are not serialized field by field: a sample's payload is its
//!   part's fields packed raw, typed once by the declaration ([`part`],
//!   §10.6). The encoder emits the payload as its own [`bytes::Bytes`]
//!   segment between small field headers ([`codec::Encoded`]), which Zenoh
//!   gathers without a copy; the decoder takes the received buffer as
//!   [`bytes::Bytes`] and the payload slices it. A 900 kB camera frame is
//!   never copied by this crate, and the concatenated segments are ordinary
//!   protobuf that any implementation reads.
//! - A fixed binary frame is the spec's Appendix A: future work, adopted
//!   only if a measurement shows protobuf prevents a latency or copy target.
//!
//! # Versioning
//!
//! See [`version`]: the generation is in the key prefix (`rlmesh/v2/...`) and
//! the protobuf package (`rlmesh.session.v2`), so generations never meet; any
//! protocol minor talks to any other, and features added in a minor are
//! feature strings checked at attach.

pub mod action;
pub mod authority;
pub mod codec;
pub mod contract;
pub mod error;
pub mod heartbeat;
pub mod ids;
pub mod keys;
pub mod lockstep;
pub mod part;
pub mod path;
pub mod report;
pub mod sample;
pub mod tick;
pub mod version;

pub use action::{ActionChunk, ChunkRef, InputRange};
pub use authority::{
    AuthoritySnapshot, BindingRecord, ClockMapping, ClockMode, Command, CtlError, CtlReply,
    CtlStatus, FenceError, Lease, Mode, ParticipantInfo, PriorityClass, Request, Role, Watermark,
};
pub use codec::Encoded;
pub use contract::{ActionGrid, ContractMode, Demand, Plan, StreamContract};
pub use error::WireError;
pub use heartbeat::{Health, Heartbeat, HeartbeatEcho, TimingSample};
pub use ids::{
    ChannelId, Epoch, Fingerprint, IdError, LeaseToken, ParticipantId, SessionId, TenantId,
};
pub use keys::SessionKeys;
pub use lockstep::{LockstepReply, LockstepReset, LockstepStep, PartValues};
pub use part::{
    ChannelPriority, CommandSpec, DType, Encoding, FieldLimit, FieldSpec, Layout, MergePolicy,
    PartSpec, SendPolicy, StreamSpec,
};
pub use path::PartPath;
pub use report::{ParticipantReport, PublisherCounters, RunCounters};
pub use sample::{Applied, EdgeMode, HistoryQuery, Sample, SampleBatch, SampleMeta, SeqRange};
pub use tick::{Step, TickState, VirtualTick};
