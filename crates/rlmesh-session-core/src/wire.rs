//! The RLMesh real-time session wire (protocol generation 2).
//!
//! What two session peers must agree on byte for byte, and nothing about how they talk:
//!
//! - [`ids`], [`path`] and [`keys`]: ids, epochs and lease tokens; part paths, the only
//!   address of a part; the Zenoh key layout of one session.
//! - [`version`]: the protocol generation, minor, kind table version and the compatibility
//!   rule.
//! - The messages: authority state and control verbs ([`authority`]), heartbeats and clock
//!   pings ([`heartbeat`]), observer telemetry ([`report`]), parts and payload packing
//!   ([`part`]), delivery contracts ([`contract`]), samples and history queries ([`sample`]),
//!   timed action chunks ([`action`]), virtual-mode ticks ([`tick`]), the lockstep fast path
//!   ([`lockstep`]), served models ([`model`]), endpoint timing ([`timing`]), machine pairing
//!   ([`pair`]), the lobby and scenes ([`scene`]) and one robot's device view ([`view`]).
//! - [`codec`]: the only module that knows the byte encoding.
//!
//! # Encoding
//!
//! Protobuf envelopes with raw payloads. A sample's payload is its part's fields packed raw,
//! typed once by the declaration ([`part`]); the encoder emits it as its own
//! [`bytes::Bytes`] segment between small field headers ([`codec::Encoded`]), and the decoder
//! slices the received buffer, so a payload is never copied and the concatenated segments are
//! ordinary protobuf.
//!
//! # Versioning
//!
//! See [`version`]: the generation is in the key prefix (`rlmesh/v2/...`) and the protobuf
//! package (`rlmesh.session.v2`), so generations never meet; any protocol minor talks to any
//! other, and features added in a minor are feature strings checked at attach.

#[allow(unused_imports)]
use crate::prelude::*;

pub mod action;
pub mod authority;
pub mod codec;
pub mod contract;
pub mod datagram;
pub mod error;
pub mod health;
pub mod heartbeat;
pub mod ids;
pub mod key;
pub mod keys;
pub mod lockstep;
pub mod model;
pub mod pair;
pub mod part;
pub mod path;
pub mod report;
pub mod sample;
pub mod scene;
pub mod tick;
pub mod timing;
pub mod version;
pub mod view;

pub use action::{ActionChunk, ChunkRef, InputRange};
pub use authority::{
    AuthoritySnapshot, BindingRecord, ClockMapping, ClockMode, Command, CtlError, CtlReply,
    CtlStatus, EstopSignal, FenceError, HandshakeAccept, HandshakeRequest, HeldRobot, LastStop,
    Lease, LeaseEnd, Mode, ParticipantInfo, PriorityClass, Reason, ReasonCode, Request, RobotClock,
    Role, Watermark,
};
pub use codec::Encoded;
pub use contract::{ActionGrid, ContractMode, Demand, Plan, StreamContract};
pub use datagram::{DatagramOffer, DatagramRoute, DatagramRoutes};
pub use error::{Unread, WireError};
pub use health::{DataPath, HostHealth, LinkHealth};
pub use heartbeat::{
    ChunkTiming, ClockRefusal, DatagramStats, Health, Heartbeat, HeartbeatEcho, LATE_BUCKETS,
    TimingSample,
};
pub use ids::{
    ChannelId, Epoch, Fingerprint, IdError, LeaseToken, ParticipantId, SessionId, TenantId,
};
pub use key::{IssuedKey, KeyBytes, KeyDelivery, KeyPurpose, KeysAck, Security};
pub use keys::SessionKeys;
pub use lockstep::{
    LockstepOp, LockstepReply, LockstepRequest, LockstepReset, LockstepResponse, LockstepResult,
    LockstepStep, PartValues,
};
pub use model::{
    ModelEpisode, ModelFault, ModelGroup, ModelHistoryRow, ModelOutcome, ModelReply, ModelRequest,
    ModelResolve, ModelResolved,
};
pub use pair::{PAIR_KEY, PairInfo, PairNonce, PairReply, PairRequest, PairStep};
pub use part::{
    ChannelPriority, CommandSpec, DType, Encoding, FieldLimit, FieldSpec, Layout, MergePolicy,
    PartSpec, SendPolicy, StreamSpec,
};
pub use path::PartPath;
pub use report::{ParticipantReport, PublisherCounters, RunCounters};
pub use sample::{Applied, EdgeMode, HistoryQuery, Sample, SampleBatch, SampleMeta, SeqRange};
pub use scene::{
    Assignment, DeviceRecord, HostOrder, HostStarted, Join, LobbyWait, MemberStage, MemberState,
    Place, Pose, RobotPin, RunStage, RunState, SceneInfo, StopOrder,
};
pub use tick::{Step, TickState, VirtualTick};
pub use timing::EndpointTiming;
pub use view::{AuthorityView, DeviceView, Member, ViewGrid, ViewReply};
