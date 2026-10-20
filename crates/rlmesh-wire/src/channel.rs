//! Channels: what a participant publishes or consumes, declared through the
//! host so the channel table is part of authority state.
//!
//! A channel's value is a list of **leaves**, each a dense little-endian
//! row-major tensor described by a [`TensorLayout`]. That is all the wire
//! needs to frame and length-check a value; what a value *means* (kind,
//! units, bounds) belongs to the session spec layer above.

use std::fmt;
use std::time::Duration;

use crate::authority::{CtlError, CtlStatus};
use crate::ids::{ChannelId, ParticipantId};

/// Which way a channel flows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Embodiment to consumers: sensors and state.
    Sense,
    /// Controller to embodiment, under a lease.
    Actuate,
    /// Anything else (reward, annotations).
    Aux,
}

/// Whether a lost message is resent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Delivery {
    /// Lost messages stay lost.
    BestEffort,
    /// Lost messages are resent (head-of-line blocking applies).
    Reliable,
}

/// The traffic class a channel rides. Higher classes are never queued behind
/// lower ones: the session maps them to separate transport sessions (see
/// `rlmesh-session`'s transport).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ChannelPriority {
    /// The control path: operator loops and safety-relevant commands.
    Control,
    /// High-rate proprioception and commands.
    Realtime,
    /// Ordinary sensors.
    Data,
    /// Cameras, point clouds, anything big.
    Bulk,
}

/// What a publisher does when its link is full.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Congestion {
    /// Drop the message rather than queue behind a full link.
    Drop,
    /// Wait for room: nothing is lost, everything behind it waits.
    Block,
}

/// What a consumer keeps, and what the host's history store holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum History {
    /// The latest value only.
    Latest,
    /// The newest `n` samples (n ≥ 2).
    Window(u32),
}

impl History {
    /// Samples kept: 1 for [`History::Latest`].
    pub fn depth(self) -> usize {
        match self {
            Self::Latest => 1,
            Self::Window(n) => n.max(1) as usize,
        }
    }

    /// The history a wire depth stands for (0 and 1 are latest).
    pub fn from_depth(depth: u32) -> Self {
        if depth <= 1 {
            Self::Latest
        } else {
            Self::Window(depth)
        }
    }
}

/// A channel's quality of service.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChannelQos {
    /// Whether lost samples are resent.
    pub delivery: Delivery,
    /// What consumers and the history store keep.
    pub history: History,
    /// A consumer treats older samples as stale.
    pub max_age: Option<Duration>,
    /// A publisher missing it raises a health event (declared, not yet
    /// enforced).
    pub deadline: Option<Duration>,
    /// The traffic class.
    pub priority: ChannelPriority,
    /// What a full link does.
    pub congestion: Congestion,
    /// The publisher's bitrate budget: media frames over it are skipped.
    pub max_bitrate_bps: Option<u64>,
}

impl ChannelQos {
    /// Best effort, latest value, dropped under congestion: the default for
    /// high-rate control loops.
    pub fn realtime(priority: ChannelPriority) -> Self {
        Self {
            delivery: Delivery::BestEffort,
            history: History::Latest,
            max_age: None,
            deadline: None,
            priority,
            congestion: Congestion::Drop,
            max_bitrate_bps: None,
        }
    }

    /// Control and real-time messages skip transport batching.
    pub fn express(&self) -> bool {
        self.priority <= ChannelPriority::Realtime
    }
}

/// How a channel's values travel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Encoding {
    /// Each leaf as its raw tensor bytes.
    Raw,
    /// One leaf: a baseline JPEG of a uint8 `[H, W, 3]` image at `quality`
    /// (1-100). Lossy: subscribers decode it back to pixels.
    Jpeg {
        /// JPEG quality, 1-100.
        quality: u8,
    },
}

impl Encoding {
    /// The name on the wire.
    pub fn wire_name(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Jpeg { .. } => "jpeg",
        }
    }
}

/// How the edge merges a new chunk into its queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MergePolicy {
    /// Keep committed actions before `replace_from`, replace from there on.
    Replace,
    /// Declared, not implemented: refused at declaration.
    Blend,
    /// Declared, not implemented: refused at declaration.
    RtcPrefix,
}

/// An actuation channel's edge contract, declared by its embodiment.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Actuation {
    /// How chunks merge.
    pub merge: MergePolicy,
    /// How long the edge waits for a valid action once its queue runs dry
    /// (or its holder is preempted) before it latches `safe_behavior` and
    /// takes the session SAFE.
    pub hold_max: Duration,
    /// The embodiment's own safe behaviour, by id ("brake", "gravity_comp").
    pub safe_behavior: String,
    /// Whether the edge may repeat the last action while it waits. Only for
    /// targets that are safe to hold (positions); off, it runs the safe
    /// behaviour while it waits. A repeated action must still be valid and
    /// under the current lease: an expired or preempted command is never
    /// continued.
    pub hold_last: bool,
}

impl Default for Actuation {
    fn default() -> Self {
        Self {
            merge: MergePolicy::Replace,
            hold_max: Duration::from_millis(100),
            safe_behavior: "stop".into(),
            hold_last: false,
        }
    }
}

/// A declared publish rate (a promise for planning, not a limiter).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rate {
    /// The nominal rate.
    pub nominal_hz: f64,
    /// The lowest acceptable rate, if any.
    pub min_hz: Option<f64>,
}

/// A tensor element type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    /// One byte, 0 or 1.
    Bool,
    /// `u8`.
    U8,
    /// `i8`.
    I8,
    /// `u16`.
    U16,
    /// `i16`.
    I16,
    /// `u32`.
    U32,
    /// `i32`.
    I32,
    /// `u64`.
    U64,
    /// `i64`.
    I64,
    /// IEEE half.
    F16,
    /// bfloat16.
    Bf16,
    /// `f32`.
    F32,
    /// `f64`.
    F64,
}

impl DType {
    /// Bytes per element.
    pub fn size(self) -> u64 {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::Bf16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }
}

/// The largest leaf a channel may declare (256 MiB).
pub const MAX_LEAF_BYTES: u64 = 256 << 20;

/// One leaf of a channel's value: a dense little-endian row-major tensor.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorLayout {
    /// The leaf's name (optional; for display and binding).
    pub name: String,
    /// The element type.
    pub dtype: DType,
    /// The shape; empty is a scalar.
    pub shape: Vec<u64>,
}

impl TensorLayout {
    /// An unnamed tensor.
    pub fn new(dtype: DType, shape: &[u64]) -> Self {
        Self {
            name: String::new(),
            dtype,
            shape: shape.to_vec(),
        }
    }

    /// An `f32` tensor of `shape`: what most sensors and commands carry.
    pub fn f32(shape: &[u64]) -> Self {
        Self::new(DType::F32, shape)
    }

    /// A `u8` `[height, width, channels]` image.
    pub fn u8_image(height: u64, width: u64, channels: u64) -> Self {
        Self::new(DType::U8, &[height, width, channels])
    }

    /// Elements in the tensor, `None` on overflow.
    pub fn elements(&self) -> Option<u64> {
        self.shape.iter().try_fold(1_u64, |n, d| n.checked_mul(*d))
    }

    /// Raw byte length, `None` on overflow.
    pub fn byte_len(&self) -> Option<u64> {
        self.elements()?.checked_mul(self.dtype.size())
    }

    /// `(height, width)` when this is a `u8` `[H, W, 3]` image.
    pub fn rgb8_dims(&self) -> Option<(u64, u64)> {
        match (self.dtype, self.shape.as_slice()) {
            (DType::U8, [h, w, 3]) if *h > 0 && *w > 0 => Some((*h, *w)),
            _ => None,
        }
    }
}

/// One declared channel.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelSpec {
    /// Stable within the session, chosen by the declarer.
    pub id: ChannelId,
    /// A human name ("wrist_cam").
    pub name: String,
    /// Which way it flows.
    pub direction: Direction,
    /// An open role registry: `image/wrist`, `proprio/eef_wrench`, ...
    pub role: String,
    /// One sample's value: one tensor per leaf.
    pub layout: Vec<TensorLayout>,
    /// Empty means SI.
    pub units: String,
    /// The coordinate frame, if any.
    pub frame: Option<String>,
    /// How values travel.
    pub encoding: Encoding,
    /// The declared rate.
    pub rate: Rate,
    /// Quality of service.
    pub qos: ChannelQos,
    /// Must be fresh for a lockstep view to tick.
    pub required: bool,
    /// Once per episode, reliable.
    pub episodic: bool,
    /// The embodiment that publishes (sense, aux) or consumes (actuate) it.
    /// The host sets it to the declaring participant.
    pub owner: Option<ParticipantId>,
    /// Actuation channels only; `None` takes [`Actuation::default`].
    pub actuation: Option<Actuation>,
}

impl ChannelSpec {
    /// A raw-encoded channel with SI units and no frame.
    pub fn new(
        id: ChannelId,
        name: impl Into<String>,
        direction: Direction,
        role: impl Into<String>,
        layout: Vec<TensorLayout>,
        nominal_hz: f64,
        qos: ChannelQos,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            direction,
            role: role.into(),
            layout,
            units: String::new(),
            frame: None,
            encoding: Encoding::Raw,
            rate: Rate {
                nominal_hz,
                min_hz: None,
            },
            qos,
            required: false,
            episodic: false,
            owner: None,
            actuation: None,
        }
    }

    /// The edge contract of an actuation channel (the default when none was
    /// declared).
    pub fn actuation(&self) -> Actuation {
        self.actuation.clone().unwrap_or_default()
    }

    /// The declared period, from the nominal rate (`None` at 0 Hz).
    pub fn period(&self) -> Option<Duration> {
        (self.rate.nominal_hz > 0.0 && self.rate.nominal_hz.is_finite())
            .then(|| Duration::from_secs_f64(1.0 / self.rate.nominal_hz))
    }

    /// The declaration rules the host applies.
    pub fn validate(&self) -> Result<(), CtlError> {
        let invalid = |message: String| CtlError::new(CtlStatus::InvalidRequest, message);
        if self.name.is_empty() {
            return Err(invalid(format!("{} has no name", self.id)));
        }
        if self.role.is_empty() {
            return Err(invalid(format!("{} ({}) has no role", self.id, self.name)));
        }
        if !(self.rate.nominal_hz.is_finite() && self.rate.nominal_hz >= 0.0) {
            return Err(invalid(format!("{} has an invalid rate", self.id)));
        }
        if let History::Window(0) = self.qos.history {
            return Err(invalid(format!("{} has an empty history window", self.id)));
        }
        if self.layout.is_empty() {
            return Err(invalid(format!("{} declares no leaves", self.id)));
        }
        for leaf in &self.layout {
            match leaf.byte_len() {
                Some(len) if len <= MAX_LEAF_BYTES => {}
                _ => {
                    return Err(invalid(format!(
                        "{} leaf {:?} is larger than {MAX_LEAF_BYTES} bytes",
                        self.id, leaf.name
                    )));
                }
            }
        }
        if let Encoding::Jpeg { quality } = self.encoding {
            if !(1..=100).contains(&quality) {
                return Err(invalid(format!("{} jpeg quality {quality}", self.id)));
            }
            if !matches!(self.layout.as_slice(), [leaf] if leaf.rgb8_dims().is_some()) {
                return Err(invalid(format!(
                    "{} is jpeg-encoded but its value is not one uint8 [H, W, 3] image",
                    self.id
                )));
            }
        }
        if let Some(actuation) = &self.actuation {
            if self.direction != Direction::Actuate {
                return Err(invalid(format!(
                    "{} declares actuation but is not an actuation channel",
                    self.id
                )));
            }
            if actuation.merge != MergePolicy::Replace {
                return Err(CtlError::new(
                    CtlStatus::Unsupported,
                    format!(
                        "{} merge policy {:?}: only replace is implemented",
                        self.id, actuation.merge
                    ),
                ));
            }
            if actuation.safe_behavior.is_empty() {
                return Err(invalid(format!("{} has no safe behaviour", self.id)));
            }
        }
        Ok(())
    }

    /// The level-2 check every received value gets: the leaf count and, for
    /// raw channels, each leaf's byte length against the declared layout.
    /// O(leaves), no per-element work.
    pub fn check_leaf_lens(
        &self,
        lens: impl ExactSizeIterator<Item = usize>,
    ) -> Result<(), String> {
        if lens.len() != self.layout.len() {
            return Err(format!(
                "{} leaves, declared {}",
                lens.len(),
                self.layout.len()
            ));
        }
        if self.encoding != Encoding::Raw {
            return Ok(());
        }
        for (i, (len, leaf)) in lens.zip(&self.layout).enumerate() {
            let want = leaf.byte_len().unwrap_or(u64::MAX);
            if len as u64 != want {
                return Err(format!("leaf {i} is {len} bytes, declared {want}"));
            }
        }
        Ok(())
    }

    /// The same declaration, ignoring the owner the host fills in.
    pub fn same_declaration(&self, other: &Self) -> bool {
        let strip = |spec: &Self| Self {
            owner: None,
            ..spec.clone()
        };
        strip(self) == strip(other)
    }
}

impl fmt::Display for ChannelSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} {:?} {} @{}Hz {:?}",
            self.id, self.name, self.direction, self.role, self.rate.nominal_hz, self.qos.priority
        )
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn camera() -> ChannelSpec {
        let mut spec = ChannelSpec::new(
            ChannelId(7),
            "wrist_cam",
            Direction::Sense,
            "image/wrist",
            vec![TensorLayout::u8_image(480, 640, 3)],
            30.0,
            ChannelQos {
                delivery: Delivery::BestEffort,
                history: History::Window(4),
                max_age: Some(Duration::from_millis(100)),
                deadline: Some(Duration::from_millis(50)),
                priority: ChannelPriority::Bulk,
                congestion: Congestion::Drop,
                max_bitrate_bps: Some(8_000_000),
            },
        );
        spec.encoding = Encoding::Jpeg { quality: 75 };
        spec.frame = Some("tool".into());
        spec.rate.min_hz = Some(10.0);
        spec.owner = Some(ParticipantId::new("ur10-sim").expect("id"));
        spec
    }

    pub(crate) fn joint_cmd(merge: MergePolicy) -> ChannelSpec {
        let mut spec = ChannelSpec::new(
            ChannelId(1),
            "joint_cmd",
            Direction::Actuate,
            "action/joint_pos",
            vec![TensorLayout::f32(&[6])],
            1000.0,
            ChannelQos::realtime(ChannelPriority::Control),
        );
        spec.actuation = Some(Actuation {
            merge,
            hold_max: Duration::from_millis(40),
            safe_behavior: "brake".into(),
            hold_last: true,
        });
        spec
    }

    #[test]
    fn declarations_are_validated() {
        camera().validate().expect("valid");
        let mut spec = camera();
        spec.name.clear();
        assert!(spec.validate().is_err());
        let mut spec = camera();
        spec.qos.history = History::Window(0);
        assert!(spec.validate().is_err());
        let mut spec = camera();
        spec.layout.clear();
        assert!(spec.validate().is_err());
        let mut spec = camera();
        spec.layout = vec![TensorLayout::f32(&[u64::MAX, 2])];
        assert!(spec.validate().is_err(), "overflowing shape");
        let mut other = camera();
        other.owner = None;
        assert!(camera().same_declaration(&other));
        other.rate.nominal_hz = 15.0;
        assert!(!camera().same_declaration(&other));
        let mut spec = camera();
        spec.encoding = Encoding::Jpeg { quality: 0 };
        assert!(spec.validate().is_err());
        let mut spec = camera();
        spec.layout = vec![TensorLayout::f32(&[6])];
        assert!(spec.validate().is_err(), "jpeg needs a uint8 image");
    }

    #[test]
    fn only_the_replace_merge_policy_is_accepted() {
        joint_cmd(MergePolicy::Replace).validate().expect("replace");
        for merge in [MergePolicy::Blend, MergePolicy::RtcPrefix] {
            let err = joint_cmd(merge).validate().expect_err("unsupported");
            assert_eq!(err.status, CtlStatus::Unsupported, "{merge:?}");
        }
        let mut sense = camera();
        sense.actuation = Some(Actuation::default());
        assert_eq!(
            sense.validate().expect_err("not actuate").status,
            CtlStatus::InvalidRequest
        );
        assert_eq!(
            joint_cmd(MergePolicy::Replace).period(),
            Some(Duration::from_millis(1))
        );
    }

    #[test]
    fn leaf_lengths_are_checked_against_the_layout() {
        let cmd = joint_cmd(MergePolicy::Replace);
        assert!(cmd.check_leaf_lens([24_usize].into_iter()).is_ok());
        assert!(cmd.check_leaf_lens([20_usize].into_iter()).is_err());
        assert!(cmd.check_leaf_lens([24_usize, 24].into_iter()).is_err());
        // A JPEG leaf has any length.
        assert!(camera().check_leaf_lens([60_000_usize].into_iter()).is_ok());
        assert_eq!(
            TensorLayout::u8_image(480, 640, 3).byte_len(),
            Some(921_600)
        );
        assert_eq!(TensorLayout::new(DType::F64, &[]).byte_len(), Some(8));
    }
}
