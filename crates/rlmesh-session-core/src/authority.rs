//! Authority state: epoch, roster, mode, e-stop latch,
//! the part table, bindings and the lease table, with the session host as
//! its single writer.
//!
//! [`Authority`] is sans-IO: every method takes the host's session clock
//! (`now_ns`) and returns what it caused, so the state machine is tested
//! without a network and the host only has to serialise calls into it.
//!
//! - **Parts** come from an embodiment's authored map, normalised by the
//!   host with the shared normaliser ([`crate::spec::normalise()`]); the
//!   snapshot carries each robot's declaration fingerprint. Channel ids are
//!   host-assigned and never reused within the session: re-declaring a path
//!   with a changed schema retires its id and assigns a new one. A
//!   commandable part is not re-declared while ARMED or ACTIVE
//!   (`INVALID_TRANSITION`); sensor parts can be added at any time.
//! - **Leases** are per commandable part, granted all or none
//!   ([`Command::GrantLease`]) and bound to the holder's current
//!   **binding**: chunks must carry its fingerprint, and a rebinding
//!   re-issues the holder's leases under a fresh token, so in-flight chunks
//!   of the old binding are fenced.
//! - **Delivery contracts** come with a binding, one per bound input
//!   stream. The host admits them ([`crate::spec::contract::admit`]), derives
//!   one publication plan per stream from every attached controller's
//!   contracts and publishes it as a [`Demand`]; the stream's publisher
//!   installs it and acknowledges the revision ([`Command::DemandAck`]).
//!   Install before authority: a lease is granted only once every stream of
//!   the requester's binding has acknowledged a revision that includes its
//!   contract (`NOT_READY`). Demand is recomputed on attach, detach,
//!   participant loss, part removal and rebinding.
//! - Priority classes: `safety > operator > model > scripted`. A higher
//!   class preempts a lower one and gets a fresh token; an equal or lower
//!   class gets `LEASE_BUSY`.
//! - **Command groups** are scoped `(robot, group)`: at most one member is
//!   leased at a time. A member held by an equal or higher class, or by the
//!   requester without releasing it in the same request, is `GROUP_BUSY`; a
//!   lower class's member is preempted. `release` switches members
//!   atomically.
//! - Leases have a TTL, renewed by every heartbeat from the holder on the
//!   current epoch. A holder that lapses, is lost or is replaced loses only
//!   the leases it held: those parts go to their safe behaviour as after a
//!   release, and the mode and every other holder go on.
//! - **Presence** is heartbeats only: a participant silent for
//!   `presence_grace` is lost. A liveliness put never makes anyone present;
//!   a delete is only a hint ([`Authority::presence_hint`]) that loses the
//!   participant unless a heartbeat arrives within `lease_grace` of it. A
//!   lost robot stays in the roster, absent with `left = ROBOT_LOST`, while
//!   its parts remain; the leases on them end with `ROBOT_LOST`.
//! - ESTOP is accepted from anyone (refusing an e-stop is never safer) and
//!   latches; only an embodiment's local clear releases the latch, and then a
//!   disarm and an arm (a fresh epoch) are needed.
//! - **Reasons**: every mode change records a structured
//!   [`Reason`] (`mode_reason`), every lease ending goes into a bounded ring
//!   of [`LeaseEnd`]s, and `last_stop` marks the last entry into SAFE,
//!   DISARMED or ESTOP, so clients never infer why. An embodiment's
//!   `EnterSafe` ends that robot's leases and holds it until a `Rearm` names
//!   it; SAFE and ESTOP stay and record its reason. The snapshot lists
//!   every robot held, with why and since when.
//! - **Places** ([`AuthorityConfig::places`]) go out in every snapshot, so
//!   every binder builds the same view of several robots; a declaration
//!   whose placed paths collide is refused (`INCOMPATIBLE`).
//! - **Scenes** (`scene`, feature `fleet`): a session a daemon runs for a
//!   scene admits only its members, each from its device, and closes in one
//!   revision (`Authority::close_scene`).
//! - In **virtual** clock mode the authority also runs the tick barrier
//!   ([`crate::virtual_clock`]); the ticks it emits are drained with
//!   [`Authority::take_ticks`].

#[allow(unused_imports)]
use crate::prelude::*;

use alloc::collections::{BTreeMap, BTreeSet, VecDeque};

#[cfg(doc)]
use crate::wire::Command;
use crate::wire::authority::fence;
use crate::wire::part::PartSpec;
use crate::wire::tick::VirtualTick;
use crate::wire::version::PROTOCOL_MINOR;
use crate::wire::{
    AuthoritySnapshot, BindingRecord, ChannelId, ClockMapping, ClockMode, Demand, Epoch,
    FenceError, Fingerprint, HeldRobot, LastStop, Lease, LeaseEnd, LeaseToken, Mode, ParticipantId,
    ParticipantInfo, Place, Reason, SessionId,
};

use crate::virtual_clock::{VirtualClock, VirtualConfig};

pub mod access;
mod barrier;
mod binding;
mod event;
mod leases;
mod mode;
#[cfg(not(feature = "fleet"))]
mod no_scene;
mod parts;
mod roster;
#[cfg(feature = "fleet")]
pub mod scene;
mod verbs;
pub use access::{Access, DeviceAccess, Verb};
pub use event::{AuthorityEvent, DetachReason};
#[cfg(feature = "fleet")]
pub use scene::{Member, SceneConfig};

/// Lease endings the snapshot keeps.
const ENDED: usize = 16;

/// The authority's rules.
#[derive(Clone, Debug, PartialEq)]
pub struct AuthorityConfig {
    /// Default lease ttl: how long the holder's heartbeats may stop before
    /// its lease lapses.
    pub lease_grace_ns: u64,
    /// How long any participant's heartbeats may stop before it is lost (at
    /// least `lease_grace_ns`).
    pub presence_grace_ns: u64,
    /// The features the session implements (see [`crate::wire::version`]).
    pub supports: Vec<String>,
    /// Virtual clock mode; `None` is wall mode.
    pub virtual_clock: Option<VirtualConfig>,
    /// The parts whose streams, chunks and lockstep traffic are encrypted; every other
    /// part is signed.
    pub seal: SealPolicy,
    /// What each device may do; `local` may do everything.
    pub access: Access,
    /// Whether a robot that went safe, restarted or came back from a loss
    /// may be leased again at once. Off by default: its leases ended and it
    /// waits for a `Rearm` naming it. Other robots never stop for it either
    /// way.
    pub rearm_robots: bool,
    /// Where each placed robot's parts appear in the view: one binding drives several
    /// robots through it. Empty: no view.
    pub places: Vec<Place>,
    /// The scene this session runs, when a daemon started it for one.
    #[cfg(feature = "fleet")]
    pub scene: Option<SceneConfig>,
}

/// Which parts a session seals (encrypts) rather than signs.
/// A robot that declares a part it can't seal (`DeclareParts.seal`) is
/// refused; a level is never lowered silently.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum SealPolicy {
    /// Every part is signed: integrity only.
    #[default]
    None,
    /// Every part is sealed.
    All,
    /// These parts (`{robot}/{path}`, or `{robot}/*` for every part of
    /// the robot) are sealed.
    Parts(Vec<String>),
}

impl SealPolicy {
    /// Whether `robot`'s part at `path` is sealed.
    pub fn seals(&self, robot: &ParticipantId, path: &str) -> bool {
        match self {
            Self::None => false,
            Self::All => true,
            Self::Parts(parts) => parts.iter().any(|p| {
                p.split_once('/')
                    .is_some_and(|(r, p)| r == robot.as_str() && (p == path || p == "*"))
            }),
        }
    }
}

impl Default for AuthorityConfig {
    fn default() -> Self {
        Self {
            lease_grace_ns: 250_000_000,
            presence_grace_ns: 1_000_000_000,
            supports: crate::wire::version::FEATURES
                .iter()
                .map(ToString::to_string)
                .collect(),
            virtual_clock: None,
            seal: SealPolicy::None,
            access: Access::default(),
            rearm_robots: false,
            places: Vec::new(),
            #[cfg(feature = "fleet")]
            scene: None,
        }
    }
}

/// Who sent a verb, as the host's handshake authenticated it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Identity {
    /// The sender's device.
    pub device: String,
    /// The sender's generation: a new one of a known participant replaces
    /// the old process.
    pub generation: u32,
}

/// What an accepted verb produced.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Applied {
    /// The leases granted, for `GrantLease`.
    pub leases: Vec<Lease>,
    /// What changed.
    pub events: Vec<AuthorityEvent>,
}

/// The authority state machine. Only the session host owns one.
#[derive(Clone, Debug)]
pub struct Authority {
    session: SessionId,
    config: AuthorityConfig,
    epoch: Epoch,
    revision: u64,
    mode: Mode,
    estop_latched: bool,
    /// Who latched the e-stop: only that robot, or `local`, clears it.
    estop_by: Option<ParticipantId>,
    mode_reason: Reason,
    /// The latest lease endings of holders still in the roster, oldest
    /// first.
    ended: VecDeque<LeaseEnd>,
    last_stop: LastStop,
    /// The host clock of the call in progress, for the reasons it records.
    now_ns: i64,
    roster: BTreeMap<ParticipantId, ParticipantInfo>,
    /// Each present participant's last heartbeat (or attach), and a pending
    /// liveliness hint.
    seen: BTreeMap<ParticipantId, Seen>,
    leases: BTreeMap<ChannelId, Lease>,
    /// Declared parts by stream id.
    parts: BTreeMap<ChannelId, PartSpec>,
    /// The last channel id assigned; ids only grow.
    last_channel: u32,
    bindings: BTreeMap<ParticipantId, BindingRecord>,
    /// Each robot's declaration fingerprint.
    contract: BTreeMap<ParticipantId, Fingerprint>,
    /// The publication plan per stream some attached binding contracts.
    demands: BTreeMap<ChannelId, Demand>,
    /// The last demand revision per stream, kept when a demand goes away so
    /// a stale acknowledgement never satisfies a later one.
    demand_revisions: BTreeMap<ChannelId, u64>,
    /// Per controller and stream, the demand revision that first included
    /// its contract: what its publisher must have acknowledged.
    needed: BTreeMap<(ParticipantId, ChannelId), u64>,
    last_token: u64,
    /// The token the verb being applied mints, reserved before its keys
    /// were delivered.
    minting: Option<LeaseToken>,
    /// The streams (parts and their applied streams) that are sealed.
    sealed: BTreeSet<ChannelId>,
    clock: Option<VirtualClock>,
    /// Virtual ticks to publish.
    ticks: Vec<VirtualTick>,
    /// Robots stopped on their own (a fallback, a restart, a loss) and not
    /// re-armed yet: their parts take no lease until a `Rearm` names them.
    held: BTreeMap<ParticipantId, Reason>,
    /// The scene run's closing and rejoin window.
    #[cfg(feature = "fleet")]
    scene: scene::SceneState,
}

impl Authority {
    /// A fresh session: DISARMED in epoch 1.
    pub fn new(session: SessionId, config: AuthorityConfig) -> Self {
        // Epoch 0 means "unchecked" on the wire, so sessions start at 1.
        let clock = config
            .virtual_clock
            .map(|virtual_clock| VirtualClock::new(virtual_clock, Epoch::FIRST, 0));
        Self {
            session,
            config,
            epoch: Epoch::FIRST,
            revision: 0,
            mode: Mode::Disarmed,
            estop_latched: false,
            estop_by: None,
            mode_reason: Reason::default(),
            ended: VecDeque::with_capacity(ENDED),
            // A fresh session is stopped (DISARMED) in its first epoch.
            last_stop: LastStop {
                epoch: Epoch::FIRST,
                revision: 0,
            },
            now_ns: 0,
            roster: BTreeMap::new(),
            seen: BTreeMap::new(),
            leases: BTreeMap::new(),
            parts: BTreeMap::new(),
            last_channel: 0,
            bindings: BTreeMap::new(),
            contract: BTreeMap::new(),
            demands: BTreeMap::new(),
            demand_revisions: BTreeMap::new(),
            needed: BTreeMap::new(),
            last_token: 0,
            minting: None,
            sealed: BTreeSet::new(),
            clock,
            ticks: Vec::new(),
            held: BTreeMap::new(),
            #[cfg(feature = "fleet")]
            scene: scene::SceneState::default(),
        }
    }

    /// Why `robot` waits for a `Rearm` before its parts take a lease, if it
    /// does.
    pub fn held(&self, robot: &ParticipantId) -> Option<&Reason> {
        self.held.get(robot)
    }

    /// The session.
    pub fn session(&self) -> SessionId {
        self.session
    }

    /// Its rules.
    pub fn config(&self) -> &AuthorityConfig {
        &self.config
    }

    /// The current epoch.
    pub fn epoch(&self) -> Epoch {
        self.epoch
    }

    /// The structural revision.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// The mode.
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// Whether an e-stop is latched.
    pub fn estop_latched(&self) -> bool {
        self.estop_latched
    }

    /// Why the mode is what it is.
    pub fn mode_reason(&self) -> &Reason {
        &self.mode_reason
    }

    /// The last entry into SAFE, DISARMED or ESTOP.
    pub fn last_stop(&self) -> LastStop {
        self.last_stop
    }

    /// The lease on `channel`.
    pub fn lease(&self, channel: ChannelId) -> Option<&Lease> {
        self.leases.get(&channel)
    }

    /// A declared part, by its stream id.
    pub fn part(&self, channel: ChannelId) -> Option<&PartSpec> {
        self.parts.get(&channel)
    }

    /// Every declared part.
    pub fn parts(&self) -> impl Iterator<Item = &PartSpec> {
        self.parts.values()
    }

    /// A controller's binding.
    pub fn binding(&self, id: &ParticipantId) -> Option<&BindingRecord> {
        self.bindings.get(id)
    }

    /// Leases held.
    pub fn lease_count(&self) -> usize {
        self.leases.len()
    }

    /// Whether `id` is in the roster and present.
    pub fn is_attached(&self, id: &ParticipantId) -> bool {
        self.roster.get(id).is_some_and(|p| p.present)
    }

    /// The virtual ticks emitted since the last call, in order.
    pub fn take_ticks(&mut self) -> Vec<VirtualTick> {
        core::mem::take(&mut self.ticks)
    }

    /// The host-side fence, expiry included.
    pub fn check_fence(
        &self,
        channel: ChannelId,
        token: LeaseToken,
        epoch: Epoch,
        now_ns: i64,
    ) -> Result<(), FenceError> {
        let lease = fence(self.epoch, self.leases.get(&channel), channel, token, epoch)?;
        if now_ns > lease.expires_ns {
            return Err(FenceError::Expired { channel });
        }
        Ok(())
    }

    /// The snapshot at `now_ns`, with the host's clock mappings.
    pub fn snapshot(&self, clocks: Vec<ClockMapping>, now_ns: i64) -> AuthoritySnapshot {
        AuthoritySnapshot {
            session: self.session,
            epoch: self.epoch,
            revision: self.revision,
            mode: self.mode,
            estop_latched: self.estop_latched,
            roster: self.roster.values().cloned().collect(),
            leases: self.leases.values().cloned().collect(),
            clocks,
            host_ns: now_ns,
            parts: self.parts.values().cloned().collect(),
            protocol_minor: PROTOCOL_MINOR,
            supports: self.config.supports.clone(),
            clock_mode: if self.clock.is_some() {
                ClockMode::Virtual
            } else {
                ClockMode::Wall
            },
            tick_ns: self.config.virtual_clock.map_or(0, |c| c.tick_ns),
            bindings: self.bindings.values().cloned().collect(),
            contract: self.contract.clone(),
            demands: self.demands.values().cloned().collect(),
            mode_reason: self.mode_reason.clone(),
            ended: self.ended.iter().cloned().collect(),
            last_stop: self.last_stop,
            places: self.config.places.clone(),
            scene: self.scene_info(),
            held: self
                .held
                .iter()
                .map(|(robot, reason)| HeldRobot {
                    robot: robot.clone(),
                    reason: reason.clone(),
                })
                .collect(),
        }
    }

    /// Every lease.
    pub fn leases(&self) -> impl Iterator<Item = &Lease> {
        self.leases.values()
    }

    /// Records the id of the `k_cmd` the lease on `channel` was committed
    /// with (its `Lease.key_id`).
    pub fn set_key_id(&mut self, channel: ChannelId, key_id: u32) {
        if let Some(lease) = self.leases.get_mut(&channel) {
            lease.key_id = key_id;
        }
    }

    /// The participant in the roster under `id`.
    pub fn participant(&self, id: &ParticipantId) -> Option<&ParticipantInfo> {
        self.roster.get(id)
    }

    /// The level of a stream (a part's or its applied stream's).
    pub fn security(&self, channel: ChannelId) -> crate::wire::Security {
        if self.sealed.contains(&channel) {
            crate::wire::Security::Seal
        } else {
            crate::wire::Security::Mac
        }
    }
}

/// When a participant was last heard from.
#[derive(Clone, Copy, Debug)]
struct Seen {
    /// Its last heartbeat (or its attach).
    last_ns: i64,
    /// When its liveliness token went away, if it did since.
    hint_ns: Option<i64>,
}

/// A `GrantLease` verb's arguments.
struct Grant {
    channels: Vec<ChannelId>,
    release: Vec<ChannelId>,
    binding: Fingerprint,
    ttl_ns: Option<u64>,
}

#[cfg(test)]
mod tests;
