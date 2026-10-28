//! One robot's device view (scene design §3.4), and the authority state an
//! edge, its fallback and a keyring read ([`AuthorityView`]): a board that
//! takes the view decodes it without the full state's decoder.

#[allow(unused_imports)]
use crate::prelude::*;

use core::time::Duration;

use crate::wire::authority::{
    AuthoritySnapshot, ClockMapping, ClockMode, FenceError, LastStop, Lease, Mode, Reason,
    ReasonCode, fence,
};
use crate::wire::contract::Demand;
use crate::wire::ids::{ChannelId, Epoch, Fingerprint, LeaseToken, ParticipantId};
use crate::wire::part::PartSpec;

/// What the edge, its fallback and a keyring read of the host's authority
/// state: an [`AuthoritySnapshot`] or one robot's [`DeviceView`].
pub trait AuthorityView {
    /// The current epoch.
    fn epoch(&self) -> Epoch;
    /// The mode.
    fn mode(&self) -> Mode;
    /// Why the mode is what it is.
    fn mode_reason(&self) -> ReasonCode;
    /// Bumps on every structural change.
    fn revision(&self) -> u64;
    /// Host session clock when it was taken.
    fn host_ns(&self) -> i64;
    /// How session time advances.
    fn clock_mode(&self) -> ClockMode;
    /// Session time per tick (0 in wall mode).
    fn tick_ns(&self) -> u64;
    /// The held leases (a view: those on its robot's parts).
    fn leases(&self) -> &[Lease];
    /// A participant's clock mapping.
    fn clock(&self, who: &ParticipantId) -> Option<&ClockMapping>;
    /// The action period the binding of `channel`'s lease declared for it.
    fn grid(&self, channel: ChannelId) -> Option<Duration>;
    /// The generation of `who` (a view: present participants only).
    fn generation(&self, who: &ParticipantId) -> Option<u32>;

    /// The lease on `channel`.
    fn lease(&self, channel: ChannelId) -> Option<&Lease> {
        self.leases().iter().find(|l| l.channel == channel)
    }

    /// The fence an edge applies to an incoming chunk: right epoch, and the
    /// token of the part's current lease.
    ///
    /// # Errors
    /// [`FenceError`].
    fn check_fence(
        &self,
        channel: ChannelId,
        token: LeaseToken,
        epoch: Epoch,
    ) -> Result<&Lease, FenceError> {
        fence(self.epoch(), self.lease(channel), channel, token, epoch)
    }
}

impl AuthorityView for AuthoritySnapshot {
    fn epoch(&self) -> Epoch {
        self.epoch
    }
    fn mode(&self) -> Mode {
        self.mode
    }
    fn mode_reason(&self) -> ReasonCode {
        self.mode_reason.code
    }
    fn revision(&self) -> u64 {
        self.revision
    }
    fn host_ns(&self) -> i64 {
        self.host_ns
    }
    fn clock_mode(&self) -> ClockMode {
        self.clock_mode
    }
    fn tick_ns(&self) -> u64 {
        self.tick_ns
    }
    fn leases(&self) -> &[Lease] {
        &self.leases
    }
    fn clock(&self, who: &ParticipantId) -> Option<&ClockMapping> {
        AuthoritySnapshot::clock(self, who)
    }
    fn grid(&self, channel: ChannelId) -> Option<Duration> {
        let lease = AuthoritySnapshot::lease(self, channel)?;
        self.binding(&lease.holder)
            .filter(|b| b.fingerprint == lease.binding)?
            .grid(channel)
    }
    fn generation(&self, who: &ParticipantId) -> Option<u32> {
        self.participant(who).map(|p| p.generation)
    }
}

/// One robot's slice of the authority state, on `host/view/{robot}`
/// (`Signed` kind 22): what its edge acts on, and no other robot's parts.
#[derive(Clone, Debug, PartialEq)]
pub struct DeviceView {
    /// Whose view.
    pub robot: ParticipantId,
    /// The current epoch.
    pub epoch: Epoch,
    /// The authority state's revision.
    pub revision: u64,
    /// The mode.
    pub mode: Mode,
    /// Why the mode is what it is (the code only).
    pub mode_reason: ReasonCode,
    /// The last entry into SAFE, DISARMED or ESTOP.
    pub last_stop: LastStop,
    /// Host session clock when it was taken.
    pub host_ns: i64,
    /// How session time advances.
    pub clock_mode: ClockMode,
    /// Session time per tick (0 in wall mode).
    pub tick_ns: u64,
    /// The robot's generation in effect (0: not attached).
    pub generation: u32,
    /// Its own parts, host-normalised.
    pub parts: Vec<PartSpec>,
    /// The leases on its parts: a token gone from here has ended.
    pub leases: Vec<Lease>,
    /// Its clock mapping and each current lease holder's.
    pub clocks: Vec<ClockMapping>,
    /// The action grid of each leased part under its lease's binding.
    pub grids: Vec<ViewGrid>,
    /// The publication plans of its streams.
    pub demands: Vec<Demand>,
    /// Every present participant generation.
    pub members: Vec<Member>,
    /// Set once the run is ending: detach and return to the lobby.
    // TODO: always `None` until the host carries a scene (scene step 5).
    pub closing: Option<Reason>,
}

/// One leased part's action grid under its lease's binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewGrid {
    /// The binding.
    pub binding: Fingerprint,
    /// The part.
    pub channel: ChannelId,
    /// The action period.
    pub dt: Duration,
}

/// A present participant generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Member {
    /// The participant.
    pub id: ParticipantId,
    /// Its generation.
    pub generation: u32,
}

impl AuthorityView for DeviceView {
    fn epoch(&self) -> Epoch {
        self.epoch
    }
    fn mode(&self) -> Mode {
        self.mode
    }
    fn mode_reason(&self) -> ReasonCode {
        self.mode_reason
    }
    fn revision(&self) -> u64 {
        self.revision
    }
    fn host_ns(&self) -> i64 {
        self.host_ns
    }
    fn clock_mode(&self) -> ClockMode {
        self.clock_mode
    }
    fn tick_ns(&self) -> u64 {
        self.tick_ns
    }
    fn leases(&self) -> &[Lease] {
        &self.leases
    }
    fn clock(&self, who: &ParticipantId) -> Option<&ClockMapping> {
        self.clocks.iter().find(|c| &c.participant == who)
    }
    fn grid(&self, channel: ChannelId) -> Option<Duration> {
        let lease = self.lease(channel)?;
        self.grids
            .iter()
            .find(|g| g.channel == channel && g.binding == lease.binding)
            .map(|g| g.dt)
    }
    fn generation(&self, who: &ParticipantId) -> Option<u32> {
        self.members
            .iter()
            .find(|m| &m.id == who)
            .map(|m| m.generation)
    }
}
