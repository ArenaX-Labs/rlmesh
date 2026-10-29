//! One robot's device view (scene design §3.4), and the authority state an
//! edge, its fallback and a keyring read ([`AuthorityView`]): a board that
//! takes the view decodes it without the full state's decoder.

#[allow(unused_imports)]
use crate::prelude::*;

use core::time::Duration;

use crate::wire::authority::{
    AuthoritySnapshot, ClockMapping, ClockMode, CtlError, FenceError, LastStop, Lease, Mode,
    Reason, ReasonCode, fence,
};
use crate::wire::contract::Demand;
use crate::wire::ids::{ChannelId, Epoch, Fingerprint, LeaseToken, ParticipantId};
use crate::wire::key::IssuedKey;
use crate::wire::part::PartSpec;
use crate::wire::scene::{Assignment, RunState};

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
    /// Why the scene run this session hosts is ending, once it is: detach
    /// and return to the lobby.
    fn closing(&self) -> Option<&Reason>;

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
    fn closing(&self) -> Option<&Reason> {
        self.scene.as_ref()?.closing.as_ref()
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
    pub closing: Option<Reason>,
}

impl DeviceView {
    /// `robot`'s view of `state`: its own parts, the leases on them, the
    /// clocks and grids those leases need, its streams' plans and every
    /// present member.
    pub fn of(state: &AuthoritySnapshot, robot: &ParticipantId) -> Self {
        let parts: Vec<PartSpec> = state
            .parts
            .iter()
            .filter(|p| &p.owner == robot)
            .cloned()
            .collect();
        let leases: Vec<Lease> = state
            .leases
            .iter()
            .filter(|l| parts.iter().any(|p| p.channel == l.channel))
            .cloned()
            .collect();
        let clocks = state
            .clocks
            .iter()
            .filter(|c| &c.participant == robot || leases.iter().any(|l| l.holder == c.participant))
            .cloned()
            .collect();
        let grids = leases
            .iter()
            .filter_map(|l| {
                Some(ViewGrid {
                    binding: l.binding,
                    channel: l.channel,
                    dt: state.grid(l.channel)?,
                })
            })
            .collect();
        let demands = state
            .demands
            .iter()
            .filter(|d| {
                parts
                    .iter()
                    .any(|p| p.channel == d.channel || p.applied_channel == Some(d.channel))
            })
            .cloned()
            .collect();
        let members = state
            .roster
            .iter()
            .filter(|p| p.present)
            .map(|p| Member {
                id: p.id.clone(),
                generation: p.generation,
            })
            .collect();
        Self {
            robot: robot.clone(),
            epoch: state.epoch,
            revision: state.revision,
            mode: state.mode,
            mode_reason: state.mode_reason.code,
            last_stop: state.last_stop,
            host_ns: state.host_ns,
            clock_mode: state.clock_mode,
            tick_ns: state.tick_ns,
            generation: state
                .participant(robot)
                .filter(|p| p.present)
                .map_or(0, |p| p.generation),
            parts,
            leases,
            clocks,
            grids,
            demands,
            members,
            closing: state.scene.as_ref().and_then(|s| s.closing.clone()),
        }
    }
}

/// The host's answer to a robot that takes its device view: a
/// [`crate::wire::CtlReply`] with the robot's view in place of the authority
/// state, which a board decodes without the state's decoder.
#[derive(Clone, Debug, PartialEq)]
pub struct ViewReply {
    /// `Ok` or the refusal.
    pub result: Result<(), CtlError>,
    /// Its view right after the verb, accepted or not (`None`: it is not
    /// attached as a robot that asked for one, as after a refused attach).
    pub view: Option<DeviceView>,
    /// The leases granted, one per part (`GrantLease`).
    pub leases: Vec<Lease>,
    /// The request this answers.
    pub request_id: u64,
    /// Keys for the caller, sealed with the reply.
    pub keys: Vec<IssuedKey>,
    /// A `LobbyWait`'s order.
    pub assignment: Option<Assignment>,
    /// A run's state.
    pub run: Option<RunState>,
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
    fn closing(&self) -> Option<&Reason> {
        self.closing.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authority::{Authority, AuthorityConfig};
    use crate::wire::BindingRecord;
    use crate::wire::authority::{ParticipantInfo, PriorityClass, Role};
    use crate::wire::contract::{ActionGrid, Plan};
    use crate::wire::ids::SessionId;
    use crate::wire::part::tests::{arm, camera, pid};

    fn member(id: &str, generation: u32, present: bool) -> ParticipantInfo {
        ParticipantInfo {
            id: pid(id),
            name: id.into(),
            roles: vec![Role::Embodiment],
            priority: PriorityClass::Scripted,
            attached_ns: 0,
            protocol_minor: 0,
            supports: Vec::new(),
            requires: Vec::new(),
            workflow_edition: String::new(),
            workflow_editions: Vec::new(),
            present,
            left: None,
            device: "local".into(),
            generation,
        }
    }

    fn clock(id: &str) -> ClockMapping {
        ClockMapping {
            participant: pid(id),
            offset_ns: 5,
            error_ns: 1,
            rtt_ns: 2,
            sample_ns: 0,
            drift_bound_ppm: 100.0,
        }
    }

    fn lease(channel: u32, holder: &str, binding: u64) -> Lease {
        Lease {
            channel: ChannelId(channel),
            holder: pid(holder),
            priority: PriorityClass::Model,
            token: LeaseToken(u64::from(channel)),
            ttl_ns: 5,
            granted_ns: 0,
            expires_ns: 5,
            binding: Fingerprint::from(binding),
            key_id: 0,
        }
    }

    /// `ur10e`'s arm (1, applied 2) leased by `policy`; `cam-1`'s camera
    /// (3) leased by `other`; `gone` lost.
    fn state() -> AuthoritySnapshot {
        let mut state = Authority::new(SessionId::named("demo"), AuthorityConfig::default())
            .snapshot(Vec::new(), 40);
        let mut cam = camera();
        cam.owner = pid("cam-1");
        state.parts = vec![arm(), cam];
        state.roster = vec![
            member("ur10e", 4, true),
            member("cam-1", 2, true),
            member("policy", 3, true),
            member("other", 5, true),
            member("gone", 1, false),
        ];
        state.leases = vec![lease(1, "policy", 7), lease(3, "other", 8)];
        state.clocks = ["ur10e", "cam-1", "policy", "other"].map(clock).to_vec();
        state.bindings = vec![BindingRecord {
            participant: pid("policy"),
            fingerprint: Fingerprint::from(7),
            revision: 1,
            table_json: "{}".into(),
            channels: vec![ChannelId(1)],
            replan_hz: 10.0,
            contracts: Vec::new(),
            grids: vec![ActionGrid {
                channel: ChannelId(1),
                dt: Duration::from_millis(20),
            }],
        }];
        state.demands = [2, 3]
            .map(|c| Demand {
                channel: ChannelId(c),
                revision: 1,
                plan: Plan::Every,
                retain: Duration::ZERO,
                acked: 0,
            })
            .to_vec();
        state
    }

    #[test]
    fn a_view_holds_its_robot_s_slice_and_answers_as_the_state_does() {
        let state = state();
        let ur = pid("ur10e");
        let view = DeviceView::of(&state, &ur);
        assert_eq!(view.parts, vec![arm()]);
        assert_eq!(view.leases, vec![lease(1, "policy", 7)]);
        assert_eq!(
            view.clocks
                .iter()
                .map(|c| c.participant.as_str())
                .collect::<Vec<_>>(),
            ["ur10e", "policy"]
        );
        assert_eq!(
            view.demands.len(),
            1,
            "its applied stream's, not the camera's"
        );
        assert_eq!(view.generation, 4);
        assert!(
            view.generation(&pid("gone")).is_none(),
            "present members only"
        );
        // What the edge and keyring read agrees with the full state.
        for (channel, who) in [(ChannelId(1), &pid("policy")), (ChannelId(1), &ur)] {
            assert_eq!(AuthorityView::lease(&view, channel), state.lease(channel));
            assert_eq!(AuthorityView::grid(&view, channel), state.grid(channel));
            assert_eq!(AuthorityView::clock(&view, who), state.clock(who));
            assert_eq!(
                AuthorityView::generation(&view, who),
                AuthorityView::generation(&state, who)
            );
        }
        assert_eq!(
            AuthorityView::grid(&view, ChannelId(1)),
            Some(Duration::from_millis(20))
        );
        assert_eq!(
            AuthorityView::check_fence(&view, ChannelId(1), LeaseToken(1), state.epoch),
            state.check_fence(ChannelId(1), LeaseToken(1), state.epoch)
        );
        // A robot that left the roster has no parts, no generation.
        let gone = DeviceView::of(&state, &pid("gone"));
        assert!(gone.parts.is_empty() && gone.leases.is_empty());
        assert_eq!(gone.generation, 0);
    }
}
