//! The holder's side of one lease (§9.7), sans-IO: a participant that
//! commands a part without the controller runtime (the browser binding,
//! through its gateway) tracks its lease here.
//!
//! The host is the only authority: this client never grants, renews or
//! extends anything on its own. It turns the host's answers into a local
//! view the holder can act on:
//!
//! - **Request.** [`LeaseClient::request`] builds the `GrantLease` verb; the
//!   host's [`CtlReply`] grants it ([`LeaseClient::on_reply`]) or refuses
//!   it.
//! - **Deadline.** The lease is usable until a *local* deadline that is a
//!   lower bound of the host's expiry in the holder's clock: the request's
//!   send time plus the ttl (the grant came after the request left), and the
//!   host's `expires_ns` converted conservatively, `expires − θ − ε`, so an
//!   offset error can only shorten it (the edge's rule, §9.1).
//! - **Renewal.** A newer authority snapshot carrying the same token moves
//!   the deadline forward. Once the deadline passes the lease is expired for
//!   good: a late snapshot never resurrects it, and only a new request (a new
//!   token) commands again.
//! - **Loss.** A newer snapshot with another token on the part (preempted),
//!   none (lapsed, released by the host, SAFE) or a new epoch revokes it.
//! - **Fencing.** Every chunk is stamped with the lease's token, binding and
//!   epoch and a seq increasing per token; the edge fences the rest.
//!
//! Times are nanoseconds: `now_ns` in the holder's clock, `expires_ns` in
//! session time.

use std::fmt;

use crate::wire::{
    ActionChunk, AuthoritySnapshot, ChannelId, Command, CtlError, CtlReply, Epoch, Fingerprint,
    InputRange, Lease, LeaseToken, Mode, ParticipantId,
};
use bytes::Bytes;

use crate::clock::OffsetEstimate;

/// Where a lease stands, as its holder sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LeaseState {
    /// Nothing requested.
    Idle,
    /// A `GrantLease` is in flight.
    Requested,
    /// Held: chunks may be sent until the local deadline.
    Held,
    /// Held, but within the renewal margin of the local deadline.
    Expiring,
    /// The local deadline passed. Terminal until a new request.
    Expired,
    /// The host took it away (preempted, lapsed, a new epoch).
    Revoked {
        /// Why, as the snapshot shows it.
        reason: String,
    },
    /// The host refused the request.
    Refused {
        /// The refusal.
        error: CtlError,
    },
}

impl LeaseState {
    /// The state's lowercase name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Requested => "requested",
            Self::Held => "held",
            Self::Expiring => "expiring",
            Self::Expired => "expired",
            Self::Revoked { .. } => "revoked",
            Self::Refused { .. } => "refused",
        }
    }
}

/// Why the holder may not send.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LeaseError {
    /// No lease is held (the state's name).
    NotHeld(&'static str),
    /// The local deadline passed.
    Expired,
}

impl fmt::Display for LeaseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotHeld(state) => write!(f, "no lease held ({state})"),
            Self::Expired => f.write_str("the lease expired by the local clock"),
        }
    }
}

impl std::error::Error for LeaseError {}

/// What a snapshot did to a held lease.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LeaseChange {
    /// The same token, with a later (or equal) host expiry.
    Renewed,
    /// Gone: the reason.
    Revoked(String),
}

#[derive(Clone, Debug)]
struct Held {
    lease: Lease,
    epoch: Epoch,
    deadline_ns: i64,
    seq: u64,
}

#[derive(Clone, Debug)]
enum State {
    Idle,
    Requested {
        sent_ns: i64,
    },
    Held(Held),
    Expired {
        lease: Lease,
        epoch: Epoch,
    },
    Revoked {
        lease: Option<Lease>,
        reason: String,
    },
    Refused {
        error: CtlError,
    },
}

/// One part's lease, from its holder's side: see the module docs.
#[derive(Clone, Debug)]
pub struct LeaseClient {
    me: ParticipantId,
    channel: ChannelId,
    margin_ns: i64,
    state: State,
    /// The newest snapshot taken, as `(revision, host_ns)`.
    seen: Option<(u64, i64)>,
}

/// `expires` (session time) as a lower bound in the holder's clock.
fn local_bound(expires_ns: i64, clock: Option<&OffsetEstimate>) -> Option<i64> {
    let clock = clock?;
    let error = i64::try_from(clock.error_ns).unwrap_or(i64::MAX);
    Some(
        expires_ns
            .saturating_sub(clock.offset_ns)
            .saturating_sub(error),
    )
}

impl LeaseClient {
    /// A client for `me`'s lease on `channel`; the lease is `Expiring` once
    /// `margin_ns` or less is left.
    pub fn new(me: ParticipantId, channel: ChannelId, margin_ns: i64) -> Self {
        Self {
            me,
            channel,
            margin_ns: margin_ns.max(0),
            state: State::Idle,
            seen: None,
        }
    }

    /// The part.
    pub fn channel(&self) -> ChannelId {
        self.channel
    }

    /// The state at `now_ns` (expiry applied).
    pub fn state(&mut self, now_ns: i64) -> LeaseState {
        self.tick(now_ns);
        match &self.state {
            State::Idle => LeaseState::Idle,
            State::Requested { .. } => LeaseState::Requested,
            State::Held(held) if held.deadline_ns.saturating_sub(now_ns) <= self.margin_ns => {
                LeaseState::Expiring
            }
            State::Held(_) => LeaseState::Held,
            State::Expired { .. } => LeaseState::Expired,
            State::Revoked { reason, .. } => LeaseState::Revoked {
                reason: reason.clone(),
            },
            State::Refused { error } => LeaseState::Refused {
                error: error.clone(),
            },
        }
    }

    /// The lease as granted (and as last renewed), while held.
    pub fn lease(&self) -> Option<&Lease> {
        match &self.state {
            State::Held(held) => Some(&held.lease),
            _ => None,
        }
    }

    /// The token of the lease this client holds or last held (zero: none).
    pub fn token(&self) -> LeaseToken {
        match &self.state {
            State::Held(held) => held.lease.token,
            State::Expired { lease, .. }
            | State::Revoked {
                lease: Some(lease), ..
            } => lease.token,
            _ => LeaseToken(0),
        }
    }

    /// The local deadline, while held.
    pub fn deadline_ns(&self) -> Option<i64> {
        match &self.state {
            State::Held(held) => Some(held.deadline_ns),
            _ => None,
        }
    }

    /// Applies the local deadline.
    pub fn tick(&mut self, now_ns: i64) {
        if let State::Held(held) = &self.state
            && now_ns >= held.deadline_ns
        {
            self.state = State::Expired {
                lease: held.lease.clone(),
                epoch: held.epoch,
            };
        }
    }

    /// Starts a request: the `GrantLease` verb for this part under
    /// `binding`, sent at `now_ns`. Any lease held before is dropped
    /// locally (release it first to tell the host).
    pub fn request(&mut self, binding: Fingerprint, ttl_ns: Option<u64>, now_ns: i64) -> Command {
        self.state = State::Requested { sent_ns: now_ns };
        Command::GrantLease {
            channels: vec![self.channel],
            release: Vec::new(),
            binding,
            ttl_ns,
        }
    }

    /// The host's reply to the request. Returns whether the lease is now
    /// held.
    pub fn on_reply(
        &mut self,
        reply: &CtlReply,
        clock: Option<&OffsetEstimate>,
        now_ns: i64,
    ) -> bool {
        let State::Requested { sent_ns } = self.state else {
            return false;
        };
        self.observe(&reply.state);
        if let Err(error) = &reply.result {
            self.state = State::Refused {
                error: error.clone(),
            };
            return false;
        }
        let Some(lease) = reply
            .leases
            .iter()
            .find(|l| l.channel == self.channel && l.holder == self.me)
        else {
            return false;
        };
        let by_request = sent_ns.saturating_add_unsigned(lease.ttl_ns);
        let deadline_ns =
            local_bound(lease.expires_ns, clock).map_or(by_request, |b| b.max(by_request));
        self.state = State::Held(Held {
            lease: lease.clone(),
            epoch: reply.state.epoch,
            deadline_ns,
            seq: 0,
        });
        self.tick(now_ns);
        matches!(self.state, State::Held(_))
    }

    /// Records `snapshot`'s order; false when it is not newer than one
    /// already taken.
    fn observe(&mut self, snapshot: &AuthoritySnapshot) -> bool {
        let order = (snapshot.revision, snapshot.host_ns);
        if self.seen.is_some_and(|seen| order <= seen) {
            return false;
        }
        self.seen = Some(order);
        true
    }

    /// Why `snapshot` shows this holder's lease `token` (of `epoch`) gone,
    /// or `None` while it is still current.
    fn loss(
        &self,
        snapshot: &AuthoritySnapshot,
        epoch: Epoch,
        token: LeaseToken,
    ) -> Option<String> {
        if snapshot.epoch != epoch {
            return Some(format!("epoch {} began", snapshot.epoch));
        }
        Some(match snapshot.lease(self.channel) {
            Some(now) if now.token == token && now.holder == self.me => return None,
            Some(now) if now.holder == self.me => {
                format!("re-issued under {} (the binding changed)", now.token)
            }
            Some(now) => format!(
                "preempted by {} ({:?}, {})",
                now.holder, now.priority, now.token
            ),
            None if snapshot.mode == Mode::Safe => "lapsed: the session is SAFE".into(),
            None if snapshot.mode == Mode::Estop => "the session is in ESTOP".into(),
            None => "ended by the host".into(),
        })
    }

    /// An authority snapshot (an update, or a reply's state). Older ones
    /// than the newest taken are ignored. Renews or revokes a held lease,
    /// and revokes one that expired locally once the host shows it gone
    /// (it is never renewed).
    pub fn on_snapshot(
        &mut self,
        snapshot: &AuthoritySnapshot,
        clock: Option<&OffsetEstimate>,
        now_ns: i64,
    ) -> Option<LeaseChange> {
        if !self.observe(snapshot) {
            return None;
        }
        self.tick(now_ns);
        let (lease, reason) = match &self.state {
            State::Held(held) => match self.loss(snapshot, held.epoch, held.lease.token) {
                Some(reason) => (held.lease.clone(), reason),
                None => {
                    let now = snapshot.lease(self.channel)?.clone();
                    let bound = local_bound(now.expires_ns, clock);
                    if let State::Held(held) = &mut self.state {
                        if let Some(bound) = bound {
                            held.deadline_ns = held.deadline_ns.max(bound);
                        }
                        held.lease = now;
                    }
                    return Some(LeaseChange::Renewed);
                }
            },
            State::Expired { lease, epoch } => {
                (lease.clone(), self.loss(snapshot, *epoch, lease.token)?)
            }
            _ => return None,
        };
        self.state = State::Revoked {
            lease: Some(lease),
            reason: reason.clone(),
        };
        Some(LeaseChange::Revoked(reason))
    }

    /// Stamps one chunk of `count` actions (packed at the command stride)
    /// with the lease's token, binding and epoch, the next seq and `source`
    /// (this holder), or refuses when the lease is not held at `now_ns`.
    /// Chunk times are session time; `publish_ns` is the holder's clock.
    #[allow(clippy::too_many_arguments)]
    pub fn chunk(
        &mut self,
        actions: Bytes,
        count: u32,
        t0_ns: i64,
        dt_ns: u64,
        valid_until_ns: i64,
        based_on: Vec<InputRange>,
        now_ns: i64,
    ) -> Result<ActionChunk, LeaseError> {
        self.tick(now_ns);
        match &mut self.state {
            State::Held(held) => {
                held.seq += 1;
                Ok(ActionChunk {
                    channel: self.channel,
                    epoch: held.epoch,
                    lease_token: held.lease.token,
                    seq: held.seq,
                    source: self.me.clone(),
                    binding: held.lease.binding,
                    based_on,
                    t0_ns,
                    dt_ns,
                    replace_from_ns: t0_ns,
                    valid_until_ns,
                    count,
                    actions,
                    publish_ns: now_ns,
                    tick: 0,
                })
            }
            State::Expired { .. } => Err(LeaseError::Expired),
            _ => Err(LeaseError::NotHeld(self.state_name())),
        }
    }

    fn state_name(&self) -> &'static str {
        match &self.state {
            State::Idle => "idle",
            State::Requested { .. } => "requested",
            State::Held(_) => "held",
            State::Expired { .. } => "expired",
            State::Revoked { .. } => "revoked",
            State::Refused { .. } => "refused",
        }
    }

    /// Gives the lease back: the `ReleaseLease` verb, when one was held (or
    /// had expired locally but may still be held at the host).
    pub fn release(&mut self) -> Option<Command> {
        let token = match &self.state {
            State::Held(held) => held.lease.token,
            State::Expired { lease, .. } => lease.token,
            _ => {
                self.state = State::Idle;
                return None;
            }
        };
        self.state = State::Idle;
        Some(Command::ReleaseLease {
            channels: vec![self.channel],
            token,
        })
    }

    /// Drops the lease locally without telling the host (the connection
    /// that held it is gone, so its token must never be used again).
    /// Returns the token it held.
    pub fn forget(&mut self) -> LeaseToken {
        let token = self.token();
        self.state = State::Idle;
        token
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::wire::{ClockMode, CtlStatus, PriorityClass, SessionId};

    use super::*;

    const MS: i64 = 1_000_000;

    fn me() -> ParticipantId {
        ParticipantId::new("web-1").expect("id")
    }

    fn lease(holder: &str, token: u64, expires_ns: i64) -> Lease {
        Lease {
            channel: ChannelId(3),
            holder: ParticipantId::new(holder).expect("id"),
            priority: PriorityClass::Operator,
            token: LeaseToken(token),
            ttl_ns: 250 * MS as u64,
            granted_ns: 0,
            expires_ns,
            binding: Fingerprint::from(7),
        }
    }

    fn snapshot(revision: u64, host_ns: i64, mode: Mode, leases: Vec<Lease>) -> AuthoritySnapshot {
        AuthoritySnapshot {
            session: SessionId::new(),
            epoch: Epoch(1),
            revision,
            mode,
            estop_latched: false,
            roster: Vec::new(),
            leases,
            clocks: Vec::new(),
            host_ns,
            parts: Vec::new(),
            protocol_minor: 0,
            supports: Vec::new(),
            clock_mode: ClockMode::Wall,
            tick_ns: 0,
            bindings: Vec::new(),
            contract: BTreeMap::new(),
            demands: Vec::new(),
        }
    }

    fn granted(client: &mut LeaseClient, clock: Option<&OffsetEstimate>) {
        let verb = client.request(Fingerprint::from(7), None, 0);
        assert!(
            matches!(verb, Command::GrantLease { ref channels, .. } if channels == &[ChannelId(3)])
        );
        let reply = CtlReply {
            result: Ok(()),
            state: snapshot(5, 0, Mode::Active, vec![lease("web-1", 9, 1_000 * MS)]),
            leases: vec![lease("web-1", 9, 1_000 * MS)],
        };
        assert!(client.on_reply(&reply, clock, MS));
    }

    /// Host time = local + 100 ms, within ± 1 ms.
    fn clock() -> OffsetEstimate {
        OffsetEstimate {
            offset_ns: 100 * MS,
            error_ns: MS as u64,
            rtt_ns: 0,
            sample_ns: 0,
        }
    }

    #[test]
    fn the_deadline_is_a_conservative_lower_bound_of_the_host_expiry() {
        let mut client = LeaseClient::new(me(), ChannelId(3), 50 * MS);
        granted(&mut client, None);
        // Without a clock mapping: the request's send time plus the ttl.
        assert_eq!(client.deadline_ns(), Some(250 * MS));
        let mut mapped = LeaseClient::new(me(), ChannelId(3), 50 * MS);
        granted(&mut mapped, Some(&clock()));
        // expires − θ − ε = 1000 − 100 − 1 ms.
        assert_eq!(mapped.deadline_ns(), Some(899 * MS));
        assert_eq!(mapped.state(800 * MS), LeaseState::Held);
        assert_eq!(mapped.state(860 * MS), LeaseState::Expiring);
        assert_eq!(mapped.state(899 * MS), LeaseState::Expired);
    }

    #[test]
    fn renewals_move_the_deadline_forward_but_never_resurrect_an_expired_lease() {
        let mut client = LeaseClient::new(me(), ChannelId(3), 0);
        granted(&mut client, Some(&clock()));
        let renewed = snapshot(6, 1, Mode::Active, vec![lease("web-1", 9, 1_200 * MS)]);
        assert_eq!(
            client.on_snapshot(&renewed, Some(&clock()), 500 * MS),
            Some(LeaseChange::Renewed)
        );
        assert_eq!(client.deadline_ns(), Some(1_099 * MS));
        // An older snapshot (an earlier expiry) is ignored.
        let stale = snapshot(6, 0, Mode::Active, vec![lease("web-1", 9, 1_000 * MS)]);
        assert_eq!(client.on_snapshot(&stale, Some(&clock()), 600 * MS), None);
        assert_eq!(client.deadline_ns(), Some(1_099 * MS));
        // Past the deadline: expired, and a late renewal does not bring it back.
        assert_eq!(client.state(1_100 * MS), LeaseState::Expired);
        let late = snapshot(7, 2, Mode::Active, vec![lease("web-1", 9, 5_000 * MS)]);
        assert_eq!(client.on_snapshot(&late, Some(&clock()), 1_101 * MS), None);
        assert_eq!(client.state(1_101 * MS), LeaseState::Expired);
        assert_eq!(
            client.chunk(
                Bytes::from_static(&[0; 8]),
                1,
                0,
                0,
                0,
                Vec::new(),
                1_101 * MS
            ),
            Err(LeaseError::Expired)
        );
        // Releasing still names the token: the host may hold it a little longer.
        assert_eq!(
            client.release(),
            Some(Command::ReleaseLease {
                channels: vec![ChannelId(3)],
                token: LeaseToken(9)
            })
        );
    }

    #[test]
    fn a_lease_that_expired_locally_is_revoked_once_the_host_ends_it() {
        let mut client = LeaseClient::new(me(), ChannelId(3), 0);
        granted(&mut client, None);
        assert_eq!(client.state(300 * MS), LeaseState::Expired);
        // Still the host's (it times out a little later): stays expired.
        let still = snapshot(6, 1, Mode::Active, vec![lease("web-1", 9, 1_000 * MS)]);
        assert_eq!(client.on_snapshot(&still, None, 301 * MS), None);
        assert_eq!(client.state(301 * MS), LeaseState::Expired);
        let lapsed = snapshot(7, 2, Mode::Safe, Vec::new());
        assert_eq!(
            client.on_snapshot(&lapsed, None, 400 * MS),
            Some(LeaseChange::Revoked("lapsed: the session is SAFE".into()))
        );
        assert!(matches!(client.state(400 * MS), LeaseState::Revoked { .. }));
        assert_eq!(client.token(), LeaseToken(9));
    }

    #[test]
    fn preemption_lapse_and_a_new_epoch_revoke() {
        let mut preempted = LeaseClient::new(me(), ChannelId(3), 0);
        granted(&mut preempted, None);
        let safety = Lease {
            priority: PriorityClass::Safety,
            ..lease("supervisor", 10, 2_000 * MS)
        };
        let change =
            preempted.on_snapshot(&snapshot(6, 1, Mode::Active, vec![safety]), None, 10 * MS);
        assert!(
            matches!(change, Some(LeaseChange::Revoked(ref r)) if r.contains("preempted by supervisor"))
        );
        assert_eq!(preempted.token(), LeaseToken(9));
        assert!(matches!(
            preempted.chunk(Bytes::from_static(&[0; 8]), 1, 0, 0, 0, Vec::new(), 11 * MS),
            Err(LeaseError::NotHeld("revoked"))
        ));

        let mut lapsed = LeaseClient::new(me(), ChannelId(3), 0);
        granted(&mut lapsed, None);
        let change = lapsed.on_snapshot(&snapshot(6, 1, Mode::Safe, Vec::new()), None, 10 * MS);
        assert_eq!(
            change,
            Some(LeaseChange::Revoked("lapsed: the session is SAFE".into()))
        );

        let mut epoch = LeaseClient::new(me(), ChannelId(3), 0);
        granted(&mut epoch, None);
        let mut next = snapshot(6, 1, Mode::Active, vec![lease("web-1", 9, 1_000 * MS)]);
        next.epoch = Epoch(2);
        assert!(matches!(
            epoch.on_snapshot(&next, None, 10 * MS),
            Some(LeaseChange::Revoked(_))
        ));
    }

    #[test]
    fn chunks_carry_the_fence_and_seqs_increase_per_token() {
        let mut client = LeaseClient::new(me(), ChannelId(3), 0);
        let refused = client.chunk(Bytes::new(), 0, 0, 0, 0, Vec::new(), 0);
        assert_eq!(refused, Err(LeaseError::NotHeld("idle")));
        granted(&mut client, None);
        let a = client
            .chunk(Bytes::from_static(&[1; 8]), 1, 5, 0, 9, Vec::new(), 2 * MS)
            .expect("held");
        let b = client
            .chunk(Bytes::from_static(&[2; 8]), 1, 6, 0, 10, Vec::new(), 3 * MS)
            .expect("held");
        assert_eq!((a.seq, b.seq), (1, 2));
        assert_eq!(a.lease_token, LeaseToken(9));
        assert_eq!(a.binding, Fingerprint::from(7));
        assert_eq!(a.source, me());
        assert_eq!(
            (a.epoch, a.replace_from_ns, a.publish_ns),
            (Epoch(1), 5, 2 * MS)
        );
        assert_eq!(client.forget(), LeaseToken(9));
        assert_eq!(client.state(4 * MS), LeaseState::Idle);
    }

    #[test]
    fn a_refusal_is_terminal_until_the_next_request() {
        let mut client = LeaseClient::new(me(), ChannelId(3), 0);
        client.request(Fingerprint::from(7), None, 0);
        let reply = CtlReply {
            result: Err(CtlError::new(CtlStatus::LeaseBusy, "held by safety")),
            state: snapshot(5, 0, Mode::Active, Vec::new()),
            leases: Vec::new(),
        };
        assert!(!client.on_reply(&reply, None, MS));
        assert!(
            matches!(client.state(MS), LeaseState::Refused { ref error } if error.status == CtlStatus::LeaseBusy)
        );
        assert_eq!(client.release(), None);
    }
}
