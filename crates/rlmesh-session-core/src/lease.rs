//! The holder's side of a lease, sans-IO: every participant that
//! commands a part tracks its lease here. The browser binding holds one
//! [`LeaseClient`] per part; the native controller and the lockstep driver
//! hold their parts together as a [`LeaseGroup`] (one all-or-none request,
//! one client per part).
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
//!   offset error can only shorten it (the edge's rule).
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

#[allow(unused_imports)]
use crate::prelude::*;

use core::fmt;

use crate::wire::{
    ActionChunk, AuthoritySnapshot, ChannelId, Command, CtlError, CtlReply, Epoch, Fingerprint,
    InputRange, Lease, LeaseToken, Mode, ParticipantId, ReasonCode,
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

impl core::error::Error for LeaseError {}

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
        /// The lease of a higher class that took it, when one did.
        by: Option<Lease>,
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

    /// The lease of a higher priority class that took this one away, while
    /// that is why it is revoked: an end the host gives as `PREEMPTED`, or
    /// (from a host that gives none) another holder of a higher class on
    /// the part in the same epoch.
    pub fn preempted_by(&self) -> Option<&Lease> {
        match &self.state {
            State::Revoked { by, .. } => by.as_ref(),
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

    /// Why `snapshot` shows this holder's `lease` (of `epoch`) gone, and
    /// the lease that preempted it, if one did; `None` while it is still
    /// current.
    fn loss(
        &self,
        snapshot: &AuthoritySnapshot,
        epoch: Epoch,
        lease: &Lease,
    ) -> Option<(String, Option<Lease>)> {
        let token = lease.token;
        let held = snapshot
            .lease(self.channel)
            .is_some_and(|now| now.token == token && now.holder == self.me);
        if held && snapshot.epoch == epoch {
            return None;
        }
        // The host says why; older hosts leave it to inference.
        if let Some(end) = snapshot
            .ended
            .iter()
            .rev()
            .find(|e| e.lease.channel == self.channel && e.lease.token == token)
        {
            let by = end
                .by
                .clone()
                .filter(|_| end.reason.code == ReasonCode::Preempted);
            return Some((end.reason.to_string(), by));
        }
        if snapshot.epoch != epoch {
            return Some((format!("epoch {} began", snapshot.epoch), None));
        }
        Some(match snapshot.lease(self.channel) {
            Some(now) if now.holder == self.me => (
                format!("re-issued under {} (the binding changed)", now.token),
                None,
            ),
            Some(now) => (
                format!(
                    "preempted by {} ({:?}, {})",
                    now.holder, now.priority, now.token
                ),
                (now.priority > lease.priority).then(|| now.clone()),
            ),
            None if snapshot.mode == Mode::Safe => ("lapsed: the session is SAFE".into(), None),
            None if snapshot.mode == Mode::Estop => ("the session is in ESTOP".into(), None),
            None => ("ended by the host".into(), None),
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
        let (lease, (reason, by)) = match &self.state {
            State::Held(held) => match self.loss(snapshot, held.epoch, &held.lease) {
                Some(loss) => (held.lease.clone(), loss),
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
            State::Expired { lease, epoch } => (lease.clone(), self.loss(snapshot, *epoch, lease)?),
            _ => return None,
        };
        self.state = State::Revoked {
            lease: Some(lease),
            reason: reason.clone(),
            by,
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
            to: None,
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

/// A binding's leases, as a controller holds them: a `GrantLease` for every
/// part ([`Self::request`]) or for one robot's ([`Self::request_some`]),
/// then one [`LeaseClient`] per part for its deadline, renewal, loss and
/// release.
#[derive(Clone, Debug)]
pub struct LeaseGroup {
    clients: Vec<LeaseClient>,
}

impl LeaseGroup {
    /// `me`'s leases on `channels`; see [`LeaseClient::new`] for the margin.
    pub fn new(me: &ParticipantId, channels: &[ChannelId], margin_ns: i64) -> Self {
        Self {
            clients: channels
                .iter()
                .map(|channel| LeaseClient::new(me.clone(), *channel, margin_ns))
                .collect(),
        }
    }

    /// The parts, in order.
    pub fn channels(&self) -> Vec<ChannelId> {
        self.clients.iter().map(LeaseClient::channel).collect()
    }

    /// Starts a request for every part at once: the one `GrantLease` verb,
    /// sent at `now_ns`.
    pub fn request(&mut self, binding: Fingerprint, ttl_ns: Option<u64>, now_ns: i64) -> Command {
        for client in &mut self.clients {
            client.request(binding, ttl_ns, now_ns);
        }
        Command::GrantLease {
            channels: self.channels(),
            release: Vec::new(),
            binding,
            ttl_ns,
        }
    }

    /// Starts a request for `channels` only (one robot's parts): their
    /// `GrantLease` verb, sent at `now_ns`. The other parts keep what they
    /// hold.
    pub fn request_some(
        &mut self,
        channels: &[ChannelId],
        binding: Fingerprint,
        ttl_ns: Option<u64>,
        now_ns: i64,
    ) -> Command {
        for client in self
            .clients
            .iter_mut()
            .filter(|c| channels.contains(&c.channel()))
        {
            client.request(binding, ttl_ns, now_ns);
        }
        Command::GrantLease {
            channels: channels.to_vec(),
            release: Vec::new(),
            binding,
            ttl_ns,
        }
    }

    /// The host's reply to [`Self::request_some`] for `channels`. Returns
    /// whether every one of them is now held.
    pub fn on_reply_some(
        &mut self,
        channels: &[ChannelId],
        reply: &CtlReply,
        clock: Option<&OffsetEstimate>,
        now_ns: i64,
    ) -> bool {
        let mut all = true;
        for client in self
            .clients
            .iter_mut()
            .filter(|c| channels.contains(&c.channel()))
        {
            all &= client.on_reply(reply, clock, now_ns);
        }
        all
    }

    /// The parts not held at `now_ns`, in order.
    pub fn unheld(&mut self, now_ns: i64) -> Vec<ChannelId> {
        self.clients
            .iter_mut()
            .filter_map(|client| {
                client.tick(now_ns);
                client.lease().is_none().then(|| client.channel())
            })
            .collect()
    }

    /// The host's reply to the request. Returns whether every lease is now
    /// held.
    pub fn on_reply(
        &mut self,
        reply: &CtlReply,
        clock: Option<&OffsetEstimate>,
        now_ns: i64,
    ) -> bool {
        // Every client sees the reply: no short circuit.
        let mut all = true;
        for client in &mut self.clients {
            all &= client.on_reply(reply, clock, now_ns);
        }
        all
    }

    /// An authority snapshot, applied to every part: what it did to each.
    pub fn on_snapshot(
        &mut self,
        snapshot: &AuthoritySnapshot,
        clock: Option<&OffsetEstimate>,
        now_ns: i64,
    ) -> Vec<(ChannelId, LeaseChange)> {
        self.clients
            .iter_mut()
            .filter_map(|client| {
                let change = client.on_snapshot(snapshot, clock, now_ns)?;
                Some((client.channel(), change))
            })
            .collect()
    }

    /// The lease of a higher class that took `channel` away, while that is
    /// why it is revoked ([`LeaseClient::preempted_by`]).
    pub fn preempted_by(&self, channel: ChannelId) -> Option<&Lease> {
        self.clients
            .iter()
            .find(|c| c.channel() == channel)?
            .preempted_by()
    }

    /// The lease on `channel`, while held at `now_ns`.
    pub fn lease(&mut self, channel: ChannelId, now_ns: i64) -> Option<&Lease> {
        let client = self.clients.iter_mut().find(|c| c.channel() == channel)?;
        client.tick(now_ns);
        client.lease()
    }

    /// The leases held at `now_ns`.
    pub fn held(&mut self, now_ns: i64) -> Vec<Lease> {
        self.clients
            .iter_mut()
            .filter_map(|client| {
                client.tick(now_ns);
                client.lease().cloned()
            })
            .collect()
    }

    /// Whether every part's lease is held at `now_ns`.
    pub fn holds_all(&mut self, now_ns: i64) -> bool {
        self.clients.iter_mut().all(|client| {
            client.tick(now_ns);
            client.lease().is_some()
        })
    }

    /// Gives every lease back: one `ReleaseLease` verb per token still held
    /// (or expired locally but maybe still held at the host).
    pub fn release(&mut self) -> Vec<Command> {
        let mut by_token: Vec<(LeaseToken, Vec<ChannelId>)> = Vec::new();
        for client in &mut self.clients {
            let Some(Command::ReleaseLease {
                channels, token, ..
            }) = client.release()
            else {
                continue;
            };
            match by_token.iter_mut().find(|(t, _)| *t == token) {
                Some((_, all)) => all.extend(channels),
                None => by_token.push((token, channels)),
            }
        }
        by_token
            .into_iter()
            .map(|(token, channels)| Command::ReleaseLease {
                channels,
                token,
                to: None,
            })
            .collect()
    }

    /// Marks the leases on `channels` given back, with no verb: the caller
    /// sent its own release (a hand-back) and the host took it.
    pub fn released(&mut self, channels: &[ChannelId]) {
        for client in self
            .clients
            .iter_mut()
            .filter(|c| channels.contains(&c.channel()))
        {
            let _ = client.release();
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeMap;

    use crate::wire::{
        ClockMode, CtlStatus, LastStop, LeaseEnd, PriorityClass, Reason, ReasonCode, SessionId,
    };

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
            key_id: 0,
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
            mode_reason: Reason::default(),
            ended: Vec::new(),
            last_stop: LastStop::default(),
            places: Vec::new(),
            scene: None,
            held: Vec::new(),
        }
    }

    fn granted(client: &mut LeaseClient, clock: Option<&OffsetEstimate>) {
        let verb = client.request(Fingerprint::from(7), None, 0);
        assert!(
            matches!(verb, Command::GrantLease { ref channels, .. } if channels == &[ChannelId(3)])
        );
        let reply = CtlReply {
            request_id: 0,
            result: Ok(()),
            state: snapshot(5, 0, Mode::Active, vec![lease("web-1", 9, 1_000 * MS)]),
            leases: vec![lease("web-1", 9, 1_000 * MS)],
            keys: Vec::new(),
            assignment: None,
            run: None,
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
                token: LeaseToken(9),
                to: None,
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

        // A host that says why is taken at its word.
        let mut told = LeaseClient::new(me(), ChannelId(3), 0);
        granted(&mut told, None);
        let mut ended = snapshot(6, 1, Mode::Safe, Vec::new());
        ended.ended.push(LeaseEnd {
            lease: lease("web-1", 9, 1_000 * MS),
            reason: Reason {
                code: ReasonCode::HeartbeatLapse,
                by: "web-1".into(),
                ..Reason::default()
            },
            by: None,
            to: None,
        });
        assert_eq!(
            told.on_snapshot(&ended, None, 10 * MS),
            Some(LeaseChange::Revoked("HEARTBEAT_LAPSE by web-1".into()))
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
            request_id: 0,
            result: Err(CtlError::new(CtlStatus::LeaseBusy, "held by safety")),
            state: snapshot(5, 0, Mode::Active, Vec::new()),
            leases: Vec::new(),
            keys: Vec::new(),
            assignment: None,
            run: None,
        };
        assert!(!client.on_reply(&reply, None, MS));
        assert!(
            matches!(client.state(MS), LeaseState::Refused { ref error } if error.status == CtlStatus::LeaseBusy)
        );
        assert_eq!(client.release(), None);
    }

    #[test]
    fn a_group_is_granted_all_or_none_and_released_per_token() {
        let mut group = LeaseGroup::new(&me(), &[ChannelId(3), ChannelId(4)], 0);
        let verb = group.request(Fingerprint::from(7), None, 0);
        assert!(
            matches!(verb, Command::GrantLease { ref channels, .. } if channels == &[ChannelId(3), ChannelId(4)])
        );
        let other = Lease {
            channel: ChannelId(4),
            ..lease("web-1", 9, 1_000 * MS)
        };
        let both = vec![lease("web-1", 9, 1_000 * MS), other.clone()];
        let reply = CtlReply {
            request_id: 0,
            result: Ok(()),
            state: snapshot(5, 0, Mode::Active, both.clone()),
            leases: both,
            keys: Vec::new(),
            assignment: None,
            run: None,
        };
        assert!(group.on_reply(&reply, None, MS));
        assert!(group.holds_all(2 * MS));
        // One part preempted: the group no longer holds all of them.
        let safety = Lease {
            priority: PriorityClass::Safety,
            ..lease("supervisor", 10, 2_000 * MS)
        };
        let changes = group.on_snapshot(
            &snapshot(6, 1, Mode::Active, vec![safety, other]),
            None,
            3 * MS,
        );
        assert!(
            matches!(changes.as_slice(), [(ChannelId(3), LeaseChange::Revoked(r)), (ChannelId(4), LeaseChange::Renewed)] if r.contains("preempted"))
        );
        assert!(!group.holds_all(3 * MS));
        assert!(group.lease(ChannelId(4), 3 * MS).is_some());
        assert_eq!(
            group.release(),
            vec![Command::ReleaseLease {
                channels: vec![ChannelId(4)],
                token: LeaseToken(9),
                to: None,
            }]
        );
    }

    fn ok_reply(leases: Vec<Lease>) -> CtlReply {
        CtlReply {
            request_id: 0,
            result: Ok(()),
            state: snapshot(5, 0, Mode::Active, leases.clone()),
            leases,
            keys: Vec::new(),
            assignment: None,
            run: None,
        }
    }

    /// What `snapshot` does to a lease on part 3 held under token 9.
    fn change(snapshot: &AuthoritySnapshot) -> Option<LeaseChange> {
        let mut client = LeaseClient::new(me(), ChannelId(3), 0);
        granted(&mut client, None);
        client.on_snapshot(snapshot, None, 10 * MS)
    }

    #[test]
    fn a_new_token_for_this_holder_revokes_the_old_lease_as_a_reissue() {
        let reissued = snapshot(6, 1, Mode::Active, vec![lease("web-1", 10, 1_000 * MS)]);
        assert_eq!(
            change(&reissued),
            Some(LeaseChange::Revoked(format!(
                "re-issued under {} (the binding changed)",
                LeaseToken(10)
            )))
        );
    }

    #[test]
    fn the_same_token_under_another_holder_is_not_this_holders_lease() {
        let taken = snapshot(6, 1, Mode::Active, vec![lease("supervisor", 9, 1_000 * MS)]);
        assert!(
            matches!(change(&taken), Some(LeaseChange::Revoked(ref r)) if r.contains("preempted by supervisor"))
        );
    }

    #[test]
    fn a_missing_lease_is_explained_by_the_mode() {
        for (mode, why) in [
            (Mode::Active, "ended by the host"),
            (Mode::Safe, "lapsed: the session is SAFE"),
            (Mode::Estop, "the session is in ESTOP"),
        ] {
            assert_eq!(
                change(&snapshot(6, 1, mode, Vec::new())),
                Some(LeaseChange::Revoked(why.into())),
                "{mode:?}"
            );
        }
    }

    #[test]
    fn only_the_end_of_this_part_under_this_token_explains_the_loss() {
        let end = |ended: Lease| LeaseEnd {
            lease: ended,
            reason: Reason {
                code: ReasonCode::HeartbeatLapse,
                by: "web-1".into(),
                ..Reason::default()
            },
            by: None,
            to: None,
        };
        let mut gone = snapshot(6, 1, Mode::Active, Vec::new());
        gone.ended.push(end(Lease {
            channel: ChannelId(4),
            ..lease("web-1", 9, 1_000 * MS)
        }));
        gone.ended.push(end(lease("web-1", 8, 1_000 * MS)));
        assert_eq!(
            change(&gone),
            Some(LeaseChange::Revoked("ended by the host".into()))
        );
    }

    /// The preempting lease, after `snapshot` revoked the lease on part 3
    /// held under token 9 (an operator's).
    fn preempted_by(snapshot: &AuthoritySnapshot) -> Option<Lease> {
        let mut client = LeaseClient::new(me(), ChannelId(3), 0);
        granted(&mut client, None);
        assert!(matches!(
            client.on_snapshot(snapshot, None, 10 * MS),
            Some(LeaseChange::Revoked(_))
        ));
        client.preempted_by().cloned()
    }

    #[test]
    fn only_a_higher_class_taking_the_part_is_a_preemption() {
        let supervisor = Lease {
            priority: PriorityClass::Safety,
            ..lease("supervisor", 10, 1_000 * MS)
        };
        let end = |code: ReasonCode, by: Option<Lease>| LeaseEnd {
            lease: lease("web-1", 9, 1_000 * MS),
            reason: Reason {
                code,
                by: "supervisor".into(),
                ..Reason::default()
            },
            by,
            to: None,
        };
        // The host's own account.
        let mut taken = snapshot(6, 1, Mode::Active, vec![supervisor.clone()]);
        taken
            .ended
            .push(end(ReasonCode::Preempted, Some(supervisor.clone())));
        assert_eq!(preempted_by(&taken), Some(supervisor.clone()));
        // Lapsed, then taken by someone else: not a preemption.
        let mut lapsed = snapshot(6, 1, Mode::Active, vec![supervisor.clone()]);
        lapsed.ended.push(end(ReasonCode::HeartbeatLapse, None));
        assert_eq!(preempted_by(&lapsed), None);
        // No account: a higher class on the part took it, the same class
        // only followed it.
        let inferred = snapshot(6, 1, Mode::Active, vec![supervisor.clone()]);
        assert_eq!(preempted_by(&inferred), Some(supervisor));
        let same_class = snapshot(6, 1, Mode::Active, vec![lease("web-2", 10, 1_000 * MS)]);
        assert_eq!(preempted_by(&same_class), None);
        // A new request forgets it.
        let mut client = LeaseClient::new(me(), ChannelId(3), 0);
        granted(&mut client, None);
        client.on_snapshot(&taken, None, 10 * MS);
        assert!(client.preempted_by().is_some());
        client.request(Fingerprint::from(7), None, 20 * MS);
        assert!(client.preempted_by().is_none());
    }

    #[test]
    fn a_reply_grants_a_client_only_its_own_part_and_holder() {
        let other_part = Lease {
            channel: ChannelId(4),
            ..lease("web-1", 8, 1_000 * MS)
        };
        let other_holder = lease("supervisor", 7, 1_000 * MS);
        let mut client = LeaseClient::new(me(), ChannelId(3), 0);
        client.request(Fingerprint::from(7), None, 0);
        let reply = ok_reply(vec![
            other_part,
            other_holder,
            lease("web-1", 9, 1_000 * MS),
        ]);
        assert!(client.on_reply(&reply, None, MS));
        assert_eq!(client.token(), LeaseToken(9));
    }

    #[test]
    fn a_group_reply_missing_a_part_is_not_all_held() {
        let parts = [ChannelId(3), ChannelId(4)];
        let reply = ok_reply(vec![lease("web-1", 9, 1_000 * MS)]);
        let mut group = LeaseGroup::new(&me(), &parts, 0);
        group.request(Fingerprint::from(7), None, 0);
        assert!(!group.on_reply(&reply, None, MS));
        assert_eq!(group.unheld(MS), vec![ChannelId(4)]);
        let mut some = LeaseGroup::new(&me(), &parts, 0);
        some.request_some(&parts, Fingerprint::from(7), None, 0);
        assert!(!some.on_reply_some(&parts, &reply, None, MS));
    }

    #[test]
    fn a_group_releases_one_verb_per_token_with_all_its_parts() {
        let parts = [ChannelId(3), ChannelId(4), ChannelId(5)];
        let on = |channel, token| Lease {
            channel,
            ..lease("web-1", token, 1_000 * MS)
        };
        let mut group = LeaseGroup::new(&me(), &parts, 0);
        group.request(Fingerprint::from(7), None, 0);
        let reply = ok_reply(vec![
            on(ChannelId(3), 9),
            on(ChannelId(4), 9),
            on(ChannelId(5), 10),
        ]);
        assert!(group.on_reply(&reply, None, MS));
        assert_eq!(
            group.release(),
            vec![
                Command::ReleaseLease {
                    channels: vec![ChannelId(3), ChannelId(4)],
                    token: LeaseToken(9),
                    to: None,
                },
                Command::ReleaseLease {
                    channels: vec![ChannelId(5)],
                    token: LeaseToken(10),
                    to: None,
                }
            ]
        );
    }

    #[test]
    fn a_group_reply_granting_every_part_asked_for_holds_them_all() {
        let mut group = LeaseGroup::new(&me(), &[ChannelId(3), ChannelId(4)], 0);
        group.request_some(&[ChannelId(3)], Fingerprint::from(7), None, 0);
        let reply = ok_reply(vec![lease("web-1", 9, 1_000 * MS)]);
        assert!(group.on_reply_some(&[ChannelId(3)], &reply, None, MS));
        assert_eq!(group.unheld(MS), vec![ChannelId(4)]);
    }
}
