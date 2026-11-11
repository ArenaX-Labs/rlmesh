//! What the host publishes: the signed state, device views, virtual ticks and lease renewals.

use crate::wire::codec;
use crate::wire::version::{VIEW_DEVICE, VIEW_DEVICE_LEAN, VIEW_DEVICE_SEALED};
use crate::wire::{AuthoritySnapshot, DeviceView, Lease, ParticipantId, Role};
use rlmesh_session_core::crypto::SignedKind;
use rlmesh_session_core::crypto::sealed::Kind;
use tracing::warn;
use zenoh::Wait;
use zenoh::query::Query;

use crate::host::{HostActor, lock};
use crate::transport::{CONTROL_PRIORITY, RESENT_CONGESTION};

/// The largest message a board's transport takes in: zenoh-pico reassembles
/// a fragmented message in a buffer of 4096 bytes (`Z_FRAG_MAX_SIZE`, as
/// boards are built) and drops a larger one without a word.
const BOARD_MESSAGE: usize = 4096;
/// What the Zenoh reply around a payload can take of that: its key
/// expression, ids and extensions.
const REPLY_ENVELOPE: usize = 512;

impl HostActor {
    /// Answers a `host/state` query with the signed state. A board asks it
    /// only for the host instance in its header, and its transport drops a
    /// state larger than [`BOARD_MESSAGE`] unseen, so it would never join.
    /// Such a state is followed by its header alone (signed, with an empty
    /// body). Every other reader keeps the state: one that takes the first
    /// reply gets it first, one that keeps the latest per key (Zenoh's
    /// default) keeps the state's timestamp over none, and a header alone
    /// fails to decode as a state.
    pub(super) async fn on_state_query(&mut self, query: Query) {
        let Some(signed) = self.sign_state(&self.snapshot()) else {
            return Self::discard(query);
        };
        if signed.len() + REPLY_ENVELOPE <= BOARD_MESSAGE {
            return Self::reply(query, signed);
        }
        if let Some(held_back) = self
            .quiet
            .admit("state too large", "", std::time::Instant::now())
        {
            warn!(
                session = %self.config.session,
                bytes = signed.len(),
                board_limit = BOARD_MESSAGE,
                held_back,
                "host_state answered with a header-only follow-up: the state is too large for a board"
            );
        }
        let header = self
            .signer
            .sign(SignedKind::AuthorityState, &self.keys.host_state(), &[]);
        let at = self.transport.control().new_timestamp();
        tokio::task::spawn_blocking(move || {
            let key = query.key_expr().clone();
            let sent = query
                .reply(key.clone(), signed)
                .timestamp(at)
                .express(true)
                .wait()
                .map_err(|err| err.to_string())
                .and_then(|()| header.map_err(|err| err.to_string()))
                .and_then(|header| {
                    query
                        .reply(key, header)
                        .express(true)
                        .wait()
                        .map_err(|err| err.to_string())
                });
            if let Err(err) = sent {
                warn!(%err, "reply failed");
            }
        });
    }

    /// Answers a robot's recovery query on `host/view/{robot}` with its
    /// current view, if it asked for views.
    pub(super) async fn on_view_query(&mut self, query: Query) {
        let Some(robot) = self.keys.view_of(query.key_expr().as_str()) else {
            Self::discard(query);
            return;
        };
        let snapshot = self.snapshot();
        if !asks_for_view(&snapshot, &robot) {
            Self::discard(query);
            return;
        }
        match self.sign_view(&self.view_for(&snapshot, &robot, false)) {
            Some(signed) => Self::reply(query, signed),
            None => Self::discard(query),
        }
    }

    /// `robot`'s view of `snapshot` as it asked for it ([`view_of`]), with
    /// its datagram routes while it is on the path.
    pub(super) fn view_for(
        &self,
        snapshot: &AuthoritySnapshot,
        robot: &ParticipantId,
        parts: bool,
    ) -> DeviceView {
        let mut view = view_of(snapshot, robot, parts);
        view.datagram = self.routes_of(robot);
        view
    }

    /// A robot's view as a `Signed` broadcast (kind 22).
    fn sign_view(&mut self, view: &DeviceView) -> Option<Vec<u8>> {
        self.signer
            .sign(
                SignedKind::DeviceView,
                &self.keys.host_view(&view.robot),
                &codec::encode_view(view),
            )
            .map_err(|err| warn!(robot = %view.robot, %err, "device view not signed"))
            .ok()
    }

    /// The views of every robot that asked for one whose content changed
    /// since the last published (a renewal moves a lease's expiry, a
    /// heartbeat its robot's clock mapping, so one dropped for a congested
    /// link is soon replaced), sealed for a robot that asked for that
    /// ([`Self::send_views`] signs the rest). A change that only refines the last
    /// one sent ([`refines`]) waits until that one is a quarter of the lease
    /// grace old: a board decodes every view on its control loop, and a
    /// renewal or a clock estimate lands on nearly every publish. Anything
    /// else (a lease granted or ended, a mode, a member, a plan) goes at
    /// once.
    fn views_due(&mut self, snapshot: &AuthoritySnapshot) -> Vec<(DeviceView, Option<Vec<u8>>)> {
        self.views.retain(|robot, _| asks_for_view(snapshot, robot));
        let refine = self.refine_ns();
        let mut due = Vec::new();
        for robot in snapshot.roster.iter().map(|p| &p.id) {
            if !asks_for_view(snapshot, robot) {
                continue;
            }
            let view = self.view_for(snapshot, robot, false);
            let wait = self.views.get(robot).is_some_and(|last| {
                let unchanged = *last
                    == DeviceView {
                        host_ns: last.host_ns,
                        ..view.clone()
                    };
                unchanged
                    || refines(last, &view) && view.host_ns.saturating_sub(last.host_ns) < refine
            });
            if wait {
                continue;
            }
            let sealed = asks_for_sealed_view(snapshot, robot)
                .then(|| self.seal_view(&view))
                .flatten();
            self.views.insert(robot.clone(), view.clone());
            due.push((view, sealed));
        }
        due
    }

    /// A robot's view sealed under its own `k_down` (kind 20), for a robot
    /// that asked ([`VIEW_DEVICE_SEALED`]): only it and the host hold the
    /// key, so it opens the view at a symmetric cipher's cost instead of
    /// checking a signature. `None` before its generation is in effect.
    fn seal_view(&self, view: &DeviceView) -> Option<Vec<u8>> {
        let peer = self
            .peers
            .read()
            .ok()
            .and_then(|p| p.current.get(&view.robot).cloned())?;
        lock(&peer.down)
            .seal(
                Kind::DeviceView,
                &self.keys.participant_view(&view.robot),
                &codec::encode_view(view),
            )
            .map_err(|err| warn!(robot = %view.robot, %err, "device view not sealed"))
            .ok()
    }

    /// The snapshot as a `Signed` broadcast.
    fn sign_state(&mut self, snapshot: &AuthoritySnapshot) -> Option<Vec<u8>> {
        self.signer
            .sign(
                SignedKind::AuthorityState,
                &self.keys.host_state(),
                &codec::encode_state(snapshot),
            )
            .map_err(|err| warn!(%err, "authority state not signed"))
            .ok()
    }

    /// Replies ride the query's priority and congestion control (Zenoh 1.x
    /// ignores per-reply QoS), so the caller's choice decides both legs. A
    /// reply to a link that stopped reading can wait there for seconds,
    /// which is why it is sent off the host task.
    pub(super) fn reply(query: Query, payload: Vec<u8>) {
        tokio::task::spawn_blocking(move || {
            if let Err(err) = query
                .reply(query.key_expr().clone(), payload)
                .express(true)
                .wait()
            {
                warn!(%err, "reply failed");
            }
        });
    }

    /// Leaves a query unanswered: dropping one sends its final message,
    /// under the same congestion control as a reply.
    pub(super) fn discard(query: Query) {
        tokio::task::spawn_blocking(move || drop(query));
    }

    pub(super) async fn publish_ticks(&mut self) {
        for tick in self.authority.take_ticks() {
            let signed = self.signer.sign(
                SignedKind::VirtualTick,
                &self.keys.host_tick(),
                &codec::encode_tick(&tick),
            );
            let sent = match signed {
                Ok(signed) => self.ticks.put(signed).await.map_err(|err| err.to_string()),
                Err(err) => Err(err.to_string()),
            };
            if let Err(err) = sent {
                warn!(%err, "tick publish failed");
            }
        }
    }

    pub(super) fn snapshot(&self) -> AuthoritySnapshot {
        let now = self.now();
        let clocks = self
            .links
            .iter()
            .filter_map(|(id, link)| {
                link.estimator.estimate(now).map(|estimate| {
                    estimate.mapping(id.clone(), link.estimator.config().drift_bound_ppm)
                })
            })
            .collect();
        self.authority.snapshot(clocks, now)
    }

    /// How long a refinement of a robot's view waits after the last one
    /// sent: a quarter of the lease grace, the cadence renewals go out at.
    fn refine_ns(&self) -> i64 {
        i64::try_from(self.config.lease_grace.as_nanos() / 4).unwrap_or(i64::MAX)
    }

    /// Heartbeats renew leases without a structural change; while any lease
    /// is held, republish often enough that an edge's local expiry never
    /// runs ahead of the host's renewals: the state every quarter grace,
    /// and the view of a robot with a leased part once its last is that
    /// old.
    pub(super) async fn publish_renewals(&mut self, every_tick: bool) {
        if self.authority.lease_count() == 0 {
            return;
        }
        let quarter = self.refine_ns();
        let now = self.now();
        if every_tick || now.saturating_sub(self.last_publish_ns) >= quarter {
            self.publish().await;
        } else if self
            .views
            .values()
            .any(|last| !last.leases.is_empty() && now.saturating_sub(last.host_ns) >= quarter)
        {
            let snapshot = self.snapshot();
            let views = self.views_due(&snapshot);
            self.send_views(views).await;
        }
    }

    pub(in crate::host) async fn publish(&mut self) {
        self.last_publish_ns = self.now();
        let snapshot = self.snapshot();
        let signed = self.sign_state(&snapshot);
        let views = self.views_due(&snapshot);
        self.snapshot_tx.send_replace(snapshot);
        if let Some(signed) = signed
            && let Err(err) = self.publisher.put(signed).await
        {
            warn!(%err, "authority publish failed");
        }
        self.send_views(views).await;
    }

    /// Publishes each view: sealed to a robot that asked for that, and
    /// otherwise signed on its `host/view/{robot}`, which only that robot
    /// subscribes to. A robot that takes sealed views drops signed ones
    /// unread, so it gets the signed view only when the sealed one could
    /// not be sent (its generation not in effect yet, or a failed put).
    async fn send_views(&mut self, views: Vec<(DeviceView, Option<Vec<u8>>)>) {
        for (view, sealed) in views {
            let robot = view.robot.clone();
            if let Some(sealed) = sealed {
                let sent = self
                    .transport
                    .control()
                    .put(self.keys.participant_view(&robot), sealed)
                    .priority(CONTROL_PRIORITY)
                    .congestion_control(RESENT_CONGESTION)
                    .express(true)
                    .await;
                match sent {
                    Ok(()) => continue,
                    Err(err) => warn!(%robot, %err, "sealed device view publish failed"),
                }
            }
            let Some(signed) = self.sign_view(&view) else {
                self.views.remove(&robot);
                continue;
            };
            let sent = self
                .transport
                .control()
                .put(self.keys.host_view(&robot), signed)
                .priority(CONTROL_PRIORITY)
                .congestion_control(RESENT_CONGESTION)
                .express(true)
                .await;
            if let Err(err) = sent {
                // Sent again with the next publish.
                self.views.remove(&robot);
                warn!(%robot, %err, "device view publish failed");
            }
        }
    }
}

/// Whether `next` differs from `last` only by refinements: each lease's
/// expiry, moved by its holder's renewals, the clock mappings of the same
/// participants re-estimated, and when it was taken. Its revision too: the
/// session's, which anyone's change bumps, and which a robot only orders
/// views by.
fn refines(last: &DeviceView, next: &DeviceView) -> bool {
    let leases = last.leases.len() == next.leases.len()
        && last.leases.iter().zip(&next.leases).all(|(a, b)| {
            *a == Lease {
                expires_ns: a.expires_ns,
                ..b.clone()
            }
        });
    let clocks = last.clocks.len() == next.clocks.len()
        && last
            .clocks
            .iter()
            .zip(&next.clocks)
            .all(|(a, b)| a.participant == b.participant);
    leases
        && clocks
        && *last
            == DeviceView {
                host_ns: last.host_ns,
                revision: last.revision,
                leases: last.leases.clone(),
                clocks: last.clocks.clone(),
                ..next.clone()
            }
}

/// `robot`'s view of `snapshot`, as it asked for it: without its parts,
/// unless `parts`, when it takes them only from the answer to its
/// declaration ([`VIEW_DEVICE_LEAN`]).
pub(super) fn view_of(
    snapshot: &AuthoritySnapshot,
    robot: &ParticipantId,
    parts: bool,
) -> DeviceView {
    let mut view = DeviceView::of(snapshot, robot);
    let lean = snapshot
        .participant(robot)
        .is_some_and(|p| p.supports.iter().any(|f| f == VIEW_DEVICE_LEAN));
    if lean && !parts {
        view.parts = Vec::new();
    }
    view
}

/// Whether `robot` also asked for its view sealed under its `k_down`
/// ([`VIEW_DEVICE_SEALED`]), from a host that lists sealed views among its
/// features.
pub(super) fn asks_for_sealed_view(snapshot: &AuthoritySnapshot, robot: &ParticipantId) -> bool {
    snapshot.supports.iter().any(|f| f == VIEW_DEVICE_SEALED)
        && asks_for_view(snapshot, robot)
        && snapshot
            .participant(robot)
            .is_some_and(|p| p.supports.iter().any(|f| f == VIEW_DEVICE_SEALED))
}

/// Whether `robot` is an embodiment in `snapshot`'s roster that asked for its
/// device view (`Attach.supports`).
pub(super) fn asks_for_view(snapshot: &AuthoritySnapshot, robot: &ParticipantId) -> bool {
    snapshot.participant(robot).is_some_and(|p| {
        p.has_role(Role::Embodiment) && p.supports.iter().any(|f| f == VIEW_DEVICE)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::{
        ChannelId, ClockMapping, Fingerprint, LeaseToken, Member, Mode, PriorityClass,
    };

    fn pid(id: &str) -> ParticipantId {
        ParticipantId::new(id).expect("id")
    }

    fn view() -> DeviceView {
        DeviceView {
            revision: 4,
            host_ns: 100,
            leases: vec![Lease {
                channel: ChannelId(1),
                holder: pid("driver"),
                priority: PriorityClass::Model,
                token: LeaseToken(9),
                ttl_ns: 250,
                granted_ns: 0,
                expires_ns: 300,
                binding: Fingerprint::from(1),
                key_id: 3,
            }],
            clocks: vec![ClockMapping {
                participant: pid("eye-1"),
                offset_ns: 5,
                error_ns: 1,
                rtt_ns: 2,
                sample_ns: 0,
                drift_bound_ppm: 100.0,
            }],
            members: vec![Member {
                id: pid("driver"),
                generation: 1,
            }],
            ..DeviceView::empty(pid("eye-1"))
        }
    }

    #[test]
    fn a_renewal_a_clock_estimate_or_anyone_elses_revision_only_refines_a_view() {
        let last = view();
        let mut next = view();
        next.host_ns = 160;
        next.revision = 7;
        next.leases[0].expires_ns = 360;
        next.clocks[0].offset_ns = 6;
        next.clocks[0].error_ns = 2;
        assert!(refines(&last, &next));

        let changed: [fn(&mut DeviceView); 6] = [
            |v| v.leases[0].token = LeaseToken(10),
            |v| v.leases.clear(),
            |v| v.clocks[0].participant = pid("other"),
            |v| v.mode = Mode::Safe,
            |v| v.members.clear(),
            |v| v.generation = 2,
        ];
        for change in changed {
            let mut next = view();
            change(&mut next);
            assert!(!refines(&last, &next), "{next:?}");
        }
    }
}
