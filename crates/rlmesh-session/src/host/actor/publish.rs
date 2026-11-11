//! What the host publishes: the signed state, device views, virtual ticks and lease renewals.

use crate::wire::codec;
use crate::wire::version::VIEW_DEVICE;
use crate::wire::{AuthoritySnapshot, DeviceView, ParticipantId, Role};
use rlmesh_session_core::crypto::SignedKind;
use tracing::warn;
use zenoh::Wait;
use zenoh::query::Query;

use crate::host::HostActor;
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
        match self.sign_view(&DeviceView::of(&snapshot, &robot)) {
            Some(signed) => Self::reply(query, signed),
            None => Self::discard(query),
        }
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
    /// link is soon replaced), signed.
    fn views_due(&mut self, snapshot: &AuthoritySnapshot) -> Vec<(ParticipantId, Vec<u8>)> {
        self.views.retain(|robot, _| asks_for_view(snapshot, robot));
        let mut due = Vec::new();
        for robot in snapshot.roster.iter().map(|p| &p.id) {
            if !asks_for_view(snapshot, robot) {
                continue;
            }
            let view = DeviceView::of(snapshot, robot);
            let unchanged = self.views.get(robot).is_some_and(|last| {
                *last
                    == DeviceView {
                        host_ns: last.host_ns,
                        ..view.clone()
                    }
            });
            if unchanged {
                continue;
            }
            if let Some(signed) = self.sign_view(&view) {
                due.push((robot.clone(), signed));
                self.views.insert(robot.clone(), view);
            }
        }
        due
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

    /// Heartbeats renew leases without a structural change; while any lease
    /// is held, republish often enough that an edge's local expiry never
    /// runs ahead of the host's renewals.
    pub(super) async fn publish_renewals(&mut self, every_tick: bool) {
        if self.authority.lease_count() == 0 {
            return;
        }
        let quarter = i64::try_from(self.config.lease_grace.as_nanos() / 4).unwrap_or(i64::MAX);
        if every_tick || self.now().saturating_sub(self.last_publish_ns) >= quarter {
            self.publish().await;
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
        for (robot, signed) in views {
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

/// Whether `robot` is an embodiment in `snapshot`'s roster that asked for its
/// device view (`Attach.supports`).
pub(super) fn asks_for_view(snapshot: &AuthoritySnapshot, robot: &ParticipantId) -> bool {
    snapshot.participant(robot).is_some_and(|p| {
        p.has_role(Role::Embodiment) && p.supports.iter().any(|f| f == VIEW_DEVICE)
    })
}
