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

const BOARD_MESSAGE: usize = 4096;
const REPLY_ENVELOPE: usize = 512;

impl HostActor {
    pub(super) async fn on_state_query(&mut self, query: Query) {
        let snapshot = self.snapshot();
        let Some(signed) = self.sign_state(&snapshot) else {
            return Self::discard(query);
        };
        // A board drops a state larger than BOARD_MESSAGE unseen and would never join, so a larger
        // state is followed by its header alone.
        if signed.len() + REPLY_ENVELOPE <= BOARD_MESSAGE {
            return Self::reply(query, signed);
        }
        if has_board(&snapshot.roster)
            && let Some(held_back) =
                self.quiet
                    .admit("state too large", "", std::time::Instant::now())
        {
            warn!(
                session = %self.config.session,
                bytes = signed.len(),
                board_limit = BOARD_MESSAGE - REPLY_ENVELOPE,
                held_back,
                "host_state answered with a header-only follow-up: the signed state is larger than \
                 a board takes in one reply"
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

    fn refine_ns(&self) -> i64 {
        i64::try_from(self.config.lease_grace.as_nanos() / 4).unwrap_or(i64::MAX)
    }

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
                self.views.remove(&robot);
                warn!(%robot, %err, "device view publish failed");
            }
        }
    }
}

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

pub(super) fn asks_for_sealed_view(snapshot: &AuthoritySnapshot, robot: &ParticipantId) -> bool {
    snapshot.supports.iter().any(|f| f == VIEW_DEVICE_SEALED)
        && asks_for_view(snapshot, robot)
        && snapshot
            .participant(robot)
            .is_some_and(|p| p.supports.iter().any(|f| f == VIEW_DEVICE_SEALED))
}

fn has_board(roster: &[crate::wire::ParticipantInfo]) -> bool {
    roster
        .iter()
        .any(|p| p.has_role(Role::Embodiment) && p.supports.iter().any(|f| f == VIEW_DEVICE))
}

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
    fn a_state_too_large_for_a_board_is_worth_a_warning_only_with_a_board_in_the_session() {
        let member = |id: &str, roles: Vec<Role>, supports: &[&str]| crate::wire::ParticipantInfo {
            id: pid(id),
            name: id.into(),
            roles,
            priority: PriorityClass::Model,
            attached_ns: 0,
            protocol_minor: 0,
            supports: supports.iter().map(ToString::to_string).collect(),
            requires: Vec::new(),
            workflow_edition: String::new(),
            workflow_editions: Vec::new(),
            present: true,
            left: None,
            device: String::new(),
            generation: 1,
        };
        let mut roster = vec![
            member("env", vec![Role::Embodiment], &[]),
            member("driver", vec![Role::Controller], &[VIEW_DEVICE]),
        ];
        assert!(!has_board(&roster), "no robot here reads device views");
        roster.push(member("eye-1", vec![Role::Embodiment], &[VIEW_DEVICE]));
        assert!(has_board(&roster));
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
