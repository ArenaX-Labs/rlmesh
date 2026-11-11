//! Heartbeats, presence and link health, and the echo publishers.

use crate::wire::codec;
use crate::wire::{HostHealth, LinkHealth, ParticipantId};
use rlmesh_session_core::crypto::SignedKind;
use tracing::{debug, warn};
use zenoh::sample::SampleKind;

use crate::host::{HostActor, lock};
use crate::metrics::{ParticipantHealth, Report};
use crate::transport::{CONTROL_PRIORITY, HEARTBEAT_CONGESTION};

impl HostActor {
    pub(super) fn on_heartbeat(
        &mut self,
        id: ParticipantId,
        heartbeat: crate::wire::Heartbeat,
        t2: i64,
    ) {
        // Only attached participants are tracked; attach precedes the first
        // heartbeat.
        let Some(link) = self.links.get_mut(&id) else {
            return;
        };
        link.stats.on_heartbeat(t2);
        link.rejected = heartbeat.rejected;
        link.clock_refused = heartbeat.clock_refused;
        link.stats.on_report(Report {
            last_seq_seen: heartbeat.last_seq_seen,
            last_chunk_accepted: heartbeat.last_chunk_accepted,
            last_chunk_applied: heartbeat.last_chunk_applied,
            queue_depth: heartbeat.queue_depth,
            health: heartbeat.health,
        });
        if let Some(sample) = heartbeat.previous
            && link.estimator.add(sample)
        {
            link.stats
                .on_round_trip(sample.rtt_ns().max(0).unsigned_abs());
        }
        self.authority.heartbeat(&id, heartbeat.epoch, t2);
    }

    /// Liveliness never makes anyone present (anyone on the network can
    /// declare a token); a delete only has the expiry tick check the
    /// participant's heartbeats sooner. A participant that rejoined under its
    /// id keeps heartbeating, so its old token's late delete loses nothing.
    pub(super) fn on_presence(&mut self, sample: &zenoh::sample::Sample) {
        let Some(id) = self.keys.participant_of(sample.key_expr().as_str()) else {
            return;
        };
        if sample.kind() == SampleKind::Delete {
            debug!(%id, "participant liveliness withdrawn: checking its heartbeats");
            self.authority.presence_hint(&id, self.now());
        }
    }

    /// What `host/health` says now.
    fn health(&self, now: i64) -> HostHealth {
        let peers = self.peers.read().ok();
        let links = self
            .links
            .iter()
            .map(|(id, link)| {
                let row = link.stats.health(id, link.estimator.estimate(now), now);
                let report = peers
                    .as_ref()
                    .and_then(|p| p.current.get(id))
                    .and_then(|peer| lock(&peer.report).clone());
                LinkHealth {
                    participant: Some(id.clone()),
                    heartbeats: row.heartbeats,
                    rtt_ns: row.rtt_ns.unwrap_or(0),
                    jitter_ns: row.jitter_ns,
                    heartbeat_gap_max_ns: row.heartbeat_gap_max_ns,
                    last_heartbeat_age_ns: row.last_heartbeat_age_ns.unwrap_or(0),
                    offset: row.offset.map(|o| (o.offset_ns, o.error_ns)),
                    last_chunk_accepted: row.report.last_chunk_accepted,
                    last_chunk_applied: row.report.last_chunk_applied,
                    queue_depth: row.report.queue_depth,
                    health: row.report.health,
                    rejected: link.rejected.clone(),
                    clock_refused: link.clock_refused.clone(),
                    report,
                    last_seq_seen: row.report.last_seq_seen,
                    ..LinkHealth::default()
                }
            })
            .collect();
        HostHealth {
            epoch: self.authority.epoch(),
            host_ns: now,
            links,
        }
    }

    /// Signs and publishes `host/health`: every attached participant's link
    /// as measured here, its edge counters and its latest observer report.
    pub(super) async fn publish_health(&mut self) {
        let now = self.now();
        let health = self.health(now);
        let signed = self.signer.sign(
            SignedKind::HostHealth,
            &self.keys.host_health(),
            &codec::encode_health(&health),
        );
        let sent = match signed {
            Ok(signed) => self
                .health_out
                .put(signed)
                .await
                .map_err(|err| err.to_string()),
            Err(err) => Err(err.to_string()),
        };
        if let Err(err) = sent {
            debug!(%err, "health publish failed");
        }
    }

    pub(super) fn health_rows(&self) -> Vec<ParticipantHealth> {
        let now = self.now();
        self.links
            .iter()
            .map(|(id, link)| {
                let mut row = link.stats.health(id, link.estimator.estimate(now), now);
                row.parts = row
                    .report
                    .last_seq_seen
                    .iter()
                    .map(|(channel, _)| (*channel, self.authority.part_name(*channel)))
                    .collect();
                row
            })
            .collect()
    }

    /// Declares the echo publisher of `id`'s generation in effect, and
    /// returns that generation.
    pub(super) async fn declare_echo(&self, id: &ParticipantId) -> u32 {
        let Some(peer) = self
            .peers
            .read()
            .ok()
            .and_then(|p| p.current.get(id).cloned())
        else {
            return 0;
        };
        if peer.echo.get().is_none() {
            match self
                .transport
                .control()
                .declare_publisher(self.keys.heartbeat_echo(id))
                .priority(CONTROL_PRIORITY)
                .congestion_control(HEARTBEAT_CONGESTION)
                .express(true)
                .await
            {
                Ok(publisher) => {
                    let _ = peer.echo.set(publisher);
                }
                Err(err) => warn!(%id, %err, "echo publisher"),
            }
        }
        peer.identity.generation
    }
}
