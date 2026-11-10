//! Stream keys: keeping them in step with the parts, and delivering them until acknowledged.

use crate::wire::codec;
use crate::wire::{CtlError, CtlStatus};
use rlmesh_session_core::auth::issuer::StreamSlot;
use rlmesh_session_core::crypto::Kind;
use tracing::warn;

use crate::host::{HostActor, OsRng, lock};
use crate::transport::{CONTROL_CONGESTION, CONTROL_PRIORITY};

impl HostActor {
    /// Keeps one stream key slot per declared part.
    pub(super) fn sync_streams(&mut self) {
        let streams: Vec<StreamSlot> = self
            .authority
            .parts()
            .map(|part| StreamSlot {
                robot: part.owner.clone(),
                path: part.path.to_string(),
                channels: [Some(part.channel), part.applied_channel]
                    .into_iter()
                    .flatten()
                    .collect(),
                security: self.authority.security(part.channel),
            })
            .collect();
        let now = self.now();
        let access = &self.authority.config().access;
        if let Err(err) = self
            .auth
            .issuer()
            .sync_streams(&mut OsRng, &streams, access, now)
        {
            warn!(%err, "stream keys");
        }
    }

    /// Sends what the key schedule has due: the host's own receivers first,
    /// then every member's delivery; then decides the grants whose robots
    /// answered (or ran out of time) and answers those a participant's end
    /// failed.
    pub(super) async fn flush_keys(&mut self) {
        let now = self.now();
        while let Some(delivery) = self.auth.issuer().host_delivery() {
            let ack = lock(&self.host_keys).apply(&delivery, now);
            self.auth.issuer().host_ack(&ack, now);
        }
        for (who, delivery) in self.auth.issuer().deliveries(now) {
            let Some(peer) = self
                .peers
                .read()
                .ok()
                .and_then(|p| p.current.get(&who).cloned())
            else {
                continue;
            };
            let key = self.keys.participant_keys(&who);
            let sealed = lock(&peer.down).seal(
                Kind::KeyDelivery,
                &key,
                &codec::encode_key_delivery(&delivery),
            );
            let sent = match sealed {
                Ok(sealed) => self
                    .transport
                    .control()
                    .put(key, sealed)
                    .priority(CONTROL_PRIORITY)
                    .congestion_control(CONTROL_CONGESTION)
                    .express(true)
                    .await
                    .map_err(|err| err.to_string()),
                Err(err) => Err(err.to_string()),
            };
            if let Err(err) = sent {
                warn!(participant = %who, %err, "key delivery failed");
            }
        }
        for id in std::mem::take(&mut self.failed) {
            let gone = CtlError::new(CtlStatus::NotAttached, "the holder or a robot left");
            self.finish_grant(id, Err(gone)).await;
        }
        let decided = self.auth.issuer().ready_grants(now);
        for (id, ready) in decided {
            let outcome = match ready {
                Ok(()) => self.commit_grant(id).await,
                Err(err) => {
                    self.auth.issuer().refuse_grant(id);
                    Err(err)
                }
            };
            self.finish_grant(id, outcome).await;
        }
    }
}
