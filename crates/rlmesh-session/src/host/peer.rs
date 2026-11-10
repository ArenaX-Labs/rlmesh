//! Each participant generation's keys, and the heartbeat and report
//! callbacks that open them off the host task.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use crate::wire::codec;
use crate::wire::{Epoch, HeartbeatEcho, ParticipantId, ParticipantReport, SessionKeys};
use rlmesh_session_core::auth::RequestWindow;
use rlmesh_session_core::crypto::{Kind, SealReceiver, SealSender};
use tokio::sync::mpsc;
use tracing::{debug, warn};
use zenoh::Wait;

use crate::authority::Identity;
use crate::clock::Clock;
use crate::host::lock;

/// One participant generation's keys: a handshake's candidate, or the
/// generation in effect. Shared with the heartbeat callback, which opens
/// heartbeats and seals echoes off the host task.
pub(super) struct Peer {
    pub(super) participant: ParticipantId,
    pub(super) identity: Identity,
    /// Host clock at the handshake.
    pub(super) created_ns: i64,
    /// `k_up`: requests, heartbeats and reports.
    pub(super) up: Mutex<SealReceiver>,
    /// `k_down`: replies and echoes. One lock per key: a seq is allocated
    /// and its message sent under it.
    pub(super) down: Mutex<SealSender>,
    /// Request ids and their replies (encoded `CtlReply`s, sealed afresh
    /// for a retry).
    pub(super) requests: Mutex<RequestWindow<Vec<u8>>>,
    /// The echo publisher, once in effect.
    pub(super) echo: std::sync::OnceLock<zenoh::pubsub::Publisher<'static>>,
    /// Its latest observer report, relayed in `host/health`.
    pub(super) report: Mutex<Option<ParticipantReport>>,
}

/// Every generation with keys.
#[derive(Default)]
pub(super) struct Peers {
    /// Candidates and generations in effect, by `k_up`'s id.
    pub(super) by_key: HashMap<u32, Arc<Peer>>,
    /// The generation in effect, by participant.
    pub(super) current: HashMap<ParticipantId, Arc<Peer>>,
}

pub(super) type SharedPeers = Arc<RwLock<Peers>>;

/// The report callback: open it under the sender's `k_up` and keep it as
/// that generation's latest. Anything that doesn't open is dropped.
pub(super) fn on_report_sample(
    keys: &SessionKeys,
    peers: &SharedPeers,
    sample: &zenoh::sample::Sample,
) {
    let key = sample.key_expr().as_str();
    let Some(id) = keys.participant_of(key) else {
        return;
    };
    let Some(peer) = peers.read().ok().and_then(|p| p.current.get(&id).cloned()) else {
        return;
    };
    let mut bytes = sample.payload().to_bytes().into_owned();
    let report = match lock(&peer.up).open(&keys.report(&id), &mut bytes) {
        Ok(opened) if opened.header.kind == Kind::ParticipantReport => {
            codec::decode_report(opened.body)
        }
        Ok(_) => return,
        Err(err) => {
            debug!(%id, %err, "report dropped");
            return;
        }
    };
    match report {
        Ok(report) if report.participant == id => *lock(&peer.report) = Some(report),
        Ok(_) => warn!(%id, "report key and body disagree"),
        Err(err) => debug!(%id, %err, "undecodable report"),
    }
}

type HeartbeatTx = mpsc::Sender<(ParticipantId, crate::wire::Heartbeat, i64)>;

/// The heartbeat callback: open it under the sender's `k_up` (anything else
/// is dropped: presence is authenticated heartbeats), echo at once (t3 as
/// late as possible), then queue the heartbeat for the host task.
pub(super) fn on_heartbeat_sample(
    keys: &SessionKeys,
    clock: &dyn Clock,
    epoch: &AtomicU64,
    peers: &SharedPeers,
    queue: &HeartbeatTx,
    sample: &zenoh::sample::Sample,
    t2: i64,
) {
    let Some(id) = keys.participant_of(sample.key_expr().as_str()) else {
        return;
    };
    // Not attached (or superseded): no keys, no presence.
    let Some(peer) = peers.read().ok().and_then(|p| p.current.get(&id).cloned()) else {
        return;
    };
    let mut bytes = sample.payload().to_bytes().into_owned();
    let heartbeat = match lock(&peer.up).open(&keys.heartbeat(&id), &mut bytes) {
        Ok(opened) => codec::decode_heartbeat(opened.body),
        Err(err) => {
            debug!(%id, %err, "heartbeat dropped");
            return;
        }
    };
    let heartbeat = match heartbeat {
        Ok(hb) if hb.participant == id => hb,
        Ok(hb) => {
            warn!(%id, body = %hb.participant, "heartbeat key and body disagree");
            return;
        }
        Err(err) => {
            warn!(%id, %err, "undecodable heartbeat");
            return;
        }
    };
    if let Some(publisher) = peer.echo.get() {
        // t3 under the lock: waiting for it is the host's hold time, never
        // network time.
        let mut down = lock(&peer.down);
        let echo = HeartbeatEcho {
            seq: heartbeat.seq,
            t1_ns: heartbeat.t1_ns,
            t2_ns: t2,
            t3_ns: clock.now_ns(),
            epoch: Epoch(epoch.load(Ordering::Relaxed)),
        };
        let sent = down
            .seal(
                Kind::HeartbeatEcho,
                &keys.heartbeat_echo(&id),
                &codec::encode_echo(&echo),
            )
            .map_err(|err| err.to_string())
            .and_then(|sealed| publisher.put(sealed).wait().map_err(|err| err.to_string()));
        if let Err(err) = sent {
            warn!(%id, %err, "heartbeat echo failed");
        }
    }
    if queue.try_send((id, heartbeat, t2)).is_err() {
        warn!("host overloaded: heartbeat dropped");
    }
}
