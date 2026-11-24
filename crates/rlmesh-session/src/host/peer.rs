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

pub(super) struct Peer {
    pub(super) participant: ParticipantId,
    pub(super) identity: Identity,
    pub(super) created_ns: i64,
    pub(super) up: Mutex<SealReceiver>,
    pub(super) down: Mutex<SealSender>,
    pub(super) requests: Mutex<RequestWindow<Vec<u8>>>,
    pub(super) echo: std::sync::OnceLock<zenoh::pubsub::Publisher<'static>>,
    pub(super) report: Mutex<Option<ParticipantReport>>,
}

#[derive(Default)]
pub(super) struct Peers {
    pub(super) by_key: HashMap<u32, Arc<Peer>>,
    pub(super) current: HashMap<ParticipantId, Arc<Peer>>,
}

pub(super) type SharedPeers = Arc<RwLock<Peers>>;

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
        let mut down = lock(&peer.down);
        let echo = HeartbeatEcho {
            seq: heartbeat.seq,
            t1_ns: heartbeat.t1_ns,
            t2_ns: t2,
            // t3 is read under the lock: waiting for it is the host's hold time, never network time.
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
