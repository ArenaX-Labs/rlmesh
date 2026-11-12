//! The host's datagram path ([`DatagramRelay`]): a robot's realtime parts
//! between the host and its board over UDP, instead of the session's TCP
//! links, where one lost frame holds back everything behind it until it is
//! sent again (see [`rlmesh_session_core::datagram`]).
//!
//! The host is the relay. Down, it subscribes to each routed part's chunk
//! key as any subscriber would and sends the sealed bytes on in a datagram,
//! the newest only: a chunk that finds the previous one still waiting for the
//! socket replaces it. Up, it opens each sample datagram under the stream's
//! key (with a replay window of its own) and publishes the bytes on their key
//! as the board would have, so policies, the recorder, history and viewers
//! see the same samples whichever path they took. It never unseals a chunk.
//!
//! It sends a robot nothing before it opened a probe from it under the
//! robot's `k_up`, and only to the address of the newest probe that opened:
//! no other key moves it, since a stream key is every reader's. It answers
//! nothing it could not open, and nothing at all to a probe.
//!
//! One relay serves every session a host process runs, on one UDP port: a
//! datagram's route names its robot and session.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rlmesh_session_core::auth::Keyring;
use rlmesh_session_core::auth::keyring::StreamWindows;
use rlmesh_session_core::crypto::Kind as SealedKind;
use rlmesh_session_core::datagram::{
    self, Arrivals, Header, Kind, Latest, Learned, MAX_RECEIVED, RouteIds,
};
use tokio::net::UdpSocket;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::host::lock;
use crate::host::peer::Peer;
use crate::metrics::RelayTiming;
use crate::wire::{ChannelId, DatagramOffer, DatagramStats, ParticipantId};

/// The host's datagram path: one UDP socket for every session this process
/// hosts. Hand it to each session in [`crate::HostConfig::datagram`].
pub struct DatagramRelay {
    inner: Arc<Inner>,
    port: u16,
    cancel: CancellationToken,
}

impl std::fmt::Debug for DatagramRelay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DatagramRelay")
            .field("port", &self.port)
            .finish_non_exhaustive()
    }
}

struct Inner {
    socket: UdpSocket,
    routes: Mutex<Routes>,
    /// A route has a chunk waiting for the socket.
    waiting: Notify,
    /// Datagrams no route took: unreadable, or for no route.
    unrouted: AtomicU64,
}

#[derive(Default)]
struct Routes {
    ids: RouteIds,
    by_id: HashMap<u16, Route>,
}

#[derive(Clone)]
enum Route {
    Probe(Arc<RobotPath>),
    Down(Arc<RobotPath>, Arc<Down>),
    Up(Arc<RobotPath>, Arc<Up>),
}

/// One part's chunks, host to board.
pub(super) struct Down {
    pub(super) id: u16,
    pub(super) channel: ChannelId,
    /// The newest datagram the socket had no room for.
    waiting: Mutex<Latest<(Vec<u8>, SocketAddr, Instant)>>,
    /// A chunk too large for a datagram arrived: the part goes back to the
    /// session.
    oversize: Mutex<Option<usize>>,
    subscriber: Mutex<Option<zenoh::pubsub::Subscriber<()>>>,
}

/// One stream's samples, board to host.
pub(super) struct Up {
    pub(super) id: u16,
    pub(super) channel: ChannelId,
    applied: bool,
    key: String,
    publisher: zenoh::pubsub::Publisher<'static>,
    windows: Mutex<StreamWindows>,
}

/// One robot generation on the datagram path.
pub(super) struct RobotPath {
    pub(super) robot: ParticipantId,
    pub(super) generation: u32,
    pub(super) probe: u16,
    probe_key: String,
    peer: Arc<Peer>,
    host_keys: Arc<Mutex<Keyring>>,
    state: Mutex<PathState>,
}

struct PathState {
    to: Learned<SocketAddr>,
    next_seq: u32,
    sent: u64,
    arrivals: Arrivals,
    /// Sample datagrams that opened (the rest of those received are
    /// probes), and when the newest probe did.
    samples: u64,
    last_probe: Option<Instant>,
    fallbacks: u32,
    active: bool,
    /// How long chunks took from the relay's subscriber to the socket.
    relay: RelayTiming,
}

/// What one end of a robot's path counted, for judging it: datagrams the
/// host sent, and the probes and samples from the board that opened.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct PathCounts {
    pub(super) sent: u64,
    pub(super) probes: u64,
    pub(super) samples: u64,
}

impl RobotPath {
    /// What this end counted.
    pub(super) fn stats(&self) -> DatagramStats {
        let state = lock(&self.state);
        DatagramStats {
            active: state.active,
            sent: state.sent,
            received: state.arrivals.received,
            lost: state.arrivals.lost,
            reordered: state.arrivals.reordered,
            unopened: state.arrivals.unopened,
            fallbacks: state.fallbacks,
        }
    }

    /// How long its chunks took through the relay.
    pub(super) fn relay_timing(&self) -> RelayTiming {
        lock(&self.state).relay
    }

    /// Its counts, for judging the path.
    pub(super) fn counts(&self) -> PathCounts {
        let state = lock(&self.state);
        PathCounts {
            sent: state.sent,
            probes: state.arrivals.received.saturating_sub(state.samples),
            samples: state.samples,
        }
    }

    /// Since the newest probe that opened (`None`: none has).
    pub(super) fn probe_age(&self, now: Instant) -> Option<std::time::Duration> {
        let state = lock(&self.state);
        state.last_probe.map(|at| now.saturating_duration_since(at))
    }

    /// The host moves the robot's parts back to the session's links.
    pub(super) fn fall_back(&self) {
        let mut state = lock(&self.state);
        if state.active {
            state.active = false;
            state.fallbacks = state.fallbacks.saturating_add(1);
        }
    }

    /// And onto the path again.
    pub(super) fn reactivate(&self) {
        lock(&self.state).active = true;
    }

    /// A probe that opens under the robot's `k_up`.
    fn open_probe(&self, payload: &[u8]) -> bool {
        let mut msg = payload.to_vec();
        matches!(
            lock(&self.peer.up).open(&self.probe_key, &mut msg),
            Ok(opened) if opened.header.kind == SealedKind::DatagramProbe
        )
    }

    /// A sample that opens under its stream's key, and is no copy of one
    /// that did.
    fn open_sample(&self, up: &Up, payload: &[u8]) -> bool {
        let mut windows = lock(&up.windows);
        let opening = lock(&self.host_keys).begin_open_stream(
            payload,
            &self.robot,
            up.channel,
            up.applied,
            &windows,
        );
        let Ok((opening, version)) = opening else {
            return false;
        };
        // Decrypted in a copy: the bytes go on as they came.
        let mut copy = payload.to_vec();
        match opening.finish(&up.key, &mut copy) {
            Ok(opened) => windows.commit(&opened.header, version).is_ok(),
            Err(_) => false,
        }
    }
}

impl Down {
    /// The chunk too large for a datagram that arrived, if one did.
    pub(super) fn oversize(&self) -> Option<usize> {
        *lock(&self.oversize)
    }
}

impl DatagramRelay {
    /// Listens on `addr` (UDP), until dropped.
    ///
    /// # Errors
    /// The socket's.
    pub async fn bind(addr: SocketAddr) -> std::io::Result<Arc<Self>> {
        let socket = UdpSocket::bind(addr).await?;
        let port = socket.local_addr()?.port();
        let inner = Arc::new(Inner {
            socket,
            routes: Mutex::default(),
            waiting: Notify::new(),
            unrouted: AtomicU64::new(0),
        });
        let cancel = CancellationToken::new();
        tokio::spawn(receive(inner.clone(), cancel.clone()));
        tokio::spawn(flush(inner.clone(), cancel.clone()));
        info!(%addr, port, "datagram path open (UDP)");
        Ok(Arc::new(Self {
            inner,
            port,
            cancel,
        }))
    }

    /// The UDP port it listens on.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// What every handshake offers.
    pub fn offer(&self) -> DatagramOffer {
        DatagramOffer {
            port: self.port,
            version: datagram::VERSION,
        }
    }

    /// Datagrams no route took, over every session.
    pub fn unrouted(&self) -> u64 {
        self.inner.unrouted.load(Ordering::Relaxed)
    }

    /// A robot generation on the path, with its probe route.
    pub(super) fn add_robot(
        &self,
        robot: ParticipantId,
        peer: Arc<Peer>,
        probe_key: String,
        host_keys: Arc<Mutex<Keyring>>,
    ) -> Option<Arc<RobotPath>> {
        let mut routes = lock(&self.inner.routes);
        let probe = routes.ids.take()?;
        let path = Arc::new(RobotPath {
            generation: peer.identity.generation,
            robot,
            probe,
            probe_key,
            peer,
            host_keys,
            state: Mutex::new(PathState {
                to: Learned::default(),
                next_seq: 0,
                sent: 0,
                arrivals: Arrivals::default(),
                samples: 0,
                last_probe: None,
                fallbacks: 0,
                relay: RelayTiming::default(),
                active: true,
            }),
        });
        routes.by_id.insert(probe, Route::Probe(path.clone()));
        Some(path)
    }

    /// Part `channel`'s chunks on `key` down to `path`'s board, from
    /// `session`'s subscriber.
    pub(super) async fn add_down(
        &self,
        path: &Arc<RobotPath>,
        channel: ChannelId,
        session: &zenoh::Session,
        key: String,
    ) -> Option<Arc<Down>> {
        let id = lock(&self.inner.routes).ids.take()?;
        let down = Arc::new(Down {
            id,
            channel,
            waiting: Mutex::default(),
            oversize: Mutex::default(),
            subscriber: Mutex::default(),
        });
        let subscriber = session
            .declare_subscriber(key)
            .callback({
                let (inner, path, down) = (
                    Arc::downgrade(&self.inner),
                    Arc::downgrade(path),
                    Arc::downgrade(&down),
                );
                move |sample| {
                    if let (Some(inner), Some(path), Some(down)) =
                        (inner.upgrade(), path.upgrade(), down.upgrade())
                    {
                        inner.forward(&path, &down, &sample.payload().to_bytes());
                    }
                }
            })
            .await;
        let subscriber = match subscriber {
            Ok(subscriber) => subscriber,
            Err(err) => {
                warn!(robot = %path.robot, %channel, %err, "no chunk subscriber for the datagram path");
                lock(&self.inner.routes).ids.give_back(id);
                return None;
            }
        };
        *lock(&down.subscriber) = Some(subscriber);
        lock(&self.inner.routes)
            .by_id
            .insert(id, Route::Down(path.clone(), down.clone()));
        Some(down)
    }

    /// Stream `channel`'s samples up from `path`'s board, published on `key`
    /// by `publisher`.
    pub(super) fn add_up(
        &self,
        path: &Arc<RobotPath>,
        channel: ChannelId,
        applied: bool,
        key: String,
        publisher: zenoh::pubsub::Publisher<'static>,
    ) -> Option<Arc<Up>> {
        let mut routes = lock(&self.inner.routes);
        let id = routes.ids.take()?;
        let up = Arc::new(Up {
            id,
            channel,
            applied,
            key,
            publisher,
            windows: Mutex::default(),
        });
        routes.by_id.insert(id, Route::Up(path.clone(), up.clone()));
        Some(up)
    }

    /// Drops `path`'s down route `down`: its subscriber goes with it.
    pub(super) fn remove_down(&self, down: &Down) {
        self.remove(down.id);
        lock(&down.subscriber).take();
    }

    /// Drops `path`'s up route `up`.
    pub(super) fn remove_up(&self, up: &Up) {
        self.remove(up.id);
    }

    /// Drops `path`'s probe route: nothing of it is routed after this.
    pub(super) fn remove_probe(&self, path: &RobotPath) {
        self.remove(path.probe);
    }

    fn remove(&self, id: u16) {
        let mut routes = lock(&self.inner.routes);
        routes.by_id.remove(&id);
        routes.ids.give_back(id);
    }
}

impl Drop for DatagramRelay {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

impl Inner {
    /// Sends one chunk on `down` to its robot, the newest only.
    fn forward(&self, path: &RobotPath, down: &Down, payload: &[u8]) {
        let got = Instant::now();
        let mut state = lock(&path.state);
        let Some(to) = state.to.get().filter(|_| state.active) else {
            return;
        };
        let seq = state.next_seq;
        state.next_seq = seq.wrapping_add(1);
        let header = Header {
            kind: Kind::Chunk,
            route: down.id,
            seq,
        };
        let Some(datagram) = header.frame(payload) else {
            *lock(&down.oversize) = Some(payload.len());
            return;
        };
        // The newest goes first: one still waiting for the socket is
        // replaced, never sent behind it.
        let mut waiting = lock(&down.waiting);
        if waiting.take().is_some() {
            waiting.replaced += 1;
        }
        match self.socket.try_send_to(&datagram, to) {
            Ok(_) => {
                state.sent += 1;
                state.relay.add(got.elapsed());
            }
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                waiting.put((datagram, to, got));
                self.waiting.notify_one();
            }
            Err(err) => debug!(robot = %path.robot, %err, "chunk datagram not sent"),
        }
    }

    /// Takes one datagram from `from`.
    async fn take(&self, datagram: &[u8], from: SocketAddr) {
        let Ok((header, payload)) = Header::parse(datagram) else {
            self.unrouted.fetch_add(1, Ordering::Relaxed);
            return;
        };
        let route = lock(&self.routes).by_id.get(&header.route).cloned();
        let Some(route) = route else {
            self.unrouted.fetch_add(1, Ordering::Relaxed);
            return;
        };
        match (route, header.kind) {
            (Route::Probe(path), Kind::Probe) => {
                let opened = path.open_probe(payload);
                let mut state = lock(&path.state);
                if !opened {
                    state.arrivals.refused();
                    return;
                }
                state.arrivals.opened(header.seq);
                state.last_probe = Some(Instant::now());
                let before = state.to.get();
                if state.to.opened(from) {
                    info!(
                        robot = %path.robot,
                        %from,
                        ?before,
                        "datagram path: sending the robot's chunks here"
                    );
                }
            }
            (Route::Up(path, up), Kind::Sample) => {
                if !path.open_sample(&up, payload) {
                    lock(&path.state).arrivals.refused();
                    return;
                }
                {
                    let mut state = lock(&path.state);
                    state.arrivals.opened(header.seq);
                    state.samples += 1;
                    if !state.active {
                        return;
                    }
                }
                if let Err(err) = up.publisher.put(payload.to_vec()).await {
                    debug!(robot = %path.robot, %err, "relayed sample not published");
                }
            }
            (Route::Probe(path) | Route::Up(path, _) | Route::Down(path, _), _) => {
                lock(&path.state).arrivals.refused();
            }
        }
    }
}

/// Takes every datagram that arrives, until cancelled.
async fn receive(inner: Arc<Inner>, cancel: CancellationToken) {
    // One byte past the largest a sender may send, so an oversize one is
    // seen whole enough to refuse.
    let mut buf = vec![0_u8; MAX_RECEIVED + 1];
    loop {
        let received = tokio::select! {
            () = cancel.cancelled() => return,
            received = inner.socket.recv_from(&mut buf) => received,
        };
        match received {
            Ok((n, from)) => inner.take(&buf[..n], from).await,
            // An ICMP error from a robot that went away, on some systems.
            Err(err) => debug!(%err, "datagram path receive"),
        }
    }
}

/// How often a chunk the socket had no room for is tried again: a socket
/// that refused one need not say when it has room.
const RETRY: std::time::Duration = std::time::Duration::from_millis(2);

/// Sends the chunks the socket had no room for when they arrived, the newest
/// of each route only, until cancelled.
async fn flush(inner: Arc<Inner>, cancel: CancellationToken) {
    loop {
        tokio::select! {
            () = cancel.cancelled() => return,
            () = inner.waiting.notified() => {}
        }
        loop {
            let downs: Vec<(Arc<RobotPath>, Arc<Down>)> = lock(&inner.routes)
                .by_id
                .values()
                .filter_map(|r| match r {
                    Route::Down(path, down) if lock(&down.waiting).is_waiting() => {
                        Some((path.clone(), down.clone()))
                    }
                    _ => None,
                })
                .collect();
            if downs.is_empty() {
                break;
            }
            tokio::select! {
                () = cancel.cancelled() => return,
                () = tokio::time::sleep(RETRY) => {}
            }
            for (path, down) in downs {
                let Some((datagram, to, got)) = lock(&down.waiting).take() else {
                    continue;
                };
                match inner.socket.try_send_to(&datagram, to) {
                    Ok(_) => {
                        let mut state = lock(&path.state);
                        state.sent += 1;
                        state.relay.add(got.elapsed());
                    }
                    Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                        let mut waiting = lock(&down.waiting);
                        if !waiting.is_waiting() {
                            waiting.put((datagram, to, got));
                        }
                    }
                    Err(err) => debug!(robot = %path.robot, %err, "chunk datagram not sent"),
                }
            }
        }
    }
}
