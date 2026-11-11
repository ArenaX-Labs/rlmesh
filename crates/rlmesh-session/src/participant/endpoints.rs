//! Endpoints by robot and path: publishers, subscribers, history, ticks and host health.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::wire::{
    ChunkRef, Command, Epoch, HistoryQuery, HostHealth, Lease, ParticipantId, SampleBatch,
    StreamSpec, VirtualTick, Watermark,
};
use rlmesh_session_core::auth;
use tokio::sync::mpsc;
use tracing::debug;

use crate::chunk::ChunkPublisher;
use crate::error::SessionError;
use crate::media::ImagePublisher;
use crate::participant::sealer::Sealer;
use crate::participant::{Participant, lock};
use crate::sample::{
    SamplePublisher, SampleSubscriber, SubscribeContext, SubscribeOptions, fetch_history,
};

impl Participant {
    /// Seals and opens under this participant's delivered keys (a lockstep
    /// link's requests and replies).
    pub fn sealer(&self) -> Sealer {
        self.shared.sealer()
    }

    /// A publisher for one of this embodiment's own parts.
    pub async fn publisher(&self, path: &str) -> Result<SamplePublisher, SessionError> {
        let part = self.own_part(path)?;
        let stream = part.stream().map_err(|reason| SessionError::WrongChannel {
            part: part.qualified(),
            reason,
        })?;
        self.stream_publisher(stream).await
    }

    /// A publisher for one of this embodiment's streams (a part's, or a
    /// commandable part's `_applied`).
    pub async fn stream_publisher(
        &self,
        stream: StreamSpec,
    ) -> Result<SamplePublisher, SessionError> {
        if stream.owner != self.shared.id {
            return Err(SessionError::WrongChannel {
                part: stream.qualified(),
                reason: format!("published by {}, not {}", stream.owner, self.shared.id),
            });
        }
        let key = self.shared.keys.data(&stream.owner, &stream.path);
        let history_key = self.shared.keys.history(&stream.owner, &stream.path);
        // An applied stream has no `send` ceiling; a part's is its rate cap.
        let ceiling_hz = if stream.applied {
            None
        } else {
            self.shared
                .view
                .borrow()
                .as_ref()
                .and_then(|v| v.part(stream.channel))
                .and_then(|p| p.send_policy().hz)
        };
        let channel = stream.channel;
        let publisher = SamplePublisher::declare(
            &self.shared.transport,
            key,
            history_key,
            stream,
            self.shared.clock_mode(),
            ceiling_hz,
            self.shared.epoch.clone(),
            self.shared.clock.clone(),
            self.shared.sealer(),
        )
        .await?;
        if let Ok(mut outlets) = self.shared.outlets.lock() {
            outlets
                .entry(channel)
                .or_default()
                .push(Arc::downgrade(publisher.outlet()));
        }
        self.shared.demand_wake.notify_one();
        Ok(publisher)
    }

    /// An image publisher for one of this embodiment's image parts, with the
    /// bitrate budget its `send` declares.
    pub async fn image_publisher(&self, path: &str) -> Result<ImagePublisher, SessionError> {
        let part = self.own_part(path)?;
        let budget = part.send_policy().max_mbps.map(|mbps| (mbps * 1e6) as u64);
        ImagePublisher::new(self.publisher(path).await?, budget)
    }

    /// Subscribes to a part of `robot`.
    pub async fn subscribe(
        &self,
        robot: &ParticipantId,
        path: &str,
        options: SubscribeOptions,
    ) -> Result<SampleSubscriber, SessionError> {
        let part = self.part(robot, path)?;
        let stream = part.stream().map_err(|reason| SessionError::WrongChannel {
            part: part.qualified(),
            reason,
        })?;
        self.subscribe_stream(stream, options).await
    }

    /// Subscribes to what the edge dispatched to a commandable part of
    /// `robot` (its `_applied` stream).
    pub async fn subscribe_applied(
        &self,
        robot: &ParticipantId,
        path: &str,
        options: SubscribeOptions,
    ) -> Result<SampleSubscriber, SessionError> {
        let part = self.part(robot, path)?;
        let stream = part
            .applied_stream()
            .ok_or_else(|| SessionError::WrongChannel {
                part: part.qualified(),
                reason: "not a commandable part".into(),
            })?
            .map_err(|reason| SessionError::WrongChannel {
                part: part.qualified(),
                reason,
            })?;
        self.subscribe_stream(stream, options).await
    }

    /// Subscribes to a stream.
    pub async fn subscribe_stream(
        &self,
        stream: StreamSpec,
        options: SubscribeOptions,
    ) -> Result<SampleSubscriber, SessionError> {
        let ctx = SubscribeContext {
            key: self.shared.keys.data(&stream.owner, &stream.path),
            keys: self.shared.keys.clone(),
            seen: self.shared.seen(stream.channel),
            spec: stream,
            me: self.shared.id.clone(),
            clock: self.shared.clock.clone(),
            view: self.shared.view.subscribe(),
            history_timeout: self.shared.history_timeout,
            sealer: self.shared.sealer(),
        };
        SampleSubscriber::declare(&self.shared.transport, ctx, options).await
    }

    /// Queries the host's history store for a stream (attach, swap and
    /// recovery only): every page of the reply, in seq order.
    pub async fn query_history(
        &self,
        stream: &StreamSpec,
        query: HistoryQuery,
    ) -> Result<Vec<SampleBatch>, SessionError> {
        fetch_history(
            &self.shared.transport,
            &self.shared.sealer(),
            stream,
            &self.shared.keys,
            query,
            self.shared.history_timeout,
        )
        .await
    }

    /// A chunk sender for a commandable part under `lease`, fenced to the
    /// view's current epoch.
    pub async fn chunk_publisher(&self, lease: &Lease) -> Result<ChunkPublisher, SessionError> {
        let part = self.part_by_id(lease.channel)?;
        let key = self.shared.keys.commands(&part.owner, part.path.as_str());
        let epoch = Epoch(self.shared.epoch.load(Ordering::Relaxed));
        ChunkPublisher::declare(
            &self.shared.transport,
            key,
            part,
            self.shared.id.clone(),
            lease,
            epoch,
            self.shared.clock_mode(),
            self.shared.clock.clone(),
            self.shared.sealer(),
        )
        .await
    }

    /// Virtual mode: this embodiment published tick `tick`'s samples, up to
    /// `watermarks`.
    pub async fn ready(&self, tick: u64, watermarks: Vec<Watermark>) -> Result<(), SessionError> {
        self.shared
            .ctl(Command::Ready { tick, watermarks })
            .await
            .map(|_| ())
    }

    /// Virtual mode: this controller's decision for tick `tick` (the chunks
    /// it sent for it; none is "no action").
    pub async fn decided(&self, tick: u64, chunks: Vec<ChunkRef>) -> Result<(), SessionError> {
        self.shared
            .ctl(Command::Decided { tick, chunks })
            .await
            .map(|_| ())
    }

    /// Virtual mode: the host's ticks from now on (`host/tick`), verified
    /// and in the host's (epoch, tick, phase) order: a tick that is not
    /// signed by the host, or not after the last one taken, is dropped.
    pub async fn virtual_ticks(&self) -> Result<TickStream, SessionError> {
        let (tx, rx) = mpsc::unbounded_channel();
        let verifier = Mutex::new(self.host_verifier().ok_or(SessionError::Closed)?);
        let key = self.shared.keys.host_tick();
        let last = Mutex::new(None);
        let subscriber = self
            .shared
            .transport
            .control()
            .declare_subscriber(key.clone())
            .callback(move |sample| {
                let bytes = sample.payload().to_bytes();
                let tick = match auth::open_tick(&mut lock(&verifier), &key, &bytes) {
                    Ok(tick) => tick,
                    Err(err) => {
                        debug!(%err, "tick dropped");
                        return;
                    }
                };
                let mut last = lock(&last);
                if last.is_none_or(|last| tick.order() > last) {
                    *last = Some(tick.order());
                    let _ = tx.send(tick);
                } else {
                    debug!(order = ?tick.order(), "a tick out of order dropped");
                }
            })
            .await?;
        Ok(TickStream {
            rx,
            _subscriber: subscriber,
        })
    }

    /// Calls `on_health` with each of the host's health messages
    /// (`host/health`, about once a second) on the transport's callback
    /// thread: every attached participant's link, clock mapping, edge
    /// counters and latest observer report, as the host sees them. A message
    /// not signed by the host, or not newer than the last one taken, is
    /// dropped. The subscription lasts as long as the returned handle.
    pub async fn host_health(
        &self,
        on_health: impl Fn(HostHealth) + Send + Sync + 'static,
    ) -> Result<HealthSubscription, SessionError> {
        let verifier = Mutex::new(self.host_verifier().ok_or(SessionError::Closed)?);
        let key = self.shared.keys.host_health();
        let subscriber = self
            .shared
            .transport
            .control()
            .declare_subscriber(key.clone())
            .callback(move |sample| {
                let bytes = sample.payload().to_bytes();
                match auth::open_health(&mut lock(&verifier), &key, &bytes) {
                    Ok(health) => on_health(health),
                    Err(err) => debug!(%err, "host health dropped"),
                }
            })
            .await?;
        Ok(HealthSubscription {
            _subscriber: subscriber,
        })
    }
}

/// A subscription to the host's health ([`Participant::host_health`]);
/// dropping it ends the subscription.
pub struct HealthSubscription {
    _subscriber: zenoh::pubsub::Subscriber<()>,
}

/// The host's virtual-mode ticks, as they arrive.
pub struct TickStream {
    rx: mpsc::UnboundedReceiver<VirtualTick>,
    _subscriber: zenoh::pubsub::Subscriber<()>,
}

impl TickStream {
    /// The next tick (`None` once the subscription is gone).
    pub async fn next(&mut self) -> Option<VirtualTick> {
        self.rx.recv().await
    }

    /// Waits for the next tick matching `predicate`, up to `timeout`.
    pub async fn wait_for(
        &mut self,
        timeout: Duration,
        mut predicate: impl FnMut(&VirtualTick) -> bool,
    ) -> Result<VirtualTick, SessionError> {
        tokio::time::timeout(timeout, async {
            while let Some(tick) = self.rx.recv().await {
                if predicate(&tick) {
                    return Ok(tick);
                }
            }
            Err(SessionError::Closed)
        })
        .await
        .map_err(|_| SessionError::Timeout(timeout, "a virtual tick".into()))?
    }
}
