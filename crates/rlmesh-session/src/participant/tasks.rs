//! The participant's background tasks: keys, retries, heartbeats, demands and stream reports.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Weak};
use std::time::Duration;

use crate::wire::{ChannelId, Command, CtlReply, Epoch, Health, Heartbeat};
use tokio::sync::{mpsc, watch};
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

#[cfg(doc)]
use crate::participant::Pending;
use crate::participant::edge::SAFE_RETRY;
use crate::participant::shared::KeyWork;
use crate::participant::{Shared, lock};
use crate::report::{ReportInputs, report_loop};
use crate::sample::Outlet;

pub(super) fn health_code(health: Health) -> u8 {
    match health {
        Health::Unknown => 0,
        Health::Ok => 1,
        Health::Degraded => 2,
        Health::Unhealthy => 3,
    }
}

fn health_of(code: u8) -> Health {
    match code {
        1 => Health::Ok,
        2 => Health::Degraded,
        3 => Health::Unhealthy,
        _ => Health::Unknown,
    }
}

/// How often a participant missing keys asks for them again.
const GET_KEYS_PERIOD: Duration = Duration::from_millis(250);

/// Acknowledges every key delivery and, when a message came under a key
/// this participant doesn't hold, asks for every key of its generation
/// (`GetKeys`, at most every [`GET_KEYS_PERIOD`]): the reply installs them.
pub(super) async fn work_keys(
    shared: Arc<Shared>,
    mut work: mpsc::UnboundedReceiver<KeyWork>,
    cancel: CancellationToken,
) {
    let mut last_recovery: Option<tokio::time::Instant> = None;
    loop {
        let item = tokio::select! {
            () = cancel.cancelled() => return,
            item = work.recv() => match item {
                Some(item) => item,
                None => return,
            },
        };
        let command = match item {
            KeyWork::Ack(ack) if !ack.is_empty() => Command::KeysAck(ack),
            KeyWork::Ack(_) => continue,
            KeyWork::Missing => {
                if last_recovery.is_some_and(|at| at.elapsed() < GET_KEYS_PERIOD) {
                    continue;
                }
                last_recovery = Some(tokio::time::Instant::now());
                Command::GetKeys
            }
            KeyWork::Unanswered(id, command, answer) => {
                tokio::spawn(retry_unanswered(
                    shared.clone(),
                    id,
                    command,
                    answer,
                    cancel.clone(),
                ));
                continue;
            }
        };
        match shared.ctl(command).await {
            Ok(_) => {
                shared.keys_changed.send_modify(|n| *n += 1);
            }
            Err(err) => debug!(participant = %shared.id, %err, "keys"),
        }
    }
}

/// Resends request `id` until it is answered, and hands the answer to its
/// [`Pending`] handle. Stops when this generation ends, or when another
/// attempt's reply answered it and no handle waits.
async fn retry_unanswered(
    shared: Arc<Shared>,
    id: u64,
    command: Command,
    answer: watch::Sender<Option<CtlReply>>,
    cancel: CancellationToken,
) {
    loop {
        tokio::select! {
            () = cancel.cancelled() => return,
            () = tokio::time::sleep(SAFE_RETRY) => {}
        }
        let open = shared
            .link
            .get()
            .is_some_and(|link| lock(link).unanswered(id));
        if (!open && answer.is_closed()) || shared.generation_ended() {
            return;
        }
        if let Ok(reply) = shared.call(Some(id), &command).await.1 {
            shared.keys_changed.send_modify(|n| *n += 1);
            answer.send_replace(Some(reply));
            return;
        }
    }
}

pub(super) async fn heartbeat_loop(
    shared: Arc<Shared>,
    period: Duration,
    cancel: CancellationToken,
) {
    let mut interval = tokio::time::interval(period);
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut seq = 0_u64;
    loop {
        tokio::select! {
            () = cancel.cancelled() => break,
            _ = interval.tick() => {}
        }
        if !shared.heartbeats_enabled.load(Ordering::Relaxed) {
            continue;
        }
        #[cfg(feature = "gateway")]
        if lock(&shared.relay).is_some() {
            continue;
        }
        seq += 1;
        let (epoch, mode) = shared.epoch_and_mode();
        let previous = shared
            .completed
            .lock()
            .ok()
            .and_then(|mut slot| slot.take());
        let last_seq_seen = shared
            .seen
            .lock()
            .map(|seen| {
                seen.iter()
                    .map(|(channel, seq)| (*channel, seq.load(Ordering::Relaxed)))
                    .collect()
            })
            .unwrap_or_default();
        let (last_chunk_accepted, last_chunk_applied, queue_depth) = shared.edge.get();
        let heartbeat = Heartbeat {
            participant: shared.id.clone(),
            epoch: Epoch(epoch),
            mode,
            seq,
            t1_ns: shared.clock.now_ns(),
            previous,
            last_seq_seen,
            last_chunk_accepted,
            last_chunk_applied,
            queue_depth,
            health: health_of(shared.health.load(Ordering::Relaxed)),
            rejected: shared.edge.rejected(),
            clock_refused: shared.edge.clock_refused(),
        };
        if let Err(err) = shared.send_heartbeat(heartbeat, true).await {
            warn!(participant = %shared.id, %err, "heartbeat failed");
        }
    }
}

/// Installs the host's demands on this embodiment's publishers and
/// acknowledges each revision once installed, on every new view and
/// every newly declared publisher.
pub(super) async fn install_demands(shared: Arc<Shared>, cancel: CancellationToken) {
    let mut view = shared.view.subscribe();
    loop {
        sync_demands(&shared).await;
        tokio::select! {
            () = cancel.cancelled() => return,
            changed = view.changed() => if changed.is_err() { return },
            () = shared.demand_wake.notified() => {}
        }
    }
}

async fn sync_demands(shared: &Shared) {
    let live: Vec<(ChannelId, Vec<Arc<Outlet>>)> = {
        let Ok(mut outlets) = shared.outlets.lock() else {
            return;
        };
        outlets.retain(|_, list| {
            list.retain(|o| o.strong_count() > 0);
            !list.is_empty()
        });
        outlets
            .iter()
            .map(|(c, list)| (*c, list.iter().filter_map(Weak::upgrade).collect()))
            .collect()
    };
    if live.is_empty() {
        return;
    }
    let acks: Vec<(ChannelId, u64)> = {
        let view = shared.view.borrow();
        let Some(view) = view.as_ref() else {
            return;
        };
        live.iter()
            .filter_map(|(channel, outlets)| {
                let demand = view.demand(*channel);
                let (revision, plan) = demand.map_or((0, None), |d| (d.revision, Some(d.plan)));
                for outlet in outlets {
                    outlet.install(revision, plan);
                }
                demand
                    .filter(|d| {
                        d.acked < d.revision && outlets.iter().all(|o| o.revision() == d.revision)
                    })
                    .map(|d| (*channel, d.revision))
            })
            .collect()
    };
    for (channel, revision) in acks {
        match shared.ctl(Command::DemandAck { channel, revision }).await {
            Ok(_) => debug!(participant = %shared.id, %channel, revision, "demand installed"),
            Err(err) => {
                let part = shared
                    .view
                    .borrow()
                    .as_ref()
                    .map_or_else(|| channel.to_string(), |v| v.part_name(channel));
                warn!(participant = %shared.id, %part, %err, "demand ack failed");
            }
        }
    }
}

/// Publishes this participant's observer report: its publishers'
/// counters and a bound controller's run counters.
pub(super) async fn report_streams(shared: Arc<Shared>, cancel: CancellationToken) {
    let inputs = ReportInputs {
        transport: shared.transport.clone(),
        key: shared.keys.report(&shared.id),
        id: shared.id.clone(),
        clock: shared.clock.clone(),
        epoch: shared.epoch.clone(),
        run: shared.run.clone(),
        seal: Box::new({
            let link = shared.link.clone();
            move |key, report| {
                let link = link.get()?;
                lock(link)
                    .report(key, report)
                    .map_err(|err| debug!(%err, "report not sealed"))
                    .ok()
            }
        }),
    };
    let streams = move || {
        let Ok(outlets) = shared.outlets.lock() else {
            return Vec::new();
        };
        outlets
            .values()
            .filter_map(|list| list.iter().find_map(Weak::upgrade))
            .map(|outlet| outlet.counters())
            .collect()
    };
    report_loop(inputs, streams, cancel).await;
}
