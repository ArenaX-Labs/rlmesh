//! Lockstep recording (§13), off the critical path.
//!
//! The env's process hands each request and reply to a [`Tee`] (two `Bytes`
//! clones and a channel send); a worker thread splits the slabs per lane
//! and writes them into the in-process recorder as `Sample` envelopes:
//! observations on `/{robot}/{path}`, commands on `{path}/_applied`, with
//! `seq` the step, `tick` the step and `capture_ns` the notional session
//! time `step × tick_ns`.

use std::sync::mpsc;

use bytes::Bytes;
use rlmesh_session::record::SampleTee;
use rlmesh_session::wire::{Layout, LockstepReply, PartSpec, PartValues, Sample};

use crate::lockstep::exchange::Request;

/// One exchange to record, or a flush to acknowledge once all before it are.
#[allow(clippy::large_enum_variant)] // Exchanges are what the channel carries.
enum Teed {
    Exchange(Request, LockstepReply),
    Flush(tokio::sync::oneshot::Sender<()>),
}

/// Hands lockstep traffic to the recorder.
#[derive(Clone)]
pub struct Tee {
    tx: mpsc::Sender<Teed>,
}

impl Tee {
    /// Tees into `tee` for a robot with `parts`, `lanes` lanes and a
    /// notional `tick_ns`.
    pub fn new(tee: SampleTee, parts: Vec<PartSpec>, lanes: usize, tick_ns: u64) -> Self {
        let (tx, rx) = mpsc::channel::<Teed>();
        std::thread::Builder::new()
            .name("rlmesh-lockstep-tee".into())
            .spawn(move || {
                while let Ok(teed) = rx.recv() {
                    match teed {
                        Teed::Exchange(request, reply) => {
                            write(&tee, &parts, lanes, tick_ns, &request, &reply);
                        }
                        Teed::Flush(done) => {
                            let _ = done.send(());
                        }
                    }
                }
            })
            .ok();
        Self { tx }
    }

    /// Records one exchange.
    pub fn record(&self, request: Request, reply: LockstepReply) {
        let _ = self.tx.send(Teed::Exchange(request, reply));
    }

    /// Waits until every exchange recorded so far is written.
    pub async fn flush(&self) {
        let (done, flushed) = tokio::sync::oneshot::channel();
        if self.tx.send(Teed::Flush(done)).is_ok() {
            let _ = flushed.await;
        }
    }
}

fn write(
    tee: &SampleTee,
    parts: &[PartSpec],
    num_envs: usize,
    tick_ns: u64,
    request: &Request,
    reply: &LockstepReply,
) {
    let (lanes, step, commands): (Vec<u32>, u64, &[PartValues]) = match request {
        Request::Reset(reset) => (reset.lanes.clone(), 0, &[]),
        Request::Step(step) => (step.lanes.clone(), step.step, &step.commands),
    };
    let lanes: Vec<u32> = if lanes.is_empty() {
        (0..num_envs as u32).collect()
    } else {
        lanes
    };
    let capture_ns = i64::try_from(step.saturating_mul(tick_ns)).unwrap_or(i64::MAX);
    let emit = |values: &PartValues, applied: bool| {
        let Some(part) = parts.iter().find(|p| p.channel == values.channel) else {
            return;
        };
        let fields = match (applied, &part.command) {
            (true, Some(command)) => &command.fields,
            _ => &part.fields,
        };
        let channel = if applied {
            part.applied_channel.unwrap_or(part.channel)
        } else {
            part.channel
        };
        let stride = match Layout::of(fields) {
            Ok(Layout::Packed { stride, .. }) => Some(stride),
            // A text slab is v1's framing: recorded whole.
            _ => None,
        };
        for (i, lane) in lanes.iter().enumerate() {
            let payload = match stride {
                Some(stride) if values.payload.len() >= (i + 1) * stride => {
                    values.payload.slice(i * stride..(i + 1) * stride)
                }
                Some(_) => continue,
                None if i == 0 => values.payload.clone(),
                None => Bytes::new(),
            };
            tee.record(
                part,
                *lane,
                num_envs as u32,
                applied,
                &Sample {
                    channel,
                    epoch: reply.epoch,
                    seq: step + 1,
                    capture_ns,
                    publish_ns: capture_ns,
                    tick: step,
                    payload,
                    applied: None,
                },
            );
        }
    };
    for command in commands {
        emit(command, true);
    }
    for observation in &reply.observations {
        emit(observation, false);
    }
}
