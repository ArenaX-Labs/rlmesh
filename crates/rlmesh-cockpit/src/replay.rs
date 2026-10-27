//! Replay of a recorded session (an MCAP file from
//! `rlmesh_session::record`), sans-IO: the recorded samples, chunks and
//! authority snapshots go through the same [`Model`] as a live feed, with
//! session time standing in for the cockpit's clock (a sample's receipt is
//! its log time, its capture its publish time).
//!
//! Playing forward ingests the messages up to the new time, keeping a copy
//! of the model every [`CHECKPOINT_EVERY`] of session time. Seeking back
//! restores the newest copy before the target and plays forward from it,
//! so a scrubbed frame shows exactly what playing straight through would
//! have shown, cumulative counters included.
//!
//! A recording holds no heartbeats or observer reports, so the counters
//! only they carry (published and produced rates, controller starts and
//! skips, clock offsets) stay empty.

use std::path::Path;
use std::time::Duration;

use rlmesh_session::record::{self, RecordError, Recorded, RecordedMessage};
use rlmesh_session::wire::ParticipantId;

use crate::model::{Input, Model, ModelConfig, ns};

/// Session time between model checkpoints.
pub const CHECKPOINT_EVERY: Duration = Duration::from_secs(10);

/// The playback speeds offered.
pub const SPEEDS: [f64; 5] = [0.25, 0.5, 1.0, 2.0, 4.0];

/// A recorded session, positioned at one instant.
pub struct Replay {
    /// `(session time, input)`, in order.
    inputs: Vec<(i64, Input)>,
    model: Model,
    /// The next input to ingest.
    next: usize,
    /// `(next, model)` before the input at `next`, every
    /// [`CHECKPOINT_EVERY`], in order. The first is the empty model.
    // TODO: one full model copy per 10 s; thin them out (or drop decoded
    // thumbnails) if hour-long camera recordings use too much memory.
    checkpoints: Vec<(usize, Model)>,
    at_ns: i64,
    /// Advancing on [`Replay::tick`].
    pub playing: bool,
    /// Session seconds per wall second.
    pub speed: f64,
}

impl Replay {
    /// Reads the recording at `path`, positioned at its start.
    pub fn open(
        path: impl AsRef<Path>,
        me: ParticipantId,
        config: ModelConfig,
    ) -> Result<Self, RecordError> {
        let inputs = record::read(path)?.into_iter().map(input).collect();
        Ok(Self::new(inputs, me, config))
    }

    /// A replay of `inputs` (`(session time, input)`, in order), positioned
    /// at the first.
    pub fn new(inputs: Vec<(i64, Input)>, me: ParticipantId, config: ModelConfig) -> Self {
        let model = Model::new(me, config);
        let mut replay = Self {
            at_ns: inputs.first().map_or(0, |(at, _)| *at),
            inputs,
            checkpoints: vec![(0, model.clone())],
            model,
            next: 0,
            playing: false,
            speed: 1.0,
        };
        replay.ingest_until(replay.at_ns);
        replay
    }

    /// The model at [`Replay::at_ns`].
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// The current instant, session time.
    pub fn at_ns(&self) -> i64 {
        self.at_ns
    }

    /// The first and last recorded instants.
    pub fn bounds(&self) -> (i64, i64) {
        match (self.inputs.first(), self.inputs.last()) {
            (Some((first, _)), Some((last, _))) => (*first, *last),
            _ => (0, 0),
        }
    }

    /// Moves to `t_ns`, clamped to the recording.
    pub fn seek(&mut self, t_ns: i64) {
        let (first, last) = self.bounds();
        let t_ns = t_ns.clamp(first, last);
        if t_ns < self.at_ns {
            let i = self.checkpoints.partition_point(|(next, _)| {
                self.inputs.get(*next).is_none_or(|(at, _)| *at <= t_ns)
            });
            let (next, model) = &self.checkpoints[i.saturating_sub(1)];
            self.next = *next;
            self.model = model.clone();
        }
        self.ingest_until(t_ns);
        self.at_ns = t_ns;
    }

    /// Advances by `wall` at [`Replay::speed`] while playing; stops at the
    /// end.
    pub fn tick(&mut self, wall: Duration) {
        if !self.playing {
            return;
        }
        let step = (wall.as_secs_f64() * self.speed * 1e9) as i64;
        self.seek(self.at_ns.saturating_add(step));
        if self.at_ns >= self.bounds().1 {
            self.playing = false;
        }
    }

    /// Moves to the next (`forward`) or previous recorded message's time.
    pub fn step(&mut self, forward: bool) {
        let at = self.at_ns;
        let target = if forward {
            let i = self.inputs.partition_point(|(t, _)| *t <= at);
            self.inputs.get(i).map(|(t, _)| *t)
        } else {
            let i = self.inputs.partition_point(|(t, _)| *t < at);
            i.checked_sub(1).map(|i| self.inputs[i].0)
        };
        if let Some(t) = target {
            self.seek(t);
        }
    }

    fn ingest_until(&mut self, t_ns: i64) {
        while let Some((at, input)) = self.inputs.get(self.next) {
            if *at > t_ns {
                break;
            }
            let (newest, _) = self.checkpoints[self.checkpoints.len() - 1];
            if self.next > newest && *at >= self.inputs[newest].0 + ns(CHECKPOINT_EVERY) {
                self.checkpoints.push((self.next, self.model.clone()));
            }
            self.model.ingest(input.clone(), *at);
            self.next += 1;
        }
    }
}

fn input(m: RecordedMessage) -> (i64, Input) {
    let input = match m.message {
        Recorded::Sample(sample) => Input::Sample {
            sample,
            receive_ns: m.log_ns,
            capture_local_ns: m.publish_ns,
        },
        Recorded::Chunk(chunk) => Input::Chunk {
            chunk,
            receive_ns: m.log_ns,
        },
        Recorded::Authority(snapshot) => Input::Authority(snapshot),
    };
    (m.log_ns, input)
}
