//! The virtual clock's run loop (§11.3): the embodiment's and the
//! controller's side of the host's tick barrier.
//!
//! The host runs the barrier (`rlmesh_session::virtual_clock`); this
//! module is what participants do around it:
//!
//! - [`robot_tick`] (steps 1, 4, 5): after publishing tick k's samples, the
//!   embodiment sends `Ready{k, watermarks}` and waits for `Step{k}`, whose
//!   chunks its edge judges before it applies what is due and advances
//!   physics by `tick_ns`.
//! - [`VirtualController::decide`] (steps 2-3): on `TickState{k}`, a due
//!   controller (the host's rule: `k mod round(1 / (replan_hz × tick_s)) =
//!   0`) decides from its inputs at the watermarks, sends its chunks and
//!   `Decided{k}`.
//! - [`LatencyModel`]: the inference cost d, a constant or a seeded draw,
//!   a pure function of `(seed, controller, k)`, so every party (and a
//!   replay) computes the same schedule. The host's barrier holds Step{k}
//!   until Decided{k} arrives (latency 0 at the host), and the controller
//!   starts its chunk at `(k + d) × tick_ns`: the decision takes effect at
//!   tick k + d, the robot runs its previous actions meanwhile, and real
//!   computation is never raced.
//! - [`Faults`]: seeded channel faults (drop, freeze, delay) keyed by
//!   `(seed, channel, seq)`, handed to the decision as faulted inputs; §8.8
//!   decides what a stale input does (`hold` or skip).
//!
//! A barrier that does not move within its timeout ends the episode
//! `interrupted: TIMEOUT` ([`Interrupted::Timeout`]); the host goes SAFE.

use std::time::Duration;

use rlmesh_session::{Participant, SessionError, TickStream};
use rlmesh_wire::tick::{Step, TickState, VirtualTick};
use rlmesh_wire::{ChannelId, ChunkRef, ParticipantId, Watermark};

/// Why a virtual episode ended early.
#[derive(Debug, thiserror::Error)]
pub enum Interrupted {
    /// A barrier did not move within its timeout (§11.3).
    #[error("interrupted: TIMEOUT (no tick {phase} for tick {tick} within {waited:?})")]
    Timeout {
        /// The tick it waited on.
        tick: u64,
        /// What it waited for.
        phase: &'static str,
        /// How long.
        waited: Duration,
    },
    /// The session.
    #[error(transparent)]
    Session(#[from] SessionError),
}

/// Steps 1, 4 and 5 for an embodiment: `Ready{k}`, then the `Step{k}`
/// listing the chunks to judge. A `Step` that does not come within
/// `timeout` (the barrier timeout) ends the episode.
pub async fn robot_tick(
    participant: &Participant,
    ticks: &mut TickStream,
    k: u64,
    watermarks: Vec<Watermark>,
    timeout: Duration,
) -> Result<Step, Interrupted> {
    participant.ready(k, watermarks).await?;
    let tick = ticks
        .wait_for(
            timeout,
            |t| matches!(t, VirtualTick::Step(s) if s.tick == k),
        )
        .await
        .map_err(|err| match err {
            SessionError::Timeout(waited, _) => Interrupted::Timeout {
                tick: k,
                phase: "step",
                waited,
            },
            other => Interrupted::Session(other),
        })?;
    match tick {
        VirtualTick::Step(step) => Ok(step),
        VirtualTick::Observe(_) => unreachable!("waited for a step"),
    }
}

/// d, the ticks between a decision and its effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatencyModel {
    /// Always `d`.
    Constant(u64),
    /// Uniform on `min..=max`, drawn from `(seed, controller, k)`.
    Uniform {
        /// The seed.
        seed: u64,
        /// Fewest ticks.
        min: u64,
        /// Most ticks.
        max: u64,
    },
}

/// A mixing hash (splitmix64): the draws are pure functions of their keys.
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn key(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}

impl LatencyModel {
    /// The latency of `controller`'s decision at tick `k`.
    pub fn draw(&self, controller: &ParticipantId, k: u64) -> u64 {
        match *self {
            Self::Constant(d) => d,
            Self::Uniform { seed, min, max } => {
                let span = max.saturating_sub(min) + 1;
                min + mix(seed ^ key(controller.as_str()) ^ mix(k)) % span
            }
        }
    }
}

/// A seeded fault on one sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    /// Never arrives.
    Drop,
    /// Arrives with the previous sample's value.
    Freeze,
    /// Arrives this many ticks late.
    Delay(u64),
}

/// Seeded channel faults (§11.3), in parts per million of samples.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Faults {
    /// The seed.
    pub seed: u64,
    /// Dropped samples.
    pub drop_ppm: u32,
    /// Frozen samples.
    pub freeze_ppm: u32,
    /// Delayed samples.
    pub delay_ppm: u32,
    /// How late a delayed sample is, in ticks.
    pub delay_ticks: u64,
}

impl Faults {
    /// The fault on sample `seq` of `channel`, if any: a pure function of
    /// `(seed, channel, seq)`.
    pub fn fault(&self, channel: ChannelId, seq: u64) -> Option<Fault> {
        let draw = (mix(self.seed ^ mix(u64::from(channel.0)) ^ mix(seq.rotate_left(17)))
            % 1_000_000) as u32;
        let (d, f, l) = (self.drop_ppm, self.freeze_ppm, self.delay_ppm);
        if draw < d {
            Some(Fault::Drop)
        } else if draw < d.saturating_add(f) {
            Some(Fault::Freeze)
        } else if draw < d.saturating_add(f).saturating_add(l) {
            Some(Fault::Delay(self.delay_ticks))
        } else {
            None
        }
    }
}

/// What a due controller decides from.
#[derive(Clone, Debug)]
pub struct Decision<'a> {
    /// The observed tick.
    pub state: &'a TickState,
    /// d for this decision.
    pub latency_ticks: u64,
    /// When its chunks start: `(k + d) × tick_ns`, session time.
    pub t0_ns: i64,
    /// The inputs whose newest sample (at the watermark) is faulted.
    pub faulted: Vec<(ChannelId, Fault)>,
}

/// A controller's side of the barrier.
#[derive(Clone, Debug)]
pub struct VirtualController {
    /// The inference cost model.
    pub latency: LatencyModel,
    /// Seeded channel faults.
    pub faults: Faults,
    /// Session time per tick.
    pub tick_ns: u64,
}

impl VirtualController {
    /// Waits for the next `TickState`; when this controller is due, calls
    /// `decide` (which sends the chunks and returns their refs) and sends
    /// `Decided{k}`. Returns the tick observed and whether it decided.
    pub async fn decide(
        &self,
        participant: &Participant,
        ticks: &mut TickStream,
        timeout: Duration,
        decide: impl FnOnce(Decision<'_>) -> Result<Vec<ChunkRef>, SessionError>,
    ) -> Result<(u64, bool), Interrupted> {
        let tick = ticks
            .wait_for(timeout, |t| matches!(t, VirtualTick::Observe(_)))
            .await
            .map_err(|err| match err {
                SessionError::Timeout(waited, _) => Interrupted::Timeout {
                    tick: 0,
                    phase: "observe",
                    waited,
                },
                other => Interrupted::Session(other),
            })?;
        let VirtualTick::Observe(state) = tick else {
            unreachable!("waited for an observe");
        };
        let k = state.tick;
        if !state.due.contains(participant.id()) {
            return Ok((k, false));
        }
        let latency_ticks = self.latency.draw(participant.id(), k);
        let t0 = k.saturating_add(latency_ticks).saturating_mul(self.tick_ns);
        let faulted = state
            .watermarks
            .iter()
            .filter_map(|w| self.faults.fault(w.channel, w.seq).map(|f| (w.channel, f)))
            .collect();
        let chunks = decide(Decision {
            state: &state,
            latency_ticks,
            t0_ns: i64::try_from(t0).unwrap_or(i64::MAX),
            faulted,
        })?;
        participant.decided(k, chunks).await?;
        Ok((k, true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draws_are_pure_functions_of_their_keys() {
        let a = ParticipantId::new("policy").expect("id");
        let b = ParticipantId::new("planner").expect("id");
        let model = LatencyModel::Uniform {
            seed: 7,
            min: 1,
            max: 4,
        };
        let draws: Vec<u64> = (0..64).map(|k| model.draw(&a, k)).collect();
        assert_eq!(
            draws,
            (0..64).map(|k| model.draw(&a, k)).collect::<Vec<_>>()
        );
        assert!(draws.iter().all(|d| (1..=4).contains(d)));
        for d in 1..=4 {
            assert!(draws.contains(&d), "{d} drawn");
        }
        assert_ne!(
            draws,
            (0..64).map(|k| model.draw(&b, k)).collect::<Vec<_>>()
        );
        assert_eq!(LatencyModel::Constant(3).draw(&a, 99), 3);

        let faults = Faults {
            seed: 11,
            drop_ppm: 100_000,
            freeze_ppm: 100_000,
            delay_ppm: 100_000,
            delay_ticks: 2,
        };
        let verdicts: Vec<Option<Fault>> = (1..=10_000)
            .map(|s| faults.fault(ChannelId(3), s))
            .collect();
        let count = |f: Option<Fault>| verdicts.iter().filter(|v| **v == f).count();
        for fault in [
            Some(Fault::Drop),
            Some(Fault::Freeze),
            Some(Fault::Delay(2)),
        ] {
            let n = count(fault);
            assert!((800..1200).contains(&n), "{fault:?}: {n} of 10000 at 10%");
        }
        assert_eq!(
            verdicts,
            (1..=10_000)
                .map(|s| faults.fault(ChannelId(3), s))
                .collect::<Vec<_>>()
        );
        assert_eq!(Faults::default().fault(ChannelId(3), 1), None);
    }
}
