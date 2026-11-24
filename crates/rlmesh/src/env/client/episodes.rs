//! The client's episode bookkeeping: per-lane episode ids, step counts and
//! returns, and the completions they produce.

use crate::binding::spaces;
use crate::env::types::StepResult as VectorStepResult;

#[derive(Clone, Debug, Default)]
pub(super) struct Episodes {
    lanes: Vec<Lane>,
    interrupted: Vec<super::super::types::EpisodeMetadata>,
}

#[derive(Clone, Debug, Default)]
enum Lane {
    #[default]
    Idle,
    Active(Episode),
    Pending {
        same_step: bool,
    },
}

#[derive(Clone, Debug)]
struct Episode {
    id: String,
    seed: Option<i64>,
    steps: i64,
    reward: f64,
    start_ns: i64,
}

fn unix_ns() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_nanos()).unwrap_or(i64::MAX))
}

impl Episode {
    fn new(id: String, seed: Option<i64>) -> Self {
        Self {
            id,
            seed,
            steps: 0,
            reward: 0.0,
            start_ns: unix_ns(),
        }
    }

    fn end(
        self,
        lane: usize,
        terminated: bool,
        truncated: bool,
    ) -> super::super::types::EpisodeMetadata {
        let end_ns = unix_ns();
        super::super::types::EpisodeMetadata {
            episode_id: self.id,
            seed: self.seed,
            env_index: i32::try_from(lane).unwrap_or(i32::MAX),
            step_count: self.steps,
            cumulative_reward: self.reward,
            terminated,
            truncated,
            start_timestamp_ns: self.start_ns,
            end_timestamp_ns: end_ns,
            duration_ms: (end_ns - self.start_ns).max(0) / 1_000_000,
            final_info: None,
        }
    }
}

impl Episodes {
    pub(super) fn new(num_envs: usize) -> Self {
        Self {
            lanes: (0..num_envs).map(|_| Lane::Idle).collect(),
            interrupted: Vec::new(),
        }
    }

    pub(super) fn reset(&mut self, lanes: &[u32], seeds: &[i64], ids: &[String]) {
        let all: Vec<u32>;
        let lanes = if lanes.is_empty() {
            all = (0..self.lanes.len() as u32).collect();
            &all
        } else {
            lanes
        };
        for (pos, &lane) in lanes.iter().enumerate() {
            let Some(slot) = self.lanes.get_mut(lane as usize) else {
                continue;
            };
            let id = ids.get(pos).cloned().unwrap_or_default();
            let started = Lane::Active(Episode::new(id, seeds.get(pos).copied()));
            if let Lane::Active(old) = std::mem::replace(slot, started) {
                self.interrupted.push(old.end(lane as usize, false, true));
            }
        }
    }

    pub(super) fn step(
        &mut self,
        result: &VectorStepResult,
        ids: &[String],
        mode: spaces::types::AutoresetMode,
    ) -> Vec<super::super::types::EpisodeMetadata> {
        use spaces::types::AutoresetMode;
        let mut completed = std::mem::take(&mut self.interrupted);
        for (lane, slot) in self.lanes.iter_mut().enumerate() {
            let terminated = result.terminated.get(lane).copied().unwrap_or(false);
            let truncated = result.truncated.get(lane).copied().unwrap_or(false);
            let id = || ids.get(lane).cloned().unwrap_or_default();
            let mut episode = match std::mem::take(slot) {
                Lane::Active(episode) => episode,
                Lane::Pending { same_step: false } => {
                    *slot = Lane::Active(Episode::new(id(), None));
                    continue;
                }
                Lane::Pending { same_step: true } => Episode::new(id(), None),
                Lane::Idle => continue,
            };
            episode.steps += 1;
            episode.reward += result.rewards.get(lane).copied().unwrap_or(0.0);
            if terminated || truncated {
                completed.push(episode.end(lane, terminated, truncated));
                *slot = match mode {
                    AutoresetMode::NextStep => Lane::Pending { same_step: false },
                    AutoresetMode::SameStep => Lane::Pending { same_step: true },
                    AutoresetMode::Disabled => Lane::Idle,
                };
            } else {
                *slot = Lane::Active(episode);
            }
        }
        completed
    }
}
