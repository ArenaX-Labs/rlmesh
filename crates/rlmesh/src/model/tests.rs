use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;

use super::*;
use crate::{Result, binding::spaces};

struct SmokeModel {
    predicts: Arc<AtomicUsize>,
    closes: Arc<AtomicUsize>,
}

#[async_trait]
impl ModelHandler for SmokeModel {
    async fn predict(&mut self, observation: ModelObservation) -> Result<Vec<spaces::SpaceValue>> {
        self.predicts.fetch_add(1, Ordering::SeqCst);
        // One action per lane; the action space is a Uint8 Box[1] (SmokeEnv).
        Ok((0..observation.num_envs)
            .map(|_| {
                spaces::SpaceValue::Box(
                    spaces::Tensor::from_vec(vec![0u8], vec![1], spaces::DType::Uint8).unwrap(),
                )
            })
            .collect())
    }

    async fn on_close(&mut self) -> Result<()> {
        self.closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[test]
fn episode_budget_resolves_like_python() {
    let base = RunLocalOptions::parse(":0/env").unwrap();
    // Neither given: one episode (Python's `episodes=None`).
    assert_eq!(base.episode_budget().unwrap(), 1);
    // Seeds only: one episode per seed.
    let seeded = base.clone().episode_seeds(vec![3, 4, 5]);
    assert_eq!(seeded.episode_budget().unwrap(), 3);
    // An explicit count wins, zero included (runs nothing).
    assert_eq!(base.clone().for_episodes(7).episode_budget().unwrap(), 7);
    assert_eq!(base.clone().for_episodes(0).episode_budget().unwrap(), 0);
    // Both given: the counts must match.
    assert_eq!(seeded.clone().for_episodes(3).episode_budget().unwrap(), 3);
    let error = seeded.for_episodes(2).episode_budget().unwrap_err();
    assert!(
        error
            .to_string()
            .contains("does not match len(episode_seeds)=3"),
        "{error}"
    );
}

/// `for_episodes(0)` runs nothing and never dials the env; a mismatched seed
/// count is refused before dialing.
#[tokio::test]
async fn run_local_zero_episodes_runs_nothing() {
    // Nothing listens on port 1: dialing would fail the run.
    let unreachable = ":1/env";
    let predicts = Arc::new(AtomicUsize::new(0));
    let report = ModelWorker::new(SmokeModel {
        predicts: Arc::clone(&predicts),
        closes: Arc::new(AtomicUsize::new(0)),
    })
    .run_local_async(RunLocalOptions::parse(unreachable).unwrap().for_episodes(0))
    .await
    .unwrap();
    assert_eq!(report.total_episodes, 0);
    assert!(report.episodes.is_empty());
    assert_eq!(predicts.load(Ordering::SeqCst), 0);

    let error = ModelWorker::new(SmokeModel {
        predicts: Arc::clone(&predicts),
        closes: Arc::new(AtomicUsize::new(0)),
    })
    .run_local_async(
        RunLocalOptions::parse(unreachable)
            .unwrap()
            .for_episodes(2)
            .episode_seeds(vec![1]),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("does not match"), "{error}");
}

#[test]
fn run_local_options_cover_all_axes() {
    let run = RunLocalOptions::parse(":0/env")
        .unwrap()
        .for_episodes(5)
        .base_seed(123)
        .execution_horizon(8)
        .prefetch_lead(2);
    assert_eq!(run.max_episodes, Some(5));
    assert_eq!(run.base_seed, Some(123));
    assert_eq!(run.execution_horizon, 8);
    assert_eq!(run.prefetch_lead, 2);
    assert_eq!(
        run.env_address,
        crate::ext::ConnectAddress::parse(":0/env").unwrap()
    );

    let default_run = RunLocalOptions::new(crate::ext::ConnectAddress::parse(":0/env").unwrap());
    assert_eq!(default_run.max_episodes, None);
    assert_eq!(default_run.base_seed, None);
    assert_eq!(default_run.execution_horizon, 1);
    assert_eq!(default_run.prefetch_lead, 0);
    assert_eq!(default_run.execution_horizon(0).execution_horizon, 1);
}
