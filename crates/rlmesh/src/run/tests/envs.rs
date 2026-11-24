use super::*;

struct Lanes {
    counts: Vec<i64>,
    ended: Vec<bool>,
}

impl crate::VectorEnv for Lanes {
    fn num_envs(&self) -> u64 {
        self.counts.len() as u64
    }
    fn single_observation_space(&self) -> Space {
        Discrete::new(5, DiscreteOptions::default()).unwrap().into()
    }
    fn single_action_space(&self) -> Space {
        Discrete::new(2, DiscreteOptions::default()).unwrap().into()
    }
    fn reset(
        &mut self,
        _seed: Option<crate::VectorSeed>,
        _options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        self.counts.iter_mut().for_each(|count| *count = 0);
        self.ended.iter_mut().for_each(|ended| *ended = false);
        Ok((self.observations(), InfoDict::new()))
    }
    fn step(
        &mut self,
        _actions: Value,
    ) -> Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)> {
        let mut rewards = Vec::new();
        let mut terminated = Vec::new();
        for lane in 0..self.counts.len() {
            if self.ended[lane] {
                self.counts[lane] = 0;
                self.ended[lane] = false;
                rewards.push(0.0);
                terminated.push(false);
            } else {
                self.counts[lane] += 1;
                self.ended[lane] = self.counts[lane] >= 3;
                rewards.push(1.0);
                terminated.push(self.ended[lane]);
            }
        }
        let truncated = vec![false; self.counts.len()];
        Ok((
            self.observations(),
            rewards,
            terminated,
            truncated,
            InfoDict::new(),
        ))
    }
    fn metadata(&self) -> Option<Metadata> {
        let mut metadata = Metadata::new();
        metadata.insert("autoreset_mode", Value::Text("NextStep".into()));
        Some(metadata)
    }
}

impl Lanes {
    fn observations(&self) -> Value {
        Value::List(self.counts.iter().map(|count| Value::Int(*count)).collect())
    }
}

#[test]
fn a_borrowed_vector_env_runs_its_lanes_on_the_native_loop() {
    let model = Model::new(
        |observation: Value| {
            let lanes = match &observation {
                Value::Tensor(tensor) => tensor.shape().first().copied().unwrap_or(1),
                Value::List(items) => items.len(),
                _ => return Ok(Value::Int(1)),
            };
            Ok(Value::List(vec![Value::Int(1); lanes]))
        },
        ModelOptions::default(),
    )
    .unwrap();
    let mut env = Lanes {
        counts: vec![0; 2],
        ended: vec![false; 2],
    };
    let result = model
        .run(
            EnvTarget::vector(&mut env),
            RunOptions {
                episodes: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(result.num_episodes(), 2);
    assert!(result.episodes.iter().all(|episode| episode.steps == 3));
    assert!(
        result
            .episodes
            .iter()
            .all(|episode| episode.trial.is_none())
    );
}

#[tokio::test]
async fn a_base_seed_derives_the_same_reset_seeds_in_every_process() {
    use crate::model::random::RandomSampleHandler;
    use crate::model::worker::{ModelWorker, RunLocalOptions};

    let env = Steps::new();
    let resets = Arc::clone(&env.resets);
    let server = crate::EnvServer::new(env, quick()).unwrap();
    server.start().unwrap();
    let options = RunLocalOptions::parse(server.address())
        .unwrap()
        .for_episodes(2)
        .base_seed(4242);
    ModelWorker::new(RandomSampleHandler::default())
        .run_local_async(options)
        .await
        .unwrap();
    server.shutdown_async().await.unwrap();
    let seeds: Vec<Option<i64>> = resets.lock().unwrap().iter().map(|reset| reset.0).collect();
    assert_eq!(seeds, [Some(3_549_056_084), Some(3_732_510_325)]);
}

#[tokio::test]
async fn a_v2_handle_connects_and_drops_inside_a_current_thread_runtime() {
    let server = crate::EnvServer::new(Steps::new(), quick()).unwrap();
    server.start().unwrap();
    let mut remote =
        crate::RemoteEnv::new_async(crate::RemoteEnvOptions::default().address(server.address()))
            .await
            .unwrap();
    remote.reset_async(None, None).await.unwrap();
    drop(remote);
    server.shutdown_async().await.unwrap();
}

#[test]
fn a_native_run_through_an_open_v2_handle_says_the_handle_holds_the_env() {
    let server = steps_server();
    let mut remote =
        crate::RemoteEnv::new(crate::RemoteEnvOptions::default().address(server.address()))
            .unwrap();
    let err = ones()
        .run(
            &mut remote,
            RunOptions {
                episodes: Some(1),
                ..Default::default()
            },
        )
        .unwrap_err();
    assert_eq!(err.class(), ErrorClass::TypeError, "{err}");
    assert!(
        err.to_string()
            .starts_with("run(RemoteEnv(address)) is gone in rlmesh 0.2"),
        "{err}"
    );
    remote.close().unwrap();
    server.shutdown().unwrap();
}
