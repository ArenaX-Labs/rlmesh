//! The Gymnasium-shaped env traits served end to end.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::*;
use crate::ext::TensorExt;
use crate::spaces::{BoxOptions, Discrete, DiscreteOptions};
use crate::{DType, Error, ErrorClass, Kwargs, Tensor};

mod factories;
mod remote_client;
mod serving;
mod shutdown;

thread_local! {
    /// One client runtime per test thread: a client's connection lives on the
    /// runtime that dialed it.
    static CLIENT: tokio::runtime::Runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
}

fn quick() -> EnvServerOptions {
    EnvServerOptions::default()
}

fn client<T>(future: impl std::future::Future<Output = T>) -> T {
    CLIENT.with(|runtime| runtime.block_on(future))
}

#[derive(Default)]
struct Counters {
    closes: AtomicUsize,
    threads: Mutex<Vec<std::thread::ThreadId>>,
}

/// A three-step counter env with scripted failures.
struct Counter {
    counters: Arc<Counters>,
    step: i64,
    fail_step: Option<Error>,
    metadata: Option<Metadata>,
    /// How long each step takes.
    slow: Option<Duration>,
}

impl Counter {
    fn new(counters: &Arc<Counters>) -> Counter {
        Counter {
            counters: Arc::clone(counters),
            step: 0,
            fail_step: None,
            metadata: None,
            slow: None,
        }
    }

    fn record(&self) {
        self.counters
            .threads
            .lock()
            .unwrap()
            .push(std::thread::current().id());
    }
}

impl Env for Counter {
    fn observation_space(&self) -> Space {
        Discrete::new(5, DiscreteOptions::default()).unwrap().into()
    }
    fn action_space(&self) -> Space {
        Discrete::new(2, DiscreteOptions::default()).unwrap().into()
    }
    fn reset(
        &mut self,
        seed: Option<i64>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        self.record();
        self.step = seed.unwrap_or(0);
        let mut info = InfoDict::new();
        if let Some(trial) = trial_index(options, None)? {
            info.insert("trial", trial);
        }
        Ok((Value::Int(self.step), info))
    }
    fn step(&mut self, action: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
        self.record();
        if let Some(err) = self.fail_step.clone() {
            return Err(err);
        }
        if let Some(slow) = self.slow {
            std::thread::sleep(slow);
        }
        self.step += 1;
        let reward = match action {
            Some(Value::Int(action)) => action as f64,
            _ => -1.0,
        };
        Ok((
            Value::Int(self.step % 5),
            reward,
            self.step >= 3,
            false,
            InfoDict::new(),
        ))
    }
    fn close(&mut self) -> Result<()> {
        self.record();
        self.counters.closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn metadata(&self) -> Option<Metadata> {
        self.metadata.clone()
    }
    fn id(&self) -> Option<String> {
        Some("Counter-v0".to_string())
    }
}

fn reset(remote: &mut RemoteEnv, seed: Option<i64>) -> (Value, InfoDict) {
    remote.reset(seed, None).unwrap()
}

#[allow(clippy::type_complexity)]
fn step(remote: &mut RemoteEnv, action: i64) -> crate::Result<(Value, f64, bool, bool, InfoDict)> {
    remote.step(Value::Int(action))
}

fn dial(address: &str) -> crate::Result<RemoteEnv> {
    RemoteEnv::new(RemoteEnvOptions::default().address(address))
}

/// A two-lane vector env over Discrete actions.
struct Pair {
    num_envs: u64,
    seen: Arc<Mutex<Vec<Value>>>,
}

impl VectorEnv for Pair {
    fn num_envs(&self) -> u64 {
        self.num_envs
    }
    fn single_observation_space(&self) -> Space {
        crate::spaces::Box::new(-1.0, 1.0, [2], BoxOptions::default())
            .unwrap()
            .into()
    }
    fn single_action_space(&self) -> Space {
        Discrete::new(3, DiscreteOptions::default()).unwrap().into()
    }
    fn reset(
        &mut self,
        seed: Option<VectorSeed>,
        _options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        self.seen
            .lock()
            .unwrap()
            .push(Value::Text(format!("{seed:?}")));
        Ok((self.observations()?, InfoDict::new()))
    }
    fn step(
        &mut self,
        actions: Value,
    ) -> Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)> {
        self.seen.lock().unwrap().push(actions);
        let n = self.num_envs as usize;
        Ok((
            self.observations()?,
            vec![0.5; n],
            vec![false; n],
            vec![false; n],
            InfoDict::new(),
        ))
    }
}

impl Pair {
    fn observations(&self) -> Result<Value> {
        let n = self.num_envs as usize;
        let data: Vec<u8> = std::iter::repeat_n(0.25f32, 2 * n)
            .flat_map(f32::to_le_bytes)
            .collect();
        Ok(Value::Tensor(Tensor::from_vec(
            data,
            &[n, 2],
            DType::Float32,
        )?))
    }
}
