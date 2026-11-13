//! Runs, bindings and hooks driven end to end against local and served envs.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::*;
use crate::spaces::{Discrete, DiscreteOptions, Space};
use crate::{
    EnvFactory, EnvServer, EnvServerOptions, ErrorClass, InfoDict, Kwargs, MadeEnv, Metadata,
    ModelOptions, Result, Value,
};

mod adapted;
mod bindings;
mod envs;
mod runs;

/// One recorded `reset(seed, options)`.
type Reset = (Option<i64>, Option<InfoDict>);

/// A three-step env: observations count steps, reward is the action.
struct Steps {
    step: i64,
    closes: Arc<AtomicUsize>,
    resets: Arc<Mutex<Vec<Reset>>>,
    metadata: Option<Metadata>,
    /// Fail the step that reaches this count, as an env whose container died.
    fail_at: Option<i64>,
}

impl Steps {
    fn new() -> Steps {
        Steps {
            step: 0,
            closes: Arc::default(),
            resets: Arc::default(),
            metadata: None,
            fail_at: None,
        }
    }
}

impl Env for Steps {
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
        self.resets.lock().unwrap().push((seed, options.cloned()));
        self.step = 0;
        Ok((Value::Int(0), InfoDict::new()))
    }
    fn step(&mut self, action: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
        self.step += 1;
        if self.fail_at == Some(self.step) {
            return Err(Error::Runtime("env gone".into()));
        }
        let reward = match action {
            Some(Value::Int(action)) => action as f64,
            _ => -1.0,
        };
        let mut info = InfoDict::new();
        info.insert("is_success", Value::Bool(true));
        Ok((Value::Int(self.step), reward, self.step >= 3, false, info))
    }
    fn close(&mut self) -> Result<()> {
        self.closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn metadata(&self) -> Option<Metadata> {
        self.metadata.clone()
    }
}

fn ones() -> Model {
    Model::new(
        |_observation: Value| Ok(Value::Int(1)),
        ModelOptions::default(),
    )
    .unwrap()
}

fn quick() -> EnvServerOptions {
    EnvServerOptions {
        address: Some(crate::tests::ENV_ADDRESS.into()),
        options: Some(crate::tests::env_serve_options()),
        ..Default::default()
    }
}

/// Records every hook call, in order.
#[derive(Default)]
struct Log {
    calls: Vec<String>,
    context: Option<Arc<dyn RunContext>>,
    async_context: Vec<bool>,
    fail_on_step: Option<i64>,
}

impl RunHooks for Log {
    fn on_run_start(&mut self, context: Arc<dyn RunContext>) -> Result<()> {
        self.calls.push(format!(
            "run_start contract={} num_envs={}",
            context.contract()?.is_some(),
            context.num_envs()?
        ));
        self.async_context.push(drive::in_async_context());
        self.context = Some(context);
        Ok(())
    }
    fn on_episode_start(&mut self, episode: i64, seed: Option<i64>) -> Result<()> {
        self.calls.push(format!("episode_start {episode} {seed:?}"));
        Ok(())
    }
    fn on_step(&mut self, event: &StepEvent) -> Result<()> {
        self.async_context.push(drive::in_async_context());
        self.calls.push(format!(
            "step {} {} obs={} action={} reward={}",
            event.episode,
            event.step,
            crate::env::factory::py_repr(&event.observation),
            crate::env::factory::py_repr(&event.action),
            event.reward
        ));
        if self.fail_on_step == Some(event.step) {
            return Err(Error::invalid_argument("hook failed"));
        }
        Ok(())
    }
    fn on_episode_end(&mut self, result: &EpisodeResult) -> Result<()> {
        self.calls.push(format!(
            "episode_end {} steps={} reward={}",
            result.index, result.steps, result.reward
        ));
        Ok(())
    }
    fn on_run_end(&mut self, result: &RunResult) -> Result<()> {
        self.calls
            .push(format!("run_end episodes={}", result.num_episodes()));
        Ok(())
    }
}

fn expected_two_episodes() -> Vec<String> {
    let mut calls = vec!["run_start contract=false num_envs=1".to_string()];
    for episode in 0..2 {
        calls.push(format!("episode_start {episode} Some({})", 10 + episode));
        for step in 0..3 {
            calls.push(format!(
                "step {episode} {step} obs={step} action=1 reward=1"
            ));
        }
        calls.push(format!("episode_end {episode} steps=3 reward=3"));
    }
    calls.push("run_end episodes=2".to_string());
    calls
}

/// A factory whose made envs report their closes.
struct Factory {
    closes: Arc<AtomicUsize>,
    made: usize,
}

impl EnvFactory for Factory {
    fn make(&mut self, _kwargs: &Kwargs) -> Result<MadeEnv> {
        self.made += 1;
        let mut env = Steps::new();
        env.closes = Arc::clone(&self.closes);
        Ok(MadeEnv::env(env))
    }
}

fn steps_server() -> EnvServer {
    let server = EnvServer::new(Steps::new(), quick()).unwrap();
    server.start().unwrap();
    server
}
