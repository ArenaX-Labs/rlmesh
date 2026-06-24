//! Serve an environment over gRPC.
//!
//! This is the Rust side of the quickstart loop. It hosts a tiny environment on
//! a socket; any RLMesh client can then step it from another process, in any
//! supported language. Drive it with the `run_model` example, or with the
//! Python client in `examples/python/quickstart/eval.py`.
//!
//! ```bash
//! cargo run -p rlmesh --example serve_env            # 127.0.0.1:5555
//! cargo run -p rlmesh --example serve_env 0.0.0.0:7000
//! ```
#![allow(clippy::print_stdout)]

use rlmesh::prelude::*;
use rlmesh::spaces::{self, SpaceValue};

/// A three-step counter. The observation is the step index, every action earns
/// a reward of 1.0, and the episode terminates after the third step.
struct CounterEnv {
    observation_space: SpaceSpec,
    action_space: SpaceSpec,
    contract: EnvContract,
    step: i64,
}

impl CounterEnv {
    fn new() -> Self {
        let observation_space = spaces::spaces::DiscreteBuilder::new(5)
            .build()
            .expect("discrete observation space spec is valid");
        let action_space = spaces::spaces::DiscreteBuilder::new(2)
            .build()
            .expect("discrete action space spec is valid");
        let contract = EnvContract {
            id: "CounterEnv-v0".to_string(),
            observation_space: Some(observation_space.clone()),
            action_space: Some(action_space.clone()),
            num_envs: 1,
            ..Default::default()
        };
        Self {
            observation_space,
            action_space,
            contract,
            step: 0,
        }
    }
}

#[async_trait::async_trait]
impl Env for CounterEnv {
    fn observation_space(&self) -> &SpaceSpec {
        &self.observation_space
    }

    fn action_space(&self) -> &SpaceSpec {
        &self.action_space
    }

    fn env_contract(&self) -> &EnvContract {
        &self.contract
    }

    async fn reset(&mut self, _req: ResetRequest) -> Result<ResetResult, EnvRuntimeError> {
        self.step = 0;
        Ok(ResetResult {
            observation: Some(SpaceValue::Discrete(self.step)),
            info: None,
            episode_id: None,
        })
    }

    async fn step(&mut self, _req: StepRequest) -> Result<StepResult, EnvRuntimeError> {
        self.step += 1;
        Ok(StepResult {
            observation: Some(SpaceValue::Discrete(self.step % 5)),
            reward: 1.0,
            terminated: self.step >= 3,
            truncated: false,
            info: None,
        })
    }

    async fn render(
        &mut self,
        _req: spaces::RenderRequest,
    ) -> Result<spaces::RenderResult, EnvRuntimeError> {
        Ok(spaces::RenderResult::default())
    }

    async fn close(&mut self, _req: spaces::CloseRequest) -> Result<CloseResult, EnvRuntimeError> {
        Ok(CloseResult)
    }
}

#[tokio::main]
async fn main() -> rlmesh::Result<()> {
    let address = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:5555".to_string());

    // Bind first so the resolved address is known before serving.
    let server = EnvServer::new(CounterEnv::new())
        .bind(BindAddress::parse(&address)?)
        .await?;
    println!("serving CounterEnv on {}", server.local_addr());
    server.serve().await
}
