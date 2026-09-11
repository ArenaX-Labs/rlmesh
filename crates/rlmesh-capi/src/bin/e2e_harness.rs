//! End-to-end loopback harness: serve a trivial environment, then run a compiled
//! C/C++ model binary (`<bin> <tcp-address> 3`) against it. A binary, not a
//! `cargo test`, so the C/C++ toolchain stays out of `cargo test --workspace`.
//!
//! Usage: `e2e_harness <model-binary> [expected-stdout-substring]`. The optional
//! second argument is asserted against the child's stdout, so a smoke that
//! prints its run report is checked for the real numbers, not just its exit code.
#![allow(clippy::print_stderr)]

use std::process::{Command, ExitCode};

use async_trait::async_trait;
use rlmesh::spaces;

/// A minimal single environment: a uint8 `Box[1]` obs/action, one step per
/// episode (reset → 0, step → 1 then terminated).
struct SmokeEnv {
    obs_space: spaces::SpaceSpec,
    action_space: spaces::SpaceSpec,
    env_contract: spaces::EnvContract,
}

impl SmokeEnv {
    fn new() -> Self {
        let obs_space = spaces::spaces::BoxSpaceBuilder::scalar(0.0, 255.0, vec![1])
            .dtype(spaces::DType::Uint8)
            .build()
            .expect("valid obs space");
        let action_space = spaces::spaces::BoxSpaceBuilder::scalar(0.0, 1.0, vec![1])
            .dtype(spaces::DType::Uint8)
            .build()
            .expect("valid action space");
        let env_contract = spaces::EnvContract {
            id: "SmokeEnv-capi-e2e".to_string(),
            observation_space: Some(obs_space.clone()),
            action_space: Some(action_space.clone()),
            num_envs: 1,
            ..Default::default()
        };
        Self {
            obs_space,
            action_space,
            env_contract,
        }
    }
}

fn u8_box(value: u8) -> spaces::SpaceValue {
    spaces::SpaceValue::Box(
        spaces::Tensor::from_vec(vec![value], vec![1], spaces::DType::Uint8).expect("tensor"),
    )
}

#[async_trait]
impl rlmesh::Env for SmokeEnv {
    fn observation_space(&self) -> &spaces::SpaceSpec {
        &self.obs_space
    }
    fn action_space(&self) -> &spaces::SpaceSpec {
        &self.action_space
    }
    fn env_contract(&self) -> &spaces::EnvContract {
        &self.env_contract
    }

    async fn reset(
        &mut self,
        _req: rlmesh::ResetRequest,
    ) -> Result<rlmesh::ResetResult, spaces::EnvRuntimeError> {
        Ok(rlmesh::ResetResult {
            observation: Some(u8_box(0)),
            info: None,
            episode_id: None,
        })
    }

    async fn step(
        &mut self,
        _req: rlmesh::StepRequest,
    ) -> Result<rlmesh::StepResult, spaces::EnvRuntimeError> {
        Ok(rlmesh::StepResult {
            observation: Some(u8_box(1)),
            reward: 1.0,
            terminated: true,
            truncated: false,
            info: None,
        })
    }

    async fn render(
        &mut self,
        _req: rlmesh::RenderRequest,
    ) -> Result<rlmesh::RenderResult, spaces::EnvRuntimeError> {
        Ok(rlmesh::RenderResult::default())
    }

    async fn close(
        &mut self,
        _req: rlmesh::CloseRequest,
    ) -> Result<rlmesh::CloseResult, spaces::EnvRuntimeError> {
        Ok(rlmesh::CloseResult)
    }
}

/// Episodes the model binary is asked to run (more than one, so episode
/// teardown and the run report carry real numbers).
const EPISODES: &str = "3";

fn main() -> ExitCode {
    let Some(binary) = std::env::args().nth(1) else {
        eprintln!("usage: e2e_harness <model-binary> [expected-stdout-substring]");
        return ExitCode::FAILURE;
    };
    let expect = std::env::args().nth(2);
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            eprintln!("failed to build runtime: {err}");
            return ExitCode::FAILURE;
        }
    };
    runtime.block_on(run(binary, expect))
}

async fn run(binary: String, expect: Option<String>) -> ExitCode {
    // Bind first: the listener is accepting before the model connects (port 0 →
    // OS-assigned), so no readiness sleep is needed.
    let bound = match rlmesh::EnvServer::new(SmokeEnv::new())
        .bind(rlmesh::BindAddress::Tcp {
            host: "127.0.0.1".to_string(),
            port: 0,
        })
        .await
    {
        Ok(bound) => bound,
        Err(err) => {
            eprintln!("failed to bind env: {err}");
            return ExitCode::FAILURE;
        }
    };
    let address = bound.local_addr().to_string();
    let server = tokio::spawn(async move {
        let _ = bound.serve().await;
    });

    let output = tokio::task::spawn_blocking(move || {
        Command::new(&binary).arg(&address).arg(EPISODES).output()
    })
    .await;
    server.abort();

    let output = match output {
        Ok(Ok(output)) => output,
        Ok(Err(err)) => {
            eprintln!("failed to spawn model binary: {err}");
            return ExitCode::FAILURE;
        }
        Err(err) => {
            eprintln!("harness join error: {err}");
            return ExitCode::FAILURE;
        }
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    eprint!("{stdout}");
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
    if !output.status.success() {
        eprintln!("model binary exited with {}", output.status);
        return ExitCode::FAILURE;
    }
    if let Some(expect) = expect
        && !stdout.contains(&expect)
    {
        eprintln!("model stdout did not contain {expect:?}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
