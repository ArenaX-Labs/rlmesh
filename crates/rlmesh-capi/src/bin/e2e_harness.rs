#![allow(clippy::print_stderr)]

use std::process::{Command, ExitCode};
use std::sync::{Arc, Mutex};

use rlmesh::spaces::{Box, BoxOptions, Space};
use rlmesh::{DType, EnvServer, EnvServerOptions, InfoDict, Tensor, Value};

const STEPS_PER_EPISODE: u8 = 2;

struct SmokeEnv {
    steps: u8,
    reset_seeds: Arc<Mutex<Vec<Option<i64>>>>,
}

fn u8_box(high: u8) -> Space {
    Box::new(
        0,
        high,
        [1],
        BoxOptions {
            dtype: DType::Uint8,
        },
    )
    .expect("valid space")
    .into()
}

fn u8_value(value: u8) -> Value {
    Value::Tensor(Tensor::new(&[value], &[1], DType::Uint8).expect("tensor"))
}

impl rlmesh::Env for SmokeEnv {
    fn observation_space(&self) -> Space {
        u8_box(255)
    }
    fn action_space(&self) -> Space {
        u8_box(1)
    }
    fn id(&self) -> Option<String> {
        Some("SmokeEnv-capi-e2e".to_string())
    }

    fn reset(
        &mut self,
        seed: Option<i64>,
        _options: Option<&InfoDict>,
    ) -> rlmesh::Result<(Value, InfoDict)> {
        self.steps = 0;
        self.reset_seeds.lock().expect("reset seed log").push(seed);
        Ok((u8_value(0), InfoDict::new()))
    }

    fn step(
        &mut self,
        _action: Option<Value>,
    ) -> rlmesh::Result<(Value, f64, bool, bool, InfoDict)> {
        self.steps += 1;
        Ok((
            u8_value(self.steps),
            1.0,
            self.steps >= STEPS_PER_EPISODE,
            false,
            InfoDict::new(),
        ))
    }
}

const EPISODES: &str = "3";

fn main() -> ExitCode {
    let Some(binary) = std::env::args().nth(1) else {
        eprintln!("usage: e2e_harness <model-binary> [expected-stdout-substring]");
        return ExitCode::FAILURE;
    };
    let expect = std::env::args().nth(2);
    let reset_seeds = Arc::new(Mutex::new(Vec::new()));
    let env = SmokeEnv {
        steps: 0,
        reset_seeds: Arc::clone(&reset_seeds),
    };
    let options = EnvServerOptions {
        address: Some(":0/env".to_string()),
        ..Default::default()
    };
    let server = match EnvServer::new(env, options).and_then(|server| {
        server.start()?;
        Ok(server)
    }) {
        Ok(server) => server,
        Err(err) => {
            eprintln!("failed to serve the env: {err}");
            return ExitCode::FAILURE;
        }
    };
    let output = Command::new(&binary)
        .arg(server.address())
        .arg(EPISODES)
        .output();
    let _ = server.shutdown();
    let output = match output {
        Ok(output) => output,
        Err(err) => {
            eprintln!("failed to spawn model binary: {err}");
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
    let seeds: Vec<i64> = reset_seeds
        .lock()
        .expect("reset seed log")
        .iter()
        .flatten()
        .copied()
        .collect();
    for seed in seeds {
        let expect = format!("predict 1 seed {}", rlmesh::predict_seed(seed, 1));
        if !stdout.contains(&expect) {
            eprintln!("model stdout did not contain {expect:?} for episode seed {seed}");
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}
