#![allow(clippy::print_stdout)]

use rlmesh::prelude::*;
use rlmesh::spaces::{Discrete, DiscreteOptions};
use rlmesh::{EnvServer, EnvServerOptions};

pub(crate) struct CounterEnv {
    observation_space: Space,
    action_space: Space,
    step: i64,
}

impl CounterEnv {
    pub(crate) fn new() -> rlmesh::Result<Self> {
        Ok(Self {
            observation_space: Discrete::new(5, DiscreteOptions::default())?.into(),
            action_space: Discrete::new(2, DiscreteOptions::default())?.into(),
            step: 0,
        })
    }
}

impl Env for CounterEnv {
    fn observation_space(&self) -> Space {
        self.observation_space.clone()
    }

    fn action_space(&self) -> Space {
        self.action_space.clone()
    }

    fn id(&self) -> Option<String> {
        Some("CounterEnv-v0".to_string())
    }

    fn reset(
        &mut self,
        _seed: Option<i64>,
        _options: Option<&InfoDict>,
    ) -> rlmesh::Result<(Value, InfoDict)> {
        self.step = 0;
        Ok((Value::Int(self.step), InfoDict::new()))
    }

    fn step(
        &mut self,
        _action: Option<Value>,
    ) -> rlmesh::Result<(Value, f64, bool, bool, InfoDict)> {
        self.step += 1;
        Ok((
            Value::Int(self.step % 5),
            1.0,
            self.step >= 3,
            false,
            InfoDict::new(),
        ))
    }
}

fn main() -> rlmesh::Result<()> {
    let address = std::env::args()
        .nth(1)
        .unwrap_or_else(|| ":5555".to_string());

    let server = EnvServer::new(
        CounterEnv::new()?,
        EnvServerOptions {
            address: Some(address),
            ..Default::default()
        },
    )?;
    println!("serving CounterEnv on {}", server.address());
    match server.serve() {
        Err(rlmesh::Error::Cancelled) => Ok(()),
        other => other,
    }
}
