#![allow(clippy::print_stdout)]

use rlmesh::prelude::*;
use rlmesh::{Model, ModelOptions, RunOptions};

struct ConstantPolicy;

impl Policy for ConstantPolicy {
    fn predict(&mut self, _observation: Value) -> rlmesh::Result<Value> {
        Ok(Value::Int(0))
    }
}

fn main() -> rlmesh::Result<()> {
    let address = std::env::args()
        .nth(1)
        .unwrap_or_else(|| ":5555".to_string());

    let model = Model::new(ConstantPolicy, ModelOptions::default())?;

    println!("driving model against {address} for 3 episodes");
    let result = model.run(
        address.as_str(),
        RunOptions {
            episodes: Some(3),
            ..Default::default()
        },
    )?;
    model.close()?;
    println!(
        "done: {} episodes, {} steps",
        result.num_episodes(),
        result.total_steps()
    );
    Ok(())
}

#[cfg(test)]
#[allow(dead_code)]
#[path = "serve_env.rs"]
mod serve_env;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_run_drives_three_counter_episodes() -> rlmesh::Result<()> {
        let server = rlmesh::EnvServer::new(
            serve_env::CounterEnv::new()?,
            rlmesh::EnvServerOptions {
                address: Some(":0/counter".into()),
                ..Default::default()
            },
        )?;
        server.start()?;
        let model = Model::new(ConstantPolicy, ModelOptions::default())?;
        let result = model.run(
            server.address(),
            RunOptions {
                episodes: Some(3),
                ..Default::default()
            },
        )?;
        model.close()?;
        server.shutdown()?;
        assert_eq!(result.num_episodes(), 3);
        assert_eq!(result.total_steps(), 9);
        Ok(())
    }
}
