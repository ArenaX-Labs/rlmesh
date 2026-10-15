//! Drive a model against a running environment server.
//!
//! This is the Rust side of the quickstart loop. Start `serve_env` first, then
//! run this: the model connects to the environment, `predict` runs on every
//! observation, and one action goes back per step until the episode budget
//! runs out. The same model can drive an environment served from any
//! language.
//!
//! ```bash
//! cargo run -p rlmesh --example serve_env    # in one terminal
//! cargo run -p rlmesh --example run_model    # in another
//! ```
#![allow(clippy::print_stdout)]

use rlmesh::prelude::*;
use rlmesh::{Model, ModelOptions};

/// A policy that always takes action 0. The codec encodes the value against
/// the env's action space, so the same `predict` works for any discrete env.
struct ConstantPolicy;

impl Policy for ConstantPolicy {
    fn predict(&mut self, _observation: Value) -> rlmesh::Result<Value> {
        Ok(Value::Int(0))
    }
}

fn main() -> rlmesh::Result<()> {
    let address = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:5555".to_string());

    let model = Model::new(ConstantPolicy, ModelOptions::default())?;

    println!("driving model against {address} for 3 episodes");
    // The in-process run path until `rlmesh::run` lands.
    rlmesh::binding::model::run_model_local(
        &model,
        rlmesh::binding::model::RunLocalOptions::parse(&address)?.for_episodes(3),
    )?;
    model.close()?;
    println!("done");
    Ok(())
}
