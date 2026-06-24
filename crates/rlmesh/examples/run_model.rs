//! Drive a model against a running environment server.
//!
//! This is the Rust side of the quickstart loop. Start `serve_env` first, then
//! run this: the worker connects to the environment, calls `predict` on every
//! observation, and returns one action per step until the episode budget runs
//! out. The same worker can drive an environment served from any language.
//!
//! ```bash
//! cargo run -p rlmesh --example serve_env    # in one terminal
//! cargo run -p rlmesh --example run_model    # in another
//! ```
#![allow(clippy::print_stdout)]

use rlmesh::prelude::*;
use rlmesh::spaces::SpaceValue;

/// A model that always takes action 0, one per lane. The codec encodes the
/// typed value against the route's action space, so the same `predict` works
/// for any discrete env without hand-packing bytes.
struct ConstantModel;

#[async_trait::async_trait]
impl ModelHandler for ConstantModel {
    async fn predict(&mut self, obs: ModelObservation) -> rlmesh::Result<Vec<SpaceValue>> {
        Ok((0..obs.num_envs).map(|_| SpaceValue::Discrete(0)).collect())
    }
}

#[tokio::main]
async fn main() -> rlmesh::Result<()> {
    let address = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:5555".to_string());

    let model = ConstantModel;

    println!("driving model against {address} for 3 episodes");
    ModelWorker::new(model)
        .run_local_async(RunLocalOptions::parse(&address)?.for_episodes(3))
        .await?;
    println!("done");
    Ok(())
}
