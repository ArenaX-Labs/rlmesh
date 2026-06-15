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

use rlmesh::encode_action;
use rlmesh::prelude::*;
use rlmesh::spaces::{self, SpaceValue};

/// A model that always takes action 0. The action is encoded against the
/// environment's action space, so the same `predict` works for any discrete
/// env without hand-packing bytes.
struct ConstantModel {
    action_space: SpaceSpec,
}

#[async_trait::async_trait]
impl ModelHandler for ConstantModel {
    async fn predict(&mut self, _obs: ModelObservation) -> rlmesh::Result<BinaryPayload> {
        encode_action(&SpaceValue::Discrete(0), &self.action_space)
    }
}

#[tokio::main]
async fn main() -> rlmesh::Result<()> {
    let address = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:5555".to_string());

    let model = ConstantModel {
        action_space: spaces::spaces::DiscreteBuilder::new(2)
            .build()
            .expect("discrete action space spec is valid"),
    };

    println!("driving model against {address} for 3 episodes");
    ModelWorker::new(model)
        .run_local_async(RunLocalOptions::parse(&address)?.for_episodes(3))
        .await?;
    println!("done");
    Ok(())
}
