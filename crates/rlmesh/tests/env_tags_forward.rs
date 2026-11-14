//! A model resolving the tags of an env written by a newer release: a field
//! its reader does not know is refused unless the env marked it ignorable
//! (`x-` / `ext.`). A model driving a robot must not silently drop a
//! modifier that changes what an action means.

#![allow(clippy::unwrap_used)]

mod common;

use std::sync::Arc;

use common::{CATALOG_MODEL_SPEC, CatalogEnv, catalog_tags};
use rlmesh::adapters::ModelSpec;
use rlmesh::prelude::*;
use rlmesh::{DType, Model, ModelOptions, RunOptions, Tensor};
use rlmesh_spaces::AutoresetMode;

/// Holds still: seven zeros.
struct Still;

impl Policy for Still {
    const CORNERS: Corners = Corners::PREDICT;

    fn predict(&mut self, _observation: Value) -> rlmesh::Result<Value> {
        Ok(Value::Tensor(Tensor::new(&[0; 28], &[7], DType::Float32)?))
    }
}

/// Runs one episode of the catalog env publishing `tags`.
fn run_against(tags: &serde_json::Value) -> rlmesh::Result<usize> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let binding = rt.block_on(rlmesh::lockstep::bind(":0/catalog")).unwrap();
    let address = binding.address();
    rt.spawn(binding.serve(
        Arc::new(CatalogEnv::new(1, AutoresetMode::Disabled, false).with_tags(tags)),
        rlmesh::lockstep::ServeOptions::default(),
        tokio_util::sync::CancellationToken::new(),
    ));
    let model = Model::new(
        Still,
        ModelOptions {
            spec: Some(ModelSpec::from_json(CATALOG_MODEL_SPEC).unwrap().into()),
            ..Default::default()
        },
    )
    .unwrap();
    let run = model.run(
        address.as_str(),
        RunOptions {
            episodes: Some(1),
            ..Default::default()
        },
    );
    model.close().unwrap();
    run.map(|result| result.num_episodes())
}

#[test]
fn a_newer_envs_ignorable_tag_fields_resolve_and_run() {
    let mut tags = catalog_tags();
    tags["observation"]["cam"]["x-lens"] = "fisheye".into();
    tags["action"]["components"][0]["ext.vendor"] = serde_json::json!({"gain": 2});
    assert_eq!(run_against(&tags).unwrap(), 1);
}

#[test]
fn a_newer_envs_bare_tag_field_is_refused_naming_it_and_the_older_side() {
    let mut tags = catalog_tags();
    // A modifier a later release might add, which 0.2 has never seen.
    tags["action"]["components"][2]["deadband"] = 0.05.into();
    let refused = run_against(&tags).expect_err("a modifier this model cannot read");
    let message = refused.to_string();
    assert!(message.contains("\"deadband\""), "{message}");
    assert!(message.contains("action component[2]"), "{message}");
    assert!(
        message.contains("the env's tags are newer than this model's rlmesh"),
        "{message}"
    );
    assert!(message.contains("`x-`"), "{message}");
}
