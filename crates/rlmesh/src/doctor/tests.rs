//! The probes end to end against an env and a model served in this process.

use std::sync::Arc;
use std::time::Duration;

use rlmesh_spaces::spaces::{BoxSpaceBuilder, DictSpaceBuilder};
use rlmesh_spaces::{DType, SpaceSpec as WireSpec};
use serde_json::json;

use super::*;
use crate::spaces::{Space, SpaceSpec};
use crate::value::space_codec;
use crate::{Env, EnvServer, EnvServerOptions, InfoDict, Value};

fn observation_space() -> WireSpec {
    DictSpaceBuilder::new()
        .insert(
            "cam",
            BoxSpaceBuilder::scalar(0.0, 255.0, vec![16, 16, 3])
                .dtype(DType::Uint8)
                .build()
                .unwrap(),
        )
        .insert(
            "pos",
            BoxSpaceBuilder::scalar(-1.0, 1.0, vec![2])
                .dtype(DType::Float32)
                .build()
                .unwrap(),
        )
        .build()
        .unwrap()
}

fn action_space() -> WireSpec {
    BoxSpaceBuilder::scalar(-1.0, 1.0, vec![2])
        .dtype(DType::Float32)
        .build()
        .unwrap()
}

/// A camera and a position, both a pure function of the reset seed and the
/// step count, so its replays reproduce.
struct Scene {
    seed: u64,
    step: u64,
}

impl Scene {
    fn observation(&self) -> crate::Result<Value> {
        let space = observation_space();
        let value = rlmesh_spaces::sample_seeded(&space, self.seed + self.step).unwrap();
        space_codec::value_from_space(&space, value)
    }
}

impl Env for Scene {
    fn observation_space(&self) -> Space {
        Space::new(SpaceSpec::from_wire(observation_space()))
    }
    fn action_space(&self) -> Space {
        Space::new(SpaceSpec::from_wire(action_space()))
    }
    fn reset(
        &mut self,
        seed: Option<i64>,
        _: Option<&InfoDict>,
    ) -> crate::Result<(Value, InfoDict)> {
        self.seed = seed.unwrap_or_default() as u64;
        self.step = 0;
        Ok((self.observation()?, InfoDict::new()))
    }
    fn step(&mut self, _: Option<Value>) -> crate::Result<(Value, f64, bool, bool, InfoDict)> {
        self.step += 1;
        Ok((self.observation()?, 0.0, false, false, InfoDict::new()))
    }
}

fn serve_scene() -> EnvServer {
    let server = EnvServer::new(
        Scene { seed: 0, step: 0 },
        EnvServerOptions {
            address: Some(crate::tests::ENV_ADDRESS.into()),
            options: Some(crate::tests::env_serve_options()),
            ..Default::default()
        },
    )
    .unwrap();
    server.start().unwrap();
    server
}

fn statuses(report: &ProbeReport) -> Vec<(&str, CheckStatus)> {
    report
        .checks
        .iter()
        .map(|check| (check.name.as_str(), check.status))
        .collect()
}

#[test]
fn an_env_that_matches_its_describe_passes_and_one_that_does_not_fails() {
    let server = serve_scene();
    let output = tempfile::tempdir().unwrap();
    let declared = |observation: serde_json::Value| {
        json!({"env_spec": {
            "observation_space": observation,
            "action_space": rlmesh_adapters::v1::space_spec_to_json(&action_space()).unwrap(),
        }})
    };
    let truthful = declared(rlmesh_adapters::v1::space_spec_to_json(&observation_space()).unwrap());
    let mut lying = truthful.clone();
    lying["env_spec"]["observation_space"]["details"]["spaces"]["pos"]["shape"] = json!([3]);

    let report = probe_env(
        server.address(),
        &EnvProbeOptions {
            describe: Some(truthful),
            output_dir: Some(output.path().to_path_buf()),
            connect_timeout: Duration::from_secs(10),
        },
    );
    use CheckStatus::{Fail, Pass, Skip};
    assert_eq!(
        statuses(&report),
        [
            ("env_handshake", Pass),
            ("env_spec_matches_describe", Pass),
            ("env_contract", Skip),
            ("env_repro", Pass),
            ("env_step_conforms", Pass),
            ("env_step_latency", Pass),
            ("preview", Pass),
        ],
        "{:#?}",
        report.checks
    );
    assert_eq!(report.observed.env_state_reproducible, Some(true));
    let preview = report.preview.as_ref().expect("a preview");
    assert!(output.path().join(&preview.video).is_file());
    assert_eq!(report.blocker(), None);

    let report = probe_env(
        server.address(),
        &EnvProbeOptions {
            describe: Some(lying),
            output_dir: None,
            connect_timeout: Duration::from_secs(10),
        },
    );
    let mismatch = &report.checks[1];
    assert_eq!(
        (mismatch.name.as_str(), mismatch.status),
        ("env_spec_matches_describe", Fail)
    );
    assert!(
        mismatch.detail.contains("pos.shape[0]: 2 vs declared 3"),
        "{}",
        mismatch.detail
    );
    server.shutdown().unwrap();
}

#[test]
fn a_model_is_checked_against_its_spec_or_its_reference_env() {
    use crate::lockstep::{ModelBinding, ModelService};

    let runtime = crate::run::drive::runtime();
    let binding = runtime.block_on(ModelBinding::bind(":0/random")).unwrap();
    let address = binding.address();
    let stop = tokio_util::sync::CancellationToken::new();
    let service = ModelService::new(
        crate::model::random::RandomSampleHandler::default(),
        String::new(),
    );
    runtime.spawn(binding.serve(Arc::new(service), Duration::from_secs(5), stop.clone()));
    let server = serve_scene();
    let spec = json!({"model_spec": {
        "input": {"state": {"type": "state", "components": [{"role": "proprio/eef_pos"}]}},
        "output": {"components": [{"role": "action/delta_eef_pos", "dim": 3}]},
    }});
    let adapter_less = json!({"model_spec": null});
    let cases = [
        (
            Some(spec),
            None,
            CheckStatus::Pass,
            SYNTHETIC_ENV_ID.to_string(),
        ),
        (
            Some(adapter_less.clone()),
            Some(server.address().to_string()),
            CheckStatus::Pass,
            server.address().to_string(),
        ),
        (Some(adapter_less), None, CheckStatus::Warn, String::new()),
        // Nothing to synthesize from: no label, and this model describes nothing.
        (None, None, CheckStatus::Fail, String::new()),
    ];
    for (describe, reference_env, handshake, env_id) in cases {
        let report = probe_model(
            &address,
            &ModelProbeOptions {
                describe,
                reference_env,
                connect_timeout: Duration::from_secs(10),
            },
        );
        let names: Vec<&str> = report.checks.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, model::MODEL_CHECK_NAMES);
        assert_eq!(report.checks[0].status, handshake, "{:#?}", report.checks);
        assert_eq!(report.env.env_id, env_id);
        if handshake == CheckStatus::Pass {
            // The random baseline answers in range, but never alike.
            for (name, status) in [
                ("resolve", CheckStatus::Pass),
                ("shape_single", CheckStatus::Pass),
                ("shape_batch", CheckStatus::Pass),
                ("determinism_floor", CheckStatus::Warn),
            ] {
                let check = report.checks.iter().find(|c| c.name == name).unwrap();
                assert_eq!(check.status, status, "{name}: {}", check.detail);
            }
            assert_eq!(report.observed.deterministic, Some(false));
        }
    }

    let bench = bench_model(
        &address,
        &BenchOptions {
            probe: ModelProbeOptions {
                describe: Some(json!({"model_spec": null})),
                reference_env: Some(server.address().to_string()),
                connect_timeout: Duration::from_secs(10),
            },
            sizes: vec![1, 2],
            iterations: 3,
            warmup: 1,
        },
    );
    let sizes: Vec<u32> = bench.report.baseline.iter().map(|p| p.batch_size).collect();
    assert_eq!(sizes, [1, 2], "{:#?}", bench.report.checks);
    assert_eq!(bench.report.observed.max_ok_batch, Some(2));
    assert!(bench.report.recommendation.max_batch_size.is_some());
    stop.cancel();
    server.shutdown().unwrap();
}
