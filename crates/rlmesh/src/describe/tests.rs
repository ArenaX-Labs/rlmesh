//! Describe envelopes for model and env factory types and instances.

use super::*;
use crate::model::{LoadKey, ModelOptions};
use crate::spaces::{Discrete, DiscreteOptions, Space};
use crate::{Env, ErrorClass, InfoDict, Variant, VariantOptions};

const SPEC: &str = r#"{
    "input": {"state": {"type": "state", "container": "list", "dtype": "float32",
                        "components": [{"role": "proprio/gripper", "dim": 1}]}},
    "output": {"components": [{"role": "action/gripper", "dim": 1, "range": [0.0, 10.0]}]}
}"#;

#[derive(Default)]
struct Planner;

impl Policy for Planner {
    const CORNERS: Corners = Corners::PREDICT.union(Corners::PREDICT_CHUNK);
    const NATIVE_CHUNK: Option<u32> = Some(4);
    const LOAD_KEYS: Option<&'static [LoadKey]> =
        Some(&[LoadKey::required("checkpoint"), LoadKey::optional("device")]);

    fn spec() -> Option<SpecArg> {
        Some(crate::ext::model_spec_from_json(SPEC).unwrap().into())
    }

    fn workflow_edition() -> Option<String> {
        Some(crate::current_workflow_edition().to_string())
    }

    fn load(&mut self, _config: &Kwargs) -> Result<()> {
        Ok(())
    }
}

fn json(text: &str) -> Json {
    serde_json::from_str(text).unwrap()
}

#[test]
fn a_model_type_describes_its_declarations_without_constructing() {
    let envelope =
        json(&describe_json(Describable::model::<Planner>(), Default::default()).unwrap());
    assert_eq!(envelope["kind"], "model");
    assert_eq!(envelope["schema_version"], 1);
    let qualname = envelope["target"]["qualname"].as_str().unwrap();
    assert!(
        qualname.starts_with("rust:") && qualname.ends_with("Planner"),
        "{qualname}"
    );
    assert_eq!(envelope["target"]["entrypoint"], Json::Null);
    assert_eq!(envelope["corners"], json(r#"["predict", "predict_chunk"]"#));
    assert_eq!(envelope["native_chunk"], 4);
    assert_eq!(
        envelope["params"],
        json(
            r#"{"param_spec": null, "signature_tier": [
                {"name": "checkpoint", "type": "str", "default": null, "required": true},
                {"name": "device", "type": "str", "default": null, "required": false}]}"#
        )
    );
    assert!(envelope["model_spec"].is_object());
    let runtime = &envelope["runtime"];
    assert_eq!(runtime["component"], "rlmesh-rust");
    assert_eq!(runtime["language"], "rust");
    assert!(runtime["preferred_workflow_edition"].is_string());
    let metadata = Model::describe::<Planner>().unwrap();
    assert_eq!(metadata.get("kind"), Some(&Value::Text("model".into())));
}

struct NoCorner;
impl Policy for NoCorner {
    const NATIVE_CHUNK: Option<u32> = Some(2);
}

struct ZeroChunk;
impl Policy for ZeroChunk {
    const CORNERS: Corners = Corners::PREDICT.union(Corners::PREDICT_CHUNK);
    const NATIVE_CHUNK: Option<u32> = Some(0);
}

#[test]
fn a_model_type_construction_refuses_is_refused() {
    let error = describe(Describable::model::<NoCorner>(), Default::default()).unwrap_err();
    assert_eq!(error.class(), ErrorClass::TypeError);
    let error = describe(Describable::model::<ZeroChunk>(), Default::default()).unwrap_err();
    assert_eq!(error.class(), ErrorClass::ValueError);
}

#[test]
fn a_built_model_reports_its_policys_items_and_its_own_spec() {
    let model = Model::new(
        |_obs: Value| Ok(Value::Int(0)),
        ModelOptions {
            spec: Some(crate::ext::model_spec_from_json(SPEC).unwrap().into()),
            ..Default::default()
        },
    )
    .unwrap();
    let envelope = json(&describe_json(&model, Default::default()).unwrap());
    assert!(envelope["model_spec"].is_object(), "the model's own spec");
    assert_eq!(envelope["corners"], json(r#"["predict"]"#));
    assert_eq!(envelope["params"]["signature_tier"], json("[]"));
}

#[test]
fn describe_refuses_a_contradicting_kind_and_a_bad_timestamp() {
    let error = describe(
        Describable::model::<Planner>(),
        DescribeOptions {
            kind: Some(DescribeKind::Env),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert_eq!(error.class(), ErrorClass::ValueError);
    let error = describe(
        Describable::model::<Planner>(),
        DescribeOptions {
            generated_at: Some("yesterday".into()),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert_eq!(error.class(), ErrorClass::ValueError);
    assert!(
        error.to_string().starts_with("invalid describe envelope"),
        "{error}"
    );
    let envelope = json(
        &describe_json(
            Describable::model::<Planner>(),
            DescribeJsonOptions {
                generated_at: Some("2026-10-15T00:00:00Z".into()),
                ..Default::default()
            },
        )
        .unwrap(),
    );
    assert_eq!(envelope["generated_at"], "2026-10-15T00:00:00Z");
}

#[test]
fn a_given_edition_is_used_as_given() {
    let edition = |given: &str| {
        let text = describe_json(
            Describable::model::<Planner>(),
            DescribeJsonOptions {
                workflow_edition: Some(given.into()),
                ..Default::default()
            },
        )
        .unwrap();
        json(&text)["runtime"]["preferred_workflow_edition"].clone()
    };
    assert_eq!(edition(" 2099.01 "), "2099.01", "never validated");
    assert_eq!(edition(""), rlmesh_proto::CURRENT_WORKFLOW_EDITION);
}

/// A three-step env whose observation space is `Discrete(size)`.
struct Sized {
    size: i64,
}

impl Env for Sized {
    fn observation_space(&self) -> Space {
        Discrete::new(self.size, DiscreteOptions::default())
            .unwrap()
            .into()
    }
    fn action_space(&self) -> Space {
        Discrete::new(2, DiscreteOptions::default()).unwrap().into()
    }
    fn reset(&mut self, _: Option<i64>, _: Option<&InfoDict>) -> Result<(Value, InfoDict)> {
        Ok((Value::Int(0), InfoDict::new()))
    }
    fn step(&mut self, _: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
        Ok((Value::Int(0), 1.0, true, false, InfoDict::new()))
    }
}

#[derive(Default)]
struct Branched {
    makes: usize,
}

impl EnvFactory for Branched {
    fn tag_params() -> Vec<TagParam> {
        vec![TagParam {
            name: "size".into(),
            default_: Value::Int(3),
            choices: vec![Value::Int(2), Value::Int(3)],
        }]
    }
    fn enumerate_variants() -> Result<Option<Vec<Variant>>> {
        Ok(Some(vec![Variant::new(
            "small",
            [("size".to_string(), Value::Int(2))].into_iter().collect(),
            VariantOptions::default(),
        )]))
    }
    fn make(&mut self, kwargs: &Kwargs) -> Result<MadeEnv> {
        self.makes += 1;
        let size = kwargs.get("size").unwrap().as_int()?;
        Ok(MadeEnv::env(Sized { size }))
    }
}

#[test]
fn a_factory_type_describes_every_contract_branch() {
    let envelope =
        json(&describe_json(Describable::env_factory::<Branched>(), Default::default()).unwrap());
    assert_eq!(envelope["kind"], "env");
    assert_eq!(
        envelope["env_spec"]["observation_space"]["details"]["n"], 3,
        "{envelope}"
    );
    assert_eq!(envelope["env_tags"], Json::Null);
    assert_eq!(
        envelope["params"]["signature_tier"],
        json(r#"[{"name": "size", "type": "int", "default": 3, "required": false}]"#)
    );
    let contracts = &envelope["env_contracts"];
    assert_eq!(contracts["discriminants"], json(r#"["size"]"#));
    let branches = contracts["branches"].as_array().unwrap();
    assert_eq!(branches.len(), 2);
    assert_eq!(branches[0]["default"], true);
    assert_eq!(branches[0]["params"], json(r#"{"size": 3}"#));
    assert_eq!(branches[0]["env_spec"], envelope["env_spec"]);
    assert_eq!(branches[1]["params"], json(r#"{"size": 2}"#));
    assert_eq!(
        branches[1]["env_spec"]["observation_space"]["details"]["n"],
        2
    );
    assert_eq!(envelope["variants"]["catalog"][0]["id"], "small");
    assert!(EnvFactoryExt_describe_is_the_type_form());
}

#[allow(non_snake_case)]
fn EnvFactoryExt_describe_is_the_type_form() -> bool {
    let metadata = <Branched as EnvFactoryExt>::describe().unwrap();
    metadata.get("kind") == Some(&Value::Text("env".into()))
}

#[test]
fn an_instance_builds_on_itself_and_a_served_env_replaces_the_default_build() {
    let mut factory = Branched::default();
    let served = Sized { size: 4 };
    let envelope = json(
        &describe_json(
            &mut factory,
            DescribeJsonOptions {
                served_env: Some(ServedEnv::env(&served)),
                ..Default::default()
            },
        )
        .unwrap(),
    );
    assert_eq!(envelope["env_spec"]["observation_space"]["details"]["n"], 4);
    assert_eq!(factory.makes, 1, "only the non-default branch was built");
    let error = describe_json(
        &mut factory,
        DescribeJsonOptions {
            served_env: Some(ServedEnv::Lanes(Vec::new())),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert_eq!(error.class(), ErrorClass::ValueError);
}

#[derive(Default)]
struct Broken;

impl EnvFactory for Broken {
    fn make(&mut self, _: &Kwargs) -> Result<MadeEnv> {
        Err(Error::Type("no simulator".into()))
    }
}

#[test]
fn a_failed_build_is_a_badge_not_a_failed_call() {
    let envelope =
        json(&describe_json(Describable::env_factory::<Broken>(), Default::default()).unwrap());
    assert_eq!(
        envelope["env_spec"],
        json(r#"{"error": "no simulator", "error_type": "TypeError"}"#)
    );
    assert!(envelope.get("env_contracts").is_none());
}
