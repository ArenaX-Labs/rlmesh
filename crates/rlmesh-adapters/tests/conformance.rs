use std::fs;
use std::path::PathBuf;

use rlmesh_adapters::v1::{
    EnvTags, FrameBuffers, ModelSpec, NoCustoms, NoEncodings, RolePolicy, SpaceView, Value,
    assemble_obs, reject_unsanctioned_roles_env, reject_unsanctioned_roles_model, resolve,
    transform_action, transform_obs,
};
use rlmesh_spaces::scalar::{Scalar, decode_scalars, encode_scalars};
use rlmesh_spaces::{DType, Tensor};
use serde_json::{Value as Json, json};

fn is_float_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::Float16 | DType::Float32 | DType::Float64)
}

fn assert_advisories(name: &str, adapter: &rlmesh_adapters::v1::ResolvedAdapter, expect: &Json) {
    let Some(expected) = expect["advisories_contain"].as_array() else {
        return;
    };
    let actual: Vec<String> = adapter
        .advisories()
        .iter()
        .map(|advisory| format!("{}: {}", advisory.severity.as_str(), advisory.message))
        .collect();
    for wanted in expected {
        let wanted = wanted.as_str().expect("advisory substring");
        assert!(
            actual.iter().any(|line| line.contains(wanted)),
            "{name}: no advisory contains {wanted:?}; got {actual:?}"
        );
    }
}

fn cases_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("conformance/v1/cases")
}

fn update_mode() -> bool {
    std::env::var("UPDATE_VECTORS").is_ok_and(|value| value == "1")
}

fn parse_inputs(case: &Json) -> (EnvTags, SpaceView, SpaceView, ModelSpec) {
    let tags: EnvTags = serde_json::from_value(case["env_tags"].clone()).expect("env_tags parses");
    let observation_space: SpaceView = serde_json::from_value(case["observation_space"].clone())
        .expect("observation_space parses");
    let action_space: SpaceView =
        serde_json::from_value(case["action_space"].clone()).expect("action_space parses");
    let model_spec: ModelSpec =
        serde_json::from_value(case["model_spec"].clone()).expect("model_spec parses");
    (tags, observation_space, action_space, model_spec)
}

fn dec(value: &Json) -> Value {
    match value["kind"].as_str().expect("value kind") {
        "text" => Value::Text(value["data"].as_str().expect("text data").to_owned()),
        "list" => Value::List(
            value["data"]
                .as_array()
                .expect("list data")
                .iter()
                .map(|item| match item {
                    Json::String(text) => Value::Text(text.clone()),
                    other => Value::Number(other.as_f64().expect("numeric item")),
                })
                .collect(),
        ),
        "map" => Value::Map(
            value["data"]
                .as_object()
                .expect("map data")
                .iter()
                .map(|(key, item)| (key.clone(), dec(item)))
                .collect(),
        ),
        "array" => {
            let dtype =
                DType::from_name(value["dtype"].as_str().expect("dtype")).expect("supported dtype");
            let shape: Vec<i64> = value["shape"]
                .as_array()
                .expect("shape")
                .iter()
                .map(|dim| dim.as_i64().expect("dim"))
                .collect();
            let scalars: Vec<Scalar> = value["data"]
                .as_array()
                .expect("array data")
                .iter()
                .map(|item| Scalar::Float(item.as_f64().expect("numeric element")))
                .collect();
            let bytes = encode_scalars(&scalars, dtype).expect("encode case array");
            Value::Tensor(Tensor::from_vec(bytes, shape, dtype).expect("case tensor"))
        }
        other => panic!("unknown value kind {other:?}"),
    }
}

fn enc(value: &Value) -> Json {
    match value {
        Value::None | Value::Bool(_) | Value::Int(_) | Value::Tuple(_) => {
            panic!("host custom value {value:?} has no conformance vector encoding")
        }
        Value::Text(text) => json!({"kind": "text", "data": text}),
        Value::Bytes(_) => panic!("bytes have no conformance vector encoding"),
        Value::Number(number) => panic!("bare number {number} has no vector encoding"),
        Value::List(items) => json!({
            "kind": "list",
            "data": items
                .iter()
                .map(|item| match item {
                    Value::Text(text) => Json::String(text.clone()),
                    Value::Number(number) => json!(number),
                    other => panic!("unsupported list item {other:?}"),
                })
                .collect::<Vec<Json>>(),
        }),
        Value::Map(entries) => json!({
            "kind": "map",
            "data": entries
                .iter()
                .map(|(key, item)| (key.clone(), enc(item)))
                .collect::<serde_json::Map<String, Json>>(),
        }),
        Value::Tensor(tensor) => {
            let dtype = tensor.dtype();
            let scalars =
                decode_scalars(&tensor.to_contiguous_bytes(), dtype).expect("decode tensor");
            let data: Vec<Json> = scalars
                .iter()
                .map(|scalar| {
                    if is_float_dtype(dtype) {
                        json!(scalar.to_f64(dtype))
                    } else {
                        json!(scalar.as_i64())
                    }
                })
                .collect();
            json!({
                "kind": "array",
                "dtype": dtype.name(),
                "shape": tensor.shape(),
                "data": data,
            })
        }
    }
}

fn assert_value(name: &str, key: &str, actual: &Value, expected: &Json, atol: f64) {
    match expected["kind"].as_str().expect("expected kind") {
        "map" => {
            let Value::Map(entries) = actual else {
                panic!("{name}/{key}: expected map, got {actual:?}");
            };
            let expected_entries = expected["data"].as_object().expect("conformance fixture");
            assert_eq!(
                entries.len(),
                expected_entries.len(),
                "{name}/{key}: map entry count"
            );
            for (entry_key, expected_item) in expected_entries {
                let item = entries
                    .get(entry_key)
                    .unwrap_or_else(|| panic!("{name}/{key}: missing map entry {entry_key:?}"));
                assert_value(
                    name,
                    &format!("{key}.{entry_key}"),
                    item,
                    expected_item,
                    atol,
                );
            }
        }
        "text" => {
            let Value::Text(text) = actual else {
                panic!("{name}/{key}: expected text, got {actual:?}");
            };
            assert_eq!(
                text,
                expected["data"].as_str().expect("conformance fixture"),
                "{name}/{key}"
            );
        }
        "list" => {
            let Value::List(items) = actual else {
                panic!("{name}/{key}: expected list, got {actual:?}");
            };
            let expected_items = expected["data"].as_array().expect("conformance fixture");
            assert_eq!(items.len(), expected_items.len(), "{name}/{key}: length");
            for (position, (item, expected_item)) in items.iter().zip(expected_items).enumerate() {
                match (item, expected_item) {
                    (Value::Text(text), Json::String(expected_text)) => {
                        assert_eq!(text, expected_text, "{name}/{key}[{position}]");
                    }
                    (Value::Number(number), other) => {
                        let expected_number = other.as_f64().expect("numeric item");
                        assert!(
                            (number - expected_number).abs() <= atol,
                            "{name}/{key}[{position}]: {number} vs {expected_number}"
                        );
                    }
                    (item, expected_item) => panic!(
                        "{name}/{key}[{position}]: mismatched kinds {item:?} vs \
                         {expected_item:?}"
                    ),
                }
            }
        }
        "array" => {
            let Value::Tensor(tensor) = actual else {
                panic!("{name}/{key}: expected array, got {actual:?}");
            };
            assert_eq!(
                tensor.dtype().name(),
                expected["dtype"].as_str().expect("conformance fixture"),
                "{name}/{key}: dtype"
            );
            let expected_shape: Vec<i64> = expected["shape"]
                .as_array()
                .expect("conformance fixture")
                .iter()
                .map(|dim| dim.as_i64().expect("conformance fixture"))
                .collect();
            assert_eq!(
                tensor.shape(),
                expected_shape.as_slice(),
                "{name}/{key}: shape"
            );
            let actual_values: Vec<f64> =
                decode_scalars(&tensor.to_contiguous_bytes(), tensor.dtype())
                    .expect("decode tensor")
                    .iter()
                    .map(|scalar| scalar.to_f64(tensor.dtype()))
                    .collect();
            let expected_values: Vec<f64> = expected["data"]
                .as_array()
                .expect("conformance fixture")
                .iter()
                .map(|item| item.as_f64().expect("conformance fixture"))
                .collect();
            assert_eq!(
                actual_values.len(),
                expected_values.len(),
                "{name}/{key}: element count"
            );
            for (position, (actual_value, expected_value)) in
                actual_values.iter().zip(&expected_values).enumerate()
            {
                assert!(
                    (actual_value - expected_value).abs() <= atol,
                    "{name}/{key}[{position}]: {actual_value} vs {expected_value}"
                );
            }
        }
        other => panic!("{name}/{key}: unknown expected kind {other:?}"),
    }
}

fn updated_case(name: &str, case: &Json) -> Json {
    let preserve_inputs = case["preserve_inputs"] == Json::Bool(true);
    let mut out = case.clone();
    match case["kind"].as_str().expect("case kind") {
        "serialization" | "role_policy" => {
            unreachable!("{name}: frozen vectors are not rewritten in update mode")
        }
        "resolve" => {
            let (tags, obs_space, action_space, model_spec) = parse_inputs(case);
            if !preserve_inputs {
                out["env_tags"] = serde_json::to_value(&tags).expect("serializes");
                out["observation_space"] = serde_json::to_value(&obs_space).expect("serializes");
                out["action_space"] = serde_json::to_value(&action_space).expect("serializes");
                out["model_spec"] = serde_json::to_value(&model_spec).expect("serializes");
            }
            out["expect"] = match resolve(&tags, &obs_space, &action_space, &model_spec, false) {
                Ok(adapter) => {
                    let mut expect = json!({"ok": true, "describe": adapter.describe()});
                    if let Some(curated) = case["expect"].get("advisories_contain") {
                        assert_advisories(name, &adapter, &case["expect"]);
                        expect["advisories_contain"] = curated.clone();
                    }
                    expect
                }
                Err(error) => {
                    let existing = case["expect"]["error_contains"].as_str();
                    let pinned = match existing {
                        Some(sub) if error.message.contains(sub) => sub.to_owned(),
                        _ => error.message.clone(),
                    };
                    json!({"error_contains": pinned})
                }
            };
        }
        "apply" => {
            let (tags, obs_space, action_space, model_spec) = parse_inputs(case);
            if !preserve_inputs {
                out["env_tags"] = serde_json::to_value(&tags).expect("serializes");
                out["observation_space"] = serde_json::to_value(&obs_space).expect("serializes");
                out["action_space"] = serde_json::to_value(&action_space).expect("serializes");
                out["model_spec"] = serde_json::to_value(&model_spec).expect("serializes");
            }
            let adapter = resolve(&tags, &obs_space, &action_space, &model_spec, false)
                .unwrap_or_else(|e| panic!("{name}: resolve failed: {e}"));
            let Value::Map(raw_obs) = dec(&case["observation"]) else {
                panic!("{name}: observation must decode to a map");
            };
            let payload = transform_obs(&adapter, &raw_obs, &NoCustoms)
                .unwrap_or_else(|e| panic!("{name}: transform_obs failed: {e}"));
            let action = transform_action(&adapter, &dec(&case["model_output"]))
                .unwrap_or_else(|e| panic!("{name}: transform_action failed: {e}"));
            let atol = case["expect"]["atol"].clone();
            out["expect"] = json!({
                "payload": enc(&payload),
                "action": enc(&Value::Tensor(action)),
                "atol": if atol.is_null() { json!(1e-6) } else { atol },
            });
        }
        "apply_sequence" => {
            let (tags, obs_space, action_space, model_spec) = parse_inputs(case);
            if !preserve_inputs {
                out["env_tags"] = serde_json::to_value(&tags).expect("serializes");
                out["observation_space"] = serde_json::to_value(&obs_space).expect("serializes");
                out["action_space"] = serde_json::to_value(&action_space).expect("serializes");
                out["model_spec"] = serde_json::to_value(&model_spec).expect("serializes");
            }
            let adapter = resolve(&tags, &obs_space, &action_space, &model_spec, false)
                .unwrap_or_else(|e| panic!("{name}: resolve failed: {e}"));
            let atol = case["expect"]["atol"].clone();
            let payloads: Vec<Json> = sequence_payloads(name, &adapter, &case["observations"])
                .iter()
                .map(enc)
                .collect();
            out["expect"] = json!({
                "payloads": payloads,
                "atol": if atol.is_null() { json!(1e-6) } else { atol },
            });
        }
        other => panic!("{name}: unknown case kind {other:?}"),
    }
    out
}

fn sequence_payloads(
    name: &str,
    adapter: &rlmesh_adapters::v1::ResolvedAdapter,
    observations: &Json,
) -> Vec<Value> {
    let mut buffers = FrameBuffers::new();
    observations
        .as_array()
        .unwrap_or_else(|| panic!("{name}: observations must be a list"))
        .iter()
        .map(|observation| {
            let Value::Map(raw_obs) = dec(observation) else {
                panic!("{name}: each observation must decode to a map");
            };
            let step = buffers.next_step("ep");
            buffers
                .advance_step("ep", step)
                .unwrap_or_else(|e| panic!("{name}: advance_step failed: {e}"));
            assemble_obs(
                adapter,
                &raw_obs,
                "ep",
                step,
                &mut buffers,
                &NoCustoms,
                &NoEncodings,
            )
            .unwrap_or_else(|e| panic!("{name}: assemble_obs failed: {e}"))
        })
        .collect()
}

fn verify_case(name: &str, case: &Json) {
    match case["kind"].as_str().expect("case kind") {
        "serialization" => {
            let doc = &case["doc"];
            let round_tripped = if case["side"] == "env" {
                let spec: EnvTags = serde_json::from_value(doc.clone())
                    .unwrap_or_else(|e| panic!("{name}: parse failed: {e}"));
                serde_json::to_value(&spec).expect("serializes")
            } else {
                let spec: ModelSpec = serde_json::from_value(doc.clone())
                    .unwrap_or_else(|e| panic!("{name}: parse failed: {e}"));
                serde_json::to_value(&spec).expect("serializes")
            };
            assert_eq!(&round_tripped, doc, "{name}: round trip mismatch");
        }
        "resolve" => {
            let (tags, obs_space, action_space, model_spec) = parse_inputs(case);
            let expect = &case["expect"];
            match resolve(&tags, &obs_space, &action_space, &model_spec, false) {
                Ok(adapter) => {
                    let expected = expect["describe"]
                        .as_str()
                        .unwrap_or_else(|| panic!("{name}: expected an error"));
                    assert_eq!(adapter.describe(), expected, "{name}: describe");
                    assert_advisories(name, &adapter, expect);
                }
                Err(error) => {
                    let expected = expect["error_contains"]
                        .as_str()
                        .unwrap_or_else(|| panic!("{name}: unexpected error: {error}"));
                    assert!(
                        error.message.contains(expected),
                        "{name}: error {:?} does not contain {:?}",
                        error.message,
                        expected
                    );
                }
            }
        }
        "apply" => {
            let (tags, obs_space, action_space, model_spec) = parse_inputs(case);
            let adapter = resolve(&tags, &obs_space, &action_space, &model_spec, false)
                .unwrap_or_else(|e| panic!("{name}: resolve failed: {e}"));
            let atol = case["expect"]["atol"].as_f64().expect("atol");

            let Value::Map(raw_obs) = dec(&case["observation"]) else {
                panic!("{name}: observation must decode to a map");
            };
            let payload = transform_obs(&adapter, &raw_obs, &NoCustoms)
                .unwrap_or_else(|e| panic!("{name}: transform_obs failed: {e}"));
            assert_value(name, "payload", &payload, &case["expect"]["payload"], atol);

            let action = transform_action(&adapter, &dec(&case["model_output"]))
                .unwrap_or_else(|e| panic!("{name}: transform_action failed: {e}"));
            assert_value(
                name,
                "action",
                &Value::Tensor(action),
                &case["expect"]["action"],
                atol,
            );
        }
        "role_policy" => {
            let policy = match case["policy"].as_str().expect("case policy") {
                "passthrough" => RolePolicy::Passthrough,
                "strict" => RolePolicy::Strict,
                "forbid" => RolePolicy::Forbid,
                other => panic!("{name}: unknown role policy {other:?}"),
            };
            let doc = case["doc"].clone();
            let outcome = if case["side"] == "env" {
                let tags: EnvTags = serde_json::from_value(doc)
                    .unwrap_or_else(|e| panic!("{name}: parse failed: {e}"));
                reject_unsanctioned_roles_env(&tags, policy)
            } else {
                let spec: ModelSpec = serde_json::from_value(doc)
                    .unwrap_or_else(|e| panic!("{name}: parse failed: {e}"));
                reject_unsanctioned_roles_model(&spec, policy)
            };
            match (outcome, case["expect"]["error_contains"].as_str()) {
                (Ok(()), None) => {}
                (Ok(()), Some(expected)) => {
                    panic!("{name}: expected rejection containing {expected:?}")
                }
                (Err(message), None) => panic!("{name}: unexpected rejection: {message}"),
                (Err(message), Some(expected)) => assert!(
                    message.contains(expected),
                    "{name}: rejection {message:?} does not contain {expected:?}"
                ),
            }
        }
        "apply_sequence" => {
            let (tags, obs_space, action_space, model_spec) = parse_inputs(case);
            let adapter = resolve(&tags, &obs_space, &action_space, &model_spec, false)
                .unwrap_or_else(|e| panic!("{name}: resolve failed: {e}"));
            let atol = case["expect"]["atol"].as_f64().expect("atol");
            let payloads = sequence_payloads(name, &adapter, &case["observations"]);
            let expected = case["expect"]["payloads"]
                .as_array()
                .unwrap_or_else(|| panic!("{name}: expect.payloads must be a list"));
            assert_eq!(payloads.len(), expected.len(), "{name}: step count");
            for (step, (payload, expected_payload)) in payloads.iter().zip(expected).enumerate() {
                assert_value(
                    name,
                    &format!("payloads[{step}]"),
                    payload,
                    expected_payload,
                    atol,
                );
            }
        }
        other => panic!("{name}: unknown case kind {other:?}"),
    }
}

#[test]
fn conformance_vectors() {
    let update = update_mode();
    let mut ran = 0usize;
    let entries = fs::read_dir(cases_dir()).expect("conformance cases directory");
    for entry in entries {
        let path = entry.expect("readable directory entry").path();
        if path.extension().is_none_or(|ext| ext != "json") {
            continue;
        }
        let name = path
            .file_stem()
            .expect("conformance fixture")
            .to_string_lossy()
            .to_string();
        let case: Json = serde_json::from_str(&fs::read_to_string(&path).expect("readable case"))
            .expect("case parses as JSON");

        if update && !matches!(case["kind"].as_str(), Some("serialization" | "role_policy")) {
            let rewritten = updated_case(&name, &case);
            let mut text = serde_json::to_string_pretty(&rewritten).expect("serializes");
            text.push('\n');
            fs::write(&path, text).expect("writable case");
        } else {
            verify_case(&name, &case);
        }
        ran += 1;
    }
    assert!(ran >= 118, "expected at least 118 vectors, ran {ran}");
}

#[test]
fn existing_pad_and_zero_fill_text_is_unchanged() {
    let path = cases_dir().join("resolve_rot6d_with_optional_zero_fill.json");
    let case: Json = serde_json::from_str(&fs::read_to_string(&path).expect("readable case"))
        .expect("case parses as JSON");
    assert_eq!(
        case["expect"]["describe"].as_str().expect("describe text"),
        "observation:\n  \"instruction\" <- text \"instruction\"\n  \"state\" <- \
         concat(eef_pos[:3], eef_quat (quat_xyzw->rot6d), gripper[:1], zeros(3)#right_arm), pad to 16\
         \naction:\n  \"action/delta_eef_pos\" <- model[0:3]\n  \"action/delta_eef_rot\" <- \
         model[3:9] (rot6d->axis_angle)\n  \"action/gripper\" <- model[9:10]\n  clip to \
         (-1.0, 1.0)\nprojections:\n  select_state: model input \"state\": state role \
         \"proprio/gripper\" reads the first 1 of the 2 dims of env feature \"gripper\"",
    );
    let (tags, obs_space, action_space, model_spec) = parse_inputs(&case);
    let adapter = resolve(&tags, &obs_space, &action_space, &model_spec, false).expect("resolve");
    assert_eq!(adapter.describe(), case["expect"]["describe"]);
}
