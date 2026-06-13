//! Run the v1 adapter conformance vectors against this implementation.
//!
//! Snapshot-style: expectations live in `conformance/v1/cases/*.json`.
//! `UPDATE_VECTORS=1 cargo test -p rlmesh-adapters` rewrites the `expect`
//! blocks (and normalizes spec documents) from current behavior — review
//! the diff before committing; a changed vector is a semantic change to
//! v1 and must be additive. New cases are authored by hand: write the
//! inputs with an empty `expect`, then run update mode once.

use std::fs;
use std::path::PathBuf;

use rlmesh_adapters::v1::{
    ArrayData, Dtype, EnvAnnotations, ModelIoSpec, NoCustoms, SpaceView, Value, resolve,
};
use serde_json::{Value as Json, json};

fn cases_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("conformance/v1/cases")
}

fn update_mode() -> bool {
    std::env::var("UPDATE_VECTORS").is_ok_and(|value| value == "1")
}

fn parse_inputs(case: &Json) -> (EnvAnnotations, SpaceView, SpaceView, ModelIoSpec) {
    let annotations: EnvAnnotations =
        serde_json::from_value(case["env_annotations"].clone()).expect("env_annotations parses");
    let observation_space: SpaceView = serde_json::from_value(case["observation_space"].clone())
        .expect("observation_space parses");
    let action_space: SpaceView =
        serde_json::from_value(case["action_space"].clone()).expect("action_space parses");
    let model_spec: ModelIoSpec =
        serde_json::from_value(case["model_spec"].clone()).expect("model_spec parses");
    (annotations, observation_space, action_space, model_spec)
}

/// Decode the conformance value encoding into an engine [`Value`].
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
                Dtype::parse(value["dtype"].as_str().expect("dtype")).expect("supported dtype");
            let shape: Vec<usize> = value["shape"]
                .as_array()
                .expect("shape")
                .iter()
                .map(|dim| dim.as_u64().expect("dim") as usize)
                .collect();
            let numbers: Vec<f64> = value["data"]
                .as_array()
                .expect("array data")
                .iter()
                .map(|item| item.as_f64().expect("numeric element"))
                .collect();
            let data = match dtype {
                Dtype::U8 => ArrayData::U8(numbers.iter().map(|&x| x as u8).collect()),
                Dtype::I32 => ArrayData::I32(numbers.iter().map(|&x| x as i32).collect()),
                Dtype::I64 => ArrayData::I64(numbers.iter().map(|&x| x as i64).collect()),
                Dtype::F32 => ArrayData::F32(numbers.iter().map(|&x| x as f32).collect()),
                Dtype::F64 => ArrayData::F64(numbers),
            };
            let array = rlmesh_adapters::v1::Array { dtype, shape, data };
            assert_eq!(
                array.shape.iter().product::<usize>(),
                array.len(),
                "array shape/data length mismatch in case input"
            );
            Value::Array(array)
        }
        other => panic!("unknown value kind {other:?}"),
    }
}

/// Encode an engine [`Value`] into the conformance value encoding
/// (inverse of [`dec`]); used by update mode to write expectations.
fn enc(value: &Value) -> Json {
    match value {
        Value::Text(text) => json!({"kind": "text", "data": text}),
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
        Value::Array(array) => {
            let data: Vec<Json> = match &array.data {
                ArrayData::U8(values) => values.iter().map(|&x| json!(x)).collect(),
                ArrayData::I32(values) => values.iter().map(|&x| json!(x)).collect(),
                ArrayData::I64(values) => values.iter().map(|&x| json!(x)).collect(),
                ArrayData::F32(values) => values.iter().map(|&x| json!(f64::from(x))).collect(),
                ArrayData::F64(values) => values.iter().map(|&x| json!(x)).collect(),
            };
            json!({
                "kind": "array",
                "dtype": array.dtype.as_str(),
                "shape": array.shape,
                "data": data,
            })
        }
    }
}

fn array_f64s(data: &ArrayData) -> Vec<f64> {
    match data {
        ArrayData::U8(values) => values.iter().map(|&x| f64::from(x)).collect(),
        ArrayData::I32(values) => values.iter().map(|&x| f64::from(x)).collect(),
        ArrayData::I64(values) => values.iter().map(|&x| x as f64).collect(),
        ArrayData::F32(values) => values.iter().map(|&x| f64::from(x)).collect(),
        ArrayData::F64(values) => values.clone(),
    }
}

/// Assert one produced value matches its expected conformance encoding.
fn assert_value(name: &str, key: &str, actual: &Value, expected: &Json, atol: f64) {
    match expected["kind"].as_str().expect("expected kind") {
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
            let Value::Array(array) = actual else {
                panic!("{name}/{key}: expected array, got {actual:?}");
            };
            assert_eq!(
                array.dtype.as_str(),
                expected["dtype"].as_str().expect("conformance fixture"),
                "{name}/{key}: dtype"
            );
            let expected_shape: Vec<usize> = expected["shape"]
                .as_array()
                .expect("conformance fixture")
                .iter()
                .map(|dim| dim.as_u64().expect("conformance fixture") as usize)
                .collect();
            assert_eq!(array.shape, expected_shape, "{name}/{key}: shape");
            let actual_values = array_f64s(&array.data);
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

/// Recompute a case's normalized spec documents and `expect` block from
/// current behavior (update mode).
///
/// Cases with `"preserve_inputs": true` keep their spec documents
/// verbatim — used by defaults-pinning cases whose specs are
/// deliberately minimal (every omitted field must resolve to the same
/// default in every implementation).
fn updated_case(name: &str, case: &Json) -> Json {
    let preserve_inputs = case["preserve_inputs"] == Json::Bool(true);
    let mut out = case.clone();
    match case["kind"].as_str().expect("case kind") {
        "serialization" => {
            let doc = &case["doc"];
            out["doc"] = if case["side"] == "env" {
                let spec: EnvAnnotations = serde_json::from_value(doc.clone())
                    .unwrap_or_else(|e| panic!("{name}: parse failed: {e}"));
                serde_json::to_value(&spec).expect("serializes")
            } else {
                let spec: ModelIoSpec = serde_json::from_value(doc.clone())
                    .unwrap_or_else(|e| panic!("{name}: parse failed: {e}"));
                serde_json::to_value(&spec).expect("serializes")
            };
        }
        "resolve" => {
            let (annotations, obs_space, action_space, model_spec) = parse_inputs(case);
            if !preserve_inputs {
                out["env_annotations"] = serde_json::to_value(&annotations).expect("serializes");
                out["observation_space"] = serde_json::to_value(&obs_space).expect("serializes");
                out["action_space"] = serde_json::to_value(&action_space).expect("serializes");
                out["model_spec"] = serde_json::to_value(&model_spec).expect("serializes");
            }
            out["expect"] =
                match resolve(&annotations, &obs_space, &action_space, &model_spec, false) {
                    Ok(adapter) => json!({"ok": true, "describe": adapter.describe()}),
                    Err(error) => {
                        // Keep a hand-curated substring when it still matches;
                        // otherwise pin the full current message.
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
            let (annotations, obs_space, action_space, model_spec) = parse_inputs(case);
            if !preserve_inputs {
                out["env_annotations"] = serde_json::to_value(&annotations).expect("serializes");
                out["observation_space"] = serde_json::to_value(&obs_space).expect("serializes");
                out["action_space"] = serde_json::to_value(&action_space).expect("serializes");
                out["model_spec"] = serde_json::to_value(&model_spec).expect("serializes");
            }
            let adapter = resolve(&annotations, &obs_space, &action_space, &model_spec, false)
                .unwrap_or_else(|e| panic!("{name}: resolve failed: {e}"));
            let Value::Map(raw_obs) = dec(&case["observation"]) else {
                panic!("{name}: observation must decode to a map");
            };
            let payload = adapter
                .transform_obs(&raw_obs, &NoCustoms)
                .unwrap_or_else(|e| panic!("{name}: transform_obs failed: {e}"));
            let action = adapter
                .transform_action(&dec(&case["model_output"]))
                .unwrap_or_else(|e| panic!("{name}: transform_action failed: {e}"));
            let atol = case["expect"]["atol"].clone();
            out["expect"] = json!({
                "payload": payload
                    .iter()
                    .map(|(key, value)| (key.clone(), enc(value)))
                    .collect::<serde_json::Map<String, Json>>(),
                "action": enc(&Value::Array(action)),
                "atol": if atol.is_null() { json!(1e-6) } else { atol },
            });
        }
        other => panic!("{name}: unknown case kind {other:?}"),
    }
    out
}

fn verify_case(name: &str, case: &Json) {
    match case["kind"].as_str().expect("case kind") {
        "serialization" => {
            let doc = &case["doc"];
            let round_tripped = if case["side"] == "env" {
                let spec: EnvAnnotations = serde_json::from_value(doc.clone())
                    .unwrap_or_else(|e| panic!("{name}: parse failed: {e}"));
                serde_json::to_value(&spec).expect("serializes")
            } else {
                let spec: ModelIoSpec = serde_json::from_value(doc.clone())
                    .unwrap_or_else(|e| panic!("{name}: parse failed: {e}"));
                serde_json::to_value(&spec).expect("serializes")
            };
            assert_eq!(&round_tripped, doc, "{name}: round trip mismatch");
        }
        "resolve" => {
            let (annotations, obs_space, action_space, model_spec) = parse_inputs(case);
            let expect = &case["expect"];
            match resolve(&annotations, &obs_space, &action_space, &model_spec, false) {
                Ok(adapter) => {
                    let expected = expect["describe"]
                        .as_str()
                        .unwrap_or_else(|| panic!("{name}: expected an error"));
                    assert_eq!(adapter.describe(), expected, "{name}: describe");
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
            let (annotations, obs_space, action_space, model_spec) = parse_inputs(case);
            let adapter = resolve(&annotations, &obs_space, &action_space, &model_spec, false)
                .unwrap_or_else(|e| panic!("{name}: resolve failed: {e}"));
            let atol = case["expect"]["atol"].as_f64().expect("atol");

            let Value::Map(raw_obs) = dec(&case["observation"]) else {
                panic!("{name}: observation must decode to a map");
            };
            let payload = adapter
                .transform_obs(&raw_obs, &NoCustoms)
                .unwrap_or_else(|e| panic!("{name}: transform_obs failed: {e}"));
            let expected_payload = case["expect"]["payload"].as_object().expect("payload");
            let mut expected_keys: Vec<&String> = expected_payload.keys().collect();
            expected_keys.sort();
            let actual_keys: Vec<&String> = payload.keys().collect();
            assert_eq!(actual_keys, expected_keys, "{name}: payload keys");
            for (key, expected) in expected_payload {
                assert_value(name, key, &payload[key], expected, atol);
            }

            let action = adapter
                .transform_action(&dec(&case["model_output"]))
                .unwrap_or_else(|e| panic!("{name}: transform_action failed: {e}"));
            assert_value(
                name,
                "action",
                &Value::Array(action),
                &case["expect"]["action"],
                atol,
            );
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

        if update {
            let rewritten = updated_case(&name, &case);
            let mut text = serde_json::to_string_pretty(&rewritten).expect("serializes");
            text.push('\n');
            fs::write(&path, text).expect("writable case");
        } else {
            verify_case(&name, &case);
        }
        ran += 1;
    }
    assert!(ran >= 13, "expected at least 13 vectors, ran {ran}");
}
