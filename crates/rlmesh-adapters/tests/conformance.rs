//! Run the v1 adapter conformance vectors against this implementation.
//!
//! Snapshot-style: expectations live in `conformance/v1/cases/*.json`.
//! `UPDATE_VECTORS=1 cargo test -p rlmesh-adapters` rewrites the `expect`
//! blocks (and normalizes resolve/apply input specs) from current
//! behavior — review the diff before committing; a changed vector is a
//! semantic change to v1 and must be additive. New resolve/apply cases are
//! authored by hand: write the inputs with an empty `expect`, then run
//! update mode once. `serialization` vectors are the FROZEN wire contract:
//! their `doc` is NEVER rewritten (auto-normalizing would let a renamed or
//! removed serde field self-heal green), so author the canonical `doc` by
//! hand and let the round-trip assertion verify it.

use std::fs;
use std::path::PathBuf;

use rlmesh_adapters::v1::{
    EnvTags, FrameBuffers, ModelSpec, NoCustoms, NoEncodings, RolePolicy, SpaceView, Value,
    assemble_obs, reject_unsanctioned_roles_env, reject_unsanctioned_roles_model, resolve,
};
use rlmesh_spaces::scalar::{Scalar, decode_scalars, encode_scalars};
use rlmesh_spaces::{DType, Tensor};
use serde_json::{Value as Json, json};

/// Whether a dtype is a float family (controls integer-vs-float JSON output).
fn is_float_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::Float16 | DType::Float32 | DType::Float64)
}

/// Assert every hand-curated `advisories_contain` substring still matches one of
/// the resolve's advisories, as `"<severity>: <message>"`.
///
/// Hand-authored and never machine-written (the `error_contains` idiom): a
/// vector states only the advisory it is *about*, so adding a case cannot
/// silently re-pin every other note a resolve happens to raise.
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

/// Encode an engine [`Value`] into the conformance value encoding
/// (inverse of [`dec`]); used by update mode to write expectations.
fn enc(value: &Value) -> Json {
    match value {
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

/// Assert one produced value matches its expected conformance encoding.
fn assert_value(name: &str, key: &str, actual: &Value, expected: &Json, atol: f64) {
    match expected["kind"].as_str().expect("expected kind") {
        "map" => {
            // The assembled obs payload is now a single `Value::Map` tree
            // (placement-keyed), pinned by update mode as one `map`-kind value;
            // recurse into each entry so a leaf mismatch still localizes.
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
                    // Curated, so carried through verbatim rather than rewritten
                    // -- and checked here so update mode cannot bless a stale one.
                    if let Some(curated) = case["expect"].get("advisories_contain") {
                        assert_advisories(name, &adapter, &case["expect"]);
                        expect["advisories_contain"] = curated.clone();
                    }
                    expect
                }
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
            let payload = adapter
                .transform_obs(&raw_obs, &NoCustoms)
                .unwrap_or_else(|e| panic!("{name}: transform_obs failed: {e}"));
            let action = adapter
                .transform_action(&dec(&case["model_output"]))
                .unwrap_or_else(|e| panic!("{name}: transform_action failed: {e}"));
            let atol = case["expect"]["atol"].clone();
            // The assembled payload is now a Value tree; the conformance schema
            // pins it as a single encoded value.
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

/// Drive a case's `observations` through the stateful assemble seam, one
/// episode, and collect the payload each step produced.
///
/// The stateful entry point ([`assemble_obs`]) rather than the stateless one:
/// what these vectors pin is the frame window across steps, which does not exist
/// in a single `transform_obs`.
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
            assemble_obs(
                adapter,
                &raw_obs,
                "ep",
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
            let payload = adapter
                .transform_obs(&raw_obs, &NoCustoms)
                .unwrap_or_else(|e| panic!("{name}: transform_obs failed: {e}"));
            // The assembled payload is now a Value tree; the conformance schema
            // pins it as a single encoded value compared structurally.
            assert_value(name, "payload", &payload, &case["expect"]["payload"], atol);

            let action = adapter
                .transform_action(&dec(&case["model_output"]))
                .unwrap_or_else(|e| panic!("{name}: transform_action failed: {e}"));
            assert_value(
                name,
                "action",
                &Value::Tensor(action),
                &case["expect"]["action"],
                atol,
            );
        }
        // The publish-gate role policy: `doc` is a spec, `policy` names the
        // tier, and the expectation is acceptance or a pinned rejection
        // substring. Frozen like `serialization` — the policy table IS the
        // contract, so update mode never rewrites it.
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
        // A frame window only exists ACROSS steps, so this kind drives a sequence
        // of observations through the stateful assemble seam and pins the payload
        // each step produced. The `apply` kind stays single-shot.
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
            // Serialization and role_policy vectors are the FROZEN v1
            // contract: never rewritten, even under UPDATE_VECTORS
            // (auto-normalizing let a renamed serde field, or a role quietly
            // added to the registry, self-heal green). verify_case is their
            // sole authority and runs in both modes.
            verify_case(&name, &case);
        }
        ran += 1;
    }
    assert!(ran >= 13, "expected at least 13 vectors, ran {ran}");
}

/// The `zeros(n)` / `, pad to N` wording predates constant parts and fills, and
/// PR-12's new describe tokens (`const(n)=v`, `fill(n)=v`, `(post_rotate)`,
/// `(*s +o)`) must render only when the new fields are used. Pinning the text
/// here as a literal catches a rewrite that `UPDATE_VECTORS=1` would happily
/// bless into the vector itself.
#[test]
fn existing_pad_and_zero_fill_text_is_unchanged() {
    let path = cases_dir().join("resolve_rot6d_with_optional_zero_fill.json");
    let case: Json = serde_json::from_str(&fs::read_to_string(&path).expect("readable case"))
        .expect("case parses as JSON");
    assert_eq!(
        case["expect"]["describe"].as_str().expect("describe text"),
        "observation:\n  \"instruction\" <- text \"instruction\"\n  \"state\" <- \
         concat(eef_pos[:3], eef_quat (quat_xyzw->rot6d), gripper[:1], zeros(3)), pad to 16\
         \naction:\n  \"action/delta_eef_pos\" <- model[0:3]\n  \"action/delta_eef_rot\" <- \
         model[3:9] (rot6d->axis_angle)\n  \"action/gripper\" <- model[9:10]\n  clip to \
         (-1.0, 1.0)",
    );
    let (tags, obs_space, action_space, model_spec) = parse_inputs(&case);
    let adapter = resolve(&tags, &obs_space, &action_space, &model_spec, false).expect("resolve");
    assert_eq!(adapter.describe(), case["expect"]["describe"]);
}
