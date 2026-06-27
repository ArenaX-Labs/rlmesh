//! The tolerant-reader relay-fidelity CI gate (design §6).
//!
//! Any hop that parses + re-serializes a spec must retain every unknown kind and
//! field. The contract is **semantic** round-trip, not byte-identical: the field
//! capture map is a `BTreeMap` and re-emits in sorted key order, but every
//! unknown key/value survives, and an `Unknown` leaf's raw object is byte-faithful
//! (it already embeds `type`). These tests pin that, plus the malformed floor that
//! stays a hard error in every mode.

use rlmesh_adapters::v1::{EnvTags, ModelSpec};

/// The READ-door canonicalizer: parse (tolerant) then re-serialize. Models one
/// relay hop. A real hop never runs `reject_unknowns` (that is the publish gate).
fn normalize_env(json: &str) -> String {
    let tags: EnvTags = serde_json::from_str(json).expect("env tags parse (tolerant)");
    serde_json::to_string(&tags).expect("env tags serialize")
}

fn normalize_model(json: &str) -> String {
    let spec: ModelSpec = serde_json::from_str(json).expect("model spec parse (tolerant)");
    serde_json::to_string(&spec).expect("model spec serialize")
}

#[test]
fn normalize_is_idempotent_and_preserves_unknown_kind_and_field() {
    // Carries an unknown *kind* (audio, with its own sub-field), an unknown
    // *field* on a recognized kind (bare `normalize`), and an `x-` field.
    let input = r#"{
        "observation": {
            "cam": {"type": "image", "role": "image/primary", "normalize": false, "x-team": 7},
            "mic": {"type": "audio", "role": "audio/mic", "sample_rate": 16000}
        },
        "action": {"components": [{"role": "a", "dim": 1}]}
    }"#;

    let once = normalize_env(input);
    // §6 gate 1: idempotence.
    assert_eq!(normalize_env(&once), once, "normalize is not idempotent");

    // §6 gate 2: every injected unknown kind and field survives verbatim.
    for needle in [
        r#""type":"audio""#,
        "sample_rate",
        "16000",
        "normalize",
        "x-team",
    ] {
        assert!(once.contains(needle), "lost {needle:?} in: {once}");
    }
}

#[test]
fn model_spec_preserves_unknown_kind_and_field() {
    let input = r#"{
        "input": {
            "vibe": {"type": "haptics", "role": "touch", "channels": 12},
            "pixels": {"type": "image", "role": "image/primary", "future_flag": true}
        },
        "output": {"components": [{"role": "g", "dim": 1}]}
    }"#;
    let once = normalize_model(input);
    assert_eq!(
        normalize_model(&once),
        once,
        "model normalize not idempotent"
    );
    for needle in [r#""type":"haptics""#, "channels", "12", "future_flag"] {
        assert!(once.contains(needle), "lost {needle:?} in: {once}");
    }
}

#[test]
fn type_never_leaks_into_a_capture_map() {
    // §6 gate 3 (partial): a known leaf's `type` is consumed by the leaf branch
    // and must not reappear as a duplicate or a captured field. Re-serializing
    // emits exactly one `type` for the leaf.
    let once = normalize_env(
        r#"{"observation":{"cam":{"type":"image","role":"image/primary","x-z":1}},
            "action":{"components":[]}}"#,
    );
    assert_eq!(once.matches(r#""type":"image""#).count(), 1, "got: {once}");
}

#[test]
fn malformed_of_known_kind_still_hard_errors_in_every_mode() {
    // §6 gate 3: tolerance never weakens the malformed floor for a *recognized*
    // kind. A reversed range, a zero dim, and a non-string `type` all hard-error
    // at parse, tolerant mode included.
    assert!(
        serde_json::from_str::<EnvTags>(
            r#"{"observation":{"s":{"type":"state","role":"r","range":[1.0,0.0]}},
                "action":{"components":[]}}"#,
        )
        .is_err(),
        "reversed range must hard-error"
    );
    assert!(
        serde_json::from_str::<EnvTags>(
            r#"{"observation":{"s":{"type":"split","fields":[{"role":"r","dim":0}]}},
                "action":{"components":[]}}"#,
        )
        .is_err(),
        "zero-dim split field must hard-error"
    );
    assert!(
        serde_json::from_str::<EnvTags>(
            r#"{"observation":{"type":7},"action":{"components":[]}}"#,
        )
        .is_err(),
        "non-string type must hard-error"
    );
}
