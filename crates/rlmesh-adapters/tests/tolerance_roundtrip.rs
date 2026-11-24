use rlmesh_adapters::v1::{EnvTags, ModelSpec};

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
    let input = r#"{
        "observation": {
            "cam": {"type": "image", "role": "image/primary", "normalize": false, "x-team": 7},
            "mic": {"type": "audio", "role": "audio/mic", "sample_rate": 16000}
        },
        "action": {"components": [{"role": "a", "dim": 1}]}
    }"#;

    let once = normalize_env(input);
    assert_eq!(normalize_env(&once), once, "normalize is not idempotent");

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
    let once = normalize_env(
        r#"{"observation":{"cam":{"type":"image","role":"image/primary","x-z":1}},
            "action":{"components":[]}}"#,
    );
    assert_eq!(once.matches(r#""type":"image""#).count(), 1, "got: {once}");
}

#[test]
fn malformed_of_known_kind_still_hard_errors_in_every_mode() {
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

#[test]
fn inner_leaves_carry_an_unknown_field_through_a_hop() {
    let env = normalize_env(
        r#"{"observation": {"s": {"type": "split",
                "fields": [{"role": "r", "dim": 7, "frame": "world"}]}},
            "action": {"components": []}}"#,
    );
    assert!(env.contains(r#""frame":"world""#), "got: {env}");
    assert_eq!(normalize_env(&env), env, "normalize is not idempotent");

    let model = normalize_model(
        r#"{"input": {"type": "state", "components": [{"role": "r", "frame": "world"}]},
            "output": {"components": []}}"#,
    );
    assert!(model.contains(r#""frame":"world""#), "got: {model}");
    assert_eq!(
        normalize_model(&model),
        model,
        "normalize is not idempotent"
    );
}

#[test]
fn reversed_range_still_hard_errors() {
    for json in [
        r#"{"observation": {"s": {"type": "split",
                "fields": [{"role": "r", "dim": 1, "range": [1.0, 0.0]}]}},
            "action": {"components": []}}"#,
        r#"{"observation": {"s": {"type": "split",
                "fields": [{"role": "r", "dim": 1, "range": [1.0, 0.0], "frame": "world"}]}},
            "action": {"components": []}}"#,
    ] {
        let err = serde_json::from_str::<EnvTags>(json).unwrap_err();
        assert!(err.to_string().contains("min <= max"), "got: {err}");
    }

    let err = serde_json::from_str::<ModelSpec>(
        r#"{"input": {"type": "state",
                "components": [{"role": "r", "range": [1.0, 0.0], "frame": "world"}]},
            "output": {"components": []}}"#,
    )
    .unwrap_err();
    assert!(err.to_string().contains("min <= max"), "got: {err}");
}
