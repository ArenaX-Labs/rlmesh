//! Author-facing error-message contract.
//!
//! The Rust codec is the single source of spec-validation errors (Python and,
//! later, the FE binding surface them verbatim with a field path). A spec
//! author should never see a Rust wire type (`u32`, `f64`, `tuple of size 2`)
//! or a Rust type name (`ImageInput`, `StateField`) in an error — those are
//! implementation leaks. This pins every parse-error class to domain language;
//! if a new field reintroduces a leak, the sweep below fails.
//!
//! Documented scope: this covers *field-level* errors. A whole-document type
//! error (the root is not an object, e.g. `from_str::<ModelSpec>("42")`) still
//! names the Rust struct (`expected struct ModelSpec`); that input is
//! unreachable through `json.dumps(dict)` and is a known, accepted limitation
//! (see `de_spec` in the PyO3 binding). The leak list below therefore does not
//! ban `struct`/the type names — only the field-level wire scalars.

use rlmesh_adapters::v1::{EnvTags, ModelSpec};

fn model_err(json: &str) -> String {
    serde_json::from_str::<ModelSpec>(json)
        .expect_err("expected a parse error")
        .to_string()
}

fn env_err(json: &str) -> String {
    serde_json::from_str::<EnvTags>(json)
        .expect_err("expected a parse error")
        .to_string()
}

/// One malformed document per author-facing parse-error class.
fn cases() -> Vec<(&'static str, String)> {
    vec![
        // counts (num.rs)
        (
            "dim negative",
            model_err(
                r#"{"inputs":[{"type":"state","key":"s","components":[{"role":"r","dim":-1}]}],"action":{"components":[]}}"#,
            ),
        ),
        (
            "dim wrong-type",
            model_err(
                r#"{"inputs":[{"type":"state","key":"s","components":[{"role":"r","dim":"x"}]}],"action":{"components":[]}}"#,
            ),
        ),
        (
            "dim overflow",
            model_err(
                r#"{"inputs":[{"type":"state","key":"s","components":[{"role":"r","dim":99999999999}]}],"action":{"components":[]}}"#,
            ),
        ),
        (
            "dim float",
            model_err(
                r#"{"inputs":[{"type":"state","key":"s","components":[{"role":"r","dim":3.0}]}],"action":{"components":[]}}"#,
            ),
        ),
        // stack bound (image.rs)
        (
            "stack low",
            model_err(
                r#"{"inputs":[{"type":"image","key":"c","role":"r","stack":0}],"action":{"components":[]}}"#,
            ),
        ),
        (
            "stack high",
            model_err(
                r#"{"inputs":[{"type":"image","key":"c","role":"r","stack":1000}],"action":{"components":[]}}"#,
            ),
        ),
        (
            "stack negative",
            model_err(
                r#"{"inputs":[{"type":"image","key":"c","role":"r","stack":-1}],"action":{"components":[]}}"#,
            ),
        ),
        // reshape elements (model/state.rs)
        (
            "reshape elem wrong-type",
            model_err(
                r#"{"inputs":[{"type":"state","key":"s","components":[{"role":"r","dim":1}],"reshape":[1,"x"]}],"action":{"components":[]}}"#,
            ),
        ),
        (
            "reshape not array",
            model_err(
                r#"{"inputs":[{"type":"state","key":"s","components":[{"role":"r","dim":1}],"reshape":5}],"action":{"components":[]}}"#,
            ),
        ),
        // state field cross-field rules (env_tags.rs)
        (
            "statefield dim zero",
            env_err(
                r#"{"observation":{"x":{"type":"layout","fields":[{"role":"r","dim":0}]}},"action":{"components":[]}}"#,
            ),
        ),
        (
            "statefield roleless skip",
            env_err(
                r#"{"observation":{"x":{"type":"layout","fields":[{"dim":3,"encoding":"rot6d"}]}},"action":{"components":[]}}"#,
            ),
        ),
        (
            "statelayout empty",
            env_err(
                r#"{"observation":{"x":{"type":"layout","fields":[]}},"action":{"components":[]}}"#,
            ),
        ),
        // cross-engine parity guards (codec rejects what the read path / resolve reject)
        (
            "statelayout dup role",
            env_err(
                r#"{"observation":{"x":{"type":"layout","fields":[{"role":"r","dim":1},{"role":"r","dim":1}]}},"action":{"components":[]}}"#,
            ),
        ),
        (
            "model dup input key",
            model_err(
                r#"{"inputs":[{"type":"text","key":"s","role":"r"},{"type":"text","key":"s","role":"r"}],"action":{"components":[]}}"#,
            ),
        ),
        (
            "state input empty",
            model_err(
                r#"{"inputs":[{"type":"state","key":"s","components":[]}],"action":{"components":[]}}"#,
            ),
        ),
        (
            "action dup role",
            model_err(
                r#"{"inputs":[],"action":{"components":[{"role":"g","dim":1},{"role":"g","dim":1}]}}"#,
            ),
        ),
        // range / clip pairs (num.rs de_opt_range)
        (
            "range too short",
            model_err(
                r#"{"inputs":[],"action":{"components":[{"role":"g","dim":1,"range":[0.0]}]}}"#,
            ),
        ),
        (
            "range too long",
            model_err(
                r#"{"inputs":[],"action":{"components":[{"role":"g","dim":1,"range":[0.0,1.0,2.0]}]}}"#,
            ),
        ),
        (
            "range elem wrong-type",
            model_err(
                r#"{"inputs":[],"action":{"components":[{"role":"g","dim":1,"range":["lo",1.0]}]}}"#,
            ),
        ),
        (
            "clip elem wrong-type",
            model_err(r#"{"inputs":[],"action":{"components":[],"clip":["lo",1.0]}}"#),
        ),
        (
            "scale wrong-type",
            model_err(
                r#"{"inputs":[],"action":{"components":[{"role":"g","dim":1,"scale":"x"}]}}"#,
            ),
        ),
        (
            "threshold wrong-type",
            model_err(
                r#"{"inputs":[],"action":{"components":[{"role":"g","dim":1,"threshold":"x"}]}}"#,
            ),
        ),
        // frozen vocab / unknown kind / unknown field / missing / wrong-type
        (
            "unknown model input",
            model_err(r#"{"inputs":[{"type":"audio","key":"c"}],"action":{"components":[]}}"#),
        ),
        (
            "unknown obs tag",
            env_err(r#"{"observation":{"x":{"type":"audio"}},"action":{"components":[]}}"#),
        ),
        (
            "unknown rotation",
            model_err(
                r#"{"inputs":[{"type":"state","key":"s","components":[{"role":"r","encoding":"rot10d"}]}],"action":{"components":[]}}"#,
            ),
        ),
        (
            "unknown layout",
            model_err(
                r#"{"inputs":[{"type":"image","key":"c","role":"r","layout":"nhwc"}],"action":{"components":[]}}"#,
            ),
        ),
        (
            "unknown field",
            model_err(
                r#"{"inputs":[],"action":{"components":[{"role":"g","dim":1,"rnge":[0,1]}]}}"#,
            ),
        ),
        (
            "missing field",
            model_err(r#"{"inputs":[],"action":{"components":[{"dim":1}]}}"#),
        ),
        (
            "string wrong-type",
            model_err(r#"{"inputs":[],"action":{"components":[{"role":5,"dim":1}]}}"#),
        ),
    ]
}

/// No author-facing message may leak a Rust wire type or internal type name.
#[test]
fn no_message_leaks_rust_internals() {
    // serde phrases its wire scalars as bare type names; our custom
    // deserializers must replace these with domain words. `1..=` is Rust range
    // syntax; the type names are our own structs.
    const LEAKS: &[&str] = &[
        "u32",
        "i64",
        "f64",
        "u64",
        "usize",
        "1..=",
        "tuple",
        "floating point",
        "ImageInput",
        "StateField",
    ];
    for (name, message) in cases() {
        for leak in LEAKS {
            assert!(
                !message.contains(leak),
                "[{name}] leaks {leak:?}: {message}"
            );
        }
    }
}

/// Spot-check the domain phrasing of the messages we rewrote, so a regression
/// in wording (not just a type leak) is caught.
#[test]
fn rewritten_messages_read_in_domain_language() {
    let by_name: std::collections::HashMap<_, _> = cases().into_iter().collect();
    let has = |name: &str, needle: &str| {
        let msg = &by_name[name];
        assert!(msg.contains(needle), "[{name}] missing {needle:?}: {msg}");
    };
    has("stack low", "stack must be between 1 and 64, got 0");
    has("stack negative", "non-negative integer");
    has("dim float", "non-negative integer");
    has("reshape elem wrong-type", "whole number");
    has("statefield dim zero", "state field dim must be >= 1");
    has("statefield roleless skip", "role-less field");
    has("statelayout empty", "at least one field");
    has("statelayout dup role", "more than once");
    has("model dup input key", "duplicate model input key");
    has("state input empty", "at least one component");
    has("action dup role", "more than once");
    has("range too short", "pair of numbers [min, max], got 1");
    has("range too long", "pair of numbers [min, max], got 3");
    has("range elem wrong-type", "expected a number");
    has("scale wrong-type", "expected a number");
    has("threshold wrong-type", "expected a number");
}
