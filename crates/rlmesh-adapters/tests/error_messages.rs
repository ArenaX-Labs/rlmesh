//! Author-facing error-message contract.
//!
//! The Rust codec is the single source of spec-validation errors (Python and,
//! later, the FE binding surface them verbatim with a field path). A spec
//! author should never see a Rust wire type (`u32`, `f64`, `tuple of size 2`)
//! or a Rust type name (`Image`, `Field`, `ObsNode`) in an error — those are
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
                r#"{"input":{"s":{"type":"state","components":[{"role":"r","dim":-1}]}},"output":{"components":[]}}"#,
            ),
        ),
        (
            "dim wrong-type",
            model_err(
                r#"{"input":{"s":{"type":"state","components":[{"role":"r","dim":"x"}]}},"output":{"components":[]}}"#,
            ),
        ),
        (
            "dim overflow",
            model_err(
                r#"{"input":{"s":{"type":"state","components":[{"role":"r","dim":99999999999}]}},"output":{"components":[]}}"#,
            ),
        ),
        (
            "dim float",
            model_err(
                r#"{"input":{"s":{"type":"state","components":[{"role":"r","dim":3.0}]}},"output":{"components":[]}}"#,
            ),
        ),
        // stack bound (image.rs)
        (
            "stack low",
            model_err(
                r#"{"input":{"c":{"type":"image","role":"r","stack":0}},"output":{"components":[]}}"#,
            ),
        ),
        (
            "stack high",
            model_err(
                r#"{"input":{"c":{"type":"image","role":"r","stack":1000}},"output":{"components":[]}}"#,
            ),
        ),
        (
            "stack negative",
            model_err(
                r#"{"input":{"c":{"type":"image","role":"r","stack":-1}},"output":{"components":[]}}"#,
            ),
        ),
        // reshape elements (model/state.rs)
        (
            "reshape elem wrong-type",
            model_err(
                r#"{"input":{"s":{"type":"state","components":[{"role":"r","dim":1}],"reshape":[1,"x"]}},"output":{"components":[]}}"#,
            ),
        ),
        (
            "reshape not array",
            model_err(
                r#"{"input":{"s":{"type":"state","components":[{"role":"r","dim":1}],"reshape":5}},"output":{"components":[]}}"#,
            ),
        ),
        // state field cross-field rules (env_tags.rs)
        (
            "statefield dim zero",
            env_err(
                r#"{"observation":{"x":{"type":"split","fields":[{"role":"r","dim":0}]}},"action":{"components":[]}}"#,
            ),
        ),
        (
            "statefield roleless skip",
            env_err(
                r#"{"observation":{"x":{"type":"split","fields":[{"dim":3,"encoding":"rot6d"}]}},"action":{"components":[]}}"#,
            ),
        ),
        (
            "statelayout empty",
            env_err(
                r#"{"observation":{"x":{"type":"split","fields":[]}},"action":{"components":[]}}"#,
            ),
        ),
        // cross-engine parity guards (codec rejects what the read path / resolve reject)
        (
            "statelayout dup role",
            env_err(
                r#"{"observation":{"x":{"type":"split","fields":[{"role":"r","dim":1},{"role":"r","dim":1}]}},"action":{"components":[]}}"#,
            ),
        ),
        // (The old "model dup input key" class is gone: in the recursive tree a
        // model input's placement is its tree position, so a JSON object cannot
        // express two inputs under the same key — there is no duplicate-key error
        // to surface.)
        (
            "state input empty",
            model_err(
                r#"{"input":{"s":{"type":"state","components":[]}},"output":{"components":[]}}"#,
            ),
        ),
        (
            "action dup role",
            model_err(
                r#"{"input":{},"output":{"components":[{"role":"g","dim":1},{"role":"g","dim":1}]}}"#,
            ),
        ),
        // range / clip pairs (num.rs de_opt_range)
        (
            "range too short",
            model_err(
                r#"{"input":{},"output":{"components":[{"role":"g","dim":1,"range":[0.0]}]}}"#,
            ),
        ),
        (
            "range too long",
            model_err(
                r#"{"input":{},"output":{"components":[{"role":"g","dim":1,"range":[0.0,1.0,2.0]}]}}"#,
            ),
        ),
        (
            "range elem wrong-type",
            model_err(
                r#"{"input":{},"output":{"components":[{"role":"g","dim":1,"range":["lo",1.0]}]}}"#,
            ),
        ),
        (
            "clip elem wrong-type",
            model_err(r#"{"input":{},"output":{"components":[],"clip":["lo",1.0]}}"#),
        ),
        (
            "scale wrong-type",
            model_err(r#"{"input":{},"output":{"components":[{"role":"g","dim":1,"scale":"x"}]}}"#),
        ),
        (
            "threshold wrong-type",
            model_err(
                r#"{"input":{},"output":{"components":[{"role":"g","dim":1,"threshold":"x"}]}}"#,
            ),
        ),
        // frozen vocab / unknown layout / missing / wrong-type.
        // An unknown *rotation encoding* is deliberately absent here: a rotation
        // field is now an accept-set that tolerates an unrecognized (future)
        // encoding at parse and rejects it at *resolve* instead (graceful
        // forward-compatible degradation). See the resolver's selection tests.
        // An unknown leaf `type` is likewise no longer a parse error under the
        // tolerant reader: it becomes an `Unknown` leaf, retained for relay, and
        // surfaces as a typed `UnsupportedKind` at *resolve* (only if a model
        // input references it). So neither an unknown obs kind nor an unknown
        // model-input kind appears in this parse-error sweep anymore.
        (
            "unknown layout",
            model_err(
                r#"{"input":{"c":{"type":"image","role":"r","layout":"nhwc"}},"output":{"components":[]}}"#,
            ),
        ),
        // (The old "unknown field" parse-error class is gone: the tolerant reader
        // captures an unrecognized field verbatim instead of failing at parse, and
        // the strict-v1 publish gate (`reject_unknowns`) rejects it post-parse —
        // see the `spec::strict` tests. There is no field-level parse error to
        // surface here anymore.)
        (
            // `role` is optional (a role-less component is an opaque actuator);
            // `dim` is the required field, so omitting it is the missing-field case.
            "missing field",
            model_err(r#"{"input":{},"output":{"components":[{"role":"g"}]}}"#),
        ),
        (
            "string wrong-type",
            model_err(r#"{"input":{},"output":{"components":[{"role":5,"dim":1}]}}"#),
        ),
    ]
}

/// No author-facing message may leak a Rust wire type or internal type name.
#[test]
fn no_message_leaks_rust_internals() {
    // serde phrases its wire scalars as bare type names; our custom
    // deserializers must replace these with domain words. `1..=` is Rust range
    // syntax; the type names are our own current spec structs/enums (capitalized
    // identifiers — author-facing domain words are lowercase, so these guard
    // against a serde-derived message naming an internal type).
    const LEAKS: &[&str] = &[
        // wire scalars
        "u32",
        "i64",
        "f64",
        "u64",
        "usize",
        "1..=",
        "tuple",
        "floating point",
        // current spec type names (the deleted ImageInput/StateField are gone)
        "Image",
        "Field",
        "State",
        "Actuator",
        "Concat",
        "Action",
        "Text",
        "Custom",
        "SplitLayout",
        "ObsLeaf",
        "ObsNode",
        "ModelLeaf",
        "InputNode",
    ];
    for (name, message) in cases() {
        // When a structurally-misplaced value (not an object/array) sits in a
        // node position, the recursive-tree dispatch describes the user-facing
        // node vocabulary ("a ... leaf, a dict of nodes, or a tuple (array) of
        // nodes"). That phrase is domain language, not a Rust wire-type leak, so
        // strip it before the sweep — otherwise its benign "tuple" trips the
        // `tuple` ban. (An unknown leaf `type` no longer reaches this phrase: it
        // is named directly by the `unknown ... kind` error.)
        let scanned = message
            .replace("tuple (array) of nodes", "")
            .replace("a tuple (array)", "");
        for leak in LEAKS {
            assert!(
                !scanned.contains(leak),
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
    has("state input empty", "at least one component");
    has("action dup role", "more than once");
    has("range too short", "pair of numbers [min, max], got 1");
    has("range too long", "pair of numbers [min, max], got 3");
    has("range elem wrong-type", "expected a number");
    has("scale wrong-type", "expected a number");
    has("threshold wrong-type", "expected a number");
    // (Unknown leaf kinds are no longer parse errors under the tolerant reader;
    // their `UnsupportedKind` resolve-time message is exercised in the resolver
    // tests, not this parse-error sweep.)
}
