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

fn cases() -> Vec<(&'static str, String)> {
    vec![
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
        (
            "statelayout dup role",
            env_err(
                r#"{"observation":{"x":{"type":"split","fields":[{"role":"r","dim":1},{"role":"r","dim":1}]}},"action":{"components":[]}}"#,
            ),
        ),
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
        (
            "unknown layout",
            model_err(
                r#"{"input":{"c":{"type":"image","role":"r","layout":"nhwc"}},"output":{"components":[]}}"#,
            ),
        ),
        (
            "missing field",
            model_err(r#"{"input":{},"output":{"components":[{"role":"g"}]}}"#),
        ),
        (
            "string wrong-type",
            model_err(r#"{"input":{},"output":{"components":[{"role":5,"dim":1}]}}"#),
        ),
    ]
}

#[test]
fn no_message_leaks_rust_internals() {
    const LEAKS: &[&str] = &[
        "u32",
        "i64",
        "f64",
        "u64",
        "usize",
        "1..=",
        "tuple",
        "floating point",
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
}
