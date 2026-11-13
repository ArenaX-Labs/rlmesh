//! Env tags this release writes, read by the released rlmesh-adapters 0.1.0.
//!
//! A 0.1 model resolves its adapter against a 0.2 env with the 0.1 reader:
//! it parses the `rlmesh.adapters.v1.env_tags` metadata as its `EnvTags`, and
//! its `resolve` first refuses any field it does not understand
//! (`reject_bare_fields_env`). Each case here is what this release emits for
//! a set of tags (parsed, checked and re-emitted by this release's writer),
//! and must pass that reader with nothing dropped.

use rlmesh_adapters::v1 as current;
use rlmesh_adapters_0_1::v1 as released;
use serde_json::{Value, json};

/// What this release publishes for `tags`: what its writer accepts,
/// re-emitted.
fn emitted(tags: &Value) -> Result<Value, String> {
    let parsed: current::EnvTags =
        serde_json::from_value(tags.clone()).map_err(|err| err.to_string())?;
    current::reject_unknowns_env(&parsed)?;
    serde_json::to_value(&parsed).map_err(|err| err.to_string())
}

/// Whether the 0.1 reader takes `wire` as a 0.1 model's resolve does, keeping
/// everything in it: what it reads back means to this release what `wire`
/// does.
fn released_reads(wire: &Value) -> Result<(), String> {
    let parsed: released::EnvTags =
        serde_json::from_value(wire.clone()).map_err(|err| format!("does not parse: {err}"))?;
    released::reject_bare_fields_env(&parsed)
        .map_err(|err| format!("refused at resolve: {err}"))?;
    let read_back = serde_json::to_value(&parsed).map_err(|err| err.to_string())?;
    let meant: current::EnvTags =
        serde_json::from_value(wire.clone()).map_err(|err| err.to_string())?;
    let understood: current::EnvTags = serde_json::from_value(read_back.clone())
        .map_err(|err| format!("reads back as tags this release refuses: {err}"))?;
    if meant != understood {
        return Err(format!(
            "parses, but drops or changes what the env declared: wrote {wire}, read {read_back}"
        ));
    }
    Ok(())
}

/// Every case `released_reads` fails, with why.
fn failures<'a>(cases: impl IntoIterator<Item = (String, &'a Value)>) -> Vec<String> {
    cases
        .into_iter()
        .filter_map(|(name, tags)| {
            let wire = emitted(tags)
                .unwrap_or_else(|err| panic!("{name}: not tags this release writes: {err}"));
            released_reads(&wire)
                .err()
                .map(|err| format!("{name}: {err}"))
        })
        .collect()
}

fn assert_read(cases: Vec<(String, Value)>) {
    assert!(!cases.is_empty());
    let failed = failures(cases.iter().map(|(name, tags)| (name.clone(), tags)));
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

fn tags(observation: Value, action: Value) -> Value {
    json!({ "observation": observation, "action": action })
}

fn one_actuator() -> Value {
    json!({ "components": [{ "role": "action/joint_pos", "dim": 7 }] })
}

fn one_state() -> Value {
    json!({ "q": { "type": "state", "role": "proprio/joint_pos" } })
}

/// The tags of every case in this release's adapter conformance corpus that
/// its writer accepts.
#[test]
fn the_conformance_corpus_reads_on_0_1() {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../rlmesh-adapters/conformance/v1/cases"
    );
    let mut cases = Vec::new();
    for entry in std::fs::read_dir(dir).expect("the conformance cases") {
        let path = entry.expect("a case").path();
        let case: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("read a case"))
                .expect("a case is JSON");
        let Some(tags) = case.get("env_tags") else {
            continue;
        };
        if emitted(tags).is_ok() {
            cases.push((path.display().to_string(), tags.clone()));
        }
    }
    assert!(cases.len() > 50, "only {} corpus cases", cases.len());
    assert_read(cases);
}

#[test]
fn image_tags_read_on_0_1() {
    let mut cases = Vec::new();
    for layout in ["hwc", "chw"] {
        for upside_down in [false, true] {
            cases.push((
                format!("image layout {layout}, upside_down {upside_down}"),
                tags(
                    json!({ "cam": { "type": "image", "role": "image/primary",
                        "layout": layout, "upside_down": upside_down } }),
                    one_actuator(),
                ),
            ));
        }
    }
    cases.push((
        "images on parts, in a tuple".to_owned(),
        tags(
            json!([
                { "type": "image", "role": "image/wrist", "part": "left_arm" },
                { "type": "image", "role": "image/wrist", "part": "right_arm" },
            ]),
            one_actuator(),
        ),
    ));
    assert_read(cases);
}

#[test]
fn state_tags_with_every_encoding_and_attribute_read_on_0_1() {
    let mut cases = Vec::new();
    for encoding in current::RotationEncoding::ALL {
        let role = if encoding.is_sink() {
            "proprio/projected_gravity"
        } else {
            "proprio/eef_rot"
        };
        cases.push((
            format!("state encoding {}", encoding.as_str()),
            tags(
                json!({ "rot": { "type": "state", "role": role, "encoding": encoding.as_str() } }),
                one_actuator(),
            ),
        ));
    }
    cases.push((
        "state accepting a set of encodings".to_owned(),
        tags(
            json!({ "rot": { "type": "state", "role": "proprio/eef_rot",
                "encoding": ["quat_xyzw", "quat_wxyz"] } }),
            one_actuator(),
        ),
    ));
    for frame in current::FRAMES {
        cases.push((
            format!("state frame {frame}"),
            tags(
                json!({ "pos": { "type": "state", "role": "proprio/eef_pos", "frame": frame } }),
                one_actuator(),
            ),
        ));
    }
    for provenance in current::PROVENANCES {
        cases.push((
            format!("state provenance {provenance}"),
            tags(
                json!({ "q": { "type": "state", "role": "proprio/joint_pos",
                    "provenance": provenance } }),
                one_actuator(),
            ),
        ));
    }
    cases.push((
        "state range, part and labels".to_owned(),
        tags(
            json!({ "q": { "type": "state", "role": "proprio/joint_pos", "range": [-3.0, 3.0],
                "part": "left_arm", "labels": ["shoulder", "elbow"] } }),
            one_actuator(),
        ),
    ));
    assert_read(cases);
}

#[test]
fn text_split_and_nested_tags_read_on_0_1() {
    let cases = vec![
        (
            "text".to_owned(),
            tags(
                json!({ "task": { "type": "text", "role": "text/instruction" } }),
                one_actuator(),
            ),
        ),
        (
            "split state with every field attribute".to_owned(),
            tags(
                json!({ "state": { "type": "split", "fields": [
                    { "role": "proprio/eef_pos", "dim": 3, "frame": "robot_base",
                        "provenance": "estimated", "range": [-1.0, 1.0],
                        "labels": ["x", "y", "z"] },
                    { "role": "proprio/eef_rot", "dim": 4, "encoding": "quat_wxyz",
                        "part": "left_arm" },
                    { "dim": 2 },
                ] } }),
                one_actuator(),
            ),
        ),
        (
            "nested dicts and tuples".to_owned(),
            tags(
                json!({ "robot": { "arm": [one_state()["q"], { "type": "image",
                    "role": "image/wrist" }] }, "task": { "type": "text",
                    "role": "text/instruction" } }),
                one_actuator(),
            ),
        ),
    ];
    assert_read(cases);
}

#[test]
fn action_tags_read_on_0_1() {
    let mut cases = Vec::new();
    for encoding in current::RotationEncoding::ALL
        .into_iter()
        .filter(|encoding| !encoding.is_sink())
    {
        cases.push((
            format!("actuator encoding {}", encoding.as_str()),
            tags(
                one_state(),
                json!({ "components": [{ "role": "action/delta_eef_rot",
                    "dim": encoding.dims(), "encoding": encoding.as_str() }] }),
            ),
        ));
    }
    cases.push((
        "every actuator attribute".to_owned(),
        tags(
            one_state(),
            json!({ "clip": [-2.0, 2.0], "components": [
                { "role": "action/delta_eef_pos", "dim": 3, "range": [-1.0, 1.0],
                    "scale": 0.5, "offset": 0.1, "clip": true, "frame": "tool",
                    "reference": "current", "part": "left_arm",
                    "labels": ["x", "y", "z"] },
                { "role": "action/joint_pos", "dim": 2, "axis_scale": [1.0, 2.0],
                    "axis_offset": [0.0, 0.5], "invert": true },
                { "role": "action/gripper", "dim": 1, "binary": true, "threshold": 0.5 },
                { "role": "action/joint_vel", "dim": 1, "optional": true, "fill": 0.25 },
                { "dim": 1 },
            ] }),
        ),
    ));
    assert_read(cases);
}

/// The tags the repo's example envs publish, as `write_example_tags.py`
/// recorded them.
#[test]
fn the_example_envs_tags_read_on_0_1() {
    let recorded: serde_json::Map<String, Value> =
        serde_json::from_str(include_str!("fixtures/example_env_tags.json"))
            .expect("the recorded example tags");
    assert_read(recorded.into_iter().collect());
}

/// `polarity` and `unit`, which 0.2 added, on each leaf that takes them.
fn convention_cases() -> Vec<(String, Value)> {
    vec![
        (
            "state polarity".to_owned(),
            tags(
                json!({ "g": { "type": "state", "role": "proprio/gripper", "polarity": "open" } }),
                one_actuator(),
            ),
        ),
        (
            "split field polarity".to_owned(),
            tags(
                json!({ "s": { "type": "split", "fields": [
                    { "role": "proprio/gripper", "dim": 1, "polarity": "close" },
                ] } }),
                one_actuator(),
            ),
        ),
        (
            "actuator polarity".to_owned(),
            tags(
                one_state(),
                json!({ "components": [{ "role": "action/gripper", "dim": 1,
                    "polarity": "open" }] }),
            ),
        ),
        (
            "actuator unit".to_owned(),
            tags(
                one_state(),
                json!({ "components": [{ "role": "action/delta_eef_pos", "dim": 3,
                    "unit": 0.05 }] }),
            ),
        ),
    ]
}

/// What happens to each 0.2-only convention on the 0.1 reader: the tags
/// parse, but resolve refuses them, since 0.1 treats a field it does not
/// know as must-understand. A 0.1 model therefore fails closed, naming the
/// field, against a 0.2 env that declares one. Kept as a record: the tag
/// format is not changed for it.
#[test]
fn the_0_2_conventions_are_refused_by_0_1() {
    let cases = convention_cases();
    let failed = failures(cases.iter().map(|(name, tags)| (name.clone(), tags)));
    assert_eq!(failed.len(), cases.len(), "{failed:#?}");
    for (failure, field) in failed
        .iter()
        .zip(["polarity", "polarity", "polarity", "unit"])
    {
        assert!(failure.contains("refused at resolve"), "{failure}");
        assert!(
            failure.contains(&format!("unrecognized field \"{field}\"")),
            "{failure}"
        );
    }
}

/// The gate a 0.2 env meets a 0.1 model at: every convention read.
#[test]
#[ignore = "the released 0.1 resolve refuses 0.2's polarity (state, split field, actuator) \
            and actuator unit as unrecognized fields: see the_0_2_conventions_are_refused_by_0_1"]
fn the_0_2_conventions_read_on_0_1() {
    assert_read(convention_cases());
}
