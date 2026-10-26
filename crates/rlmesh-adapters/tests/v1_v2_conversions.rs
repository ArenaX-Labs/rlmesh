//! Golden: each conversion gives the same values through the binder's two
//! doors, a Gym binding (env tags and a model spec, lowered, bound and
//! applied) and a v2 binding (a robot map and a policy map, bound and
//! applied), including the inputs where they once differed: a quaternion
//! with w < 0, gimbal lock, the input affine and the binary snap.

use std::collections::BTreeMap;

use rlmesh_adapters::v1::{
    EnvTags, ModelSpec, NoCustoms, SpaceView, Value, resolve, transform_action, transform_obs,
};
use rlmesh_session_core::spec::apply::Value as V2Value;
use rlmesh_session_core::spec::{Binding, Policy, RunOptions, bind, convert, normalise};
use rlmesh_session_core::wire::part::{Layout, PartSpec};
use rlmesh_session_core::wire::version::SPEC_VERSION;
use rlmesh_session_core::wire::{ChannelId, ParticipantId};
use rlmesh_spaces::{DType, Tensor};
use serde_json::json;

/// Quaternions (xyzw) the codecs used to disagree on: w < 0, a 200° turn,
/// a gimbal-locked pose (as an f32 quaternion), a half turn, the identity.
const QUATS: [[f32; 4]; 6] = [
    [0.0, 0.0, 0.1, -0.995],
    [0.001_257_12, 0.305_294_78, -0.280_147_64, -0.910_115_84],
    [0.0, 0.0, 0.984_807_7, -0.173_648_18],
    [0.455_530_7, -0.540_825_07, 0.455_530_7, 0.540_825_07],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

const H: usize = 48;
const W: usize = 64;

fn frame() -> Vec<u8> {
    let mut seed = 0x1357_9bdf_u32;
    (0..H * W * 3)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed >> 24) as u8
        })
        .collect()
}

// v1: a LIBERO-shaped env and model.

fn v1_adapter(model: serde_json::Value) -> rlmesh_adapters::v1::ResolvedAdapter {
    let tags: EnvTags = serde_json::from_value(json!({
        "observation": {
            "agentview": {"type": "image", "role": "image/primary", "layout": "hwc", "upside_down": true},
            "wrist": {"type": "image", "role": "image/wrist", "layout": "hwc", "upside_down": true},
            "eef_pos": {"type": "state", "role": "proprio/eef_pos"},
            "eef_quat": {"type": "state", "role": "proprio/eef_rot", "encoding": "quat_xyzw"},
            "fingers": {"type": "state", "role": "proprio/gripper"},
            "instruction": {"type": "text", "role": "text/instruction"}
        },
        "action": {"components": [
            {"role": "action/delta_eef_pos", "dim": 3},
            {"role": "action/delta_eef_rot", "dim": 3, "encoding": "axis_angle"},
            {"role": "action/gripper", "dim": 1, "range": [-1.0, 1.0], "binary": true}
        ]}
    }))
    .expect("tags");
    let image = json!({"kind": "box", "shape": [H, W, 3], "dtype": "uint8"});
    let vec = |n: usize| json!({"kind": "box", "shape": [n], "dtype": "float32"});
    let obs: SpaceView = serde_json::from_value(json!({
        "kind": "dict", "dtype": "unspecified",
        "keys": ["agentview", "eef_pos", "eef_quat", "fingers", "instruction", "wrist"],
        "children": [image, vec(3), vec(4), vec(2), {"kind": "text", "dtype": "unspecified"}, image]
    }))
    .expect("obs");
    let action: SpaceView = serde_json::from_value(vec(7)).expect("action");
    let spec: ModelSpec = serde_json::from_value(model).expect("model");
    resolve(&tags, &obs, &action, &spec, false).expect("v1 resolves")
}

fn f32_tensor(values: &[f32]) -> Value {
    let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    Value::Tensor(Tensor::from_vec(bytes, vec![values.len() as i64], DType::Float32).expect("t"))
}

fn v1_obs(quat: [f32; 4], fingers: [f32; 2]) -> BTreeMap<String, Value> {
    let image = Value::Tensor(
        Tensor::from_vec(frame(), vec![H as i64, W as i64, 3], DType::Uint8).expect("image"),
    );
    BTreeMap::from([
        ("agentview".to_owned(), image.clone()),
        ("wrist".to_owned(), image),
        ("eef_pos".to_owned(), f32_tensor(&[0.1, 0.2, 0.3])),
        ("eef_quat".to_owned(), f32_tensor(&quat)),
        ("fingers".to_owned(), f32_tensor(&fingers)),
        ("instruction".to_owned(), Value::Text("pick".to_owned())),
    ])
}

fn v1_bytes(payload: &Value, key: &str) -> Vec<u8> {
    let Value::Map(map) = payload else {
        panic!("not a map: {payload:?}")
    };
    match &map[key] {
        Value::Tensor(t) => t.to_contiguous_bytes().into_owned(),
        Value::List(items) => items
            .iter()
            .flat_map(|v| match v {
                Value::Number(n) => (*n as f32).to_le_bytes(),
                other => panic!("{other:?}"),
            })
            .collect(),
        other => panic!("{key}: {other:?}"),
    }
}

// v2: the same scene as a robot map, the same model as a policy map.

fn v2_binding(policy: serde_json::Value) -> Binding {
    let robot = json!({
        "scene/cam": {"kind": "camera", "width": W, "height": H, "hz": 20, "mount": "upside_down"},
        "wrist/cam": {"kind": "camera", "width": W, "height": H, "hz": 20, "mount": "upside_down"},
        "arm/tcp": {"kind": "pose", "frame": "robot_base", "child": "tcp", "hz": 20,
                    "command": "delta", "reference": "current"},
        "gripper/fingers": {"kind": "joints", "count": 2, "fields": ["pos"], "hz": 20,
                            "prismatic": true},
        "gripper": {"kind": "gripper", "hz": 20, "command": "position", "binary": true},
        "task": {"kind": "text"}
    });
    let mut parts = normalise(
        &ParticipantId::new("libero").expect("id"),
        SPEC_VERSION,
        &robot.to_string(),
    )
    .expect("robot");
    let mut next = 0;
    for part in &mut parts {
        next += 1;
        part.channel = ChannelId(next);
        if part.command.is_some() {
            next += 1;
            part.applied_channel = Some(ChannelId(next));
        }
    }
    let policy = Policy::parse(&policy.to_string()).expect("policy");
    bind(&policy, &parts, &RunOptions::named("golden")).expect("v2 binds")
}

fn packed(part: &PartSpec, values: &[(&str, &[f32])]) -> Vec<u8> {
    let layout = Layout::of(&part.fields).expect("layout");
    let mut out = vec![0_u8; layout.stride().expect("packed")];
    for (i, f) in part.fields.iter().enumerate() {
        if let Some((_, v)) = values.iter().find(|(n, _)| *n == f.name) {
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            out[layout.field(i).expect("field")].copy_from_slice(&bytes);
        }
    }
    out
}

fn v2_input(b: &Binding, entry: &str, sample: &[u8]) -> Vec<u8> {
    match b.input(entry).expect(entry).apply(&[sample]).expect(entry) {
        V2Value::Tensor(t) => t.data,
        other => panic!("{entry}: {other:?}"),
    }
}

/// Byte equality, reporting the first difference rather than both buffers.
fn same(v1: &[u8], v2: &[u8], what: &str) {
    assert_eq!(v1.len(), v2.len(), "{what}: lengths");
    if let Some(i) = v1.iter().zip(v2).position(|(a, b)| a != b) {
        panic!(
            "{what}: byte {i} is {} through v1, {} through v2",
            v1[i], v2[i]
        );
    }
}

fn part(b: &Binding, entry: &str) -> PartSpec {
    b.input(entry).expect(entry).part.clone().expect("bound")
}

#[test]
fn a_libero_pair_converts_identically_through_v1_and_v2() {
    // Images: 180° rotation, a 0.9 slice crop, area resize to 32x40 with a
    // cover fit, BGR, CHW, normalised into [-1, 1] as float32; the wrist
    // stays u8 HWC, bilinear_aa, letterboxed. State: the pose's rotation as
    // axis_angle; the fingers through the affine `x * -2 + 1`.
    let v1 = v1_adapter(json!({
        "input": {
            "image": {"type": "image", "role": "image/primary", "height": 32, "width": 40,
                      "resample": "area", "fit": "crop", "crop": 0.9, "crop_mode": "slice",
                      "channel_order": "bgr", "layout": "chw", "dtype": "float32",
                      "normalize": [-1.0, 1.0]},
            "wrist": {"type": "image", "role": "image/wrist", "height": 30, "width": 30,
                      "resample": "bilinear_aa", "fit": "pad"},
            "eef": {"type": "state", "dtype": "float32", "components": [
                "proprio/eef_pos", {"role": "proprio/eef_rot", "encoding": "axis_angle"}]},
            "fingers": {"type": "state", "dtype": "float32", "components": [
                {"role": "proprio/gripper", "dim": 2, "scale": -2.0, "offset": 1.0}]},
            "prompt": {"type": "text", "role": "text/instruction"}
        },
        "output": {"components": [
            {"role": "action/delta_eef_pos", "dim": 3},
            {"role": "action/delta_eef_rot", "dim": 3, "encoding": "axis_angle"},
            {"role": "action/gripper", "dim": 1, "range": [-1.0, 1.0]}
        ]}
    }));
    let v2 = v2_binding(json!({
        "hz": 10,
        "inputs": {
            "image": {"kind": "camera", "at": "scene/cam", "size": [32, 40], "resample": "area",
                      "crop": 0.9, "format": "bgr8", "layout": "chw", "dtype": "float32",
                      "range": [-1.0, 1.0], "allow": ["crop"]},
            "wrist": {"kind": "camera", "at": "wrist/cam", "size": [30, 30],
                      "resample": "bilinear_aa", "allow": ["pad"]},
            "eef": {"kind": "pose", "at": "arm/tcp", "fields": ["pos", "rot"],
                    "rotation": "axis_angle"},
            "fingers": {"kind": "joints", "at": "gripper/fingers", "fields": ["pos"],
                        "count": 2, "scale": -2.0, "offset": 1.0},
            "prompt": {"kind": "text", "at": "task"}
        },
        "actions": {
            "eef": {"kind": "pose", "at": "arm/tcp", "command": "delta", "reference": "current"},
            "gripper": {"kind": "gripper", "at": "gripper", "command": "position",
                        "open": -1.0, "closed": 1.0}
        },
        "allow": ["resize", "rotate180", "rotation", "rescale", "bgr"]
    }));
    let frame = frame();
    let (tcp, fingers) = (part(&v2, "eef"), part(&v2, "fingers"));
    for quat in QUATS {
        let finger_values = [0.031_f32, -0.0287];
        let payload = transform_obs(&v1, &v1_obs(quat, finger_values), &NoCustoms).expect("v1 obs");
        same(
            &v1_bytes(&payload, "image"),
            &v2_input(&v2, "image", &frame),
            "image",
        );
        same(
            &v1_bytes(&payload, "wrist"),
            &v2_input(&v2, "wrist", &frame),
            "wrist",
        );
        let pose = packed(&tcp, &[("pos", &[0.1, 0.2, 0.3]), ("rot", &quat)]);
        assert_eq!(
            v1_bytes(&payload, "eef"),
            v2_input(&v2, "eef", &pose),
            "eef {quat:?}"
        );
        let joints = packed(&fingers, &[("pos", &finger_values)]);
        assert_eq!(
            v1_bytes(&payload, "fingers"),
            v2_input(&v2, "fingers", &joints),
            "fingers"
        );
    }
}

#[test]
fn the_v1_gripper_snap_and_the_edge_snap_are_one_rule() {
    // v1: model [0, 1] mapped onto the env's [-1, 1], then ±1 at 0. The
    // edge: a v2 binary command, 0/1 at 0.5. Both are the midpoint rule.
    let v1 = v1_adapter(json!({
        "input": {"prompt": {"type": "text", "role": "text/instruction"}},
        "output": {"components": [
            {"role": "action/delta_eef_pos", "dim": 3},
            {"role": "action/delta_eef_rot", "dim": 3, "encoding": "axis_angle"},
            {"role": "action/gripper", "dim": 1, "range": [0.0, 1.0], "binary": true}
        ]}
    }));
    for g in [0.0_f32, 0.25, 0.4999, 0.5, 0.5001, 0.75, 1.0] {
        let action = f32_tensor(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, g]);
        let out = transform_action(&v1, &action).expect("v1 action");
        let got = f32::from_le_bytes(out.to_contiguous_bytes()[24..28].try_into().expect("f32"));
        let edge = convert::binary(f64::from(g), 0.0, 1.0);
        // The neutral raw 0.0 at the exact midpoint is v1's one exception.
        if g == 0.5 {
            assert_eq!(got, 0.0);
            continue;
        }
        assert_eq!(f64::from(got), edge * 2.0 - 1.0, "g = {g}");
    }
}
