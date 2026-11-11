//! The normaliser's kinds, custom fields, refusals and fingerprints.

use super::*;
use crate::spec::canonical;
use crate::spec::units::NORMALIZED;
use crate::wire::part::{ChannelPriority, DType, Encoding, MAX_PARTS, PartSpec};
use crate::wire::version::SPEC_VERSION;
use crate::wire::{CtlError, CtlStatus, ParticipantId};

fn robot() -> ParticipantId {
    ParticipantId::new("ur10e").expect("id")
}

fn parts(json: &str) -> Vec<PartSpec> {
    normalise(&robot(), SPEC_VERSION, json).expect("normalise")
}

fn refused(json: &str) -> CtlError {
    normalise(&robot(), SPEC_VERSION, json).expect_err("refused")
}

#[test]
fn authored_paths_keep_the_maps_order_and_refuse_duplicates() {
    let json = r#"{"z": {"kind": "text"}, "a/b": {"kind": "text"}, "m": {}}"#;
    assert_eq!(authored_paths(json).expect("paths"), ["z", "a/b", "m"]);
    let twice = r#"{"z": {"kind": "text"}, "z": {"kind": "text"}}"#;
    assert!(authored_paths(twice).is_err());
    assert!(authored_paths("[1]").is_err());
}

fn shapes(part: &PartSpec) -> Vec<(String, DType, Vec<u32>, String)> {
    part.fields
        .iter()
        .map(|f| (f.name.clone(), f.dtype, f.shape.clone(), f.unit.clone()))
        .collect()
}

#[test]
fn core_kinds_expand_to_their_fields() {
    let ur = parts(
        r#"{
            "arm": {"kind": "joints", "names": ["a", "b", "c"], "fields": ["effort", "pos"],
                    "hz": 500, "command": "position", "limits": {"pos": [-6, 6]}, "group": "arm",
                    "units": {"pos": "deg"}},
            "arm/tcp": {"kind": "pose", "frame": "base", "child": "tool0", "hz": 500,
                        "command": "delta", "group": "arm",
                        "limits": {"pos": {"max_norm": 0.01}, "rot": {"max_norm": 0.05}}},
            "wrist/cam": {"kind": "camera", "width": 640, "height": 480, "hz": 30,
                          "encoding": "jpeg", "quality": 75, "mount": "upside_down"},
            "head/depth": {"kind": "depth", "width": 64, "height": 48, "hz": 30, "dtype": "uint16"},
            "gripper": {"kind": "gripper", "hz": 100, "command": "position", "width_m": [0, 0.085],
                        "units": {"pos": "m"}, "binary": true},
            "base/imu": {"kind": "imu", "frame": "base", "hz": 500, "rotation": "quat_wxyz",
                         "fields": ["lin_acc", "orientation"]},
            "base/twist": {"kind": "velocity", "frame": "base", "hz": 50, "command": "velocity"},
            "hand/tactile": {"kind": "tactile", "shape": [9, 12], "hz": 100},
            "task": {"kind": "text"},
            "reward": {"kind": "reward"},
            "success": {"kind": "success"}
        }"#,
    );
    let arm = &ur[0];
    assert_eq!(
        shapes(arm),
        [
            ("pos".into(), DType::F32, vec![3], "deg".into()),
            ("effort".into(), DType::F32, vec![3], "N*m".into())
        ],
        "table order, declared units"
    );
    let command = arm.command.as_ref().expect("commandable");
    assert_eq!(
        command.fields[0].unit, "deg",
        "units apply to command fields"
    );
    assert!(command.hold_last && arm.control_hz == 500.0);
    let tcp = ur[1].command.as_ref().expect("delta");
    assert_eq!(
        (
            ur[1].fields[1].rotation.as_str(),
            tcp.fields[1].rotation.as_str()
        ),
        ("quat_xyzw", "axis_angle")
    );
    assert_eq!(tcp.reference, "current");
    assert!(!tcp.hold_last && ur[1].attrs[attrs::CHILD] == "tool0");
    let cam = &ur[2];
    assert_eq!(shapes(cam)[0].2, [480, 640, 3]);
    assert_eq!(cam.encoding, Encoding::Jpeg { quality: 75 });
    assert_eq!(
        (
            cam.priority,
            cam.send.as_str(),
            cam.attrs[attrs::MOUNT].as_str()
        ),
        (ChannelPriority::Bulk, "latest(hz=30)", "upside_down")
    );
    assert_eq!(
        shapes(&ur[3])[0],
        ("depth".into(), DType::U16, vec![48, 64], "mm".into())
    );
    let gripper = &ur[4];
    assert_eq!(gripper.fields[0].unit, "m");
    assert_eq!(gripper.attrs[attrs::WIDTH_M], "[0,0.085]");
    assert!(gripper.command.as_ref().is_some_and(|c| c.binary));
    let imu = &ur[5];
    assert_eq!(
        imu.fields
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>(),
        ["orientation", "lin_acc"]
    );
    assert_eq!(imu.fields[0].rotation, "quat_wxyz");
    assert_eq!(shapes(&ur[6])[0].2, [2], "planar by default");
    assert_eq!(
        (ur[7].priority, ur[7].fields[0].unit.as_str()),
        (ChannelPriority::Data, NORMALIZED)
    );
    for event in &ur[8..] {
        assert!(event.event && event.hz == 0.0 && event.priority == ChannelPriority::Data);
    }
    assert_eq!(ur[8].fields[0].dtype, DType::Utf8);
}

#[test]
fn custom_kinds_keep_their_declared_field_order() {
    let go2 = parts(
        r#"{"feet/force": {"kind": "unitree/foot_force", "hz": 500,
                "fields": {"force": ["int16", [4]], "contact": ["bool", [4]]}, "units": {"force": "raw"}},
                "x": {"kind": "x/blob", "hz": 10, "fields": {"b": ["uint8", [8192]]}}}"#,
    );
    assert_eq!(
        shapes(&go2[0]),
        [
            ("force".into(), DType::I16, vec![4], "raw".into()),
            ("contact".into(), DType::Bool, vec![4], String::new())
        ]
    );
    assert_eq!(go2[0].priority, ChannelPriority::Data);
    assert_eq!(
        (go2[1].priority, go2[1].send.as_str()),
        (ChannelPriority::Bulk, "latest(hz=10)")
    );
    // The canonical form keeps the order in an array, so swapping the
    // two fields changes the fingerprint.
    let swapped = parts(
        r#"{"feet/force": {"kind": "unitree/foot_force", "hz": 500,
                "fields": {"contact": ["bool", [4]], "force": ["int16", [4]]}, "units": {"force": "raw"}},
                "x": {"kind": "x/blob", "hz": 10, "fields": {"b": ["uint8", [8192]]}}}"#,
    );
    assert_ne!(fingerprint(&go2), fingerprint(&swapped));
}

#[test]
fn the_fingerprint_is_of_the_normalised_form() {
    let a = parts(r#"{"arm": {"kind": "joints", "count": 2, "hz": 100}}"#);
    let b = parts(
        r#"{"arm": {"hz": 100.0, "fields": ["vel", "pos"], "kind": "joints", "count": 2,
                "send": "every", "dtype": "float32"}}"#,
    );
    assert_eq!(fingerprint(&a), fingerprint(&b), "defaults expanded");
    let c = parts(r#"{"arm": {"kind": "joints", "count": 2, "hz": 101}}"#);
    assert_ne!(fingerprint(&a), fingerprint(&c));
    let text = canonical::to_string(&canonical(&a));
    assert!(text.starts_with(r#"[{"attrs":{},"command":null,"control_hz":0,"encoding":"raw","event":false,"fields":[{"dtype":"float32","name":"pos""#), "{text}");
}

#[test]
fn declarations_failing_the_rules_are_refused_naming_the_path_and_key() {
    let cases = [
        (r#"{"Arm": {"kind": "joints", "count": 1, "hz": 1}}"#, "Arm"),
        (
            r#"{"_session/x": {"kind": "joints", "count": 1, "hz": 1}}"#,
            "reserved",
        ),
        (
            r#"{"arm": {"kind": "rlmesh/a", "hz": 1, "fields": {"v": ["float32", [1]]}}}"#,
            "reserved",
        ),
        (r#"{"arm": {"kind": "points", "hz": 1}}"#, "core kind"),
        (
            r#"{"arm": {"kind": "joints", "count": 1}}"#,
            "hz is required",
        ),
        (
            r#"{"arm": {"kind": "joints", "count": 1, "hz": 20000}}"#,
            "hz",
        ),
        (r#"{"arm": {"kind": "joints", "hz": 1}}"#, "names"),
        (
            r#"{"arm": {"kind": "joints", "names": ["a", "a"], "hz": 1}}"#,
            "repeated",
        ),
        (
            r#"{"arm": {"kind": "joints", "names": ["a"], "count": 2, "hz": 1}}"#,
            "count",
        ),
        (r#"{"arm": {"kind": "hand", "count": 2, "hz": 1}}"#, "names"),
        (
            r#"{"arm": {"kind": "joints", "count": 1, "hz": 1, "fields": ["pos", "acc"]}}"#,
            "acc",
        ),
        (
            r#"{"arm": {"kind": "joints", "count": 1, "hz": 1, "width": 3}}"#,
            "width does not apply",
        ),
        (
            r#"{"arm": {"kind": "joints", "count": 1, "hz": 1, "color": "red"}}"#,
            "color",
        ),
        (
            r#"{"arm": {"kind": "joints", "count": 1, "hz": 1, "safe": "stop"}}"#,
            "without command",
        ),
        (
            r#"{"arm": {"kind": "joints", "count": 1, "hz": 1, "command": "delta"}}"#,
            "position, velocity, torque",
        ),
        (
            r#"{"arm": {"kind": "joints", "count": 1, "hz": 1, "units": {"pos": "mm"}}}"#,
            "angle",
        ),
        (
            r#"{"arm": {"kind": "joints", "count": 1, "hz": 1, "prismatic": true, "units": {"pos": "deg"}}}"#,
            "length",
        ),
        (
            r#"{"arm": {"kind": "joints", "count": 2, "hz": 1, "command": "position", "limits": {"pos": [[0, 1]]}}}"#,
            "pairs",
        ),
        (
            r#"{"arm": {"kind": "joints", "count": 2, "hz": 1, "command": "position", "limits": {"pos": [1, 0]}}}"#,
            "low <= high",
        ),
        (
            r#"{"arm": {"kind": "joints", "count": 2, "hz": 1, "command": "position", "limits": {"pos": {"max_norm": 1}}}}"#,
            "max_norm",
        ),
        (
            r#"{"arm": {"kind": "joints", "count": 2, "hz": 1, "command": "position", "limits": {"vel": [0, 1]}}}"#,
            "not a command field",
        ),
        (
            r#"{"arm": {"kind": "joints", "count": 2, "hz": 1, "command": "position", "hold_max_ms": 70000}}"#,
            "hold_max_ms",
        ),
        (
            r#"{"arm": {"kind": "joints", "count": 2, "hz": 1, "range": {"pos": [1, 1]}}}"#,
            "low < high",
        ),
        (
            r#"{"cam": {"kind": "camera", "width": 4, "hz": 1}}"#,
            "width and height",
        ),
        (
            r#"{"cam": {"kind": "camera", "width": 4, "height": 4, "hz": 1, "format": "mono8", "encoding": "jpeg"}}"#,
            "rgb8",
        ),
        (
            r#"{"cam": {"kind": "camera", "width": 4, "height": 4, "hz": 1, "quality": 50}}"#,
            "quality without jpeg",
        ),
        (
            r#"{"cam": {"kind": "camera", "width": 4096, "height": 4096, "hz": 1}}"#,
            "cap",
        ),
        (r#"{"ft": {"kind": "wrench", "hz": 1}}"#, "frame"),
        (
            r#"{"imu": {"kind": "imu", "frame": "base", "hz": 1}}"#,
            "lists its fields",
        ),
        (
            r#"{"imu": {"kind": "imu", "frame": "base", "hz": 1, "fields": ["ang_vel"], "rotation": "quat_wxyz"}}"#,
            "orientation",
        ),
        (
            r#"{"tcp": {"kind": "pose", "frame": "base", "hz": 1}}"#,
            "child",
        ),
        (
            r#"{"tcp": {"kind": "pose", "frame": "base", "child": "t", "hz": 1, "rotation": "gravity_xyz"}}"#,
            "source",
        ),
        (
            r#"{"tcp": {"kind": "pose", "frame": "base", "child": "t", "hz": 1, "command": "absolute", "reference": "target"}}"#,
            "delta only",
        ),
        (
            r#"{"g": {"kind": "gripper", "hz": 1, "units": {"pos": "m"}}}"#,
            "width_m",
        ),
        (r#"{"t": {"kind": "text", "hz": 10}}"#, "hz does not apply"),
        (r#"{"x": {"kind": "x/a", "hz": 1}}"#, "declares its fields"),
        (
            r#"{"x": {"kind": "x/a", "hz": 1, "fields": {"t": ["utf8", []], "v": ["float32", [1]]}}}"#,
            "utf8",
        ),
        (
            r#"{"x": {"kind": "x/a", "hz": 1, "fields": {"v": ["float32", [1]], "v": ["float32", [1]]}}}"#,
            "duplicate",
        ),
        (
            r#"{"a": {"kind": "joints", "count": 2, "hz": 1, "command": "velocity", "group": "g"},
                 "b": {"kind": "joints", "count": 2, "hz": 1, "command": "velocity", "group": "g", "safe": "brake"}}"#,
            "same safe",
        ),
        ("[]", "robot map"),
        ("{", "robot map"),
        (r#"{} {}"#, "robot map"),
    ];
    for (json, needle) in cases {
        let err = refused(json);
        assert_eq!(err.status, CtlStatus::InvalidRequest, "{json}: {err}");
        assert!(err.message.contains(needle), "{json}: {}", err.message);
    }
    assert_eq!(
        refused(r#"{"a": {"kind": "joints", "count": 1, "hz": 1, "command": "position", "merge": "blend"}}"#).status,
        CtlStatus::Unsupported
    );
    let many: Vec<String> = (0..=MAX_PARTS)
        .map(|i| format!(r#""p{i}": {{"kind": "text"}}"#))
        .collect();
    assert!(
        refused(&format!("{{{}}}", many.join(",")))
            .message
            .contains("at most 256")
    );
}

#[test]
fn a_robot_on_an_older_or_equal_kind_table_is_accepted() {
    for (declared, own) in [(1, 1), (1, 3), (2, 3), (3, 3)] {
        assert_eq!(
            check_spec_version(declared, own),
            Ok(()),
            "{declared} on {own}"
        );
    }
    assert!(normalise(&robot(), SPEC_VERSION, "{}").is_ok());
}

#[test]
fn a_robot_on_a_newer_kind_table_is_told_to_upgrade_the_host() {
    let err = check_spec_version(4, 3).expect_err("newer");
    assert_eq!(err.status, CtlStatus::Incompatible);
    for needle in ["4", "3", "upgrade the host"] {
        assert!(err.message.contains(needle), "{}", err.message);
    }
    let err = normalise(&robot(), SPEC_VERSION + 1, "{}").expect_err("newer");
    assert_eq!(err.status, CtlStatus::Incompatible);
    assert!(err.message.contains("upgrade the host"), "{}", err.message);
}

#[test]
fn kind_table_versions_start_at_one() {
    let err = normalise(&robot(), 0, "{}").expect_err("zero");
    assert_eq!(err.status, CtlStatus::Incompatible);
}
