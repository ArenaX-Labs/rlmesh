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
                    "hz": 500, "command": "position", "safe": "stop", "limits": {"pos": [-6, 6]}, "group": "arm",
                    "units": {"pos": "deg"}},
            "arm/tcp": {"kind": "pose", "frame": "base", "child": "tool0", "hz": 500,
                        "command": "delta", "safe": "stop", "group": "arm",
                        "limits": {"pos": {"max_norm": 0.01}, "rot": {"max_norm": 0.05}}},
            "wrist/cam": {"kind": "camera", "width": 640, "height": 480, "hz": 30,
                          "encoding": "jpeg", "quality": 75, "mount": "upside_down"},
            "head/depth": {"kind": "depth", "width": 64, "height": 48, "hz": 30, "dtype": "uint16"},
            "gripper": {"kind": "gripper", "hz": 100, "command": "position", "safe": "stop", "width_m": [0, 0.085],
                        "units": {"pos": "m"}, "binary": true},
            "base/imu": {"kind": "imu", "frame": "base", "hz": 500, "rotation": "quat_wxyz",
                         "fields": ["lin_acc", "orientation"]},
            "base/twist": {"kind": "velocity", "frame": "base", "hz": 50, "command": "velocity", "safe": "stop"},
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
fn a_declared_priority_is_kept_and_carried_in_the_canonical_form() {
    let robot = parts(r#"{"log": {"kind": "text", "priority": "bulk"}}"#);
    assert_eq!(robot[0].priority, ChannelPriority::Bulk);
    let text = canonical::to_string(&canonical(&robot));
    assert!(text.contains(r#""priority":"bulk""#), "{text}");
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
    // Command durations are carried in milliseconds.
    let commanded = parts(
        r#"{"arm": {"kind": "joints", "count": 1, "hz": 50, "command": "position", "safe": "hold",
                "hold_max_ms": 250, "max_action_age_ms": 400, "max_clock_error_ms": 12.5}}"#,
    );
    let text = canonical::to_string(&canonical(&commanded));
    for ms in [
        r#""hold_max_ms":250,"#,
        r#""max_action_age_ms":400,"#,
        r#""max_clock_error_ms":12.5,"#,
    ] {
        assert!(text.contains(ms), "{ms} in {text}");
    }
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
            r#"{"arm": {"kind": "joints", "count": 2, "hz": 1, "command": "position", "safe": "stop", "limits": {"pos": [[0, 1]]}}}"#,
            "pairs",
        ),
        (
            r#"{"arm": {"kind": "joints", "count": 2, "hz": 1, "command": "position", "safe": "stop", "limits": {"pos": [1, 0]}}}"#,
            "low <= high",
        ),
        (
            r#"{"arm": {"kind": "joints", "count": 2, "hz": 1, "command": "position", "safe": "stop", "limits": {"pos": {"max_norm": 1}}}}"#,
            "max_norm",
        ),
        (
            r#"{"arm": {"kind": "joints", "count": 2, "hz": 1, "command": "position", "safe": "stop", "limits": {"vel": [0, 1]}}}"#,
            "not a command field",
        ),
        (
            r#"{"arm": {"kind": "joints", "count": 2, "hz": 1, "command": "position", "safe": "stop", "hold_max_ms": 70000}}"#,
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
            r#"{"tcp": {"kind": "pose", "frame": "base", "child": "t", "hz": 1, "command": "absolute", "safe": "stop", "reference": "target"}}"#,
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
            r#"{"a": {"kind": "joints", "count": 2, "hz": 1, "command": "velocity", "safe": "stop", "group": "g"},
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
    let missing =
        refused(r#"{"arm": {"kind": "joints", "count": 1, "hz": 1, "command": "position"}}"#);
    assert_eq!(missing.status, CtlStatus::InvalidRequest);
    for needle in ["arm", "safe", "stop", "hold", "firmware"] {
        assert!(missing.message.contains(needle), "{}", missing.message);
    }
    assert_eq!(
        refused(r#"{"a": {"kind": "joints", "count": 1, "hz": 1, "command": "position", "safe": "stop", "merge": "blend"}}"#).status,
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
    let full = format!("{{{}}}", many[..MAX_PARTS].join(","));
    assert_eq!(
        parts(&full).len(),
        MAX_PARTS,
        "exactly the limit is accepted"
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

#[test]
fn a_safe_behaviour_is_any_identifier_passed_through_uninterpreted() {
    for safe in ["stop", "hold", "park"] {
        let parts = parts(&format!(
            r#"{{"arm": {{"kind": "joints", "count": 1, "hz": 1, "command": "position", "safe": "{safe}"}}}}"#
        ));
        assert_eq!(
            parts[0].command.as_ref().expect("command").safe_behavior,
            safe
        );
    }
    let bad = refused(
        r#"{"arm": {"kind": "joints", "count": 1, "hz": 1, "command": "position", "safe": "Not Ok"}}"#,
    );
    assert!(bad.message.contains("safe"), "{}", bad.message);
}

#[test]
fn a_switch_is_one_bool_and_holds_its_target() {
    let lamp = parts(
        r#"{"led": {"kind": "switch", "hz": 10, "command": "set", "safe": "stop"},
            "button": {"kind": "switch", "hz": 50}}"#,
    );
    assert_eq!(
        shapes(&lamp[0]),
        [("on".into(), DType::Bool, vec![], String::new())]
    );
    let command = lamp[0].command.as_ref().expect("commanded");
    assert_eq!(
        (
            command.mode.as_str(),
            command.fields.len(),
            command.hold_last
        ),
        ("set", 1, true)
    );
    assert_eq!(
        (lamp[0].priority, lamp[0].send.as_str()),
        (ChannelPriority::Realtime, "every")
    );
    assert!(lamp[1].command.is_none(), "a button is observe-only");
}

#[test]
fn a_value_is_one_float_with_an_optional_unit_and_range() {
    let room = parts(
        r#"{"temp": {"kind": "value", "hz": 1, "unit": "degC", "range": [-40, 125]},
            "dimmer": {"kind": "value", "hz": 20, "command": "set", "safe": "hold"},
            "bare": {"kind": "value", "hz": 1}}"#,
    );
    assert_eq!(
        shapes(&room[0]),
        [("value".into(), DType::F32, vec![], "degC".into())]
    );
    assert_eq!(room[0].attrs[attrs::RANGE], r#"{"value":[-40,125]}"#);
    assert!(room[0].command.is_none(), "a thermometer is observe-only");
    let command = room[1].command.as_ref().expect("commanded");
    assert_eq!((command.mode.as_str(), command.hold_last), ("set", true));
    assert_eq!(shapes(&room[2])[0].3, "", "unitless by default");
}

#[test]
fn switches_and_values_refuse_what_does_not_apply() {
    let cases = [
        (
            r#"{"s": {"kind": "switch", "hz": 1, "command": "position", "safe": "stop"}}"#,
            "takes set",
        ),
        (
            r#"{"s": {"kind": "switch", "hz": 1, "command": "set"}}"#,
            "safe",
        ),
        (
            r#"{"s": {"kind": "switch", "hz": 1, "unit": "V"}}"#,
            "unit does not apply",
        ),
        (r#"{"s": {"kind": "switch"}}"#, "hz is required"),
        (
            r#"{"s": {"kind": "switch", "hz": 1, "command": "set", "safe": "stop", "limits": {"on": [0, 1]}}}"#,
            "only on float32",
        ),
        (
            r#"{"v": {"kind": "value", "hz": 1, "dtype": "float64"}}"#,
            "dtype does not apply",
        ),
        (
            r#"{"v": {"kind": "value", "hz": 1, "units": {"value": "V"}}}"#,
            "units does not apply",
        ),
        (
            r#"{"v": {"kind": "value", "hz": 1, "range": {"value": [0, 1]}}}"#,
            "range is [low, high]",
        ),
        (
            r#"{"v": {"kind": "value", "hz": 1, "range": [1, 0]}}"#,
            "low < high",
        ),
        (r#"{"v": {"kind": "value", "hz": 1, "unit": ""}}"#, "unit"),
        (
            r#"{"a": {"kind": "joints", "count": 1, "hz": 1, "range": [0, 1]}}"#,
            "range is {field",
        ),
    ];
    for (json, needle) in cases {
        let err = refused(json);
        assert_eq!(err.status, CtlStatus::InvalidRequest, "{json}: {err}");
        assert!(err.message.contains(needle), "{json}: {}", err.message);
    }
}

#[test]
fn latch_defaults_on_for_switches_and_values_and_off_elsewhere() {
    let robot = parts(
        r#"{"relay": {"kind": "switch", "hz": 10, "command": "set", "safe": "stop"},
            "dimmer": {"kind": "value", "hz": 10, "command": "set", "safe": "hold"},
            "manual": {"kind": "switch", "hz": 10, "command": "set", "safe": "stop", "latch": false},
            "arm": {"kind": "joints", "count": 1, "hz": 10, "command": "position", "safe": "stop"},
            "brake": {"kind": "joints", "count": 1, "hz": 10, "command": "position", "safe": "stop",
                      "latch": true}}"#,
    );
    let latches: Vec<bool> = robot
        .iter()
        .map(|p| p.command.as_ref().expect("command").latch)
        .collect();
    assert_eq!(latches, [true, true, false, false, true]);
    let text = canonical::to_string(&canonical(&robot));
    assert!(text.contains(r#""latch":true"#), "{text}");
}

#[test]
fn latch_is_refused_on_a_mode_that_does_not_hold_its_target() {
    for part in [
        r#"{"kind": "joints", "count": 1, "hz": 1, "command": "velocity""#,
        r#"{"kind": "joints", "count": 1, "hz": 1, "command": "torque""#,
        r#"{"kind": "velocity", "frame": "base", "hz": 1, "command": "velocity""#,
        r#"{"kind": "pose", "frame": "b", "child": "t", "hz": 1, "command": "delta""#,
        r#"{"kind": "x/a", "fields": {"v": ["float32", [1]]}, "hz": 1, "command": "go""#,
    ] {
        let err = refused(&format!(
            r#"{{"p": {part}, "safe": "stop", "latch": true}}}}"#
        ));
        assert_eq!(err.status, CtlStatus::InvalidRequest, "{part}: {err}");
        assert!(err.message.contains("run away"), "{part}: {}", err.message);
    }
    let off = parts(
        r#"{"p": {"kind": "joints", "count": 1, "hz": 1, "command": "velocity", "safe": "stop",
                  "latch": false}}"#,
    );
    assert!(!off[0].command.as_ref().expect("command").latch);
    let idle = refused(r#"{"p": {"kind": "switch", "hz": 1, "latch": true}}"#);
    assert!(
        idle.message.contains("latch without command"),
        "{}",
        idle.message
    );
}

fn clock_bound(extra: &str) -> Result<Duration, CtlError> {
    let map = format!(
        r#"{{"arm": {{"kind": "joints", "count": 2, "hz": 50, "command": "position",
                     "safe": "stop"{extra}}}}}"#
    );
    let parts = normalise(&robot(), SPEC_VERSION, &map)?;
    Ok(parts[0].command.as_ref().expect("command").max_clock_error)
}

#[test]
fn the_clock_bound_defaults_to_a_quarter_of_the_age_and_at_least_5_ms() {
    assert_eq!(clock_bound("").expect("map"), Duration::from_millis(50));
    assert_eq!(
        clock_bound(r#", "max_action_age_ms": 1000"#).expect("map"),
        Duration::from_millis(250)
    );
    assert_eq!(
        clock_bound(r#", "max_action_age_ms": 8"#).expect("map"),
        Duration::from_millis(5),
        "the floor holds even above half of a very short age"
    );
}

#[test]
fn an_explicit_clock_bound_is_kept_up_to_half_the_age() {
    assert_eq!(
        clock_bound(r#", "max_clock_error_ms": 12.5"#).expect("map"),
        Duration::from_micros(12_500)
    );
    assert_eq!(
        clock_bound(r#", "max_clock_error_ms": 100"#).expect("half"),
        Duration::from_millis(100)
    );
    let err = clock_bound(r#", "max_clock_error_ms": 100.5"#).expect_err("above half");
    assert!(err.message.contains("meaningless"), "{}", err.message);
    assert!(
        err.message.contains("max_action_age_ms (200)"),
        "{}",
        err.message
    );
    for bad in ["0", "-1"] {
        let err = clock_bound(&format!(r#", "max_clock_error_ms": {bad}"#)).expect_err(bad);
        assert!(
            err.message.contains("max_clock_error_ms"),
            "{}",
            err.message
        );
    }
}

#[test]
fn declarations_at_their_limits_bind_and_past_them_are_refused() {
    let long = |n: usize| "a".repeat(n);
    let camera = |extra: &str| {
        format!(
            r#"{{"c": {{"kind": "camera", "width": 4, "height": 4, "hz": 1, "frame": "f"{extra}}}}}"#
        )
    };
    let tactile = |layout: &str| {
        format!(r#"{{"t": {{"kind": "tactile", "shape": [2], "hz": 1, "layout": "{layout}"}}}}"#)
    };
    let tcp = |max_norm: &str| {
        format!(
            r#"{{"tcp": {{"kind": "pose", "frame": "b", "child": "t", "hz": 1, "command": "delta",
                "safe": "stop", "limits": {{"pos": {{"max_norm": {max_norm}}}}}}}}}"#
        )
    };
    for ok in [
        format!(
            r#"{{"arm": {{"kind": "joints", "names": ["{}"], "hz": 1}}}}"#,
            long(128)
        ),
        format!(r#"{{"t": {{"kind": "text", "doc": "{}"}}}}"#, long(4096)),
        tactile(&long(4096)),
        camera(r#", "fov_deg": 90, "intrinsics": [500, 500, 2, 2]"#),
        tcp("0.01"),
        r#"{"a": {"kind": "joints", "count": 1, "hz": 1, "command": "position", "safe": "hold",
            "hold_max_ms": 0}}"#
            .to_string(),
        r#"{"x": {"kind": "x/box", "hz": 1, "fields": {"v": ["float32", [1]]}, "x_vendor": 7}}"#
            .to_string(),
    ] {
        normalise(&robot(), SPEC_VERSION, &ok).unwrap_or_else(|err| panic!("{ok}: {err}"));
    }
    for (bad, needle) in [
        (
            r#"{"arm": {"kind": "joints", "names": [""], "hz": 1}}"#.to_string(),
            "is empty",
        ),
        (
            format!(
                r#"{{"arm": {{"kind": "joints", "names": ["{}"], "hz": 1}}}}"#,
                long(129)
            ),
            "longer than 128",
        ),
        (
            format!(r#"{{"t": {{"kind": "text", "doc": "{}"}}}}"#, long(4097)),
            "doc is longer",
        ),
        (tactile(&long(4097)), "layout is longer"),
        (camera(r#", "fov_deg": 0"#), "fov_deg 0"),
        (camera(r#", "fov_deg": 360"#), "fov_deg 360"),
        (tcp("0"), "max_norm 0"),
        (
            r#"{"arm": {"kind": "joints", "count": 1, "hz": 0}}"#.to_string(),
            "hz",
        ),
    ] {
        let err = refused(&bad);
        assert!(err.message.contains(needle), "{bad}: {}", err.message);
    }
}

#[test]
fn a_wrench_takes_wrench_commands_and_an_imu_publishes_acceleration() {
    let robot = parts(
        r#"{"ft": {"kind": "wrench", "frame": "tool0", "hz": 100, "command": "wrench", "safe": "stop"},
            "sensor": {"kind": "wrench", "frame": "tool0", "hz": 100},
            "imu": {"kind": "imu", "frame": "base", "hz": 100, "fields": ["lin_acc"]}}"#,
    );
    let command: Vec<&str> = robot[0]
        .command
        .as_ref()
        .expect("commandable")
        .fields
        .iter()
        .map(|f| f.name.as_str())
        .collect();
    assert_eq!(command, ["force", "torque"]);
    assert!(robot[1].command.is_none());
    assert_eq!(
        shapes(&robot[2]),
        [("lin_acc".into(), DType::F32, vec![3], "m/s^2".into())]
    );
}
