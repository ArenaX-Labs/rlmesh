//! Conformance fixtures (Appendix B, wave 2): every §15 policy bound against
//! the §14 reference robots, and every §6.7 refusal, as golden binding
//! tables and refusal messages under `tests/golden/`.
//!
//! `RLMESH_BLESS=1 cargo test -p rlmesh-session-core --test bind` rewrites the
//! golden files; review the diff like code.

use std::path::PathBuf;
use std::time::Duration;

use rlmesh_session_core::spec::bind::{BindError, Binding, Code, Status};
use rlmesh_session_core::spec::policy::Allow;
use rlmesh_session_core::spec::{Policy, RunOptions, bind, normalise};
use rlmesh_session_core::wire::version::SPEC_VERSION;
use rlmesh_session_core::wire::{ChannelId, ContractMode, PartSpec, ParticipantId};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

/// A robot's parts as the host would hold them: normalised, with channel
/// ids assigned in authored order (each commandable part's applied stream
/// right after it).
fn robot(name: &str) -> Vec<PartSpec> {
    robot_as(name, name)
}

fn robot_as(file: &str, id: &str) -> Vec<PartSpec> {
    let json = std::fs::read_to_string(fixtures().join(format!("fixtures/robots/{file}.json")))
        .expect("robot fixture");
    let mut parts = normalise(&ParticipantId::new(id).expect("id"), SPEC_VERSION, &json)
        .unwrap_or_else(|err| panic!("{file}: {err}"));
    let mut next = 0;
    for part in &mut parts {
        next += 1;
        part.channel = ChannelId(next);
        if part.command.is_some() {
            next += 1;
            part.applied_channel = Some(ChannelId(next));
        }
    }
    parts
}

fn policy(name: &str) -> Policy {
    let json = std::fs::read_to_string(fixtures().join(format!("fixtures/policies/{name}.json")))
        .expect("policy fixture");
    Policy::parse(&json).unwrap_or_else(|err| panic!("{name}: {err}"))
}

/// Compares `text` with `tests/golden/{name}.txt`.
fn golden(name: &str, text: &str) {
    let path = fixtures().join(format!("golden/{name}.txt"));
    if std::env::var_os("RLMESH_BLESS").is_some() {
        std::fs::write(&path, text).expect("bless");
        return;
    }
    let want = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("no golden file {}; run with RLMESH_BLESS=1", path.display()));
    assert_eq!(text, want, "{name} differs from {}", path.display());
}

fn bound(policy_name: &str, robot_name: &str, options: RunOptions) -> Binding {
    let parts = robot(robot_name);
    match bind(&policy(policy_name), &parts, &options.clone()) {
        Ok(binding) => binding,
        Err(err) => panic!("{policy_name} on {robot_name}: {err}"),
    }
}

fn refused(
    policy_name: &str,
    robot_name: &str,
    options: RunOptions,
) -> rlmesh_session_core::spec::Refusal {
    let parts = robot(robot_name);
    match bind(&policy(policy_name), &parts, &options) {
        Ok(binding) => panic!("{policy_name} on {robot_name} bound:\n{binding}"),
        Err(BindError::Refused(refusal)) => *refusal,
        Err(err) => panic!("{err}"),
    }
}

fn plan(binding: &Binding, entry: &str) -> String {
    binding
        .input(entry)
        .map(|r| r.plan.to_string())
        .unwrap_or_else(|| panic!("no input {entry}"))
}

fn codes(refusal: &rlmesh_session_core::spec::Refusal) -> Vec<(Code, String)> {
    refusal
        .problems
        .iter()
        .map(|p| (p.code, p.entry.clone()))
        .collect()
}

#[test]
fn pi0_single_on_ur10e() {
    let b = bound("pi0_single", "ur10e", RunOptions::named("pi0_single"));
    assert_eq!(
        plan(&b, "image"),
        "decode(jpeg) → pad(1280x720 → 224x224, bilinear)"
    );
    assert_eq!(
        plan(&b, "wrist_image"),
        "decode(jpeg) → pad(640x480 → 224x224, bilinear)"
    );
    assert_eq!(plan(&b, "state"), "fields pos; names identical");
    assert_eq!(plan(&b, "gripper"), "none");
    assert_eq!(plan(&b, "prompt"), "none");
    let arm = b.action("arm").expect("arm");
    assert_eq!(
        (arm.horizon, arm.dt),
        (10, Duration::from_nanos(66_666_667))
    );
    assert_eq!(arm.plan.to_string(), "none; names identical");
    assert_eq!(
        b.action("gripper").expect("gripper").plan.to_string(),
        "rescale(-1…1 → 0…1)"
    );
    // The gripper keeps the default 200 ms open-loop budget: the binder
    // warns, never stretches.
    assert_eq!(b.warnings.len(), 1, "{:?}", b.warnings);
    golden("pi0_single__ur10e", &b.table());
}

#[test]
fn pi0_single_on_aloha2_needs_positional_names() {
    let options = RunOptions::named("pi0_single")
        .under("right")
        .expect("under");
    let refusal = refused("pi0_single", "aloha2", options.clone());
    assert_eq!(
        codes(&refusal),
        [
            (Code::Incompatible, "state".into()),
            (Code::Incompatible, "arm".into())
        ]
    );
    golden("pi0_single__aloha2__refused", &refusal.to_string());

    let b = bound(
        "pi0_single",
        "aloha2",
        options.positional("state").positional("arm"),
    );
    let parts: Vec<String> = b
        .inputs
        .iter()
        .filter_map(|r| r.part.as_ref().map(|p| p.path.to_string()))
        .collect();
    assert_eq!(
        parts,
        [
            "scene/cam",
            "right/wrist/cam",
            "right/arm",
            "right/gripper",
            "task"
        ]
    );
    assert_eq!(b.input("state").expect("state").status, Status::Override);
    assert_eq!(
        b.action("gripper")
            .expect("gripper")
            .part
            .as_ref()
            .map(|p| p.path.as_str()),
        Some("right/gripper")
    );
    golden("pi0_single__aloha2__positional", &b.table());
}

#[test]
fn pi0_single_on_go2_is_unbound_everywhere() {
    let refusal = refused("pi0_single", "go2", RunOptions::named("pi0_single"));
    assert_eq!(
        codes(&refusal),
        [
            (Code::Unbound, "image".into()),
            (Code::Unbound, "wrist_image".into()),
            (Code::Unbound, "state".into()),
            (Code::Unbound, "gripper".into()),
            (Code::Unbound, "prompt".into()),
            (Code::Unbound, "arm".into()),
            (Code::Unbound, "gripper".into()),
        ]
    );
    golden("pi0_single__go2__refused", &refusal.to_string());
}

#[test]
fn pi0_aloha_binds_every_entry_and_blanks_a_missing_camera() {
    let b = bound("pi0_aloha", "aloha2", RunOptions::named("pi0_aloha"));
    assert!(
        b.inputs
            .iter()
            .chain([].iter())
            .all(|r| r.status == Status::Bound)
            && b.actions.iter().all(|r| r.status == Status::Bound)
    );
    // 50 actions at 50 Hz: exactly the robot's 1000 ms budget, no warning.
    assert!(b.warnings.is_empty(), "{:?}", b.warnings);
    golden("pi0_aloha__aloha2", &b.table());

    let b = bound(
        "pi0_aloha",
        "aloha2_no_cam_low",
        RunOptions::named("pi0_aloha"),
    );
    let low = b.input("cam_low").expect("cam_low");
    assert_eq!(low.status, Status::Blank);
    assert_eq!(
        low.plan.to_string(),
        "blank(u8 224x224x3 black, then declared preprocessing)"
    );
    assert_eq!(low.delivers.to_string(), "u8 [224, 224, 3]");
    golden("pi0_aloha__aloha2_no_cam_low", &b.table());
}

#[test]
fn force_policy_on_ur10e() {
    let b = bound("force_policy", "ur10e", RunOptions::named("force_policy"));
    let ft = b.input("ft").expect("ft");
    assert_eq!(ft.delivers.to_string(), "f32 [50, 6]");
    assert_eq!(
        plan(&b, "ft"),
        "fields force, torque → window(50 by seq); warmup at episode start"
    );
    assert_eq!(b.input("q").expect("q").delivers.to_string(), "f32 [12]");
    assert_eq!(
        plan(&b, "wrist"),
        "decode(jpeg) → crop(640x480 → 96x96, bilinear) → auto layout(chw) → normalise(u8 → 0…1 f32)"
    );
    assert_eq!(
        b.input("wrist").expect("wrist").delivers.to_string(),
        "f32 [3, 96, 96]"
    );
    let last = b.input("last").expect("last");
    assert!(last.applied);
    assert_eq!(
        last.channel,
        last.part.as_ref().and_then(|p| p.applied_channel)
    );
    let modes: Vec<(String, ContractMode, f64)> = b
        .contracts
        .iter()
        .map(|c| {
            let entry = b
                .inputs
                .iter()
                .find(|r| r.channel == Some(c.channel))
                .map(|r| r.entry.clone())
                .unwrap_or_default();
            (entry, c.mode, c.delivery_hz)
        })
        .collect();
    assert_eq!(
        modes,
        [
            ("ft".into(), ContractMode::History, 50.0),
            ("q".into(), ContractMode::Latest, 50.0),
            // D = 50 is at or above the camera's S = 30: every (§8.2).
            ("wrist".into(), ContractMode::Every, 50.0),
            ("last".into(), ContractMode::Latest, 50.0),
        ]
    );
    // The derived freshness bound for q: max(3000 / 50, 5) = 60 ms.
    assert_eq!(
        b.input("q").expect("q").max_age,
        Some(Duration::from_millis(60))
    );
    let arm = b.action("arm").expect("arm");
    assert_eq!(
        (arm.dt, arm.horizon, arm.valid),
        (Duration::from_millis(10), 20, Duration::from_millis(150))
    );
    golden("force_policy__ur10e", &b.table());
}

#[test]
fn force_policy_refuses_a_base_frame_wrench_and_robots_without_one() {
    let rtde = refused(
        "force_policy",
        "ur10e_rtde",
        RunOptions::named("force_policy"),
    );
    assert_eq!(codes(&rtde), [(Code::Incompatible, "ft".into())]);
    golden("force_policy__ur10e_rtde__refused", &rtde.to_string());
    for robot in ["go2", "aloha2"] {
        let refusal = refused("force_policy", robot, RunOptions::named("force_policy"));
        assert_eq!(refusal.problems[0].code, Code::Unbound);
        assert_eq!(refusal.problems[0].entry, "ft");
        golden(
            &format!("force_policy__{robot}__refused"),
            &refusal.to_string(),
        );
    }
}

#[test]
fn a_haptic_operator_and_a_20_hz_policy_share_the_sim() {
    let operator = bound(
        "haply_operator",
        "ur10e_sim",
        RunOptions::named("haply_operator"),
    );
    for c in &operator.contracts {
        assert_eq!(c.mode, ContractMode::Every, "{c:?}");
    }
    assert_eq!(
        operator.input("ft").expect("ft").max_age,
        Some(Duration::from_millis(3))
    );
    assert_eq!(plan(&operator, "cam"), "decode(jpeg)");
    golden("haply_operator__ur10e_sim", &operator.table());

    let policy = bound(
        "force_policy_20hz",
        "ur10e_sim",
        RunOptions::named("force_policy_20hz"),
    );
    let ft = policy
        .contracts
        .iter()
        .find(|c| Some(c.channel) == policy.input("ft").and_then(|r| r.channel))
        .expect("ft contract");
    assert_eq!(
        (ft.mode, ft.delivery_hz, ft.history),
        (ContractMode::History, 20.0, 50)
    );
    golden("force_policy_20hz__ur10e_sim", &policy.table());
}

#[test]
fn loco_go2_reorders_isaac_joints_and_needs_rotation_for_gravity() {
    let b = bound("loco_go2", "go2", RunOptions::named("loco_go2"));
    assert_eq!(
        plan(&b, "gravity"),
        "fields orientation → rotation(orientation quat_wxyz → gravity_xyz)"
    );
    assert_eq!(
        b.input("gravity").expect("gravity").delivers.to_string(),
        "f32 [3]"
    );
    assert_eq!(
        b.input("command").expect("command").delivers.to_string(),
        "f32 [3]"
    );
    assert_eq!(
        plan(&b, "q"),
        "fields pos → auto reorder(by names) → offset(12 values)"
    );
    assert_eq!(
        b.action("legs").expect("legs").plan.to_string(),
        "scale(0.25) → offset(12 values) → auto reorder(by names)"
    );
    golden("loco_go2__go2", &b.table());

    let refusal = refused("loco_go2_no_rotation", "go2", RunOptions::named("loco_go2"));
    assert_eq!(codes(&refusal), [(Code::NotAllowed, "gravity".into())]);
    golden("loco_go2__go2__refused", &refusal.to_string());
}

#[test]
fn dex_grasp_needs_a_side() {
    let refusal = refused("dex_grasp", "g1", RunOptions::named("dex_grasp"));
    assert_eq!(codes(&refusal), [(Code::AmbiguousCapture, "hand".into())]);
    golden("dex_grasp__g1__refused", &refusal.to_string());
    let b = bound(
        "dex_grasp",
        "g1",
        RunOptions::named("dex_grasp")
            .capture("side", "left")
            .expect("capture"),
    );
    assert_eq!(b.captures, [("side".into(), "left".into())]);
    assert_eq!(
        b.action("hand")
            .and_then(|a| a.part.as_ref())
            .map(|p| p.path.as_str()),
        Some("left/hand")
    );
    // 8 actions at 30 Hz outlast the hands' default 200 ms budget.
    assert_eq!(b.warnings.len(), 1, "{:?}", b.warnings);
    golden("dex_grasp__g1__left", &b.table());
}

#[test]
fn timing_contracts_for_a_vla_and_a_planner() {
    let vla = bound("vla_20hz", "ur10e_sim", RunOptions::named("vla_20hz"));
    let modes: Vec<(ContractMode, u32)> =
        vla.contracts.iter().map(|c| (c.mode, c.history)).collect();
    assert_eq!(
        modes,
        [
            (ContractMode::Latest, 1),
            (ContractMode::Latest, 1),
            (ContractMode::History, 50)
        ]
    );
    assert_eq!(
        vla.action("arm").expect("arm").dt,
        Duration::from_millis(10)
    );
    golden("vla_20hz__ur10e_sim", &vla.table());

    let planner = bound("planner_2hz", "go2", RunOptions::named("planner_2hz"));
    assert!(planner.actions.is_empty() && planner.channels().is_empty());
    assert!(
        planner
            .contracts
            .iter()
            .all(|c| c.mode == ContractMode::Latest && c.delivery_hz == 2.0)
    );
    golden("planner_2hz__go2", &planner.table());
}

#[test]
fn pi05_libero_on_a_v2_native_libero_robot() {
    let b = bound("pi05_libero", "libero", RunOptions::named("pi05_libero"));
    assert_eq!(
        plan(&b, "observation/image"),
        "rotate180 → resize(bilinear, 224x224)"
    );
    assert_eq!(
        plan(&b, "eef"),
        "fields pos, rot → rotation(rot quat_xyzw → axis_angle)"
    );
    assert_eq!(b.input("eef").expect("eef").delivers.to_string(), "f32 [6]");
    assert_eq!(
        b.action("gripper").expect("gripper").plan.to_string(),
        "rescale(1…-1 → 0…1)"
    );
    golden("pi05_libero__libero", &b.table());

    let refusal = refused(
        "pi05_libero",
        "libero_joint_only",
        RunOptions::named("pi05_libero"),
    );
    assert_eq!(codes(&refusal), [(Code::Incompatible, "eef".into())]);
    golden(
        "pi05_libero__libero_joint_only__refused",
        &refusal.to_string(),
    );
}

#[test]
fn an_ambiguous_selector_and_an_unallowed_resize() {
    let refusal = refused("my_vla", "ur10e", RunOptions::named("my_vla"));
    assert_eq!(
        codes(&refusal),
        [
            (Code::Ambiguous, "image".into()),
            (Code::NotAllowed, "wrist_image".into())
        ]
    );
    golden("my_vla__ur10e__refused", &refusal.to_string());
    // Allowing the conversions at run time and picking the camera binds.
    let b = bound(
        "my_vla",
        "ur10e",
        RunOptions::named("my_vla")
            .at("image", "scene/cam")
            .expect("at")
            .allow(Allow::Resize)
            .allow(Allow::Pad),
    );
    assert_eq!(b.input("image").expect("image").status, Status::Override);
}

/// Inline policies for the refusals the fixtures do not reach.
fn inline(json: &str, robot_name: &str) -> Result<Binding, BindError> {
    bind(
        &Policy::parse(json).expect("policy"),
        &robot(robot_name),
        &RunOptions::named("inline"),
    )
}

fn refusal_text(json: &str, robot_name: &str) -> String {
    match inline(json, robot_name) {
        Ok(b) => panic!("bound:\n{b}"),
        Err(err) => err.to_string(),
    }
}

#[test]
fn two_actions_on_one_command_group_are_refused() {
    let text = refusal_text(
        r#"{"hz": 10, "actions": {
            "joints": {"kind": "joints", "at": "arm", "horizon": 1},
            "cartesian": {"kind": "pose", "at": "arm/tcp", "command": "delta", "horizon": 1}}}"#,
        "ur10e",
    );
    assert!(text.contains("[duplicate_target] actions \"joints\" and \"cartesian\" command arm and arm/tcp, members of one command group \"arm\"."), "{text}");
}

#[test]
fn conversions_need_allow_unless_lossless() {
    // Degrees convert automatically; millimetres need allow units.
    let ok = inline(
        r#"{"hz": 10, "inputs": {"q": {"kind": "joints", "at": "arm", "fields": ["pos"],
            "units": {"pos": "deg"}}}}"#,
        "ur10e",
    )
    .expect("angles are automatic");
    assert_eq!(plan(&ok, "q"), "fields pos → auto units(pos rad → deg)");
    let text = refusal_text(
        r#"{"hz": 10, "inputs": {"p": {"kind": "pose", "at": "arm/tcp", "fields": ["pos"],
            "units": {"pos": "mm"}}}}"#,
        "ur10e",
    );
    assert!(text.contains("[not_allowed] input \"p\": arm/tcp publishes pos in m, the model wants mm\n    needs: units"), "{text}");
    // Quaternion order is automatic; the depth camera's millimetres are not.
    let ok = inline(
        r#"{"hz": 10, "inputs": {"o": {"kind": "imu", "at": "pelvis/imu", "fields": ["orientation"],
            "rotation": "quat_xyzw"}}}"#,
        "g1",
    )
    .expect("quaternion order");
    assert_eq!(
        plan(&ok, "o"),
        "fields orientation → auto rotation(orientation quat_wxyz → quat_xyzw)"
    );
    let text = refusal_text(
        r#"{"hz": 10, "inputs": {"d": {"kind": "depth", "at": "head/depth"}}}"#,
        "g1",
    );
    assert!(text.contains("needs: units"), "{text}");
    let ok = inline(
        r#"{"hz": 10, "inputs": {"d": {"kind": "depth", "at": "head/depth", "size": [240, 320]}},
            "allow": ["units", "resize"]}"#,
        "g1",
    )
    .expect("allowed");
    assert_eq!(
        plan(&ok, "d"),
        "units(depth mm → m, 0 → NaN) → resize(nearest, 320x240)"
    );
    assert_eq!(
        ok.input("d").expect("d").delivers.to_string(),
        "f32 [240, 320]"
    );
    // A subset of joints needs select; enlarging needs upscale.
    let text = refusal_text(
        r#"{"hz": 10, "inputs": {"q": {"kind": "joints", "at": "arm", "fields": ["pos"],
            "names": ["elbow_joint", "shoulder_pan_joint"]}}}"#,
        "ur10e",
    );
    assert!(
        text.contains("arm has 6 joints, the model names 2 of them\n    needs: select"),
        "{text}"
    );
    let text = refusal_text(
        r#"{"hz": 10, "inputs": {"i": {"kind": "camera", "at": "wrist/cam", "size": [960, 1280]}},
            "allow": ["resize"]}"#,
        "ur10e",
    );
    assert!(
        text.contains("which enlarges it\n    needs: upscale"),
        "{text}"
    );
    // An inexact cast needs dtype; bool and float never convert.
    let text = refusal_text(
        r#"{"hz": 10, "inputs": {"f": {"kind": "unitree/foot_force", "at": "feet/force",
            "fields": {"force": ["int8", [4]]}}}}"#,
        "go2",
    );
    assert!(
        text.contains("publishes int16, the model wants int8 (not exact)\n    needs: dtype"),
        "{text}"
    );
    let ok = inline(
        r#"{"hz": 10, "inputs": {"f": {"kind": "unitree/foot_force", "at": "feet/force",
            "fields": {"force": ["float32", [4]]}}}}"#,
        "go2",
    )
    .expect("int16 to float32 is exact");
    assert_eq!(plan(&ok, "f"), "auto cast(i16 → f32)");
}

#[test]
fn mismatches_with_no_conversion_are_incompatible() {
    for (json, robot_name, needle) in [
        (
            r#"{"hz": 10, "actions": {"a": {"kind": "joints", "at": "arm", "command": "velocity"}}}"#,
            "ur10e",
            "arm accepts \"position\".\n    RLMesh does not convert between command modes.",
        ),
        (
            r#"{"hz": 10, "actions": {"t": {"kind": "pose", "at": "arm/tcp", "reference": "target"}}}"#,
            "ur10e",
            "relative to the target pose",
        ),
        (
            r#"{"hz": 1000, "actions": {"t": {"kind": "pose", "at": "arm/tcp", "hz": 1000}}}"#,
            "ur10e",
            "faster than arm/tcp's control rate of 500 Hz",
        ),
        (
            r#"{"hz": 50, "inputs": {"ft": {"kind": "wrench", "at": "wrist/ft", "history": 50,
                "history_hz": 1000}}}"#,
            "ur10e",
            "the window was trained at 1000 Hz; wrist/ft publishes at 500 Hz",
        ),
        (
            r#"{"hz": 20, "inputs": {"ft": {"kind": "wrench", "at": "wrist/ft",
                "max_age_ms": 5}}}"#,
            "ur10e",
            "max_age_ms 5 is below the 50 ms between samples of latest@20",
        ),
        (
            r#"{"hz": 20, "inputs": {"ft": {"kind": "wrench", "at": "wrist/ft", "history": 50,
                "max_age_ms": 1}}}"#,
            "ur10e",
            "max_age_ms 1 is below the 2 ms between samples of history@20",
        ),
        (
            r#"{"hz": 20, "inputs": {"c": {"kind": "camera", "at": "wrist/cam", "history": 4}}}"#,
            "ur10e",
            "a window of 4 needs every sample, but wrist/cam publishes latest(max_mbps=20)",
        ),
        (
            r#"{"hz": 20, "inputs": {"c": {"kind": "camera", "at": "wrist/cam", "format": "mono8"}}}"#,
            "ur10e",
            "Mono ↔ RGB conversion is Later",
        ),
        (
            r#"{"hz": 20, "inputs": {"t": {"kind": "tactile", "at": "left/hand/tactile", "shape": [108]}}}"#,
            "g1",
            "RLMesh does not reshape",
        ),
        (
            r#"{"hz": 20, "inputs": {"f": {"kind": "unitree/foot_force", "at": "feet/force",
                "fields": {"force": ["int16", [2]]}}}}"#,
            "go2",
            "do not match the declared schema",
        ),
        (
            r#"{"hz": 20, "actions": {"l": {"kind": "joints", "at": "legs", "names": ["FL_hip_joint"]}}}"#,
            "go2",
            "Commanding a subset of a part (scatter) is Later",
        ),
        (
            r#"{"hz": 20, "inputs": {"q": {"kind": "joints", "at": "arm", "names": "positional", "count": 7}}}"#,
            "ur10e",
            "the model has 7 joints; arm has 6.",
        ),
    ] {
        let text = refusal_text(json, robot_name);
        assert!(text.contains("[incompatible]"), "{json}:\n{text}");
        assert!(text.contains(needle), "{json}:\n{text}");
    }
}

#[test]
fn optional_entries_blank_or_drop_only_when_allowed() {
    let text = refusal_text(
        r#"{"hz": 10, "inputs": {"low": {"kind": "camera", "at": "scene/cam_low", "optional": true,
            "size": [224, 224]}}}"#,
        "ur10e",
    );
    assert!(text.contains("[not_allowed] input \"low\": ur10e has no camera part at \"scene/cam_low\"\n    needs: blank (it is optional)"), "{text}");
    let text = refusal_text(
        r#"{"hz": 10, "inputs": {"low": {"kind": "camera", "at": "scene/cam_low", "optional": true}},
            "allow": ["blank"]}"#,
        "ur10e",
    );
    assert!(text.contains("cannot blank it without a size"), "{text}");
    let b = inline(
        r#"{"hz": 10, "inputs": {"q": {"kind": "joints", "at": "arm", "fields": ["pos"]}},
            "actions": {"hand": {"kind": "hand", "at": "hand", "optional": true}},
            "allow": ["drop"]}"#,
        "ur10e",
    )
    .expect("dropped");
    assert_eq!(b.action("hand").expect("hand").status, Status::Dropped);
    assert!(b.channels().is_empty());
}

#[test]
fn captures_scope_and_qualified_selectors() {
    // An unfilled capture is inferred when one value binds every required
    // entry (only the right hand exists here).
    let mut g1 = robot("g1");
    g1.retain(|p| !p.path.as_str().starts_with("left/hand"));
    let policy = policy("dex_grasp");
    let b = bind(&policy, &g1, &RunOptions::named("dex_grasp")).expect("inferred");
    assert_eq!(b.captures, [("side".into(), "right".into())]);
    // A capture too broad to enumerate.
    let mut wide = String::from("{");
    for i in 0..70 {
        wide.push_str(&format!(r#""a{i}/b{i}": {{"kind": "text"}},"#));
    }
    wide.push_str(r#""z": {"kind": "text"}}"#);
    let parts = normalise(
        &ParticipantId::new("wide").expect("id"),
        SPEC_VERSION,
        &wide,
    )
    .expect("map");
    let err = bind(
        &Policy::parse(
            r#"{"hz": 1, "inputs": {"x": {"kind": "text", "at": "{p}/{q}"},
                                    "y": {"kind": "text", "at": "{r}/{s}"}}}"#,
        )
        .expect("policy"),
        &parts,
        &RunOptions::named("wide"),
    )
    .expect_err("too broad");
    assert!(err.to_string().contains("[pattern_too_broad]"), "{err}");
    // Two robots: one input qualified by robot, actions on the robot given.
    let mut session = robot("go2");
    session.extend(
        robot_as("go2", "supervisor")
            .into_iter()
            .filter(|p| p.path.as_str() == "goal/cmd_vel"),
    );
    let policy = Policy::parse(
        r#"{"hz": 50, "inputs": {"goal": {"kind": "velocity", "at": "supervisor:goal/cmd_vel"}},
            "actions": {"legs": {"kind": "joints", "at": "legs", "horizon": 1}}}"#,
    )
    .expect("policy");
    let err = bind(&policy, &session, &RunOptions::named("loco")).expect_err("which robot");
    assert!(err.to_string().contains("pass robot="), "{err}");
    let b = bind(
        &policy,
        &session,
        &RunOptions::named("loco").robot("go2").expect("robot"),
    )
    .expect("bound");
    assert_eq!(
        b.input("goal")
            .and_then(|r| r.part.as_ref())
            .map(|p| p.owner.as_str()),
        Some("supervisor")
    );
}

#[test]
fn fingerprints_follow_the_table() {
    let a = bound("pi0_single", "ur10e", RunOptions::named("pi0_single"));
    let b = bound("pi0_single", "ur10e", RunOptions::named("pi0_single"));
    assert_eq!(a.fingerprint, b.fingerprint, "deterministic");
    let renamed = bound("pi0_single", "ur10e", RunOptions::named("pi0_other"));
    assert_ne!(a.fingerprint, renamed.fingerprint);
    // The grids carry each action's dt for chunks that send dt = 0.
    let grids = a.grids();
    assert_eq!(grids.len(), 2);
    assert!(
        grids
            .iter()
            .all(|g| g.dt == Duration::from_nanos(66_666_667))
    );
    assert_eq!(
        rlmesh_session_core::spec::canonical::fingerprint(&a.table_json()),
        a.fingerprint
    );
}
