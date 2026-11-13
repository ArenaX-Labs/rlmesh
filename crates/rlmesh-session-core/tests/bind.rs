//! Conformance fixtures: every policy bound against
//! the reference robots, and every refusal, as golden binding
//! tables and refusal messages under `tests/golden/`.
//!
//! `RLMESH_BLESS=1 cargo test -p rlmesh-session-core --test bind` rewrites the
//! golden files; review the diff like code.

use std::path::PathBuf;
use std::time::Duration;

use rlmesh_session_core::spec::bind::{BindError, Binding, Code, Status};
use rlmesh_session_core::spec::policy::Allow;
use rlmesh_session_core::spec::{Policy, RunOptions, Target, bind, bind_in, normalise};
use rlmesh_session_core::wire::version::SPEC_VERSION;
use rlmesh_session_core::wire::{
    ChannelId, ContractMode, PartSpec, ParticipantId, ParticipantInfo, Place, Pose, PriorityClass,
    Role,
};

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
    assert_eq!((arm.chunk, arm.dt), (10, Duration::from_nanos(66_666_667)));
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
            // D = 50 is at or above the camera's S = 30: deliver every sample.
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
        (arm.dt, arm.chunk, arm.valid),
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

/// One fix with every option repeated: what the Python test of the hints
/// reads, so no keyword goes unchecked.
#[test]
fn a_fix_repeats_every_option_the_call_gave() {
    let options = RunOptions::named("every_option")
        .under("right")
        .expect("under")
        .at("wrist", "wrist/cam")
        .expect("at")
        .positional("state")
        .allow(Allow::Resize);
    let options = RunOptions {
        robot: Some(ParticipantId::new("aloha2").expect("id")),
        ..options
    };
    let policy = Policy::parse(
        r#"{"hz": 20,
            "inputs": {
                "image": {"kind": "camera", "at": "*/cam"},
                "wrist": {"kind": "camera"},
                "state": {"kind": "joints", "at": "arm", "fields": ["pos"]}},
            "actions": {}}"#,
    )
    .expect("policy");
    let parts = robot("aloha2");
    let Err(BindError::Refused(refusal)) = bind(&policy, &parts, &options) else {
        panic!("expected a refusal");
    };
    golden("every_option__aloha2__refused", &refusal.to_string());
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
fn a_custom_schema_binds_in_any_authored_field_order() {
    let robot = r#"{"pose": {"kind": "pose/coco17", "hz": 15,
        "fields": {"xy": ["float32", [17, 2]], "conf": ["float32", [17]]}}}"#;
    let mut parts =
        normalise(&ParticipantId::new("cam").expect("id"), SPEC_VERSION, robot).expect("robot");
    parts[0].channel = ChannelId(1);
    for fields in [
        r#"{"xy": ["float32", [17, 2]], "conf": ["float32", [17]]}"#,
        r#"{"conf": ["float32", [17]], "xy": ["float32", [17, 2]]}"#,
    ] {
        let policy = format!(
            r#"{{"hz": 15, "inputs": {{"pose": {{"kind": "pose/coco17", "at": "pose", "fields": {fields}}}}}}}"#
        );
        bind(
            &Policy::parse(&policy).expect("policy"),
            &parts,
            &RunOptions::named("inline"),
        )
        .unwrap_or_else(|err| panic!("{fields}: {err}"));
    }
}

#[test]
fn two_actions_on_one_command_group_are_refused() {
    let text = refusal_text(
        r#"{"hz": 10, "actions": {
            "joints": {"kind": "joints", "at": "arm", "chunk": 1},
            "cartesian": {"kind": "pose", "at": "arm/tcp", "command": "delta", "chunk": 1}}}"#,
        "ur10e",
    );
    assert!(text.contains("[duplicate_target] actions \"joints\" and \"cartesian\" command ur10e:arm and ur10e:arm/tcp, members of one command group \"arm\"."), "{text}");
    let Err(BindError::Refused(refusal)) = inline(
        r#"{"hz": 10, "actions": {
            "joints": {"kind": "joints", "at": "arm", "chunk": 1},
            "cartesian": {"kind": "pose", "at": "arm/tcp", "command": "delta", "chunk": 1}}}"#,
        "ur10e",
    ) else {
        panic!("bound");
    };
    assert_eq!(
        codes(&refusal),
        [(Code::DuplicateTarget, "cartesian".into())],
        "the later action carries it"
    );
}

#[test]
fn a_part_that_cannot_hold_its_target_needs_chunks_that_outlast_a_replan() {
    // A velocity command runs its safe behaviour whenever nothing is due:
    // one 20 Hz action per 20 Hz replan leaves it waiting between chunks.
    let text = refusal_text(
        r#"{"hz": 20, "actions": {"walk": {"kind": "velocity", "at": "base/twist",
            "hz": 20, "chunk": 1}}}"#,
        "go2",
    );
    assert!(
        text.contains("can't hold its target between chunks") && text.contains("\"chunk\": 3"),
        "{text}"
    );
    inline(
        r#"{"hz": 20, "actions": {"walk": {"kind": "velocity", "at": "base/twist",
            "hz": 20, "chunk": 3}}}"#,
        "go2",
    )
    .expect("three actions outlast a replan");
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
    assert!(text.contains("[not_allowed] input \"p\": ur10e:arm/tcp publishes pos in m, the model wants mm\n    needs: units"), "{text}");
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
        text.contains("ur10e:arm has 6 joints, the model names 2 of them\n    needs: select"),
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
            "ur10e:arm accepts \"position\".\n    RLMesh does not convert between command modes.",
        ),
        (
            r#"{"hz": 10, "actions": {"t": {"kind": "pose", "at": "arm/tcp", "reference": "target"}}}"#,
            "ur10e",
            "relative to the target pose",
        ),
        (
            r#"{"hz": 1000, "actions": {"t": {"kind": "pose", "at": "arm/tcp", "hz": 1000}}}"#,
            "ur10e",
            "faster than ur10e:arm/tcp's control rate of 500 Hz",
        ),
        (
            r#"{"hz": 50, "inputs": {"ft": {"kind": "wrench", "at": "wrist/ft", "history": 50,
                "history_hz": 1000}}}"#,
            "ur10e",
            "the window was trained at 1000 Hz; ur10e:wrist/ft publishes at 500 Hz.\n    RLMesh does not resample a window to another rate yet: resample in the model, or declare the part at 1000 Hz on the robot.",
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
            "a window of 4 needs every sample, but ur10e:wrist/cam publishes latest(max_mbps=20)",
        ),
        (
            r#"{"hz": 20, "inputs": {"c": {"kind": "camera", "at": "wrist/cam", "format": "mono8"}}}"#,
            "ur10e",
            "RLMesh does not convert between mono and RGB yet: convert in the model, or declare the other format on the robot.",
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
            "RLMesh does not command a subset of a part yet: command every joint from the model, or declare the subset as its own part on the robot.",
        ),
        (
            r#"{"hz": 20, "inputs": {"q": {"kind": "joints", "at": "arm", "names": "positional", "count": 7}}}"#,
            "ur10e",
            "the model has 7 joints; ur10e:arm has 6.",
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
    assert!(text.contains("[not_allowed] input \"low\" (kind camera, at \"scene/cam_low\") is optional, but nothing allows it to read blank when its part is absent.\n    needs: blank (it is optional)\n    fix: allow=[\"blank\"] on the entry, the policy or the bind call (or drop optional)"), "{text}");
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
    assert!(b.table().contains("drop_action"), "{}", b.table());
    assert!(b.channels().is_empty());
}

/// `optional` without `blank` (or `drop`) allowed is refused at once, even
/// when the robot it names has not appeared: no robot could make it stand.
#[test]
fn an_optional_entry_that_may_not_stay_unbound_is_refused_at_once() {
    let parts = robot("ur10e");
    let options = RunOptions::named("reader").robot("ur10e").expect("robot");
    for (json, allow) in [
        (
            r#"{"hz": 20, "inputs": {"left": {"kind": "joints", "at": "gimbal-left:servos",
                "fields": ["pos"], "optional": true, "allow": ["hold"]}}}"#,
            "blank",
        ),
        (
            r#"{"hz": 20, "actions": {"eye": {"kind": "joints", "at": "gimbal-left:servos",
                "command": "position", "optional": true}}}"#,
            "drop",
        ),
        (
            r#"{"hz": 20, "inputs": {"q": {"kind": "joints", "at": "arm", "fields": ["pos"],
                "optional": true}}}"#,
            "blank",
        ),
    ] {
        let policy = Policy::parse(json).expect("policy");
        let err = bind(&policy, &parts, &options).expect_err("refused");
        assert_eq!(err.waiting_for(), None, "{err}");
        let text = err.to_string();
        assert!(
            text.contains("[not_allowed]") && text.contains("is optional"),
            "{text}"
        );
        assert!(text.contains(&format!("fix: allow=[{allow:?}]")), "{text}");
    }
    // The bind call's allowance counts.
    let policy = Policy::parse(
        r#"{"hz": 20, "inputs": {"q": {"kind": "joints", "at": "arm", "fields": ["pos"],
            "optional": true}}}"#,
    )
    .expect("policy");
    let allowed = options.clone().allow(Allow::Blank);
    assert!(bind(&policy, &parts, &allowed).is_ok());
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
            "actions": {"legs": {"kind": "joints", "at": "legs", "chunk": 1}}}"#,
    )
    .expect("policy");
    // No robot named: the session's view, every robot under its name, so
    // an unqualified path names no part and the refusal says which.
    let err = bind(&policy, &session, &RunOptions::named("loco")).expect_err("which robot");
    assert!(
        err.to_string().contains(r#"at={"legs": "go2/legs"}"#),
        "{err}"
    );
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

/// The eyeball rig: two eyes, a skull and a fixed camera,
/// each its own robot with channel ids unique across the session, placed
/// in one view; `extra` robots join unplaced.
fn eyeball_rig(extra: &[(&str, &str)]) -> (Vec<PartSpec>, Vec<Place>) {
    let robots = [
        ("eye", "eye-l"),
        ("eye", "eye-r"),
        ("skull", "skull"),
        ("fixed_cam", "cam-1"),
    ];
    let mut parts: Vec<PartSpec> = Vec::new();
    for (file, id) in robots.iter().chain(extra) {
        for mut part in robot_as(file, id) {
            let next = parts
                .iter()
                .map(|p| p.applied_channel.unwrap_or(p.channel).0)
                .max();
            part.channel = ChannelId(next.unwrap_or(0) + 1);
            part.applied_channel = part.command.as_ref().map(|_| ChannelId(part.channel.0 + 1));
            parts.push(part);
        }
    }
    let place = |robot: &str, at: &str, frame: &str, xyz: [f64; 3], quat_xyzw: [f64; 4]| Place {
        robot: ParticipantId::new(robot).expect("id"),
        at: at.into(),
        from: String::new(),
        frame: frame.into(),
        pose: Pose { xyz, quat_xyzw },
    };
    let identity = [0.0, 0.0, 0.0, 1.0];
    let places = vec![
        place("eye-l", "eyes/left", "", [0.035, 0.0, 0.30], identity),
        place("eye-r", "eyes/right", "", [-0.035, 0.0, 0.30], identity),
        place("skull", "skull", "", [0.0, 0.0, 0.22], identity),
        place(
            "cam-1",
            "scene",
            "cam_optical",
            [0.0, -0.6, 0.5],
            [-0.612_372_4, 0.0, 0.0, 0.790_569_4],
        ),
    ];
    (parts, places)
}

#[test]
fn one_policy_binds_across_placed_robots() {
    let (parts, places) = eyeball_rig(&[]);
    let target = Target {
        parts: &parts,
        places: &places,
        ..Target::default()
    };
    let b = bind_in(
        &policy("eyeball_tracker"),
        &target,
        &RunOptions::named("tracker"),
    )
    .unwrap_or_else(|err| panic!("{err}"));
    assert_eq!(b.robot.as_str(), "scene");
    // Rows keep each part as declared, under its own robot and path: what
    // its stream keys and leases are made of.
    let owners = |rows: Vec<Option<&PartSpec>>| -> Vec<String> {
        rows.into_iter()
            .map(|p| p.map_or("-".into(), |p| format!("{}:{}", p.owner, p.path)))
            .collect()
    };
    assert_eq!(
        owners(b.inputs.iter().map(|r| r.part.as_ref()).collect()),
        ["cam-1:cam", "eye-l:gaze"]
    );
    assert_eq!(
        owners(b.actions.iter().map(|r| r.part.as_ref()).collect()),
        ["eye-l:gaze", "eye-r:gaze", "skull:jaw"]
    );
    assert_eq!(b.table_json()["actions"][1]["robot"], "eye-r");
    golden("eyeball_tracker__eyeball_rig", &b.table());

    // robot= binds that one robot as today.
    let jaw = bind_in(
        &Policy::parse(r#"{"hz": 30, "actions": {"jaw": {"kind": "joints", "at": "jaw"}}}"#)
            .expect("policy"),
        &target,
        &RunOptions::named("jaw").robot("skull").expect("robot"),
    )
    .unwrap_or_else(|err| panic!("{err}"));
    assert!(jaw.places.is_empty() && jaw.table_json()["actions"][0].get("robot").is_none());
}

#[test]
fn a_view_refuses_ambiguous_unplaced_and_lockstep_spans() {
    let (parts, places) = eyeball_rig(&[("ur10e", "ur10e")]);
    let target = Target {
        parts: &parts,
        places: &places,
        ..Target::default()
    };
    let spans = Policy::parse(
        r#"{"hz": 30, "actions": {
            "gaze": {"kind": "joints", "at": "eyes/*/gaze"},
            "arm": {"kind": "joints", "at": "ur10e:arm"}}}"#,
    )
    .expect("policy");
    let refusal = match bind_in(&spans, &target, &RunOptions::named("spans")) {
        Err(BindError::Refused(refusal)) => *refusal,
        other => panic!("{other:?}"),
    };
    assert_eq!(
        codes(&refusal),
        [
            (Code::Ambiguous, "gaze".into()),
            (Code::Incompatible, "arm".into())
        ]
    );
    golden("spans__eyeball_rig__refused", &refusal.to_string());
}

#[test]
fn a_views_refusal_names_a_part_as_its_robot_declared_it() {
    let (parts, places) = eyeball_rig(&[]);
    let target = Target {
        parts: &parts,
        places: &places,
        ..Target::default()
    };
    let policy = Policy::parse(
        r#"{"hz": 30, "actions": {"gaze": {"kind": "joints", "at": "eyes/left/gaze",
            "names": "positional", "count": 7}}}"#,
    )
    .expect("policy");
    let refusal = match bind_in(&policy, &target, &RunOptions::named("seven")) {
        Err(BindError::Refused(refusal)) => *refusal,
        other => panic!("{other:?}"),
    };
    let text = refusal.to_string();
    assert!(
        text.contains("the model has 7 joints; eye-l:gaze has 2."),
        "{text}"
    );
}

#[test]
fn only_a_robot_that_has_not_appeared_is_waited_for() {
    let parts = robot("ur10e");
    let options = RunOptions::named("reader").robot("ur10e").expect("robot");
    let refusal = |json: &str| {
        let policy = Policy::parse(json).expect("policy");
        bind(&policy, &parts, &options).expect_err("refused")
    };
    let absent = refusal(
        r#"{"hz": 20, "inputs": {
            "q": {"kind": "joints", "at": "arm", "fields": ["pos"]},
            "target": {"kind": "joints", "at": "tracker:target", "fields": ["pos"]}}}"#,
    );
    assert_eq!(
        absent.waiting_for().as_deref(),
        Some(
            "robot tracker to declare its parts; input \"target\" (kind joints, at \
             \"tracker:target\") needs it"
        )
    );
    assert!(
        absent
            .to_string()
            .contains("tracker has not declared any parts yet.\n    fix: start tracker"),
        "{absent}"
    );
    // A present robot's missing part stands, alone or beside an absent robot.
    let missing = refusal(r#"{"hz": 20, "inputs": {"c": {"kind": "gripper", "at": "nope"}}}"#);
    assert_eq!(missing.waiting_for(), None, "{missing}");
    let both = refusal(
        r#"{"hz": 20, "inputs": {
            "c": {"kind": "gripper", "at": "nope"},
            "target": {"kind": "joints", "at": "tracker:target", "fields": ["pos"]}}}"#,
    );
    assert_eq!(both.waiting_for(), None, "{both}");
    // An optional entry's robot is not waited for: absent binds blank.
    let policy = Policy::parse(
        r#"{"hz": 20, "inputs": {"target": {"kind": "joints", "at": "tracker:target",
            "fields": ["pos"], "count": 2, "optional": true}},
            "allow": ["blank"]}"#,
    )
    .expect("policy");
    let bound = bind(&policy, &parts, &options).unwrap_or_else(|err| panic!("{err}"));
    assert_eq!(bound.input("target").expect("row").status, Status::Blank);
}

#[test]
fn switches_and_values_bind_by_kind() {
    let binding = bound("lamp_keeper", "lamp", RunOptions::named("lamp_keeper"));
    golden("lamp_keeper__lamp", &binding.table());
    let relay = binding.action("relay").expect("relay");
    assert_eq!(relay.delivers.to_string(), "bool [1]");
    assert_eq!(relay.chunk, 1);
}

#[test]
fn a_custom_kind_blanks_from_its_fields_schema() {
    let policy = Policy::parse(
        r#"{"hz": 15, "inputs": {"pose": {"kind": "pose/coco17", "at": "tracker:aim",
            "optional": true, "allow": ["blank"],
            "fields": {"xy": ["float32", [17, 2]], "conf": ["float32", [17]]}}}}"#,
    )
    .expect("policy");
    let bound = bind(&policy, &robot("ur10e"), &RunOptions::named("inline"))
        .unwrap_or_else(|err| panic!("{err}"));
    let row = bound.input("pose").expect("pose");
    assert_eq!(row.status, Status::Blank);
    assert_eq!(
        row.delivers.to_string(),
        "{conf: f32 [17], xy: f32 [17, 2]}",
        "{}",
        bound.table()
    );
    let rlmesh_session_core::spec::apply::Value::Dict(fields) = row.apply(&[]).expect("blank")
    else {
        panic!("not a dict");
    };
    let shapes: Vec<(String, Vec<u32>)> = fields
        .iter()
        .map(|(name, t)| (name.clone(), t.shape.clone()))
        .collect();
    assert_eq!(
        shapes,
        [
            ("conf".to_string(), vec![17]),
            ("xy".to_string(), vec![17, 2])
        ]
    );
}

/// An inline robot's parts, channels assigned as [`robot`] does.
fn inline_robot(json: &str) -> Vec<PartSpec> {
    let mut parts =
        normalise(&ParticipantId::new("bot").expect("id"), SPEC_VERSION, json).expect("robot");
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

fn on_inline(robot: &str, policy: &str) -> Result<Binding, BindError> {
    bind(
        &Policy::parse(policy).expect("policy"),
        &inline_robot(robot),
        &RunOptions::named("inline"),
    )
}

fn delivers(binding: &Binding, entry: &str) -> String {
    match (binding.input(entry), binding.action(entry)) {
        (Some(row), _) => row.delivers.to_string(),
        (_, Some(row)) => row.delivers.to_string(),
        _ => panic!("no entry {entry}"),
    }
}

#[test]
fn a_range_rescales_only_when_it_differs_and_only_when_allowed() {
    let gripper = |dir: &str, range: &str, allow: &str| {
        format!(
            r#"{{"hz": 10, "{dir}": {{"g": {{"kind": "gripper", "at": "gripper", "range": {range}}}}},
                "allow": [{allow}]}}"#
        )
    };
    let same = inline(&gripper("actions", "[0, 1]", ""), "ur10e").expect("same range");
    assert_eq!(same.action("g").expect("g").plan.to_string(), "none");
    let text = refusal_text(&gripper("actions", "[-1, 1]", ""), "ur10e");
    assert!(
        text.contains("the model emits -1…1, ur10e:gripper takes 0…1\n    needs: rescale"),
        "{text}"
    );
    let same = inline(&gripper("inputs", "[0, 1]", ""), "ur10e").expect("same range");
    assert_eq!(plan(&same, "g"), "none");
    let text = refusal_text(&gripper("inputs", "[-1, 1]", ""), "ur10e");
    assert!(
        text.contains("ur10e:gripper ranges 0…1, the model wants -1…1\n    needs: rescale"),
        "{text}"
    );
    let rescaled = inline(&gripper("inputs", "[-1, 1]", r#""rescale""#), "ur10e").expect("allowed");
    assert_eq!(plan(&rescaled, "g"), "rescale(0…1 → -1…1)");
    // An integer field is rescaled into float32.
    let knob = r#"{"knob": {"kind": "acme/knob", "hz": 10, "fields": {"v": ["int16", [1]]},
        "range": {"v": [0, 100]}}}"#;
    let b = on_inline(
        knob,
        r#"{"hz": 10, "inputs": {"k": {"kind": "acme/knob", "at": "knob",
            "fields": {"v": ["int16", [1]]}, "range": [0, 1]}}, "allow": ["rescale"]}"#,
    )
    .expect("allowed");
    assert_eq!(delivers(&b, "k"), "f32 [1]");
}

#[test]
fn an_action_converts_units_and_rotation_into_the_parts_and_may_run_at_its_control_rate() {
    let b = inline(
        r#"{"hz": 10, "actions": {"q": {"kind": "joints", "at": "arm", "units": {"pos": "deg"},
            "chunk": 10}}}"#,
        "ur10e",
    )
    .expect("units");
    assert_eq!(
        b.action("q").expect("q").plan.to_string(),
        "auto units(pos deg → rad)"
    );
    let b = inline(
        r#"{"hz": 10, "actions": {"t": {"kind": "pose", "at": "arm/tcp", "command": "delta",
            "rotation": "quat_xyzw", "chunk": 10}}, "allow": ["rotation"]}"#,
        "ur10e",
    )
    .expect("rotation");
    let t = b.action("t").expect("t");
    assert_eq!(t.plan.to_string(), "rotation(rot quat_xyzw → axis_angle)");
    assert_eq!(t.delivers.to_string(), "f32 [10, 7]");
    // One delta per control tick at most: exactly the control rate binds.
    inline(
        r#"{"hz": 500, "actions": {"t": {"kind": "pose", "at": "arm/tcp", "command": "delta",
            "hz": 500, "chunk": 10}}}"#,
        "ur10e",
    )
    .expect("at the control rate");
}

#[test]
fn a_custom_action_binds_by_its_fields_schema_up_to_the_chunk_cap() {
    let bot = |fields: &str, units: &str| {
        format!(
            r#"{{"x": {{"kind": "acme/arm", "hz": 100, "fields": {{"q": ["float32", [1]]}},
                "command": "go", "safe": "hold", "command_fields": {fields}{units}}}}}"#
        )
    };
    let action = |fields: &str, chunk: u32| {
        format!(
            r#"{{"hz": 1, "actions": {{"x": {{"kind": "acme/arm", "at": "x", "fields": {fields},
                "chunk": {chunk}}}}}}}"#
        )
    };
    let two = r#"{"q": ["float32", [4]], "v": ["float32", [2]]}"#;
    for fields in [two, r#"{"v": ["float32", [2]], "q": ["float32", [4]]}"#] {
        on_inline(&bot(two, ""), &action(fields, 10)).unwrap_or_else(|e| panic!("{fields}: {e}"));
    }
    for fields in [
        r#"{"q": ["float32", [4]], "v": ["float32", [3]]}"#,
        r#"{"q": ["float32", [4]]}"#,
    ] {
        let err = on_inline(&bot(two, ""), &action(fields, 10)).expect_err(fields);
        assert!(
            err.to_string().contains("do not match the declared schema"),
            "{err}"
        );
    }
    // A command field keeps its own unit: no conversion to declare.
    let b = on_inline(
        &bot(r#"{"q": ["float32", [2]]}"#, r#", "units": {"q": "tick"}"#),
        &action(r#"{"q": ["float32", [2]]}"#, 10),
    )
    .expect("own unit");
    assert_eq!(b.action("x").expect("x").plan.to_string(), "none");
    // A chunk of exactly 4 MiB binds; one more element per action does not.
    let field = |n: u32| format!(r#"{{"q": ["float32", [{n}]]}}"#);
    on_inline(&bot(&field(1024), ""), &action(&field(1024), 1024)).expect("4 MiB");
    let err = on_inline(&bot(&field(1026), ""), &action(&field(1026), 1024)).expect_err("over");
    assert!(
        err.to_string().contains("over the 4 MiB chunk cap"),
        "{err}"
    );
}

#[test]
fn image_fits_and_conversions_need_their_allows() {
    let camera = |extra: &str, allow: &str| {
        format!(
            r#"{{"hz": 10, "inputs": {{"c": {{"kind": "camera", "at": "wrist/cam"{extra}}}}},
                "allow": [{allow}]}}"#
        )
    };
    for (extra, allow, want) in [
        (r#", "fields": ["image"]"#, "", "decode(jpeg)"),
        (
            r#", "crop": 0.5"#,
            "",
            "decode(jpeg) → crop(0.5 of 640x480 → 320x240)",
        ),
        // Scaled by 1 to fit, then padded: nothing is enlarged.
        (
            r#", "size": [640, 640]"#,
            r#""resize", "pad""#,
            "decode(jpeg) → pad(640x480 → 640x640, bilinear)",
        ),
        (
            r#", "size": [480, 480]"#,
            r#""resize", "crop""#,
            "decode(jpeg) → crop(640x480 → 480x480, bilinear)",
        ),
        (
            r#", "size": [240, 640]"#,
            r#""resize", "stretch""#,
            "decode(jpeg) → stretch(640x480 → 640x240, bilinear)",
        ),
        (
            r#", "size": [480, 320]"#,
            r#""resize", "stretch""#,
            "decode(jpeg) → stretch(640x480 → 320x480, bilinear)",
        ),
        (r#", "format": "bgr8""#, r#""bgr""#, "decode(jpeg) → bgr"),
    ] {
        let b = inline(&camera(extra, allow), "ur10e").unwrap_or_else(|e| panic!("{extra}: {e}"));
        assert_eq!(plan(&b, "c"), want, "{extra}");
    }
    for (extra, allow, needle) in [
        (
            r#", "fields": ["depth"]"#,
            "",
            "fields are not selectable on an image",
        ),
        (
            r#", "size": [800, 400]"#,
            r#""resize", "stretch""#,
            "(1:2), which enlarges it\n    needs: upscale",
        ),
        (
            r#", "size": [240, 320]"#,
            "",
            "the model wants 320x240 (4:3)\n    needs: resize\n",
        ),
        (
            r#", "size": [224, 224]"#,
            r#""resize""#,
            "needs: one of crop | pad | stretch to change the aspect ratio",
        ),
        (
            r#", "format": "bgr8""#,
            "",
            "is rgb8, the model wants bgr8\n    needs: bgr",
        ),
    ] {
        let text = refusal_text(&camera(extra, allow), "ur10e");
        assert!(text.contains(needle), "{extra}: {text}");
    }
    // A depth image already in the model's unit needs no conversion.
    let b = inline(
        r#"{"hz": 10, "inputs": {"d": {"kind": "depth", "at": "head/depth", "units": {"depth": "mm"}}}}"#,
        "g1",
    )
    .expect("mm");
    assert_eq!(
        (plan(&b, "d"), delivers(&b, "d")),
        ("none".into(), "u16 [480, 640]".into())
    );
}

#[test]
fn input_fields_keep_their_dtype_and_affines_count_elements() {
    let b = on_inline(
        r#"{"arm": {"kind": "joints", "count": 2, "hz": 10, "dtype": "float64"}}"#,
        r#"{"hz": 10, "inputs": {"q": {"kind": "joints", "at": "arm", "fields": ["pos"],
            "units": {"pos": "deg"}}}}"#,
    )
    .expect("units");
    assert_eq!(delivers(&b, "q"), "f64 [2]");
    let b = inline(
        r#"{"hz": 10, "inputs": {"q": {"kind": "joints", "at": "arm", "fields": ["pos", "vel"],
            "scale": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}}}"#,
        "ur10e",
    )
    .expect("one scale per element of the concatenation");
    assert_eq!(plan(&b, "q"), "fields pos, vel → scale(12 values)");
    let text = refusal_text(
        r#"{"hz": 10, "inputs": {"q": {"kind": "joints", "at": "arm", "fields": ["pos", "vel"],
            "scale": [1, 2, 3, 4, 5, 6]}}}"#,
        "ur10e",
    );
    assert!(
        text.contains("scale has 6 values for 12 elements"),
        "{text}"
    );
    // One field counts all its elements; mixed dtypes take one value each.
    let pose = |fields: &str, scale: usize| {
        let scale = vec!["1"; scale].join(", ");
        on_inline(
            &format!(r#"{{"pose": {{"kind": "pose/coco17", "hz": 15, "fields": {fields}}}}}"#),
            &format!(
                r#"{{"hz": 15, "inputs": {{"p": {{"kind": "pose/coco17", "at": "pose",
                    "fields": {fields}, "scale": [{scale}]}}}}}}"#
            ),
        )
    };
    pose(r#"{"a": ["float32", [17, 2]]}"#, 34).expect("34 elements");
    let err =
        pose(r#"{"a": ["float32", [17, 2]], "b": ["int16", [17]]}"#, 51).expect_err("per field");
    assert!(
        err.to_string().contains("51 values for per-field elements"),
        "{err}"
    );
    // An integer field under an affine is delivered as float32.
    let b = inline(
        r#"{"hz": 10, "inputs": {"f": {"kind": "unitree/foot_force", "at": "feet/force",
            "fields": {"force": ["int16", [4]]}, "scale": [0.5]}}}"#,
        "go2",
    )
    .expect("scaled");
    assert_eq!(delivers(&b, "f"), "f32 [4]");
    let b = inline(
        r#"{"hz": 10, "inputs": {"q": {"kind": "joints", "at": "arm", "fields": ["pos"],
            "names": ["elbow_joint", "shoulder_pan_joint"]}}, "allow": ["select"]}"#,
        "ur10e",
    )
    .expect("select");
    assert_eq!(delivers(&b, "q"), "f32 [2]");
    // A window trained within 1% of the source rate binds.
    inline(
        r#"{"hz": 50, "inputs": {"ft": {"kind": "wrench", "at": "wrist/ft", "history": 50,
            "history_hz": 505}}}"#,
        "ur10e",
    )
    .expect("within 1%");
}

#[test]
fn an_absent_optional_input_blanks_to_its_declared_shape_and_preprocessing() {
    use rlmesh_session_core::spec::plan::Step;
    let blank = |entry: &str| {
        let policy = format!(r#"{{"hz": 10, "inputs": {{"x": {entry}}}, "allow": ["blank"]}}"#);
        let b = inline(&policy, "ur10e").unwrap_or_else(|e| panic!("{entry}: {e}"));
        let fill = match b.input("x").expect("x").plan.steps.as_slice() {
            [Step::Blank { fill, .. }] => fill.clone(),
            other => panic!("{other:?}"),
        };
        (delivers(&b, "x"), fill)
    };
    let absent = r#""at": "nothing", "optional": true"#;
    for (entry, want, fill) in [
        (format!(r#"{{"kind": "text", {absent}}}"#), "str", vec![0.0]),
        (
            format!(
                r#"{{"kind": "joints", {absent}, "names": ["a", "b"], "fields": ["pos"],
                "history": 4, "offset": [1]}}"#
            ),
            "f32 [4, 2]",
            vec![0.0],
        ),
        // Black (the range's low end) through the per-channel scale.
        (
            format!(
                r#"{{"kind": "camera", {absent}, "size": [2, 2], "range": [-1, 1],
                "scale": [2, 3, 4]}}"#
            ),
            "f32 [2, 2, 3]",
            vec![-2.0, -3.0, -4.0],
        ),
        (
            format!(
                r#"{{"kind": "acme/thing", {absent},
                "fields": {{"a": ["int16", [3]], "b": ["float32", []]}}}}"#
            ),
            "{a: i16 [3], b: f32 []}",
            vec![0.0],
        ),
        (
            format!(
                r#"{{"kind": "acme/thing", {absent}, "fields": {{"a": ["int16", [3]]}},
                "shape": [5], "dtype": "float32"}}"#
            ),
            "f32 [5]",
            vec![0.0],
        ),
    ] {
        assert_eq!(blank(&entry), (want.to_string(), fill), "{entry}");
    }
    let text = refusal_text(
        &format!(
            r#"{{"hz": 10, "inputs": {{"x": {{"kind": "acme/thing", {absent},
                "fields": {{"a": ["int16", [3]]}}, "shape": [5]}}}}, "allow": ["blank"]}}"#
        ),
        "ur10e",
    );
    assert!(
        text.contains("cannot blank it without a shape and a dtype"),
        "{text}"
    );
}

#[test]
fn a_bindings_leased_channels_and_wire_table_follow_its_actions() {
    let b = bound("pi0_single", "ur10e", RunOptions::named("pi0_single"));
    let parts = robot("ur10e");
    let channel = |path: &str| {
        parts
            .iter()
            .find(|p| p.path.as_str() == path)
            .map(|p| p.channel)
            .expect(path)
    };
    assert_eq!(b.channels(), [channel("arm"), channel("gripper")]);
    let text: serde_json::Value = serde_json::from_str(&b.table_text()).expect("json");
    assert_eq!(
        rlmesh_session_core::spec::canonical::fingerprint(&text),
        b.fingerprint,
        "the table text sent on the wire is what the fingerprint covers"
    );
}

#[test]
fn units_name_their_conversion_and_gripper_metres_need_a_width() {
    let b = inline(
        r#"{"hz": 10, "inputs": {"p": {"kind": "pose", "at": "arm/tcp", "fields": ["pos"],
            "units": {"pos": "mm"}}}, "allow": ["units"]}"#,
        "ur10e",
    )
    .expect("allowed");
    assert_eq!(plan(&b, "p"), "fields pos → units(pos m → mm)");
    // A robot publishing millimetres on a pose converts as plain units.
    let b = on_inline(
        r#"{"tcp": {"kind": "pose", "frame": "b", "child": "t", "hz": 10, "units": {"pos": "mm"}}}"#,
        r#"{"hz": 10, "inputs": {"p": {"kind": "pose", "at": "tcp", "fields": ["pos"]}},
            "allow": ["units"]}"#,
    )
    .expect("allowed");
    assert_eq!(plan(&b, "p"), "fields pos → units(pos mm → m)");
    let gripper = |allow: &str| {
        inline(
            &format!(
                r#"{{"hz": 10, "inputs": {{"g": {{"kind": "gripper", "at": "gripper",
                    "units": {{"pos": "m"}}}}}}, "allow": [{allow}]}}"#
            ),
            "ur10e",
        )
    };
    let err = gripper("").expect_err("needs units");
    assert!(err.to_string().contains("needs: units"), "{err}");
    let b = gripper(r#""units""#).expect("allowed");
    assert_eq!(plan(&b, "g"), "units(pos 0…1 → 0…0.085 m)");
}

#[test]
fn a_joint_count_must_match_the_robots() {
    let text = refusal_text(
        r#"{"hz": 20, "inputs": {"q": {"kind": "joints", "at": "arm", "count": 7}}}"#,
        "ur10e",
    );
    assert!(
        text.contains("the model has 7 joints; ur10e:arm has 6."),
        "{text}"
    );
}

#[test]
fn a_given_capture_narrows_the_others_candidates() {
    let parts = inline_robot(r#"{"l/x": {"kind": "text"}, "r/y": {"kind": "text"}}"#);
    let policy = Policy::parse(r#"{"hz": 10, "inputs": {"t": {"kind": "text", "at": "{s}/{t}"}}}"#)
        .expect("policy");
    for (name, value) in [("s", "l"), ("t", "x")] {
        let options = RunOptions::named("inline")
            .capture(name, value)
            .expect("capture");
        let b = bind(&policy, &parts, &options).unwrap_or_else(|e| panic!("{name}={value}: {e}"));
        assert_eq!(
            b.input("t")
                .and_then(|r| r.part.as_ref())
                .map(|p| p.path.as_str()),
            Some("l/x"),
            "{name}={value}"
        );
    }
}

#[test]
fn a_capture_only_optional_entries_use_is_not_inferred() {
    let text = refusal_text(
        r#"{"hz": 10, "inputs": {"q": {"kind": "joints", "at": "{a}", "fields": ["pos"]},
            "c": {"kind": "camera", "at": "{side}/cam", "optional": true, "size": [480, 640]}},
            "allow": ["blank"]}"#,
        "ur10e",
    );
    assert!(
        text.contains("{side} is used only by optional entries, so it is not inferred."),
        "{text}"
    );
}

fn member(id: &str, present: bool) -> ParticipantInfo {
    ParticipantInfo {
        id: ParticipantId::new(id).expect("id"),
        name: id.into(),
        roles: vec![Role::Embodiment],
        priority: PriorityClass::Scripted,
        attached_ns: 0,
        protocol_minor: 0,
        supports: Vec::new(),
        requires: Vec::new(),
        workflow_edition: String::new(),
        workflow_editions: Vec::new(),
        present,
        left: None,
        device: "local".into(),
        generation: 1,
    }
}

#[test]
fn an_optional_entry_blanks_instead_of_binding_a_departed_robots_part() {
    let parts = robot("ur10e");
    let policy = Policy::parse(
        r#"{"hz": 10, "inputs": {"c": {"kind": "camera", "at": "wrist/cam", "optional": true,
            "size": [480, 640]}}, "allow": ["blank"]}"#,
    )
    .expect("policy");
    let options = RunOptions::named("inline").robot("ur10e").expect("robot");
    for (present, status) in [(true, Status::Bound), (false, Status::Blank)] {
        let roster = [member("ur10e", present)];
        let target = Target {
            parts: &parts,
            roster: &roster,
            ..Target::default()
        };
        let b = bind_in(&policy, &target, &options).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(b.input("c").expect("c").status, status, "present {present}");
    }
    // A required entry keeps its part through the robot's absence, even
    // where blank is allowed.
    let required = Policy::parse(
        r#"{"hz": 10, "inputs": {"c": {"kind": "camera", "at": "wrist/cam"}},
            "allow": ["blank"]}"#,
    )
    .expect("policy");
    let roster = [member("ur10e", false)];
    let target = Target {
        parts: &parts,
        roster: &roster,
        ..Target::default()
    };
    let b = bind_in(&required, &target, &options).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(b.input("c").expect("c").status, Status::Bound);
}

#[test]
fn a_view_waits_for_a_placed_robot_that_has_declared_nothing() {
    let (parts, mut places) = eyeball_rig(&[]);
    places.push(Place {
        robot: ParticipantId::new("eye-x").expect("id"),
        at: "eyes/extra".into(),
        ..places[0].clone()
    });
    let target = Target {
        parts: &parts,
        places: &places,
        ..Target::default()
    };
    let policy = Policy::parse(
        r#"{"hz": 30, "actions": {"gaze": {"kind": "joints", "at": "eyes/extra/gaze"}}}"#,
    )
    .expect("policy");
    let err = bind_in(&policy, &target, &RunOptions::named("extra")).expect_err("waits");
    assert_eq!(
        err.waiting_for().as_deref(),
        Some(
            "robot eye-x to declare its parts; action \"gaze\" (kind joints, at \"eyes/extra/gaze\") needs it"
        )
    );
}

#[test]
fn an_optional_entry_on_an_absent_robot_without_blank_is_refused_not_waited_for() {
    let policy = Policy::parse(
        r#"{"hz": 20, "inputs": {"target": {"kind": "joints", "at": "tracker:target",
            "fields": ["pos"], "count": 2, "optional": true}}}"#,
    )
    .expect("policy");
    let options = RunOptions::named("reader").robot("ur10e").expect("robot");
    let err = bind(&policy, &robot("ur10e"), &options).expect_err("refused");
    assert_eq!(err.waiting_for(), None);
    let text = err.to_string();
    assert!(
        text.contains("is optional, but nothing allows it to read blank"),
        "{text}"
    );
}

#[test]
fn windows_over_the_ring_budget_are_refused() {
    let text = match on_inline(
        r#"{"cam": {"kind": "camera", "width": 1920, "height": 1080, "hz": 30, "frame": "f",
            "send": "every"}}"#,
        r#"{"hz": 10, "inputs": {"c": {"kind": "camera", "at": "cam", "history": 300}}}"#,
    ) {
        Ok(b) => panic!("bound:\n{b}"),
        Err(err) => err.to_string(),
    };
    assert!(text.contains("at most 512 MiB per controller"), "{text}");
}
