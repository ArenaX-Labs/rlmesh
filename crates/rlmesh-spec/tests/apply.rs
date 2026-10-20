//! Bound plans applied to samples and chunks (§7.4), on the conformance
//! robots and policies.

use std::path::PathBuf;

use rlmesh_spec::apply::{Tensor, Value};
use rlmesh_spec::{Binding, Policy, RunOptions, bind, normalise};
use rlmesh_wire::part::{DType, Layout, PartSpec};
use rlmesh_wire::version::SPEC_VERSION;
use rlmesh_wire::{ChannelId, ParticipantId};

fn fixture(kind: &str, name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(kind)
        .join(format!("{name}.json"));
    std::fs::read_to_string(path).expect("fixture")
}

fn binding(policy: &str, robot: &str) -> Binding {
    let mut parts = normalise(
        &ParticipantId::new(robot).expect("id"),
        SPEC_VERSION,
        &fixture("robots", robot),
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
    let policy = Policy::parse(&fixture("policies", policy)).expect("policy");
    bind(&policy, &parts, &RunOptions::named("test")).expect("bound")
}

/// A packed sample of `part` with each named field's f32 values.
fn sample(part: &PartSpec, values: &[(&str, &[f32])]) -> Vec<u8> {
    let layout = Layout::of(&part.fields).expect("layout");
    let mut out = vec![0_u8; layout.stride().expect("packed")];
    for (i, f) in part.fields.iter().enumerate() {
        if let Some((_, v)) = values.iter().find(|(n, _)| *n == f.name) {
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            let range = layout.field(i).expect("field");
            out[range].copy_from_slice(&bytes);
        }
    }
    out
}

fn tensor(value: Value) -> Tensor {
    match value {
        Value::Tensor(t) => t,
        other => panic!("not a tensor: {other:?}"),
    }
}

fn f32s(t: &Tensor) -> Vec<f32> {
    assert_eq!(t.dtype, DType::F32);
    t.data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

#[test]
fn windows_and_concatenations_on_the_force_policy() {
    let b = binding("force_policy", "ur10e");
    let ft = b.input("ft").expect("ft");
    let part = ft.part.as_ref().expect("part");
    let samples: Vec<Vec<u8>> = (0..50)
        .map(|i| {
            let i = i as f32;
            sample(
                part,
                &[("force", &[i, i + 0.1, i + 0.2]), ("torque", &[-i, -i, -i])],
            )
        })
        .collect();
    let refs: Vec<&[u8]> = samples.iter().map(Vec::as_slice).collect();
    let window = tensor(ft.apply(&refs).expect("window"));
    assert_eq!(window.shape, [50, 6]);
    let values = f32s(&window);
    assert_eq!(&values[6..12], &[1.0, 1.1, 1.2, -1.0, -1.0, -1.0]);
    assert!(ft.apply(&refs[..3]).is_err(), "a window is never short");

    let q = b.input("q").expect("q");
    let arm = q.part.as_ref().expect("arm");
    let pos = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0];
    let vel = [10.0_f32, 11.0, 12.0, 13.0, 14.0, 15.0];
    let packed = sample(arm, &[("pos", &pos), ("vel", &vel), ("effort", &[9.0; 6])]);
    let state = tensor(q.apply(&[&packed]).expect("q"));
    assert_eq!(state.shape, [12]);
    assert_eq!(f32s(&state), [pos, vel].concat());
}

#[test]
fn a_wrist_camera_is_cropped_resized_laid_out_and_normalised() {
    let b = binding("force_policy", "ur10e");
    let wrist = b.input("wrist").expect("wrist");
    // Left half black, right half white; the cover crop keeps the middle
    // 480 columns, so the left and right edges of the 96x96 result differ.
    let mut frame = vec![0_u8; 480 * 640 * 3];
    for row in 0..480 {
        for col in 320..640 {
            let i = (row * 640 + col) * 3;
            frame[i..i + 3].copy_from_slice(&[255, 255, 255]);
        }
    }
    let image = tensor(wrist.apply(&[&frame]).expect("image"));
    assert_eq!(image.shape, [3, 96, 96]);
    let values = f32s(&image);
    assert!(values.iter().all(|v| (0.0..=1.0).contains(v)));
    // Channel 0 plane, middle row: black on the left, white on the right.
    let row = &values[48 * 96..49 * 96];
    assert_eq!((row[0], row[95]), (0.0, 1.0));
    let flat = vec![51_u8; 480 * 640 * 3];
    assert!(
        f32s(&tensor(wrist.apply(&[&flat]).expect("flat")))
            .iter()
            .all(|v| *v == 0.2)
    );
}

#[test]
fn loco_go2_reorders_offsets_and_derives_gravity() {
    let b = binding("loco_go2", "go2");
    let legs = b.input("q").expect("q").part.clone().expect("legs");
    // SDK order: FR, FL, RR, RL (hip, thigh, calf each); value = SDK index.
    let pos: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let packed = sample(&legs, &[("pos", &pos)]);
    let q = f32s(&tensor(
        b.input("q").expect("q").apply(&[&packed]).expect("q"),
    ));
    // Isaac order: FL_hip (SDK 3), FR_hip (0), RL_hip (9), RR_hip (6), ...
    let default = [
        0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5,
    ];
    let isaac = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8];
    for i in 0..12 {
        assert!(
            (q[i] - (isaac[i] as f32 - default[i] as f32)).abs() < 1e-6,
            "{i}: {}",
            q[i]
        );
    }
    let imu = b
        .input("gravity")
        .expect("gravity")
        .part
        .clone()
        .expect("imu");
    // quat_wxyz identity: gravity points down the sensor's z.
    let packed = sample(&imu, &[("orientation", &[1.0, 0.0, 0.0, 0.0])]);
    let gravity = f32s(&tensor(
        b.input("gravity")
            .expect("gravity")
            .apply(&[&packed])
            .expect("g"),
    ));
    assert_eq!(gravity, [0.0, 0.0, -1.0]);
    let cmd = b
        .input("command")
        .expect("command")
        .part
        .clone()
        .expect("goal");
    let packed = sample(&cmd, &[("linear", &[0.5, 0.1]), ("angular", &[0.3])]);
    let command = f32s(&tensor(
        b.input("command")
            .expect("command")
            .apply(&[&packed])
            .expect("c"),
    ));
    assert_eq!(command, [0.5, 0.1, 0.3]);

    // The action: Isaac order out of the model, scaled and offset, packed in
    // SDK order.
    let legs_out = b.action("legs").expect("legs");
    let model: Vec<f64> = (0..12).map(|i| i as f64).collect();
    let packed = legs_out
        .pack(&Value::Tensor(Tensor::from_f64(
            DType::F32,
            vec![1, 12],
            &model,
        )))
        .expect("pack");
    let sdk: Vec<f32> = packed
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    for (model_i, sdk_i) in isaac.iter().enumerate() {
        let want = model_i as f32 * 0.25 + default[model_i] as f32;
        assert!((sdk[*sdk_i] - want).abs() < 1e-6, "{model_i}");
    }
}

#[test]
fn grippers_rescale_and_poses_re_encode() {
    let b = binding("pi0_single", "ur10e");
    let gripper = b.action("gripper").expect("gripper");
    let chunk: Vec<f64> = (0..10)
        .map(|i| if i % 2 == 0 { -1.0 } else { 1.0 })
        .collect();
    let packed = gripper
        .pack(&Value::Tensor(Tensor::from_f64(
            DType::F32,
            vec![10, 1],
            &chunk,
        )))
        .expect("pack");
    assert_eq!(
        packed.len(),
        10 * 8,
        "one f32 per action at an 8-byte stride"
    );
    let opening: Vec<f32> = packed
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(&opening[..2], &[0.0, 1.0], "closed at -1, open at 1");
    assert!(
        gripper
            .pack(&Value::Tensor(Tensor::from_f64(
                DType::F32,
                vec![9, 1],
                &chunk[..9]
            )))
            .is_err(),
        "a chunk is the declared horizon"
    );

    let libero = binding("pi05_libero", "libero");
    let eef = libero.input("eef").expect("eef");
    let tcp = eef.part.clone().expect("tcp");
    let half = std::f32::consts::FRAC_PI_4;
    let packed = sample(
        &tcp,
        &[
            ("pos", &[0.1, 0.2, 0.3]),
            ("rot", &[0.0, 0.0, half.sin(), half.cos()]),
        ],
    );
    let state = f32s(&tensor(eef.apply(&[&packed]).expect("eef")));
    assert_eq!(state.len(), 6);
    assert!(
        (state[5] - std::f32::consts::FRAC_PI_2).abs() < 1e-6,
        "{state:?}"
    );
    // The upside-down cameras are rotated back.
    let image = libero.input("observation/image").expect("image");
    let mut frame = vec![0_u8; 256 * 256 * 3];
    frame[..3].copy_from_slice(&[200, 100, 50]);
    let out = tensor(image.apply(&[&frame]).expect("image"));
    assert_eq!(out.shape, [224, 224, 3]);
    let last = &out.data[out.data.len() - 3..];
    assert!(
        last[0] > 100 && last[0] > last[1],
        "the corner moved to the far end: {last:?}"
    );
}

#[test]
fn a_blank_camera_is_black_after_preprocessing() {
    let b = binding("pi0_aloha", "aloha2_no_cam_low");
    let low = tensor(
        b.input("cam_low")
            .expect("cam_low")
            .apply(&[])
            .expect("blank"),
    );
    assert_eq!(
        (low.dtype, low.shape.clone()),
        (DType::U8, vec![224, 224, 3])
    );
    assert!(low.data.iter().all(|v| *v == 0));
}

#[test]
fn untouched_u8_images_pass_through_as_bytes_hwc_or_chw() {
    let mut parts = normalise(
        &ParticipantId::new("bot").expect("id"),
        SPEC_VERSION,
        r#"{"cam": {"kind": "camera", "width": 3, "height": 2, "hz": 10}}"#,
    )
    .expect("robot");
    parts[0].channel = ChannelId(1);
    let policy = Policy::parse(
        r#"{"hz": 10, "inputs": {"hwc": {"kind": "camera", "at": "cam"},
                                "chw": {"kind": "camera", "at": "cam", "layout": "chw"}}}"#,
    )
    .expect("policy");
    let b = bind(&policy, &parts, &RunOptions::named("test")).expect("bound");
    // Pixel (row, col) has channels (10 * row + col) + {0, 100, 200}.
    let pixels: Vec<u8> = (0..2_u8)
        .flat_map(|r| (0..3_u8).flat_map(move |c| [10 * r + c, 10 * r + c + 100, 10 * r + c + 200]))
        .collect();
    let hwc = tensor(b.input("hwc").expect("hwc").apply(&[&pixels]).expect("hwc"));
    assert_eq!(
        (hwc.dtype, hwc.shape.as_slice()),
        (DType::U8, &[2_u32, 3, 3][..])
    );
    assert_eq!(hwc.data, pixels);
    let chw = tensor(b.input("chw").expect("chw").apply(&[&pixels]).expect("chw"));
    assert_eq!(chw.shape, [3, 2, 3]);
    assert_eq!(
        chw.data,
        [
            0, 1, 2, 10, 11, 12, 100, 101, 102, 110, 111, 112, 200, 201, 202, 210, 211, 212
        ]
    );
}
