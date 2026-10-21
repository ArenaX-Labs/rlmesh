#![cfg(feature = "lockstep")]
//! A Gym env as a robot (§12.5): paths, kinds, attrs, and bit-exact slabs.

use bytes::Bytes;
use rlmesh::lockstep::map::{self, GymEnvInfo, GymMap};
use rlmesh_spaces::{
    AutoresetMode, BoxBounds, BoxSpec, DType, DictSpec, DiscreteSpec, EnvContract, MetaMap,
    MetaValue, MultiBinarySpec, MultiDiscreteSpec, SpaceKind, SpaceSpec, TextSpec, TupleSpec,
    UniformBounds,
};

fn boxed(dtype: DType, shape: &[i64]) -> SpaceSpec {
    SpaceSpec {
        shape: shape.to_vec(),
        dtype,
        spec: Some(SpaceKind::Box(BoxSpec {
            bounds: Some(BoxBounds::Uniform(UniformBounds {
                low: -1.0,
                high: 1.0,
            })),
        })),
    }
}

fn dict(items: Vec<(&str, SpaceSpec)>) -> SpaceSpec {
    SpaceSpec {
        shape: Vec::new(),
        dtype: DType::Unspecified,
        spec: Some(SpaceKind::Dict(DictSpec {
            keys: items.iter().map(|(k, _)| (*k).to_string()).collect(),
            spaces: items.into_iter().map(|(_, s)| s).collect(),
        })),
    }
}

fn native(value: &serde_json::Value) -> MetaValue {
    match value {
        serde_json::Value::Null => MetaValue::Null,
        serde_json::Value::Bool(b) => MetaValue::Bool(*b),
        serde_json::Value::Number(n) => match n.as_i64() {
            Some(i) => MetaValue::Int(i),
            None => MetaValue::Float(n.as_f64().unwrap_or(0.0)),
        },
        serde_json::Value::String(s) => MetaValue::String(s.clone()),
        serde_json::Value::Array(items) => MetaValue::List(items.iter().map(native).collect()),
        serde_json::Value::Object(map) => MetaValue::Map(
            map.iter()
                .map(|(k, v)| (k.clone(), native(v)))
                .collect::<MetaMap>(),
        ),
    }
}

/// Every v1 leaf kind and dtype, tagged and untagged, nested in Dict and
/// Tuple.
fn catalog_contract(num_envs: u32) -> EnvContract {
    let observation = dict(vec![
        ("Image", boxed(DType::Uint8, &[4, 6, 3])),
        ("joint_pos", boxed(DType::Float32, &[7])),
        ("joint_vel", boxed(DType::Float32, &[7])),
        ("ft", boxed(DType::Float32, &[6])),
        ("gripper", boxed(DType::Float64, &[1])),
        ("odd-f32", boxed(DType::Float32, &[3])),
        ("half", boxed(DType::Float16, &[3])),
        ("flags", boxed(DType::Bool, &[2])),
        (
            "mode",
            SpaceSpec {
                shape: Vec::new(),
                dtype: DType::Int64,
                spec: Some(SpaceKind::Discrete(DiscreteSpec { n: 4, start: 0 })),
            },
        ),
        (
            "md",
            SpaceSpec {
                shape: vec![3],
                dtype: DType::Int32,
                spec: Some(SpaceKind::MultiDiscrete(MultiDiscreteSpec {
                    nvec: vec![2, 3, 4],
                })),
            },
        ),
        (
            "mb",
            SpaceSpec {
                shape: vec![5],
                dtype: DType::Int8,
                spec: Some(SpaceKind::MultiBinary(MultiBinarySpec)),
            },
        ),
        (
            "instruction",
            SpaceSpec {
                shape: Vec::new(),
                dtype: DType::Uint8,
                spec: Some(SpaceKind::Text(TextSpec {
                    min_length: 0,
                    max_length: 64,
                    charset: String::new(),
                })),
            },
        ),
        (
            "pair",
            SpaceSpec {
                shape: Vec::new(),
                dtype: DType::Unspecified,
                spec: Some(SpaceKind::Tuple(TupleSpec {
                    spaces: vec![boxed(DType::Int8, &[3]), boxed(DType::Uint64, &[2])],
                })),
            },
        ),
        (
            "Nested",
            dict(vec![
                ("i16", boxed(DType::Int16, &[5])),
                ("u32", boxed(DType::Uint32, &[1])),
                ("u16", boxed(DType::Uint16, &[3])),
                ("f64", boxed(DType::Float64, &[2, 2])),
            ]),
        ),
    ]);
    let tags = serde_json::json!({
        "observation": {
            "Image": {"type": "image", "role": "image/primary"},
            "joint_pos": {"type": "state", "role": "proprio/joint_pos"},
            "joint_vel": {"type": "state", "role": "proprio/joint_vel"},
            "ft": {"type": "state", "role": "proprio/eef_wrench", "frame": "tool0"},
            "gripper": {"type": "state", "role": "proprio/gripper"},
            "instruction": {"type": "text", "role": "text/instruction"},
        },
        "action": {"components": [{"role": "action/joint_pos", "dim": 4}]},
    });
    let mut metadata = MetaMap::new();
    metadata.insert(
        rlmesh::lockstep::adapters::ENV_METADATA_KEY.to_string(),
        native(&tags),
    );
    EnvContract {
        id: "Catalog-v0".into(),
        action_space: Some(boxed(DType::Float32, &[4])),
        observation_space: Some(observation),
        metadata: Some(metadata),
        render_mode: String::new(),
        num_envs,
        autoreset_mode: AutoresetMode::NextStep,
    }
}

fn info(contract: EnvContract) -> GymEnvInfo {
    GymEnvInfo {
        contract,
        workflow_edition: "2026.06".into(),
        supported_workflow_editions: vec!["2026.06".into()],
        capabilities: Vec::new(),
    }
}

/// Deterministic bytes, every value representable (bools are 0/1).
fn leaf_bytes(leaf: &map::Leaf, lanes: usize, salt: u8) -> Bytes {
    match leaf.lane_bytes {
        None => {
            // v1's text slab: [u64 count][u64 len x count][bodies].
            let bodies: Vec<String> = (0..lanes)
                .map(|i| format!("lane {i} salt {salt}"))
                .collect();
            let mut out = Vec::new();
            out.extend_from_slice(&(lanes as u64).to_le_bytes());
            for b in &bodies {
                out.extend_from_slice(&(b.len() as u64).to_le_bytes());
            }
            for b in &bodies {
                out.extend_from_slice(b.as_bytes());
            }
            Bytes::from(out)
        }
        Some(lb) => {
            let bool_like = matches!(leaf.spec.spec, Some(SpaceKind::MultiBinary(_)))
                || leaf.spec.dtype == DType::Bool;
            Bytes::from(
                (0..lb * lanes)
                    .map(|i| {
                        if bool_like {
                            (i as u8 ^ salt) & 1
                        } else {
                            (i as u8).wrapping_mul(31).wrapping_add(salt)
                        }
                    })
                    .collect::<Vec<u8>>(),
            )
        }
    }
}

#[test]
fn a_catalog_env_maps_to_a_robot_and_its_slabs_round_trip_bit_exact() {
    for lanes in [1_usize, 3] {
        let mut gym = GymMap::new(info(catalog_contract(lanes as u32)), 1000.0).expect("map");
        gym.number_channels();
        let paths: Vec<(&str, &str)> = gym
            .parts()
            .map(|p| (p.spec.path.as_str(), p.spec.kind.as_str()))
            .collect();
        assert_eq!(
            paths,
            [
                ("scene/cam", "camera"),
                ("arm", "joints"),
                ("wrist/ft", "wrench"),
                ("gripper", "gripper"),
                ("odd_f32", "gym/box"),
                ("half", "gym/box"),
                ("flags", "gym/box"),
                ("mode", "gym/discrete"),
                ("md", "gym/multi_discrete"),
                ("mb", "gym/multi_binary"),
                ("task", "text"),
                ("pair/item0", "gym/box"),
                ("pair/item1", "gym/box"),
                ("nested/i16", "gym/box"),
                ("nested/u32", "gym/box"),
                ("nested/u16", "gym/box"),
                ("nested/f64", "gym/box"),
                ("action", "gym/box"),
            ]
        );
        // The f64 gripper is not a float32 gripper part.
        let gripper = gym.part("gripper").expect("gripper");
        assert_eq!(
            gripper.spec.fields[0].dtype,
            rlmesh_session::wire::DType::F64
        );
        let arm = gym.part("arm").expect("arm");
        let names: Vec<&str> = arm.spec.fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["pos", "vel"]);
        let wrench = gym.part("wrist/ft").expect("wrench");
        assert_eq!(wrench.spec.frame, "tool0");
        assert!(!wrench.zero_copy, "two fields of one leaf");
        assert!(gym.part("arm").is_some_and(|p| !p.zero_copy));
        assert!(gym.part("nested/f64").is_some_and(|p| p.zero_copy));
        assert!(
            gym.part("odd_f32").is_some_and(|p| !p.zero_copy),
            "12 bytes pad to 16"
        );
        let cam = gym.part("scene/cam").expect("camera");
        assert!(cam.spec.attrs.contains_key("gym_env"));
        assert!(cam.spec.attrs["env_tags"].contains("image/primary"));
        assert_eq!(cam.spec.attrs["gym_key"], "[[\"Image\"]]");
        assert!(gym.part("action").is_some_and(|p| p.spec.command.is_some()));

        // The info round-trips through its attr exactly as through a v1
        // handshake (which carries the contract as the same protobuf).
        let back = GymEnvInfo::from_attr(&cam.spec.attrs["gym_env"]).expect("gym_env");
        let handshake = rlmesh_grpc::wire::env_contract_from_proto(
            rlmesh_grpc::wire::env_contract_to_proto(&gym.info.contract),
        )
        .expect("v1 round trip");
        assert_eq!(back.contract, handshake);
        assert_eq!(back.capabilities, gym.info.capabilities);

        for salt in [0_u8, 7, 200] {
            let leaves: Vec<Bytes> = gym
                .observation
                .leaves
                .iter()
                .map(|leaf| leaf_bytes(leaf, lanes, salt))
                .collect();
            let slabs = map::pack(&gym.observation, &leaves, lanes).expect("pack");
            for (part, slab) in gym.observation.parts.iter().zip(&slabs) {
                if let Some(stride) = part.layout.stride() {
                    assert_eq!(slab.len(), stride * lanes, "{}", part.spec.path);
                    assert_eq!(stride % 8, 0);
                }
            }
            let back = map::unpack(&gym.observation, &slabs, lanes).expect("unpack");
            assert_eq!(back, leaves, "bit-exact at {lanes} lanes");
            let action = vec![leaf_bytes(&gym.action.leaves[0], lanes, salt)];
            let slabs = map::pack(&gym.action, &action, lanes).expect("pack action");
            assert_eq!(
                map::unpack(&gym.action, &slabs, lanes).expect("unpack"),
                action
            );
        }
        // A short slab is refused.
        let leaves: Vec<Bytes> = gym
            .observation
            .leaves
            .iter()
            .map(|leaf| leaf_bytes(leaf, lanes + 1, 0))
            .collect();
        assert!(map::pack(&gym.observation, &leaves, lanes).is_err());
    }
}

#[test]
fn an_untagged_box_env_is_obs_and_action_and_keys_that_escape_alike_keep_apart() {
    let contract = EnvContract {
        id: "Box-v0".into(),
        action_space: Some(boxed(DType::Float32, &[6])),
        observation_space: Some(boxed(DType::Float32, &[18])),
        metadata: None,
        render_mode: String::new(),
        num_envs: 1,
        autoreset_mode: AutoresetMode::Disabled,
    };
    let gym = GymMap::new(info(contract.clone()), 1000.0).expect("map");
    let paths: Vec<&str> = gym.parts().map(|p| p.spec.path.as_str()).collect();
    assert_eq!(paths, ["obs", "action"]);
    assert!(gym.parts().all(|p| p.zero_copy));

    // `Pos` and `pos` escape alike, and an observation key `action` meets
    // the action root: the later one takes the hash of its original keys.
    let clash = EnvContract {
        observation_space: Some(dict(vec![
            ("Pos", boxed(DType::Float32, &[3])),
            ("pos", boxed(DType::Float32, &[3])),
            ("action", boxed(DType::Float32, &[1])),
        ])),
        ..contract
    };
    let gym = GymMap::new(info(clash), 1000.0).expect("map");
    let paths: Vec<&str> = gym.parts().map(|p| p.spec.path.as_str()).collect();
    assert_eq!(paths.len(), 4);
    assert_eq!((paths[0], paths[3]), ("pos", "action"));
    assert!(
        paths[1].starts_with("pos_") && paths[2].starts_with("action_"),
        "{paths:?}"
    );
}
