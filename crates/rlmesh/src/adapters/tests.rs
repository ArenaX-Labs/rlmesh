use super::*;
use crate::ErrorClass;
fn action() -> Action {
    Action::new([Actuator {
        role: Some(ACTION_GRIPPER.into()),
        dim: 1,
        ..Default::default()
    }])
}
#[test]
fn typed_vocabularies_compare_by_python_spelling() {
    assert_eq!(
        Tolerant::Other("crop".into()),
        Tolerant::Known(FitMode::Crop)
    );
    assert_eq!(Tolerant::<Resample>::parse("future").as_str(), "future");
    assert_eq!(
        Tolerant::<crate::DType>::parse("float32").known(),
        Some(&crate::DType::Float32)
    );
    assert_eq!(
        ROTATION_DIMS
            .iter()
            .map(|(e, _)| e.as_str())
            .collect::<Vec<_>>(),
        vec![
            "axis_angle",
            "euler_xyz",
            "gravity_xyz",
            "quat_wxyz",
            "quat_xyzw",
            "rot6d",
            "rot6d_rowmajor"
        ]
    );
}
#[test]
fn dict_equality_ignores_authored_order_but_accessors_and_repr_keep_it() {
    let a = ModelSpec::new(
        InputNode::dict([
            ("z", Text::new(INSTRUCTION).into()),
            ("a", Image::new(IMAGE_PRIMARY).into()),
        ]),
        action(),
    )
    .unwrap();
    let b = ModelSpec::new(
        InputNode::dict([
            ("a", Image::new(IMAGE_PRIMARY).into()),
            ("z", Text::new(INSTRUCTION).into()),
        ]),
        action(),
    )
    .unwrap();
    assert_eq!(a, b);
    assert_eq!(a.to_json().unwrap(), b.to_json().unwrap());
    assert!(a.to_string().starts_with("ModelSpec(input={'z': Text("));
    assert_eq!(ModelSpec::from_json(&a.to_json().unwrap()).unwrap(), a);
}
#[test]
fn leaf_checks_run_at_composition_and_codec_bounds_stay_at_serialization() {
    let spec = ModelSpec::new(
        Image {
            stack: 65,
            ..Image::new(IMAGE_PRIMARY)
        },
        action(),
    )
    .unwrap();
    assert_eq!(spec.to_dict().unwrap_err().class(), ErrorClass::ValueError);
    assert!(
        ModelSpec::new(
            State {
                dim: Some(2),
                index: Some(0),
                ..State::new(JOINT_POS)
            },
            action()
        )
        .is_err()
    );
    assert!(ModelSpec::new(Custom::default(), action()).is_err());
    assert!(
        ModelSpec::new(
            InputNode::dict([
                ("x", Text::new(INSTRUCTION).into()),
                ("x", Text::new(INSTRUCTION).into())
            ]),
            action()
        )
        .is_err()
    );
}
#[test]
fn roles_keep_declaration_order_and_duplicate_identities_fail() {
    let tags = EnvTags::new(
        ObsNode::dict([
            ("z", ImageTag::new(IMAGE_SECONDARY).into()),
            ("a", ImageTag::new(IMAGE_PRIMARY).into()),
        ]),
        action(),
    )
    .unwrap();
    assert_eq!(
        tags.observation_roles().images,
        vec![IMAGE_SECONDARY, IMAGE_PRIMARY]
    );
    assert_eq!(EnvTags::from_json(&tags.to_json().unwrap()).unwrap(), tags);
    assert!(
        EnvTags::new(
            ObsNode::tuple([
                StateTag::new(JOINT_POS).into(),
                StateTag::new(JOINT_POS).into()
            ]),
            action()
        )
        .is_err()
    );
}
#[test]
fn custom_equality_uses_callable_identity_and_publish_refuses_inputs() {
    let f = ObsTransform::from(|v: &Value| Ok(v.clone()));
    let a = Custom {
        transform: Some(f.clone()),
        entrypoint: None,
    };
    assert_eq!(
        a,
        Custom {
            transform: Some(f),
            entrypoint: None
        }
    );
    assert_ne!(
        a,
        Custom {
            transform: Some(ObsTransform::from(|v: &Value| Ok(v.clone()))),
            entrypoint: None
        }
    );
    assert!(ModelSpec::new(a, action()).unwrap().to_dict().is_err());
}
#[test]
fn metadata_type_errors_are_resolution_errors_and_absence_is_none() {
    let mut m = Metadata::new();
    m.insert(ENV_METADATA_KEY, 4_i64);
    assert_eq!(
        EnvTags::from_metadata(&m).unwrap_err().class(),
        ErrorClass::AdapterResolutionError
    );
    assert_eq!(EnvTags::from_metadata(&Metadata::new()).unwrap(), None);
    assert_eq!(
        Error::adapter_resolution("authored").class(),
        ErrorClass::AdapterResolutionError
    );
}
#[test]
fn image_sugar_and_python_reprs() {
    let i = Image {
        stack: 4,
        ..Image::new(IMAGE_PRIMARY)
    }
    .with_stride(2)
    .unwrap()
    .with_size(224)
    .unwrap();
    assert_eq!(i.offsets, Some(vec![-6, -4, -2, 0]));
    assert_eq!(i.height, Some(224));
    assert!(i.with_stride(1).is_err());
    assert_eq!(
        Text::new(INSTRUCTION).to_string(),
        "Text(role='text/instruction', container='str', fill=None)"
    );
    assert_eq!(Constant::default().to_string(), "Constant(dim=1, fill=0.0)");
    assert_eq!(
        ObservationRoles::default().to_string(),
        "ObservationRoles(images=(), states=(), texts=())"
    );
}
#[test]
fn state_encodings_and_axis_values_round_trip() {
    let s = State {
        dim: Some(3),
        scale: Some(AxisValue::PerAxis(vec![1.0, 2.0, 3.0])),
        encoding: Some(StateEncoding::Native(vec![
            Tolerant::parse("future"),
            RotationEncoding::AxisAngle.into(),
        ])),
        ..State::new(EEF_ROT)
    };
    let spec = ModelSpec::new(s, action()).unwrap();
    let text = spec.to_json().unwrap();
    assert_eq!(ModelSpec::from_json(&text).unwrap(), spec);
}

fn box_spec(width: usize) -> crate::spaces::SpaceSpec {
    crate::spaces::Box::new(-100.0, 100.0, vec![width as i64], Default::default())
        .unwrap()
        .spec()
        .clone()
}
fn floats(v: &[f32]) -> Value {
    Value::Tensor(
        crate::Tensor::new(
            &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
            &[v.len()],
            crate::DType::Float32,
        )
        .unwrap(),
    )
}
fn flat_tags() -> EnvTags {
    EnvTags::new(StateTag::new(GRIPPER_POS), action()).unwrap()
}
#[test]
fn adapter_previous_action_and_clone_state_are_shared_and_reset() {
    let spec = ModelSpec::new(
        Concat::new([
            ConcatPart::Role(GRIPPER_POS.into()),
            Previous(ACTION_GRIPPER, Default::default()).into(),
        ]),
        action(),
    )
    .unwrap();
    let mut a = resolve(
        &flat_tags(),
        &box_spec(1),
        &box_spec(1),
        &spec,
        Default::default(),
    )
    .unwrap();
    assert_eq!(a.transform_obs(floats(&[2.])).unwrap(), floats(&[2., 0.]));
    a.transform_action(floats(&[7.])).unwrap();
    let mut shared = a.clone();
    assert_eq!(
        shared.transform_obs(floats(&[3.])).unwrap(),
        floats(&[3., 7.])
    );
    shared.reset().unwrap();
    assert_eq!(a.transform_obs(floats(&[4.])).unwrap(), floats(&[4., 0.]));
}
#[test]
fn custom_input_gets_full_envelope_and_preserves_authored_errors() {
    let spec = ModelSpec::new(
        Custom {
            transform: Some(ObsTransform::from(|v: &Value| {
                assert!(v.as_dict()?.contains_key("<obs>"));
                Err(Error::Runtime("authored callback".into()))
            })),
            entrypoint: None,
        },
        action(),
    )
    .unwrap();
    let mut a = resolve(
        &flat_tags(),
        &box_spec(1),
        &box_spec(1),
        &spec,
        Default::default(),
    )
    .unwrap();
    assert_eq!(
        a.transform_obs(floats(&[1.])).unwrap_err(),
        Error::Runtime("authored callback".into())
    );
}
#[test]
fn registry_resolves_once_and_replacement_affects_only_later_adapters() {
    crate::ext::register_entrypoint(
        "parity:test:input",
        ObsTransform::from(|_: &Value| Ok(Value::Int(1))),
    )
    .unwrap();
    let spec = ModelSpec::new(
        Custom {
            transform: None,
            entrypoint: Some("parity:test:input".into()),
        },
        action(),
    )
    .unwrap();
    assert_eq!(
        resolve(
            &flat_tags(),
            &box_spec(1),
            &box_spec(1),
            &spec,
            Default::default()
        )
        .err()
        .unwrap()
        .class(),
        ErrorClass::AdapterResolutionError
    );
    let mut a = resolve(
        &flat_tags(),
        &box_spec(1),
        &box_spec(1),
        &spec,
        ResolveOptions::default().trust_entrypoints(true),
    )
    .unwrap();
    crate::ext::register_entrypoint(
        "parity:test:input",
        ObsTransform::from(|_: &Value| Ok(Value::Int(2))),
    )
    .unwrap();
    assert_eq!(a.transform_obs(floats(&[1.])).unwrap(), Value::Int(1));
    let mut b = resolve(
        &flat_tags(),
        &box_spec(1),
        &box_spec(1),
        &spec,
        ResolveOptions::default().trust_entrypoints(true),
    )
    .unwrap();
    assert_eq!(b.transform_obs(floats(&[1.])).unwrap(), Value::Int(2));
}
#[test]
fn inverse_checks_are_value_distinct_and_fail_with_resolution_class() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    let calls = Arc::new(AtomicUsize::new(0));
    let count = calls.clone();
    let t = RotationTransform::from(move |v: &Value| {
        count.fetch_add(1, Ordering::SeqCst);
        Ok(v.clone())
    });
    let c = CustomEncoding {
        from_base: Some(EncodingArm::Transform(t.clone())),
        to_base: Some(EncodingArm::Transform(t)),
        ..CustomEncoding::new(RotationEncoding::QuatXyzw)
    };
    let s = State {
        encoding: Some(StateEncoding::Custom(c)),
        ..State::new(EEF_ROT)
    };
    let spec = ModelSpec::new(InputNode::tuple([s.clone().into(), s.into()]), action()).unwrap();
    let tags = EnvTags::new(
        StateTag {
            encoding: Some(vec![RotationEncoding::QuatXyzw.into()]),
            ..StateTag::new(EEF_ROT)
        },
        action(),
    )
    .unwrap();
    let mut adapter =
        resolve(&tags, &box_spec(4), &box_spec(1), &spec, Default::default()).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let value = adapter.transform_obs(floats(&[0., 0., 0., 1.])).unwrap();
    assert!(matches!(value, Value::Tuple(_)));
}

#[test]
fn custom_values_preserve_scalar_kinds_none_and_tuples() {
    let expected = Value::Tuple(vec![Value::Int(i64::MAX), Value::Bool(true), Value::None]);
    let output = expected.clone();
    let spec = ModelSpec::new(
        Custom {
            transform: Some(ObsTransform::from(move |raw: &Value| {
                assert_eq!(raw.as_dict()?.get("<obs>"), Some(&Value::Int(4)));
                Ok(output.clone())
            })),
            entrypoint: None,
        },
        action(),
    )
    .unwrap();
    let mut adapter = resolve(
        &flat_tags(),
        &box_spec(1),
        &box_spec(1),
        &spec,
        Default::default(),
    )
    .unwrap();
    assert_eq!(adapter.transform_obs(Value::Int(4)).unwrap(), expected);
}
#[test]
fn codec_refuses_nonfinite_fields_and_normalizes_known_tolerant_spellings() {
    let spec = ModelSpec::new(
        State {
            scale: Some(AxisValue::Scalar(f64::NAN)),
            ..State::new(GRIPPER_POS)
        },
        action(),
    )
    .unwrap();
    assert!(spec.to_json().is_err());
    let tags = EnvTags::new(
        StateTag {
            frame: Some(Tolerant::Other("world".into())),
            ..StateTag::new(EEF_POS)
        },
        action(),
    )
    .unwrap();
    let ObsNode::Leaf(ObsLeaf::StateTag(tag)) = tags.observation() else {
        panic!()
    };
    assert_eq!(tag.frame.as_ref().unwrap().known(), Some(&Frame::World));
}

#[test]
fn inverse_check_preserves_intermediate_shape_and_per_step_checks_the_arm() {
    let from =
        RotationTransform::from(|v: &Value| v.as_tensor()?.reshape(&[2, 2]).map(Value::Tensor));
    let to = RotationTransform::from(|v: &Value| v.as_tensor()?.reshape(&[4]).map(Value::Tensor));
    let encoding = CustomEncoding {
        from_base: Some(EncodingArm::Transform(from)),
        to_base: Some(EncodingArm::Transform(to)),
        ..CustomEncoding::new(RotationEncoding::QuatXyzw)
    };
    let spec = ModelSpec::new(
        State {
            encoding: Some(StateEncoding::Custom(encoding)),
            ..State::new(EEF_ROT)
        },
        action(),
    )
    .unwrap();
    let tags = EnvTags::new(
        StateTag {
            encoding: Some(vec![RotationEncoding::QuatXyzw.into()]),
            ..StateTag::new(EEF_ROT)
        },
        action(),
    )
    .unwrap();
    let mut adapter =
        resolve(&tags, &box_spec(4), &box_spec(1), &spec, Default::default()).unwrap();
    assert_eq!(
        adapter
            .transform_obs(floats(&[0., 0., 0., 1.]))
            .unwrap_err()
            .class(),
        ErrorClass::ValueError
    );
}
#[test]
fn readers_return_canonical_order_while_constructors_keep_authored_order() {
    let text = r#"{"input":{"z":{"type":"text","role":"text/z"},"a":{"type":"text","role":"text/a"}},"output":{"components":[{"role":"action/gripper","dim":1}]}}"#;
    let spec = ModelSpec::from_json(text).unwrap();
    let InputNode::Dict(pairs) = spec.input() else {
        panic!()
    };
    assert_eq!(pairs[0].0, "a");
    let authored = ModelSpec::new(
        InputNode::dict([
            ("z", Text::new("text/z").into()),
            ("a", Text::new("text/a").into()),
        ]),
        action(),
    )
    .unwrap();
    let InputNode::Dict(pairs) = authored.input() else {
        panic!()
    };
    assert_eq!(pairs[0].0, "z");
    assert_eq!(spec, authored);
    assert_eq!(
        ModelSpec::from_dict(&authored.to_dict().unwrap()).unwrap(),
        spec
    );
}

#[test]
fn typed_metadata_refuses_nonfinite_numbers_instead_of_converting_them_to_null() {
    let mut data = ModelSpec::new(
        State {
            dim: Some(1),
            ..State::new(GRIPPER_POS)
        },
        action(),
    )
    .unwrap()
    .to_dict()
    .unwrap();
    let Value::Dict(input) = data.get_mut("input").unwrap() else {
        panic!()
    };
    let Value::List(parts) = input.get_mut("components").unwrap() else {
        panic!()
    };
    let Value::Dict(part) = &mut parts[0] else {
        panic!()
    };
    part.insert("scale", f64::NAN);
    assert_eq!(
        ModelSpec::from_dict(&data).unwrap_err().class(),
        ErrorClass::ValueError
    );
}

#[test]
fn previous_action_source_round_trips_as_a_single_state_leaf() {
    let spec = ModelSpec::new(
        Previous(ACTION_GRIPPER, PreviousOptions::default().fill(2.0)),
        action(),
    )
    .unwrap();
    assert_eq!(
        ModelSpec::from_json(&spec.to_json().unwrap()).unwrap(),
        spec
    );
}

#[test]
fn custom_encoding_numeric_tensors_do_not_pass_through_json() {
    let from = RotationTransform::from(|_: &Value| {
        Ok(Value::Tensor(crate::Tensor::new(
            &[f64::NAN, 0.0, 0.0, 1.0]
                .into_iter()
                .flat_map(f64::to_le_bytes)
                .collect::<Vec<_>>(),
            &[4],
            crate::DType::Float64,
        )?))
    });
    let encoding = CustomEncoding {
        from_base: Some(EncodingArm::Transform(from)),
        ..CustomEncoding::new(RotationEncoding::QuatXyzw)
    };
    let spec = ModelSpec::new(
        State {
            encoding: Some(StateEncoding::Custom(encoding)),
            dtype: crate::DType::Float64.into(),
            ..State::new(EEF_ROT)
        },
        action(),
    )
    .unwrap();
    let tags = EnvTags::new(
        StateTag {
            encoding: Some(vec![RotationEncoding::QuatXyzw.into()]),
            ..StateTag::new(EEF_ROT)
        },
        action(),
    )
    .unwrap();
    let mut adapter =
        resolve(&tags, &box_spec(4), &box_spec(1), &spec, Default::default()).unwrap();
    let output = adapter.transform_obs(floats(&[0., 0., 0., 1.])).unwrap();
    let tensor = output.as_tensor().unwrap();
    assert!(f64::from_le_bytes(tensor.tobytes()[0..8].try_into().unwrap()).is_nan());
}

#[test]
fn trusted_unregistered_custom_entrypoint_is_resolution_error() {
    let spec = ModelSpec::new(
        Custom {
            transform: None,
            entrypoint: Some("parity:unregistered:input".into()),
        },
        action(),
    )
    .unwrap();
    let err = resolve(
        &flat_tags(),
        &box_spec(1),
        &box_spec(1),
        &spec,
        ResolveOptions::default().trust_entrypoints(true),
    )
    .err()
    .unwrap();
    assert_eq!(err.class(), ErrorClass::AdapterResolutionError);
    assert!(
        err.to_string()
            .contains("unregistered custom input entrypoint")
    );
}

#[test]
fn unregister_compares_identity_and_keeps_resolved_routes_alive() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    struct Freed(Arc<AtomicUsize>);
    impl Drop for Freed {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let name = "parity:unregister:routes";
    let frees = Arc::new(AtomicUsize::new(0));
    let owner = Freed(frees.clone());
    let a = ObsTransform::from(|_: &Value| Ok(Value::Int(1)));
    let b = ObsTransform::from(move |_: &Value| {
        let _keep = &owner;
        Ok(Value::Int(2))
    });
    crate::ext::register_entrypoint(name, a.clone()).unwrap();
    crate::ext::register_entrypoint(name, b.clone()).unwrap();
    assert!(!crate::ext::unregister_entrypoint(name, Some(&a)).unwrap());
    let spec = ModelSpec::new(
        Custom {
            transform: None,
            entrypoint: Some(name.into()),
        },
        action(),
    )
    .unwrap();
    let mut route = resolve(
        &flat_tags(),
        &box_spec(1),
        &box_spec(1),
        &spec,
        ResolveOptions::default().trust_entrypoints(true),
    )
    .unwrap();
    // A cloned facade transform compares by Arc identity too.
    assert!(crate::ext::unregister_entrypoint(name, Some(&b.clone())).unwrap());
    drop(b);
    assert_eq!(frees.load(Ordering::SeqCst), 0);
    assert_eq!(route.transform_obs(floats(&[1.])).unwrap(), Value::Int(2));
    assert_eq!(
        resolve(
            &flat_tags(),
            &box_spec(1),
            &box_spec(1),
            &spec,
            ResolveOptions::default().trust_entrypoints(true)
        )
        .err()
        .unwrap()
        .class(),
        ErrorClass::AdapterResolutionError
    );
    assert!(!crate::ext::unregister_entrypoint(name, None).unwrap());
    drop(route);
    assert_eq!(frees.load(Ordering::SeqCst), 1);
    assert_eq!(
        crate::ext::unregister_entrypoint("", None)
            .unwrap_err()
            .class(),
        ErrorClass::ValueError
    );
}

#[test]
fn registry_replace_resolve_and_unregister_complete_safely_under_contention() {
    use std::sync::{Arc, Barrier};
    let name = "parity:unregister:concurrent";
    let a = ObsTransform::from(|_: &Value| Ok(Value::Int(1)));
    let b = ObsTransform::from(|_: &Value| Ok(Value::Int(2)));
    let barrier = Arc::new(Barrier::new(3));
    std::thread::scope(|scope| {
        let writer_start = barrier.clone();
        let (writer_a, writer_b) = (a.clone(), b.clone());
        scope.spawn(move || {
            writer_start.wait();
            for i in 0..200 {
                crate::ext::register_entrypoint(
                    name,
                    if i % 2 == 0 {
                        writer_a.clone()
                    } else {
                        writer_b.clone()
                    },
                )
                .unwrap();
            }
        });
        let remover_start = Arc::clone(&barrier);
        let expected = a.clone();
        scope.spawn(move || {
            remover_start.wait();
            for i in 0..200 {
                crate::ext::unregister_entrypoint(name, (i % 2 == 0).then_some(&expected)).unwrap();
            }
        });
        // Resolve clones under the same lock; execution needs no registry lock.
        barrier.wait();
        let spec = ModelSpec::new(
            Custom {
                transform: None,
                entrypoint: Some(name.into()),
            },
            action(),
        )
        .unwrap();
        for _ in 0..200 {
            match resolve(
                &flat_tags(),
                &box_spec(1),
                &box_spec(1),
                &spec,
                ResolveOptions::default().trust_entrypoints(true),
            ) {
                Ok(mut route) => assert!(matches!(
                    route.transform_obs(floats(&[1.])).unwrap(),
                    Value::Int(1 | 2)
                )),
                Err(error) => assert_eq!(error.class(), ErrorClass::AdapterResolutionError),
            }
        }
    });
    crate::ext::unregister_entrypoint(name, None).unwrap();
}

#[test]
fn registry_drops_removed_and_replaced_transforms_outside_its_lock() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    struct Reenter(Arc<AtomicUsize>);
    impl Drop for Reenter {
        fn drop(&mut self) {
            let nested = "parity:unregister:reentrant:nested";
            crate::ext::register_entrypoint(nested, ObsTransform::from(|v: &Value| Ok(v.clone())))
                .unwrap();
            assert!(crate::ext::unregister_entrypoint(nested, None).unwrap());
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let name = "parity:unregister:reentrant";
    let frees = Arc::new(AtomicUsize::new(0));
    for replace in [false, true] {
        let owner = Reenter(frees.clone());
        let transform = ObsTransform::from(move |v: &Value| {
            let _keep = &owner;
            Ok(v.clone())
        });
        crate::ext::register_entrypoint(name, transform).unwrap();
        if replace {
            crate::ext::register_entrypoint(name, ObsTransform::from(|v: &Value| Ok(v.clone())))
                .unwrap();
        } else {
            assert!(crate::ext::unregister_entrypoint(name, None).unwrap());
        }
    }
    assert_eq!(frees.load(Ordering::SeqCst), 2);
    crate::ext::unregister_entrypoint(name, None).unwrap();
}
