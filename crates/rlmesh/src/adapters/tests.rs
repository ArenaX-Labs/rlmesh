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
