use super::*;
#[test]
fn defaults_shorthands_and_validation_match_python() {
    assert_eq!(
        View::default().to_string(),
        "View(backend='terminal', port=8008, fps=30, source=None, format='jpeg', quality=75, hold=False, step_hz=None)"
    );
    for text in ["", " TTY ", "term", "terminal"] {
        assert_eq!(View::parse(text).unwrap(), View::default());
    }
    for text in ["http", "HTTP:", " HTTP:8008 "] {
        assert_eq!(View::parse(text).unwrap().backend, ViewBackend::Http);
    }
    assert_eq!("http:65535".parse::<View>().unwrap().port, 65535);
    assert_eq!(View::parse("both").unwrap().backend, ViewBackend::Both);
    for text in [
        "http:0",
        "http:65536",
        "http:+80",
        "http:-80",
        "http:8_0",
        "http: 80",
        "http:８０",
        "other",
    ] {
        assert_eq!(
            View::parse(text).unwrap_err().class(),
            crate::ErrorClass::ValueError,
            "{text}"
        );
    }
    for view in [
        View {
            port: 0,
            ..Default::default()
        },
        View {
            fps: 0,
            ..Default::default()
        },
        View {
            quality: 101,
            ..Default::default()
        },
        View {
            step_hz: Some(f64::INFINITY),
            ..Default::default()
        },
        View {
            step_hz: Some(f64::NAN),
            ..Default::default()
        },
        View {
            step_hz: Some(-1.),
            ..Default::default()
        },
    ] {
        assert!(view.validate().is_err());
    }
    assert_eq!(ViewHold::from(true), ViewHold::UntilQuit);
    assert_eq!(
        ViewHold::from(Duration::ZERO),
        ViewHold::For(Duration::ZERO)
    );
}
#[test]
fn normalization_matches_python_for_tensor_and_nested_images() {
    let tensor = crate::Tensor::new(
        &[0., 0.5, 1.]
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect::<Vec<_>>(),
        &[1, 1, 3],
        crate::DType::Float64,
    )
    .unwrap();
    assert_eq!(frame(&Value::Tensor(tensor)).unwrap().data, [0, 127, 255]);
    let nested = Value::List(vec![Value::List(vec![Value::List(vec![
        Value::Float(-1.),
        Value::Float(0.),
        Value::Float(1.),
    ])])]);
    assert_eq!(frame(&nested).unwrap().data, [0, 127, 255]);
    let uint8 = crate::Tensor::new(&[0, 1, 2], &[1, 1, 3], crate::DType::Uint8).unwrap();
    assert_eq!(frame(&Value::Tensor(uint8)).unwrap().data, [0, 1, 2]);
    let nan = crate::Tensor::new(
        &[f64::NAN, 0.5, 1.]
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect::<Vec<_>>(),
        &[1, 1, 3],
        crate::DType::Float64,
    )
    .unwrap();
    assert_eq!(frame(&Value::Tensor(nan)).unwrap().data, [0, 0, 0]);
    assert!(frame(&Value::List(vec![Value::Int(1)])).is_none());
    let empty = crate::Tensor::new(&[], &[0, 1, 3], crate::DType::Uint8).unwrap();
    assert!(frame(&Value::Tensor(empty)).is_none());
}
