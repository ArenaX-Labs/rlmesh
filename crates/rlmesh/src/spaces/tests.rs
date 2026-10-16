use super::*;
use crate::ErrorClass;
use crate::ext::SpaceExt;

fn unit_box() -> Box {
    Box::new(-1.0, 1.0, [3], BoxOptions::default()).expect("box")
}

#[test]
fn shape_example_round_trips() {
    let pos = unit_box();
    let id = Discrete::new(
        4,
        DiscreteOptions {
            start: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let obs = Dict::new([("pos", SpaceSpec::from(&pos)), ("id", SpaceSpec::from(&id))]).unwrap();
    assert_eq!(obs.seed(Some(7)), 7);
    let x = obs.sample().unwrap();
    assert!(obs.contains(&x));
    // Keys sort at construction, as Python's builder does.
    assert_eq!(obs.keys().collect::<Vec<_>>(), ["id", "pos"]);
    assert_eq!(
        obs["pos"].as_box().unwrap().low(),
        Some(BoxBound::Uniform(Bound::Float(-1.0)))
    );
    assert_eq!(obs.len(), 2);
    assert!(obs.contains_key("id") && !obs.contains_key("nope"));
    assert!(obs.get("nope").is_none());
    assert_eq!((&obs).into_iter().rev().collect::<Vec<_>>(), ["pos", "id"]);
    assert_eq!(obs.items().len(), 2);
}

#[test]
fn seeding_reproduces_and_clones_share_the_rng() {
    let space = unit_box();
    let other = space.clone();
    space.seed(Some(3));
    let a = space.sample().unwrap();
    other.seed(Some(3));
    let b = space.sample().unwrap();
    assert_eq!(a, b, "a clone seeds the same RNG");
    // A fresh space from the same spec with the same seed matches too.
    let fresh = space.spec().to_space();
    fresh.seed(Some(3));
    assert_eq!(fresh.sample().unwrap(), a);
    // `seed(None)` draws a seed, applies it and returns it.
    let drawn = space.seed(None);
    let first = space.sample().unwrap();
    space.seed(Some(drawn));
    assert_eq!(space.sample().unwrap(), first);
}

#[test]
fn dict_children_keep_their_own_rng_and_are_shared() {
    let obs = Dict::new([("a", SpaceSpec::from(&unit_box()))]).unwrap();
    let child = obs["a"].clone();
    child.seed(Some(9));
    let expected = child.sample().unwrap();
    child.seed(Some(9));
    obs.seed(Some(1));
    let _ = obs.sample().unwrap();
    // The parent's seed and sample leave the child's stream alone, and a
    // clone of the Dict sees the same child.
    let clone = obs.clone();
    assert_eq!(clone["a"].sample().unwrap(), expected);
}

#[test]
fn integer_bounds_are_exact_and_read_in_the_dtype_domain() {
    let opts = |dtype| BoxOptions { dtype };
    let u = Box::new(0u64, u64::MAX, [2], opts(DType::Uint64)).unwrap();
    assert_eq!(u.bounds_kind(), Some(BoundsKind::TypedUniform));
    assert_eq!(u.high(), Some(BoxBound::Uniform(Bound::Uint(u64::MAX))));
    assert_eq!(u.low(), Some(BoxBound::Uniform(Bound::Uint(0))));
    let i = Box::new(-5, 5.0, [1], opts(DType::Int32)).unwrap();
    assert_eq!(i.low(), Some(BoxBound::Uniform(Bound::Int(-5))));
    assert_eq!(i.high(), Some(BoxBound::Uniform(Bound::Int(5))));
    let b = Box::new(false, true, [1], opts(DType::Bool)).unwrap();
    assert_eq!(b.high(), Some(BoxBound::Uniform(Bound::Bool(true))));
    let open = Box::new(f64::NEG_INFINITY, f64::INFINITY, [1], opts(DType::Int64)).unwrap();
    assert_eq!(open.bounds_kind(), Some(BoundsKind::Unbounded));
    assert_eq!(open.low(), None);
}

#[test]
fn box_refuses_fractional_integer_bounds_and_nan() {
    let int32 = BoxOptions {
        dtype: DType::Int32,
    };
    for (low, high) in [(0.5, 1.0), (0.0, 3.7), (0.5, f64::INFINITY)] {
        let err = Box::new(low, high, [1], int32.clone()).unwrap_err();
        assert_eq!(err.class(), ErrorClass::ValueError, "{low}..{high}");
    }
    let err = Box::new(f64::NAN, 1.0, [1], BoxOptions::default()).unwrap_err();
    assert_eq!(err.class(), ErrorClass::ValueError);
}

#[test]
fn from_spec_and_downcasts_check_the_kind() {
    let spec = SpaceSpec::from(&unit_box());
    let err = Discrete::from_spec(spec.clone()).unwrap_err();
    assert_eq!(err.class(), ErrorClass::ValueError);
    assert_eq!(err.to_string(), "expected discrete SpaceSpec, got 'box'");
    assert!(Dict::from_spec(spec.clone()).is_err());
    let space = Space::new(spec.clone());
    assert!(space.as_box().is_some());
    assert!(space.as_dict().is_none());
    let boxed = Box::try_from(space.clone()).unwrap();
    assert_eq!(boxed, space);
    assert_eq!(space, boxed);
    assert!(Discrete::try_from(space).is_err());
    let back: Space = boxed.into();
    assert_eq!(back.kind(), SpaceKind::Box);
}

#[test]
fn equality_is_by_spec_and_dict_order_insensitive() {
    assert_eq!(unit_box(), unit_box());
    let a = Discrete::new(3, DiscreteOptions::default()).unwrap();
    assert!(*a != *unit_box());
    let d1 = Dict::new([
        ("x", SpaceSpec::from(&a)),
        ("y", SpaceSpec::from(&unit_box())),
    ])
    .unwrap();
    let d2 = Dict::new([
        ("y", SpaceSpec::from(&unit_box())),
        ("x", SpaceSpec::from(&a)),
    ])
    .unwrap();
    assert_eq!(d1, d2);
}

#[test]
fn display_is_gymnasium_style() {
    assert_eq!(unit_box().to_string(), "Box(-1.0, 1.0, (3,), float32)");
    let d = Discrete::new(
        4,
        DiscreteOptions {
            start: -1,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(d.to_string(), "Discrete(4, start=-1)");
    assert_eq!((d.n(), d.start()), (4, -1));
    assert_eq!(d.dtype(), DType::Int64);
}

#[test]
fn discrete_refuses_other_dtypes() {
    let err = Discrete::new(
        3,
        DiscreteOptions {
            dtype: DType::Float32,
            ..Default::default()
        },
    )
    .unwrap_err();
    assert_eq!(err.class(), ErrorClass::ValueError);
}

#[test]
fn dict_refuses_duplicate_and_empty_keys() {
    let spec = SpaceSpec::from(&unit_box());
    assert!(Dict::new([("a", spec.clone()), ("a", spec.clone())]).is_err());
    assert!(Dict::new([("", spec)]).is_err());
}

#[test]
fn unknown_specs_hold_but_do_not_sample() {
    let spec = SpaceSpec::from_wire(rlmesh_spaces::SpaceSpec::default());
    assert_eq!(spec.kind(), SpaceKind::Unknown);
    assert_eq!(spec.dtype(), DType::Float32);
    let space = spec.to_space();
    assert_eq!(space.sample().unwrap_err().class(), ErrorClass::ValueError);
    assert!(!space.contains(&Value::Int(0)));
    assert!(space_from_spec(spec).is_err());
}

#[test]
fn contains_detail_explains_a_refusal() {
    let space = unit_box();
    assert!(!space.contains(&Value::Int(0)));
    assert!(space.contains_detail(&Value::Int(0)).is_err());
    let sample = space.sample().unwrap();
    assert!(space.contains_detail(&sample).is_ok());
}

#[test]
fn kind_spellings_match_python() {
    assert_eq!(SpaceKind::MultiBinary.as_str(), "multi_binary");
    assert_eq!(
        BoundsKind::TypedElementwise.to_string(),
        "typed_elementwise"
    );
}

#[test]
fn samples_are_values_of_python_kinds() {
    let pos = unit_box();
    let id = Discrete::new(4, DiscreteOptions::default()).unwrap();
    let obs = Dict::new([("pos", SpaceSpec::from(&pos)), ("id", SpaceSpec::from(&id))]).unwrap();
    obs.seed(Some(1));
    let value = obs.sample().unwrap();
    let map = value.as_dict().unwrap();
    // A Dict samples in the space's declaration order.
    assert_eq!(
        map.keys().collect::<Vec<_>>(),
        obs.keys().collect::<Vec<_>>()
    );
    let tensor = map.get("pos").unwrap().as_tensor().unwrap();
    assert_eq!(
        (tensor.shape(), tensor.dtype()),
        (&[3usize][..], DType::Float32)
    );
    let n = map.get("id").unwrap().as_int().unwrap();
    assert!((0..4).contains(&n));
}

#[test]
fn contains_takes_the_values_python_accepts() {
    let ints = Box::new(
        0,
        5,
        [2],
        BoxOptions {
            dtype: DType::Int32,
        },
    )
    .unwrap();
    assert!(ints.contains(&Value::List(vec![Value::Int(1), Value::Bool(true)])));
    // Integer dtypes never take a float, even an integral one.
    assert!(!ints.contains(&Value::List(vec![Value::Float(1.0), Value::Int(1)])));
    // The exact shape: a bare number is no member of a (2,) Box.
    assert!(!ints.contains(&Value::Int(1)));
    assert!(!ints.contains(&Value::List(vec![Value::Int(9), Value::Int(1)])));
    let floats = unit_box();
    assert!(floats.contains(&Value::Tuple(vec![
        Value::Int(0),
        Value::Float(0.5),
        Value::Bool(false)
    ])));
    let wrong_dtype = crate::Tensor::new(&[0; 12], &[3], DType::Int32).unwrap();
    assert!(!floats.contains(&Value::Tensor(wrong_dtype)));
    let flags = Box::new(false, true, [1], BoxOptions { dtype: DType::Bool }).unwrap();
    assert!(flags.contains(&Value::List(vec![Value::Float(0.5)])));

    let id = Discrete::new(
        3,
        DiscreteOptions {
            start: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(id.contains(&Value::Int(3)));
    assert!(id.contains(&Value::Bool(true)));
    assert!(id.contains(&Value::Float(2.0)));
    assert!(!id.contains(&Value::Float(2.5)));
    assert!(!id.contains(&Value::Int(0)));
    let one = crate::Tensor::new(&2i64.to_le_bytes(), &[1], DType::Int64).unwrap();
    assert!(id.contains(&Value::Tensor(one)));
    assert!(!id.contains(&Value::from("2")));

    let obs = Dict::new([("id", SpaceSpec::from(&id))]).unwrap();
    let extra: crate::Metadata = [("id", Value::Int(1)), ("other", Value::None)]
        .into_iter()
        .collect();
    assert!(obs.contains(&Value::Dict(extra)));
    assert!(!obs.contains(&Value::Dict(crate::Metadata::new())));
}
