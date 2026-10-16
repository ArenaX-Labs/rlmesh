use super::*;
use crate::{DType, ErrorClass};

#[test]
fn reads_are_strict_and_fail_with_type_error() {
    assert_eq!(Value::from(3).as_int().unwrap(), 3);
    assert_eq!(Value::from(1.5f32).as_float().unwrap(), 1.5);
    assert_eq!(Value::from("hi").as_text().unwrap(), "hi");
    assert_eq!(Value::from(vec![1u8, 2]).as_bytes().unwrap(), &[1, 2]);
    assert!(Value::None.is_none());
    let err = Value::Int(1).as_float().unwrap_err();
    assert_eq!(err.class(), ErrorClass::TypeError);
    assert_eq!(err.to_string(), "expected a float value, got int");
    assert_eq!(
        Value::Bool(true).as_int().unwrap_err().class(),
        ErrorClass::TypeError
    );
    assert!(Value::List(vec![]).as_tuple().is_err());
    assert_eq!(Value::Tuple(vec![Value::None]).kind(), ValueKind::Tuple);
}

#[test]
fn equality_is_kind_strict() {
    assert_ne!(Value::Int(1), Value::Float(1.0));
    assert_ne!(Value::Bool(true), Value::Int(1));
    assert_ne!(Value::Float(f64::NAN), Value::Float(f64::NAN));
    assert_ne!(
        Value::List(vec![Value::Int(1)]),
        Value::Tuple(vec![Value::Int(1)])
    );
    let tensor = Tensor::new(&[1, 0], &[2], DType::Bool).unwrap();
    assert_eq!(Value::from(tensor.clone()), Value::from(tensor.copy()));
}

#[test]
fn metadata_keeps_insertion_order_and_compares_without_it() {
    let mut map = Metadata::new();
    map.insert("b", 1);
    map.insert("a", 2);
    map.insert("c", 3);
    assert_eq!(map.keys().collect::<Vec<_>>(), ["b", "a", "c"]);
    // Replacing keeps the position.
    assert_eq!(map.insert("b", 9), Some(Value::Int(1)));
    assert_eq!(map.keys().collect::<Vec<_>>(), ["b", "a", "c"]);
    // Removing shifts, never swaps.
    assert_eq!(map.remove("b"), Some(Value::Int(9)));
    assert_eq!(map.keys().collect::<Vec<_>>(), ["a", "c"]);
    assert!(map.contains_key("a") && !map.contains_key("b"));
    *map.get_mut("a").unwrap() = Value::from("x");
    assert_eq!(map.get("a"), Some(&Value::from("x")));

    let reordered: Metadata = [("c", Value::Int(3)), ("a", Value::from("x"))]
        .into_iter()
        .collect();
    assert_eq!(map, reordered);
    assert_ne!(
        map,
        [("c", Value::Float(3.0)), ("a", Value::from("x"))]
            .into_iter()
            .collect::<Metadata>()
    );
}

#[test]
fn collecting_a_repeated_key_is_dict_of_pairs() {
    let map: Metadata = [("k", 1), ("j", 2), ("k", 3)].into_iter().collect();
    assert_eq!(map.len(), 2);
    assert_eq!(
        map.iter().collect::<Vec<_>>(),
        [("k", &Value::Int(3)), ("j", &Value::Int(2))]
    );
    let owned: Vec<(String, Value)> = map.clone().into_iter().collect();
    assert_eq!(owned[0].0, "k");
    assert_eq!(
        format!("{map:?}"),
        r#"Metadata({"k": Int(3), "j": Int(2)})"#
    );
}
