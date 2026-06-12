use crate::errors::{SpaceError, err_space};
use crate::spaces::{SpaceKind, SpaceSpec, SpaceValue, contains_at};

pub(crate) fn contains_dict(
    space: &SpaceSpec,
    value: &SpaceValue,
    path: &str,
) -> Result<(), SpaceError> {
    let dict_val = match value {
        SpaceValue::Dict(d) => d,
        _ => return err_space!(path, "expected Dict value"),
    };

    let d = match &space.spec {
        Some(SpaceKind::Dict(d)) => d,
        _ => return err_space!(path, "space is not Dict"),
    };

    for (i, key) in d.keys.iter().enumerate() {
        match dict_val.get(key) {
            Some(sub_val) => {
                contains_at(&d.spaces[i], sub_val, &format!("{path}.{key}"))?;
            }
            None => {
                return err_space!(path, format!("missing key '{}'", key));
            }
        }
    }

    for key in dict_val.keys() {
        if !d.keys.contains(key) {
            return err_space!(path, format!("unexpected key '{}'", key));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::DType;
    use crate::spaces::composite::DictSpaceBuilder;
    use crate::spaces::fundamental::{BoxSpaceBuilder, DiscreteBuilder};
    use crate::spaces::{SpaceValue, contains};
    use crate::tensor::Tensor;

    #[test]
    fn test_dict_contains() {
        let box_space = BoxSpaceBuilder::scalar(0.0, 1.0, vec![3]).build().unwrap();
        let discrete = DiscreteBuilder::new(4).build().unwrap();

        let space = DictSpaceBuilder::new()
            .insert("obs", box_space)
            .insert("action", discrete)
            .build()
            .unwrap();

        let valid = SpaceValue::Dict(BTreeMap::from([
            (
                "obs".to_string(),
                SpaceValue::Box(
                    Tensor::from_vec(vec![0u8; 12], vec![3], DType::Float32).expect("valid tensor"),
                ),
            ),
            ("action".to_string(), SpaceValue::Discrete(2)),
        ]));
        assert!(contains(&space, &valid).is_ok());

        let missing = SpaceValue::Dict(BTreeMap::from([(
            "obs".to_string(),
            SpaceValue::Box(
                Tensor::from_vec(vec![0u8; 12], vec![3], DType::Float32).expect("valid tensor"),
            ),
        )]));
        assert!(contains(&space, &missing).is_err());

        let extra = SpaceValue::Dict(BTreeMap::from([
            (
                "obs".to_string(),
                SpaceValue::Box(
                    Tensor::from_vec(vec![0u8; 12], vec![3], DType::Float32).expect("valid tensor"),
                ),
            ),
            ("action".to_string(), SpaceValue::Discrete(2)),
            ("extra".to_string(), SpaceValue::Discrete(0)),
        ]));
        assert!(contains(&space, &extra).is_err());
    }
}
