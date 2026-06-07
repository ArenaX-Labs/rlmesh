use crate::errors::{SpaceError, err_space};
use crate::v1::spaces::{SpaceSpec, SpaceValue, contains_at, space_spec};

pub(crate) fn contains_tuple(
    space: &SpaceSpec,
    value: &SpaceValue,
    path: &str,
) -> Result<(), SpaceError> {
    let tuple_val = match value {
        SpaceValue::Tuple(t) => t,
        _ => return err_space!(path, "expected Tuple value"),
    };

    let t = match &space.spec {
        Some(space_spec::Spec::Tuple(t)) => t,
        _ => return err_space!(path, "space is not Tuple"),
    };

    if tuple_val.len() != t.spaces.len() {
        return err_space!(
            path,
            format!(
                "tuple length mismatch: expected {}, got {}",
                t.spaces.len(),
                tuple_val.len()
            )
        );
    }

    for (i, (sub_space, sub_val)) in t.spaces.iter().zip(tuple_val.iter()).enumerate() {
        contains_at(sub_space, sub_val, &format!("{path}[{i}]"))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::v1::DType;
    use crate::v1::spaces::composite::TupleSpaceBuilder;
    use crate::v1::spaces::fundamental::{BoxSpaceBuilder, BoxValue, DiscreteBuilder};
    use crate::v1::spaces::{SpaceValue, contains};

    #[test]
    fn test_tuple_contains() {
        let box_space = BoxSpaceBuilder::scalar(0.0, 1.0, vec![3]).build().unwrap();
        let discrete = DiscreteBuilder::new(4).build().unwrap();

        let space = TupleSpaceBuilder::new()
            .with(box_space)
            .with(discrete)
            .build()
            .unwrap();

        let valid = SpaceValue::Tuple(vec![
            SpaceValue::Box(BoxValue::new(vec![0u8; 12], vec![3], DType::Float32)),
            SpaceValue::Discrete(2),
        ]);
        assert!(contains(&space, &valid).is_ok());

        // Wrong length
        let wrong_len = SpaceValue::Tuple(vec![SpaceValue::Discrete(2)]);
        assert!(contains(&space, &wrong_len).is_err());
    }
}
