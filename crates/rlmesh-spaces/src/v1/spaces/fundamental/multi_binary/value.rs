use crate::errors::{SpaceError, err_space};
use crate::v1::multi_binary_spec;
use crate::v1::spaces::{SpaceSpec, SpaceValue, space_spec};

pub(crate) fn contains_multibinary(
    space: &SpaceSpec,
    value: &SpaceValue,
    path: &str,
) -> Result<(), SpaceError> {
    let vals = match value {
        SpaceValue::MultiBinary(v) => v,
        _ => return err_space!(path, "expected MultiBinary value"),
    };

    let mb = match &space.spec {
        Some(space_spec::Spec::MultiBinary(mb)) => mb,
        _ => return err_space!(path, "space is not MultiBinary"),
    };

    // Get expected size from the space
    let expected_size = match &mb.n {
        Some(multi_binary_spec::N::Size(n)) => *n as usize,
        Some(multi_binary_spec::N::Dims(dims)) => dims.data.iter().map(|&d| d as usize).product(),
        None => return err_space!(path, "MultiBinary.n not set"),
    };

    if vals.len() != expected_size {
        return err_space!(
            path,
            format!(
                "MultiBinary size mismatch: expected {}, got {}",
                expected_size,
                vals.len()
            )
        );
    }

    // Values are bools, always valid
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::v1::spaces::fundamental::MultiBinaryBuilder;
    use crate::v1::spaces::{SpaceValue, contains};

    #[test]
    fn test_multibinary_contains() {
        let space = MultiBinaryBuilder::scalar(3).build().unwrap();

        assert!(contains(&space, &SpaceValue::MultiBinary(vec![true, false, true])).is_ok());
        assert!(contains(&space, &SpaceValue::MultiBinary(vec![true, false])).is_err());
        assert!(contains(&space, &SpaceValue::Discrete(1)).is_err());
    }
}
