use crate::MultiBinaryDims;
use crate::errors::{SpaceError, err_space};
use crate::spaces::{SpaceKind, SpaceSpec, SpaceValue};

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
        Some(SpaceKind::MultiBinary(mb)) => mb,
        _ => return err_space!(path, "space is not MultiBinary"),
    };

    // Get expected size from the space
    let expected_size = match &mb.n {
        Some(MultiBinaryDims::Size(n)) => *n as usize,
        Some(MultiBinaryDims::Dims(dims)) => dims.iter().map(|&d| d as usize).product(),
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
    use crate::spaces::fundamental::MultiBinaryBuilder;
    use crate::spaces::{SpaceValue, contains};

    #[test]
    fn test_multibinary_contains() {
        let space = MultiBinaryBuilder::scalar(3).build().unwrap();

        assert!(contains(&space, &SpaceValue::MultiBinary(vec![true, false, true])).is_ok());
        assert!(contains(&space, &SpaceValue::MultiBinary(vec![true, false])).is_err());
        assert!(contains(&space, &SpaceValue::Discrete(1)).is_err());
    }
}
