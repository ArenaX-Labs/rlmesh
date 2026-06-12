use crate::errors::{SpaceError, err_space};
use crate::spaces::{SpaceSpec, SpaceValue, space_spec};

pub(crate) fn contains_discrete(
    space: &SpaceSpec,
    value: &SpaceValue,
    path: &str,
) -> Result<(), SpaceError> {
    let val = match value {
        SpaceValue::Discrete(v) => *v,
        _ => return err_space!(path, "expected Discrete value"),
    };

    let d = match &space.spec {
        Some(space_spec::Spec::Discrete(d)) => d,
        _ => return err_space!(path, "space is not Discrete"),
    };

    // Check value is in range [start, start + n)
    if val < d.start || val >= d.start + d.n {
        return err_space!(
            path,
            format!(
                "value {} not in range [{}, {})",
                val,
                d.start,
                d.start + d.n
            )
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::spaces::fundamental::DiscreteBuilder;
    use crate::spaces::{SpaceValue, contains};

    #[test]
    fn test_discrete_contains() {
        let space = DiscreteBuilder::new(4).build().unwrap();

        assert!(contains(&space, &SpaceValue::Discrete(0)).is_ok());
        assert!(contains(&space, &SpaceValue::Discrete(3)).is_ok());
        assert!(contains(&space, &SpaceValue::Discrete(4)).is_err());
        assert!(contains(&space, &SpaceValue::Discrete(-1)).is_err());
    }

    #[test]
    fn test_discrete_with_start() {
        let space = DiscreteBuilder::new(4).start(10).build().unwrap();

        assert!(contains(&space, &SpaceValue::Discrete(10)).is_ok());
        assert!(contains(&space, &SpaceValue::Discrete(13)).is_ok());
        assert!(contains(&space, &SpaceValue::Discrete(9)).is_err());
        assert!(contains(&space, &SpaceValue::Discrete(14)).is_err());
    }
}
