use crate::errors::{SpaceError, err_space};
use crate::v1::multi_discrete_spec;
use crate::v1::spaces::{SpaceSpec, SpaceValue, space_spec};

pub(crate) fn contains_multidiscrete(
    space: &SpaceSpec,
    value: &SpaceValue,
    path: &str,
) -> Result<(), SpaceError> {
    let vals = match value {
        SpaceValue::MultiDiscrete(v) => v,
        _ => return err_space!(path, "expected MultiDiscrete value"),
    };

    let md = match &space.spec {
        Some(space_spec::Spec::MultiDiscrete(md)) => md,
        _ => return err_space!(path, "space is not MultiDiscrete"),
    };

    // Get nvec from the space
    let nvec: Vec<i64> = match &md.nvec {
        Some(multi_discrete_spec::Nvec::Flat(v)) => v.data.clone(),
        Some(multi_discrete_spec::Nvec::Shaped(m)) => {
            m.data.iter().flat_map(|row| row.data.clone()).collect()
        }
        None => return err_space!(path, "MultiDiscrete.nvec not set"),
    };

    if vals.len() != nvec.len() {
        return err_space!(
            path,
            format!(
                "MultiDiscrete size mismatch: expected {}, got {}",
                nvec.len(),
                vals.len()
            )
        );
    }

    // Check each value is in range [0, nvec[i])
    for (i, (&val, &n)) in vals.iter().zip(nvec.iter()).enumerate() {
        if val < 0 || val >= n {
            return err_space!(
                path,
                format!("value[{}] = {} not in range [0, {})", i, val, n)
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::v1::spaces::fundamental::MultiDiscreteBuilder;
    use crate::v1::spaces::{SpaceValue, contains};

    #[test]
    fn test_multidiscrete_contains() {
        let space = MultiDiscreteBuilder::vector(vec![2, 3]).build().unwrap();

        assert!(contains(&space, &SpaceValue::MultiDiscrete(vec![0, 2])).is_ok());
        assert!(contains(&space, &SpaceValue::MultiDiscrete(vec![1])).is_err());
        assert!(contains(&space, &SpaceValue::MultiDiscrete(vec![2, 0])).is_err());
        assert!(contains(&space, &SpaceValue::Discrete(1)).is_err());
    }
}
