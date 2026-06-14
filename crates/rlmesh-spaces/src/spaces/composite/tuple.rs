use crate::errors::{SpaceError, err_space};
use crate::spaces::{
    SpaceKind, SpaceSpec, SpaceValue, contains_at, validate_space, validate_space_at,
};
use crate::{DType, TupleSpec};

#[must_use = "a space builder does nothing until .build() is called"]
pub struct TupleSpaceBuilder {
    spaces: Vec<SpaceSpec>,
}

impl Default for TupleSpaceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TupleSpaceBuilder {
    pub fn new() -> Self {
        Self { spaces: Vec::new() }
    }

    pub fn with(mut self, space: SpaceSpec) -> Self {
        self.spaces.push(space);
        self
    }

    pub fn extend<I: IntoIterator<Item = SpaceSpec>>(mut self, spaces: I) -> Self {
        self.spaces.extend(spaces);
        self
    }

    pub fn build(self) -> Result<SpaceSpec, SpaceError> {
        make_tuple_space(self.spaces)
    }
}

fn make_tuple_space(spaces: Vec<SpaceSpec>) -> Result<SpaceSpec, SpaceError> {
    let spec = SpaceSpec {
        shape: vec![],
        dtype: DType::Unspecified,
        spec: Some(SpaceKind::Tuple(TupleSpec { spaces })),
    };

    validate_space(&spec)?;
    Ok(spec)
}

pub(crate) fn validate_tuple_at(spec: &SpaceSpec, path: &str) -> Result<(), SpaceError> {
    if !spec.shape.is_empty() {
        return err_space!(path, "Tuple", "shape must be empty");
    }
    if spec.dtype != DType::Unspecified {
        return err_space!(path, "Tuple", "dtype must be 'UNSPECIFIED'");
    }

    let t = match &spec.spec {
        Some(SpaceKind::Tuple(t)) => t,
        _ => return err_space!(path, "Tuple", "spec.tuple must be set"),
    };

    for (i, child) in t.spaces.iter().enumerate() {
        validate_space_at(child, &format!("{path}[{i}]"))?;
    }

    Ok(())
}

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
        Some(SpaceKind::Tuple(t)) => t,
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
    use crate::DType;
    use crate::spaces::composite::TupleSpaceBuilder;
    use crate::spaces::fundamental::{BoxSpaceBuilder, DiscreteBuilder};
    use crate::spaces::{SpaceValue, contains};
    use crate::tensor::Tensor;

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
            SpaceValue::Box(
                Tensor::from_vec(vec![0u8; 12], vec![3], DType::Float32).expect("valid tensor"),
            ),
            SpaceValue::Discrete(2),
        ]);
        assert!(contains(&space, &valid).is_ok());

        // Wrong length
        let wrong_len = SpaceValue::Tuple(vec![SpaceValue::Discrete(2)]);
        assert!(contains(&space, &wrong_len).is_err());
    }
}
