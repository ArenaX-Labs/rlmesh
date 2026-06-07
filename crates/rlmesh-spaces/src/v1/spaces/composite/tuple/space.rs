use crate::errors::{SpaceError, err_space};
use crate::v1::spaces::{SpaceSpec, space_spec, validate_space, validate_space_at};
use crate::v1::{DType, TupleSpec};

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
        spec: Some(space_spec::Spec::Tuple(TupleSpec { spaces })),
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
        Some(space_spec::Spec::Tuple(t)) => t,
        _ => return err_space!(path, "Tuple", "spec.tuple must be set"),
    };

    for (i, child) in t.spaces.iter().enumerate() {
        validate_space_at(child, &format!("{path}[{i}]"))?;
    }

    Ok(())
}
