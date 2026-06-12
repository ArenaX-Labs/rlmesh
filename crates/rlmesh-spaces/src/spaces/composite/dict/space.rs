use crate::errors::{SpaceError, err_space};
use crate::spaces::{SpaceKind, SpaceSpec, validate_space, validate_space_at};
use crate::{DType, DictSpec};
use std::collections::BTreeMap;
use std::collections::HashSet;

#[must_use = "a space builder does nothing until .build() is called"]
pub struct DictSpaceBuilder {
    entries: BTreeMap<String, SpaceSpec>,
}

impl Default for DictSpaceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DictSpaceBuilder {
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    pub fn insert(mut self, key: impl Into<String>, space: SpaceSpec) -> Self {
        self.entries.insert(key.into(), space);
        self
    }

    pub fn extend<I, K>(mut self, entries: I) -> Self
    where
        I: IntoIterator<Item = (K, SpaceSpec)>,
        K: Into<String>,
    {
        for (k, v) in entries {
            self.entries.insert(k.into(), v);
        }
        self
    }

    pub fn build(self) -> Result<SpaceSpec, SpaceError> {
        let (keys, spaces): (Vec<String>, Vec<SpaceSpec>) = self.entries.into_iter().unzip();

        make_dict_space(keys, spaces)
    }
}

fn make_dict_space(keys: Vec<String>, spaces: Vec<SpaceSpec>) -> Result<SpaceSpec, SpaceError> {
    let spec = SpaceSpec {
        shape: vec![],
        dtype: DType::Unspecified,
        spec: Some(SpaceKind::Dict(DictSpec { keys, spaces })),
    };

    validate_space(&spec)?;
    Ok(spec)
}

pub(crate) fn validate_dict_at(spec: &SpaceSpec, path: &str) -> Result<(), SpaceError> {
    if !spec.shape.is_empty() {
        return err_space!(path, "Dict", "shape must be empty");
    }

    if spec.dtype != DType::Unspecified {
        return err_space!(path, "Dict", "dtype must be 'UNSPECIFIED'");
    }

    let d = match &spec.spec {
        Some(SpaceKind::Dict(d)) => d,
        _ => return err_space!(path, "Dict", "spec.dict must be set"),
    };

    if d.keys.len() != d.spaces.len() {
        return err_space!(path, "Dict", "keys/spaces length mismatch");
    }

    let mut seen = HashSet::with_capacity(d.keys.len());

    for i in 0..d.keys.len() {
        let key = &d.keys[i];

        if key.is_empty() {
            return err_space!(path, "Dict", "keys must be non-empty");
        }
        if !seen.insert(key.clone()) {
            return err_space!(
                path,
                "Dict",
                format!("keys must be unique; duplicate: {key}")
            );
        }

        let child = &d.spaces[i];
        validate_space_at(child, &format!("{path}.{key}"))?;
    }

    Ok(())
}
