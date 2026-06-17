use crate::errors::{SpaceError, err_space};
use crate::spaces::{
    Conformance, SpaceKind, SpaceSpec, SpaceValue, conform_at, validate_space, validate_space_at,
};
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

pub(crate) fn conform_dict(space: &SpaceSpec, value: &SpaceValue, path: &str) -> Conformance {
    let dict_val = match value {
        SpaceValue::Dict(d) => d,
        _ => return Conformance::Structural(SpaceError::invalid(path, "expected Dict value")),
    };

    let d = match &space.spec {
        Some(SpaceKind::Dict(d)) => d,
        _ => return Conformance::Structural(SpaceError::invalid(path, "space is not Dict")),
    };

    // A structural deviation in any child outranks a range deviation anywhere, so
    // return immediately on the first structural one but keep the first range one.
    let mut range: Option<SpaceError> = None;
    for (i, key) in d.keys.iter().enumerate() {
        match dict_val.get(key) {
            Some(sub_val) => match conform_at(&d.spaces[i], sub_val, &format!("{path}.{key}")) {
                Conformance::Structural(err) => return Conformance::Structural(err),
                Conformance::Range(err) => {
                    if range.is_none() {
                        range = Some(err);
                    }
                }
                Conformance::Ok => {}
            },
            None => {
                return Conformance::Structural(SpaceError::invalid(
                    path,
                    format!("missing key '{key}'"),
                ));
            }
        }
    }

    for key in dict_val.keys() {
        if !d.keys.contains(key) {
            return Conformance::Structural(SpaceError::invalid(
                path,
                format!("unexpected key '{key}'"),
            ));
        }
    }

    match range {
        Some(err) => Conformance::Range(err),
        None => Conformance::Ok,
    }
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
