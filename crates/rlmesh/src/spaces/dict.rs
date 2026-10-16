//! `Dict` and its read-only mapping.

use std::collections::HashSet;
use std::iter::FusedIterator;
use std::ops::Index;

use rlmesh_spaces::SpaceSpec as WireSpec;
use rlmesh_spaces::spaces::DictSpaceBuilder;
use rlmesh_spaces::types::SpaceKind as WireKind;

use super::{Space, SpaceSpec, space_newtype};
use crate::{Error, Result};

/// Named child spaces: Python's `rlmesh.spaces.Dict`.
///
/// Read it as a mapping in the spec's declaration order (a Dict built here
/// sorts its keys, as Python's does; a remote spec keeps its own order).
/// Each child is a space of its own with its own sampler, shared by every
/// handle of this Dict: seeding or sampling the Dict leaves them alone.
#[derive(Clone)]
#[repr(transparent)]
pub struct Dict(Space);

space_newtype!(Dict, Dict);

impl Dict {
    /// `Dict(spaces)`: each child's spec is copied into a fresh child space.
    /// A duplicate or empty key is the ValueError class.
    pub fn new<K: Into<String>, S: Into<SpaceSpec>>(
        spaces: impl IntoIterator<Item = (K, S)>,
    ) -> Result<Dict> {
        let mut seen = HashSet::new();
        let mut builder = DictSpaceBuilder::new();
        for (key, spec) in spaces {
            let key = key.into();
            if !seen.insert(key.clone()) {
                return Err(Error::invalid_argument(format!(
                    "Dict keys must be unique; duplicate: {key}"
                )));
            }
            builder = builder.insert(key, spec.into().wire().clone());
        }
        let spec = builder
            .build()
            .map_err(|err| Error::invalid_argument(err.to_string()))?;
        Ok(Dict(Space::new(SpaceSpec::from_wire(spec))))
    }

    /// The children as a read-only mapping (Python's `.spaces`).
    pub fn spaces(&self) -> &DictSpaces {
        self.0.dict_children()
    }

    /// The keys, in declaration order.
    pub fn keys(&self) -> DictKeys<'_> {
        self.spaces().keys()
    }

    /// The child spaces, in declaration order.
    pub fn values(&self) -> DictValues<'_> {
        self.spaces().values()
    }

    /// The `(key, space)` pairs, in declaration order.
    pub fn items(&self) -> DictItems<'_> {
        self.spaces().items()
    }

    /// The keys (Python's `iter(d)`).
    pub fn iter(&self) -> DictKeys<'_> {
        self.spaces().iter()
    }

    /// The child space at `key` (`d[key]`), or `None`.
    pub fn get(&self, key: &str) -> Option<&Space> {
        self.spaces().get(key)
    }

    /// The number of children (`len(d)`).
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.spaces().len()
    }

    /// Whether `key` names a child (`key in d`).
    pub fn contains_key(&self, key: &str) -> bool {
        self.spaces().contains_key(key)
    }
}

/// `d[key]`; panics when the key is absent (use [`Dict::get`]).
impl Index<&str> for Dict {
    type Output = Space;

    fn index(&self, key: &str) -> &Space {
        &self.spaces()[key]
    }
}

impl<'a> IntoIterator for &'a Dict {
    type Item = &'a str;
    type IntoIter = DictKeys<'a>;

    fn into_iter(self) -> DictKeys<'a> {
        self.iter()
    }
}

/// A Dict's children: an ordered, read-only map (Python's
/// `MappingProxyType`).
pub struct DictSpaces {
    keys: Vec<String>,
    values: Vec<Space>,
}

impl DictSpaces {
    /// The child spaces of a Dict spec (empty for another kind).
    pub(super) fn of(spec: &WireSpec) -> DictSpaces {
        match &spec.spec {
            Some(WireKind::Dict(dict)) => DictSpaces {
                keys: dict.keys.clone(),
                values: dict
                    .spaces
                    .iter()
                    .map(|child| Space::new(SpaceSpec::from_wire(child.clone())))
                    .collect(),
            },
            _ => DictSpaces {
                keys: Vec::new(),
                values: Vec::new(),
            },
        }
    }

    /// The keys, in declaration order.
    pub fn keys(&self) -> DictKeys<'_> {
        DictKeys(self.keys.iter())
    }

    /// The child spaces, in declaration order.
    pub fn values(&self) -> DictValues<'_> {
        DictValues(self.values.iter())
    }

    /// The `(key, space)` pairs, in declaration order.
    pub fn items(&self) -> DictItems<'_> {
        DictItems(self.keys.iter().zip(self.values.iter()))
    }

    /// The keys.
    pub fn iter(&self) -> DictKeys<'_> {
        self.keys()
    }

    /// The child space at `key`, or `None`.
    pub fn get(&self, key: &str) -> Option<&Space> {
        let index = self.keys.iter().position(|k| k == key)?;
        self.values.get(index)
    }

    /// The number of children.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether `key` names a child.
    pub fn contains_key(&self, key: &str) -> bool {
        self.keys.iter().any(|k| k == key)
    }
}

impl Index<&str> for DictSpaces {
    type Output = Space;

    fn index(&self, key: &str) -> &Space {
        match self.get(key) {
            Some(space) => space,
            None => panic!("no Dict key {key:?}"),
        }
    }
}

impl<'a> IntoIterator for &'a DictSpaces {
    type Item = &'a str;
    type IntoIter = DictKeys<'a>;

    fn into_iter(self) -> DictKeys<'a> {
        self.keys()
    }
}

/// A Dict's keys, in declaration order.
#[derive(Clone)]
pub struct DictKeys<'a>(std::slice::Iter<'a, String>);

impl<'a> Iterator for DictKeys<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        self.0.next().map(String::as_str)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl DoubleEndedIterator for DictKeys<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back().map(String::as_str)
    }
}

impl ExactSizeIterator for DictKeys<'_> {}
impl FusedIterator for DictKeys<'_> {}

/// A Dict's child spaces, in declaration order.
#[derive(Clone)]
pub struct DictValues<'a>(std::slice::Iter<'a, Space>);

impl<'a> Iterator for DictValues<'a> {
    type Item = &'a Space;

    fn next(&mut self) -> Option<&'a Space> {
        self.0.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl DoubleEndedIterator for DictValues<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }
}

impl ExactSizeIterator for DictValues<'_> {}
impl FusedIterator for DictValues<'_> {}

/// A Dict's `(key, space)` pairs, in declaration order.
#[derive(Clone)]
pub struct DictItems<'a>(std::iter::Zip<std::slice::Iter<'a, String>, std::slice::Iter<'a, Space>>);

impl<'a> Iterator for DictItems<'a> {
    type Item = (&'a str, &'a Space);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(key, space)| (key.as_str(), space))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl DoubleEndedIterator for DictItems<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back().map(|(key, space)| (key.as_str(), space))
    }
}

impl ExactSizeIterator for DictItems<'_> {}
impl FusedIterator for DictItems<'_> {}
