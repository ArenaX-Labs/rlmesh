//! Values, metadata dictionaries and environment info.

use indexmap::IndexMap;

use crate::{Error, Result, Tensor};

pub(crate) mod meta_codec;
pub(crate) mod space_codec;

/// A value exchanged with a model or environment.
///
/// Supported kinds: the six primitives, [`Tensor`], `list`, `tuple` and
/// `dict`. A Discrete action is an `Int`, a Box observation a `Tensor`;
/// `bool` is not an `Int`.
///
/// Reads are strict: [`as_float`](Self::as_float) refuses an `Int` and
/// [`as_int`](Self::as_int) refuses a `Bool`, each with the TypeError class.
/// Equality is kind-strict too: `Int(1) != Float(1.0)` and `Bool(true) !=
/// Int(1)`; a `Float` compares as IEEE `f64` (NaN is unequal to itself), a
/// `Dict` ignores order.
#[derive(Clone, PartialEq, Debug)]
#[non_exhaustive]
pub enum Value {
    /// `None`.
    None,
    /// `bool`.
    Bool(bool),
    /// `int`.
    Int(i64),
    /// `float`.
    Float(f64),
    /// `str`.
    Text(String),
    /// `bytes`.
    Bytes(Vec<u8>),
    /// `rlmesh.Tensor`.
    Tensor(Tensor),
    /// `list`.
    List(Vec<Value>),
    /// `tuple`.
    Tuple(Vec<Value>),
    /// `dict`, in insertion order.
    Dict(Metadata),
}

/// A value's kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ValueKind {
    /// `None`.
    None,
    /// `bool`.
    Bool,
    /// `int`.
    Int,
    /// `float`.
    Float,
    /// `str`.
    Text,
    /// `bytes`.
    Bytes,
    /// `rlmesh.Tensor`.
    Tensor,
    /// `list`.
    List,
    /// `tuple`.
    Tuple,
    /// `dict`.
    Dict,
}

impl ValueKind {
    /// The diagnostic type name (`"int"`, `"Tensor"`, `"NoneType"`).
    pub(crate) fn python_name(self) -> &'static str {
        match self {
            ValueKind::None => "NoneType",
            ValueKind::Bool => "bool",
            ValueKind::Int => "int",
            ValueKind::Float => "float",
            ValueKind::Text => "str",
            ValueKind::Bytes => "bytes",
            ValueKind::Tensor => "Tensor",
            ValueKind::List => "list",
            ValueKind::Tuple => "tuple",
            ValueKind::Dict => "dict",
        }
    }
}

impl Value {
    /// The value's kind.
    pub fn kind(&self) -> ValueKind {
        match self {
            Value::None => ValueKind::None,
            Value::Bool(_) => ValueKind::Bool,
            Value::Int(_) => ValueKind::Int,
            Value::Float(_) => ValueKind::Float,
            Value::Text(_) => ValueKind::Text,
            Value::Bytes(_) => ValueKind::Bytes,
            Value::Tensor(_) => ValueKind::Tensor,
            Value::List(_) => ValueKind::List,
            Value::Tuple(_) => ValueKind::Tuple,
            Value::Dict(_) => ValueKind::Dict,
        }
    }

    /// Whether this is `None`.
    pub fn is_none(&self) -> bool {
        matches!(self, Value::None)
    }

    /// A `Bool`'s value.
    pub fn as_bool(&self) -> Result<bool> {
        match self {
            Value::Bool(value) => Ok(*value),
            other => Err(other.wrong_kind(ValueKind::Bool)),
        }
    }

    /// An `Int`'s value; a `Bool` is refused.
    pub fn as_int(&self) -> Result<i64> {
        match self {
            Value::Int(value) => Ok(*value),
            other => Err(other.wrong_kind(ValueKind::Int)),
        }
    }

    /// A `Float`'s value; an `Int` is refused.
    pub fn as_float(&self) -> Result<f64> {
        match self {
            Value::Float(value) => Ok(*value),
            other => Err(other.wrong_kind(ValueKind::Float)),
        }
    }

    /// A `Text`'s value.
    pub fn as_text(&self) -> Result<&str> {
        match self {
            Value::Text(value) => Ok(value),
            other => Err(other.wrong_kind(ValueKind::Text)),
        }
    }

    /// A `Bytes`'s value.
    pub fn as_bytes(&self) -> Result<&[u8]> {
        match self {
            Value::Bytes(value) => Ok(value),
            other => Err(other.wrong_kind(ValueKind::Bytes)),
        }
    }

    /// A `Tensor`'s value.
    pub fn as_tensor(&self) -> Result<&Tensor> {
        match self {
            Value::Tensor(value) => Ok(value),
            other => Err(other.wrong_kind(ValueKind::Tensor)),
        }
    }

    /// A `List`'s items.
    pub fn as_list(&self) -> Result<&[Value]> {
        match self {
            Value::List(items) => Ok(items),
            other => Err(other.wrong_kind(ValueKind::List)),
        }
    }

    /// A `Tuple`'s items.
    pub fn as_tuple(&self) -> Result<&[Value]> {
        match self {
            Value::Tuple(items) => Ok(items),
            other => Err(other.wrong_kind(ValueKind::Tuple)),
        }
    }

    /// A `Dict`'s map.
    pub fn as_dict(&self) -> Result<&Metadata> {
        match self {
            Value::Dict(map) => Ok(map),
            other => Err(other.wrong_kind(ValueKind::Dict)),
        }
    }

    fn wrong_kind(&self, expected: ValueKind) -> Error {
        Error::Type(format!(
            "expected a {} value, got {}",
            expected.python_name(),
            self.kind().python_name()
        ))
    }
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Value::Bool(value)
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Value::Int(value)
    }
}

impl From<i32> for Value {
    fn from(value: i32) -> Self {
        Value::Int(i64::from(value))
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Value::Float(value)
    }
}

impl From<f32> for Value {
    fn from(value: f32) -> Self {
        Value::Float(f64::from(value))
    }
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Value::Text(value.to_string())
    }
}

impl From<String> for Value {
    fn from(value: String) -> Self {
        Value::Text(value)
    }
}

impl From<Vec<u8>> for Value {
    fn from(value: Vec<u8>) -> Self {
        Value::Bytes(value)
    }
}

impl From<&[u8]> for Value {
    fn from(value: &[u8]) -> Self {
        Value::Bytes(value.to_vec())
    }
}

impl From<Tensor> for Value {
    fn from(value: Tensor) -> Self {
        Value::Tensor(value)
    }
}

impl From<Vec<Value>> for Value {
    fn from(value: Vec<Value>) -> Self {
        Value::List(value)
    }
}

impl From<Metadata> for Value {
    fn from(value: Metadata) -> Self {
        Value::Dict(value)
    }
}

/// An insertion-ordered string → [`Value`] map.
///
/// Iteration follows insertion order; lookup and equality ignore order.
/// [`insert`](Self::insert) on an existing key replaces the value in place
/// and keeps its position; [`remove`](Self::remove) keeps the order of the
/// remaining keys; collecting pairs with a repeated key keeps the first
/// position and the last value.
#[derive(Clone, Debug, Default)]
pub struct Metadata(IndexMap<String, Value>);

/// An environment's `info` dictionary.
pub type InfoDict = Metadata;

/// Named construction or factory arguments.
pub type Kwargs = Metadata;

impl Metadata {
    /// An empty map.
    pub fn new() -> Metadata {
        Metadata(IndexMap::new())
    }

    /// The value at `key`.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.0.get(key)
    }

    /// The value at `key`, mutably.
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        self.0.get_mut(key)
    }

    /// Whether `key` is present (`key in d`).
    pub fn contains_key(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }

    /// Set `key` (`d[key] = value`), returning the value it replaced. An
    /// existing key keeps its position.
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<Value>) -> Option<Value> {
        self.0.insert(key.into(), value.into())
    }

    /// Remove `key` (`d.pop(key, None)`), keeping the order of the rest.
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        self.0.shift_remove(key)
    }

    /// The entries in insertion order.
    pub fn iter(&self) -> MetadataIter<'_> {
        MetadataIter(self.0.iter())
    }

    /// The keys in insertion order.
    pub fn keys(&self) -> impl ExactSizeIterator<Item = &str> + '_ {
        self.0.keys().map(String::as_str)
    }

    /// The values in insertion order.
    pub fn values(&self) -> impl ExactSizeIterator<Item = &Value> + '_ {
        self.0.values()
    }

    /// The number of entries.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Order-insensitive equality; values compare kind-strictly.
impl PartialEq for Metadata {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .all(|(key, value)| other.get(key) == Some(value))
    }
}

impl<K: Into<String>, V: Into<Value>> FromIterator<(K, V)> for Metadata {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut map = Metadata::new();
        for (key, value) in iter {
            map.insert(key, value);
        }
        map
    }
}

impl IntoIterator for Metadata {
    type Item = (String, Value);
    type IntoIter = indexmap::map::IntoIter<String, Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a Metadata {
    type Item = (&'a str, &'a Value);
    type IntoIter = MetadataIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// The entries of a [`Metadata`], in insertion order.
#[derive(Clone, Debug)]
pub struct MetadataIter<'a>(indexmap::map::Iter<'a, String, Value>);

impl<'a> Iterator for MetadataIter<'a> {
    type Item = (&'a str, &'a Value);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(key, value)| (key.as_str(), value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl ExactSizeIterator for MetadataIter<'_> {}

impl DoubleEndedIterator for MetadataIter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back().map(|(key, value)| (key.as_str(), value))
    }
}

#[cfg(test)]
mod tests;
