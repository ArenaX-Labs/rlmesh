//! Spaces: Python's `rlmesh.spaces`.
//!
//! A [`SpaceSpec`] is immutable data describing a space. A [`Space`] is a
//! spec plus its sampler, a shared handle whose clones drive one RNG.
//! [`Box`], [`Discrete`] and [`Dict`] are typed views of a [`Space`] (they
//! deref to it) that add their kind's accessors.
//!
//! ```
//! use rlmesh::spaces::{Box, BoxOptions, Dict, Discrete, DiscreteOptions, SpaceSpec};
//!
//! # fn main() -> rlmesh::Result<()> {
//! let pos = Box::new(-1.0, 1.0, [3], BoxOptions::default())?;
//! let id = Discrete::new(4, DiscreteOptions { start: 1, ..Default::default() })?;
//! let obs = Dict::new([("pos", SpaceSpec::from(&pos)), ("id", SpaceSpec::from(&id))])?;
//! obs.seed(Some(7));
//! let x = obs.sample()?;
//! assert!(obs.contains(&x));
//! # Ok(())
//! # }
//! ```
//!
//! This module is not in the prelude: [`spaces::Box`](Box) shadows `std`'s
//! `Box`.
//!
//! Values are [`Value`]s: a Box samples a `Tensor`, a Discrete an `Int`, a
//! Dict a dict in its declaration order, a Text a `Text`.

use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

use rand::SeedableRng;
use rlmesh_spaces::types::SpaceKind as WireKind;
use rlmesh_spaces::{ChaCha12Rng, SpaceSpec as WireSpec, SpaceType};

use crate::value::space_codec;
use crate::{DType, Error, Result, Value};

pub(crate) mod box_space;
mod cast;
mod dict;
mod discrete;

pub use box_space::{Bound, BoundsKind, Box, BoxBound, BoxOptions};
pub use dict::{Dict, DictItems, DictKeys, DictSpaces, DictValues};
pub use discrete::{Discrete, DiscreteOptions};

/// A space's kind: Python's `kind` strings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SpaceKind {
    /// `"box"`.
    Box,
    /// `"discrete"`.
    Discrete,
    /// `"dict"`.
    Dict,
    /// `"multi_binary"` (Experimental).
    MultiBinary,
    /// `"multi_discrete"` (Experimental).
    MultiDiscrete,
    /// `"text"` (Experimental).
    Text,
    /// `"tuple"` (Experimental).
    Tuple,
    /// `"unknown"`: a spec with no payload.
    Unknown,
}

impl SpaceKind {
    /// Python's spelling (`"multi_binary"`).
    pub fn as_str(self) -> &'static str {
        match self {
            SpaceKind::Box => "box",
            SpaceKind::Discrete => "discrete",
            SpaceKind::Dict => "dict",
            SpaceKind::MultiBinary => "multi_binary",
            SpaceKind::MultiDiscrete => "multi_discrete",
            SpaceKind::Text => "text",
            SpaceKind::Tuple => "tuple",
            SpaceKind::Unknown => "unknown",
        }
    }

    pub(crate) fn of(spec: &WireSpec) -> SpaceKind {
        match spec.space_type() {
            SpaceType::Box => SpaceKind::Box,
            SpaceType::Discrete => SpaceKind::Discrete,
            SpaceType::Dict => SpaceKind::Dict,
            SpaceType::MultiBinary => SpaceKind::MultiBinary,
            SpaceType::MultiDiscrete => SpaceKind::MultiDiscrete,
            SpaceType::Text => SpaceKind::Text,
            SpaceType::Tuple => SpaceKind::Tuple,
            SpaceType::Unspecified => SpaceKind::Unknown,
        }
    }
}

impl fmt::Display for SpaceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Immutable data describing a space: Python's `SpaceSpec`. Clones share.
///
/// Get one from a space (`SpaceSpec::from(&space)`), a remote contract, or
/// [`Space::spec`]; [`to_space`](Self::to_space) makes a fresh space from it.
/// There is no `PartialEq`: Python compares specs by identity (compare
/// spaces instead).
#[derive(Clone)]
pub struct SpaceSpec(Arc<WireSpec>);

impl SpaceSpec {
    pub(crate) fn from_wire(spec: WireSpec) -> SpaceSpec {
        SpaceSpec(Arc::new(spec))
    }

    pub(crate) fn wire(&self) -> &WireSpec {
        &self.0
    }

    /// The space's kind.
    pub fn kind(&self) -> SpaceKind {
        SpaceKind::of(&self.0)
    }

    /// The element shape (empty for scalars and composites).
    pub fn shape(&self) -> &[i64] {
        &self.0.shape
    }

    /// The element dtype; a space without one (a Dict) reads `Float32`.
    pub fn dtype(&self) -> DType {
        DType::from_core(self.0.dtype)
    }

    /// A fresh space for this spec, with its own sampler.
    pub fn to_space(&self) -> Space {
        Space::new(self.clone())
    }
}

/// Gymnasium style, as Python's `repr`: `Box(-1.0, 1.0, (3,), float32)`.
impl fmt::Display for SpaceSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&*self.0, f)
    }
}

impl fmt::Debug for SpaceSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SpaceSpec({self})")
    }
}

struct Inner {
    spec: SpaceSpec,
    rng: Mutex<ChaCha12Rng>,
    /// A Dict's child spaces, built on first access and shared by every
    /// handle of this space.
    children: OnceLock<DictSpaces>,
}

/// A space: a spec plus its sampler (Python's `Space`).
///
/// Clones share one handle, so two names drive one RNG, as in Python. `seed`,
/// `sample` and `contains` may run concurrently.
#[derive(Clone)]
pub struct Space(Arc<Inner>);

impl Space {
    /// A space for `spec`, seeded from OS entropy. Accepts every kind,
    /// including a spec with no payload (which cannot be sampled).
    pub fn new(spec: SpaceSpec) -> Space {
        Space(Arc::new(Inner {
            spec,
            rng: Mutex::new(ChaCha12Rng::seed_from_u64(rand::random())),
            children: OnceLock::new(),
        }))
    }

    /// The space's spec.
    pub fn spec(&self) -> &SpaceSpec {
        &self.0.spec
    }

    /// The space's kind.
    pub fn kind(&self) -> SpaceKind {
        self.0.spec.kind()
    }

    /// The element shape.
    pub fn shape(&self) -> &[i64] {
        self.0.spec.shape()
    }

    /// The element dtype.
    pub fn dtype(&self) -> DType {
        self.0.spec.dtype()
    }

    /// Restart the sample stream from `seed` and return it; `None` draws a
    /// seed from OS entropy. A Dict's children keep their own streams.
    pub fn seed(&self, seed: Option<u64>) -> u64 {
        let seed = seed.unwrap_or_else(rand::random);
        *self.rng() = ChaCha12Rng::seed_from_u64(seed);
        seed
    }

    /// Draw a value. Fails (ValueError class) for a spec that cannot be
    /// sampled, such as one with no payload.
    pub fn sample(&self) -> Result<Value> {
        let wire = self.sample_wire()?;
        Ok(space_codec::value_from_space(self.0.spec.wire(), wire))
    }

    /// Whether `value` is a member of this space; never fails. A Box takes a
    /// `Tensor` of its dtype or a number or nested list of numbers of its
    /// exact shape; a Discrete an `Int`, a `Bool`, an integral `Float` or a
    /// one-element `Tensor`; a Dict a dict holding every key (others are
    /// ignored); a Tuple a tuple or list.
    pub fn contains(&self, value: &Value) -> bool {
        space_codec::space_contains(self.0.spec.wire(), value)
    }

    pub(crate) fn sample_wire(&self) -> Result<rlmesh_spaces::SpaceValue> {
        rlmesh_spaces::sample_with(self.0.spec.wire(), &mut *self.rng())
            .map_err(|err| Error::invalid_argument(err.to_string()))
    }

    /// This space as a `Box`, when it is one.
    pub fn as_box(&self) -> Option<&Box> {
        (self.kind() == SpaceKind::Box).then(|| cast::as_box(self))
    }

    /// This space as a `Discrete`, when it is one.
    pub fn as_discrete(&self) -> Option<&Discrete> {
        (self.kind() == SpaceKind::Discrete).then(|| cast::as_discrete(self))
    }

    /// This space as a `Dict`, when it is one.
    pub fn as_dict(&self) -> Option<&Dict> {
        (self.kind() == SpaceKind::Dict).then(|| cast::as_dict(self))
    }

    fn rng(&self) -> std::sync::MutexGuard<'_, ChaCha12Rng> {
        // A panic while sampling must not brick the RNG.
        self.0.rng.lock().unwrap_or_else(|err| err.into_inner())
    }

    fn dict_children(&self) -> &DictSpaces {
        self.0
            .children
            .get_or_init(|| DictSpaces::of(self.0.spec.wire()))
    }

    fn require_kind(self, kind: SpaceKind) -> Result<Space> {
        if self.kind() == kind {
            Ok(self)
        } else {
            Err(wrong_kind(kind, self.kind()))
        }
    }
}

/// The ValueError a `from_spec` raises for a spec of another kind.
fn wrong_kind(expected: SpaceKind, actual: SpaceKind) -> Error {
    // Python's text: `expected dict SpaceSpec, got 'box'`.
    Error::invalid_argument(format!("expected {expected} SpaceSpec, got '{actual}'"))
}

/// Normalized spec equality (Python's `spec_to_dict` comparison): Dict
/// children compare by key, whatever their order.
fn spec_eq(a: &WireSpec, b: &WireSpec) -> bool {
    if a.shape != b.shape || DType::from_core(a.dtype) != DType::from_core(b.dtype) {
        return false;
    }
    match (&a.spec, &b.spec) {
        (Some(WireKind::Dict(a)), Some(WireKind::Dict(b))) => {
            a.keys.len() == b.keys.len()
                && a.keys.iter().zip(&a.spaces).all(|(key, child)| {
                    b.keys
                        .iter()
                        .position(|other| other == key)
                        .is_some_and(|index| spec_eq(child, &b.spaces[index]))
                })
        }
        (Some(WireKind::Tuple(a)), Some(WireKind::Tuple(b))) => {
            a.spaces.len() == b.spaces.len()
                && a.spaces.iter().zip(&b.spaces).all(|(a, b)| spec_eq(a, b))
        }
        (a, b) => a == b,
    }
}

/// Spaces compare by spec, as Python's `Space.__eq__`; there is no `Hash`.
impl PartialEq for Space {
    fn eq(&self, other: &Self) -> bool {
        spec_eq(self.0.spec.wire(), other.0.spec.wire())
    }
}

impl fmt::Display for Space {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0.spec, f)
    }
}

impl fmt::Debug for Space {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0.spec, f)
    }
}

impl From<Space> for SpaceSpec {
    fn from(space: Space) -> Self {
        space.0.spec.clone()
    }
}

impl From<&Space> for SpaceSpec {
    fn from(space: &Space) -> Self {
        space.0.spec.clone()
    }
}

/// The typed space for `spec` (Python's `space_from_spec`). Refuses a spec
/// with no payload (`unknown`) with the ValueError class.
pub fn space_from_spec(spec: SpaceSpec) -> Result<Space> {
    if spec.kind() == SpaceKind::Unknown {
        return Err(Error::invalid_argument(
            "unsupported RLMesh space kind: unknown",
        ));
    }
    Ok(Space::new(spec))
}

/// The newtype pattern shared by `Box`, `Discrete` and `Dict`: deref to
/// `Space`, delegate the traits, convert both ways.
macro_rules! space_newtype {
    ($ty:ident, $kind:ident) => {
        impl std::ops::Deref for $ty {
            type Target = crate::spaces::Space;

            fn deref(&self) -> &crate::spaces::Space {
                &self.0
            }
        }

        impl std::fmt::Display for $ty {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Display::fmt(&self.0, f)
            }
        }

        impl std::fmt::Debug for $ty {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Debug::fmt(&self.0, f)
            }
        }

        impl PartialEq for $ty {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }

        impl PartialEq<crate::spaces::Space> for $ty {
            fn eq(&self, other: &crate::spaces::Space) -> bool {
                self.0 == *other
            }
        }

        impl PartialEq<$ty> for crate::spaces::Space {
            fn eq(&self, other: &$ty) -> bool {
                *self == other.0
            }
        }

        impl From<$ty> for crate::spaces::Space {
            fn from(value: $ty) -> Self {
                value.0
            }
        }

        impl From<$ty> for crate::spaces::SpaceSpec {
            fn from(value: $ty) -> Self {
                value.0.spec().clone()
            }
        }

        impl From<&$ty> for crate::spaces::SpaceSpec {
            fn from(value: &$ty) -> Self {
                value.0.spec().clone()
            }
        }

        /// Downcast, keeping the handle and its RNG; another kind is the
        /// ValueError class.
        impl TryFrom<crate::spaces::Space> for $ty {
            type Error = crate::Error;

            fn try_from(space: crate::spaces::Space) -> crate::Result<Self> {
                space.require_kind(crate::spaces::SpaceKind::$kind).map($ty)
            }
        }

        impl $ty {
            /// A space for `spec`; a spec of another kind is the ValueError
            /// class (Python's constructor given a spec).
            pub fn from_spec(spec: crate::spaces::SpaceSpec) -> crate::Result<Self> {
                Self::try_from(crate::spaces::Space::new(spec))
            }
        }
    };
}
use space_newtype;

#[cfg(test)]
mod tests;
