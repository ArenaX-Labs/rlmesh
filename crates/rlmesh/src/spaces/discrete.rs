//! `Discrete`.

use rlmesh_spaces::spaces::DiscreteBuilder;
use rlmesh_spaces::types::SpaceKind as WireKind;

use super::{Space, SpaceSpec, space_newtype};
use crate::{DType, Error, Result};

/// `Discrete`'s keyword arguments.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscreteOptions {
    /// The first value (`0`).
    pub start: i64,
    /// Element dtype: `Int64` (the default), `Int32` or `Uint8`.
    pub dtype: DType,
}

impl Default for DiscreteOptions {
    fn default() -> Self {
        Self {
            start: 0,
            dtype: DType::Int64,
        }
    }
}

/// The integers `start .. start + n`: Python's `rlmesh.spaces.Discrete`.
#[derive(Clone)]
#[repr(transparent)]
pub struct Discrete(Space);

space_newtype!(Discrete, Discrete);

impl Discrete {
    /// `Discrete(n, start, dtype)`. `n <= 0`, a dtype other than `Int64`,
    /// `Int32` or `Uint8`, or a range the dtype cannot hold is the ValueError
    /// class.
    pub fn new(n: i64, options: DiscreteOptions) -> Result<Discrete> {
        let spec = DiscreteBuilder::new(n)
            .start(options.start)
            .dtype(options.dtype.to_core())
            .build()
            .map_err(|err| Error::invalid_argument(err.to_string()))?;
        Ok(Discrete(Space::new(SpaceSpec::from_wire(spec))))
    }

    /// The number of values.
    pub fn n(&self) -> i64 {
        self.payload().0
    }

    /// The first value.
    pub fn start(&self) -> i64 {
        self.payload().1
    }

    fn payload(&self) -> (i64, i64) {
        match &self.spec().wire().spec {
            Some(WireKind::Discrete(spec)) => (spec.n, spec.start),
            _ => (0, 0),
        }
    }
}
