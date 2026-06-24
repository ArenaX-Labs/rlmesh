mod composite;
mod fundamental;
mod space;
mod value;
mod walk;

pub use crate::{SpaceKind, SpaceSpec, SpaceType};
pub use composite::{DictSpaceBuilder, TupleSpaceBuilder};
pub use fundamental::{
    BoxSpaceBuilder, DiscreteBuilder, MultiBinaryBuilder, MultiDiscreteBuilder, TextBuilder,
};
pub use space::validate_space;
pub(crate) use space::validate_space_at;
pub(crate) use value::conform_at;
pub use value::{Conformance, PolicyOutcome, SpaceValue, ValidationPolicy, conform, contains};
pub use walk::{assemble_value, flatten_leaves, leaf_specs};
