mod dict;
mod tuple;

pub use dict::DictSpaceBuilder;
pub(crate) use dict::{conform_dict, validate_dict_at};
pub use tuple::TupleSpaceBuilder;
pub(crate) use tuple::{conform_tuple, validate_tuple_at};
