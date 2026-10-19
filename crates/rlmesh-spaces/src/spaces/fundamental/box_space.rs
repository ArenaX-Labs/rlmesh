mod space;
mod value;

#[cfg(test)]
mod value_reference_tests;

pub use space::BoxSpaceBuilder;
pub(crate) use space::validate_box_at;
pub(crate) use value::conform_box;
