//! The value conventions a gripper and a delta command carry: `polarity` and `unit`.
//!
//! Each side declares its own convention and the resolver compares them; a difference is refused
//! unless the model spec allows the projection that converts it (`flip_gripper`,
//! `convert_units`). A side that declares nothing is not checked. Both describe a side's raw
//! value: the model's output before its own `invert` and `scale`, and the env's input at its
//! declared range.

pub use rlmesh_session_core::spec::role::input::Polarity;

pub(crate) fn check_unit(unit: Option<f64>, locus: &str) -> Result<(), String> {
    match unit {
        Some(unit) if !(unit.is_finite() && unit > 0.0) => Err(format!(
            "{locus}: unit must be a positive number (the SI size of one command unit), got {unit}"
        )),
        _ => Ok(()),
    }
}
