//! The value conventions a gripper and a delta command carry: `polarity` and
//! `unit`.
//!
//! Two sides can agree on a role, a width and a range and still disagree on
//! what the numbers mean: one env's `+1` gripper command opens and another's
//! closes, one env's `1.0` position delta is 5 cm and another's is a metre.
//! Each side declares its own convention, the resolver compares them, and a
//! difference is refused unless the model spec allows the named projection
//! that converts it (`flip_gripper`, `convert_units`). A side that declares
//! nothing is not checked.
//!
//! Both describe a side's *raw* value: the model's output as it emits it
//! (its own `invert` and `scale` are corrections on top), and the env's
//! input at its declared range, before the env's own corrections.

pub use rlmesh_session_core::spec::role::input::Polarity;

/// Validates a declared `unit`: the SI size (metres, or radians for a
/// rotation) of one unit of a delta command. Finite and positive.
pub(crate) fn check_unit(unit: Option<f64>, locus: &str) -> Result<(), String> {
    match unit {
        Some(unit) if !(unit.is_finite() && unit > 0.0) => Err(format!(
            "{locus}: unit must be a positive number (the SI size of one command unit), got {unit}"
        )),
        _ => Ok(()),
    }
}
