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

use serde::{Deserialize, Serialize};

/// Which way a gripper moves as its value rises: `open` (a higher command
/// opens it, a higher reading is more open: openness) or `close` (closedness).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Polarity {
    /// A higher value is more open.
    Open,
    /// A higher value is more closed.
    Close,
}

impl Polarity {
    /// Its wire spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Close => "close",
        }
    }

    /// The other polarity.
    pub fn flipped(self) -> Self {
        match self {
            Self::Open => Self::Close,
            Self::Close => Self::Open,
        }
    }

    /// The polarity after a side's own sign corrections: `invert` and a
    /// negative `scale` each reverse it.
    pub fn through(self, invert: bool, scale: Option<f64>) -> Self {
        if invert ^ scale.is_some_and(|scale| scale < 0.0) {
            self.flipped()
        } else {
            self
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::Polarity;

    #[test]
    fn polarity_is_open_or_close_and_reverses_through_a_sign_correction() {
        let open: Polarity = serde_json::from_str(r#""open""#).unwrap();
        assert_eq!(open, Polarity::Open);
        assert!(serde_json::from_str::<Polarity>(r#""opened""#).is_err());
        assert_eq!(open.through(true, None), Polarity::Close);
        assert_eq!(open.through(true, Some(-2.0)), Polarity::Open);
        assert_eq!(open.through(false, Some(0.5)), Polarity::Open);
    }
}
