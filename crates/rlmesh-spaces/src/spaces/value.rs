//! Space values and conformance.
//!
//! [`SpaceValue`] is the runtime value carried by a space. [`conform`] checks it
//! against a [`SpaceSpec`] in one pass, separating structural deviations (always
//! rejected) from range deviations the serving [`ValidationPolicy`] governs.

use std::collections::BTreeMap;

use crate::errors::SpaceError;
use crate::spaces::composite::{conform_dict, conform_tuple};
use crate::spaces::fundamental::{
    conform_box, conform_text, contains_discrete, contains_multibinary, contains_multidiscrete,
};
use crate::spaces::{SpaceSpec, SpaceType};
use crate::tensor::Tensor;

/// Runtime value carried by an RLMesh space.
#[derive(Debug, Clone, PartialEq)]
pub enum SpaceValue {
    /// Continuous tensor.
    Box(Tensor),

    /// Single integer.
    Discrete(i64),

    /// Boolean array.
    MultiBinary(Vec<bool>),

    /// Integer array.
    MultiDiscrete(Vec<i64>),

    /// String value.
    Text(String),

    /// Named child values.
    Dict(BTreeMap<String, SpaceValue>),

    /// Ordered child values.
    Tuple(Vec<SpaceValue>),
}

/// Classified outcome of checking a value against a space.
///
/// Structural deviations — wrong shape, dtype, arity, or domain, a missing Dict
/// key, a NaN element — are always rejected. Range deviations — Box bounds, Text
/// charset and length — are governed by the serving side's validation policy.
/// Structural outranks range: a structural deviation anywhere in the value masks
/// any range deviation, so a NaN element is never delivered just because some
/// other element happened to be out of bounds.
#[derive(Debug)]
pub enum Conformance {
    /// The value is a full member of the space.
    Ok,
    /// A deviation that must be rejected regardless of policy.
    Structural(SpaceError),
    /// A bounds/charset/length deviation the serving side may tolerate.
    Range(SpaceError),
}

impl Conformance {
    #[must_use]
    pub fn is_ok(&self) -> bool {
        matches!(self, Conformance::Ok)
    }

    fn into_result(self) -> Result<(), SpaceError> {
        match self {
            Conformance::Ok => Ok(()),
            Conformance::Structural(err) | Conformance::Range(err) => Err(err),
        }
    }
}

/// Validate that `value` belongs to `space`: structural and range conformance
/// must both hold. This is the strict check used wherever a value must be a full
/// member of its space (sampling, tests, internal invariants); the serving side
/// instead inspects [`conform`] so it can apply its range policy.
pub fn contains(space: &SpaceSpec, value: &SpaceValue) -> Result<(), SpaceError> {
    conform(space, value).into_result()
}

/// Classify `value` against `space` in a single traversal, separating structural
/// deviations (always rejected) from range deviations (policy-governed).
pub fn conform(space: &SpaceSpec, value: &SpaceValue) -> Conformance {
    conform_at(space, value, "$")
}

pub(crate) fn conform_at(space: &SpaceSpec, value: &SpaceValue, path: &str) -> Conformance {
    match space.space_type() {
        SpaceType::Box => conform_box(space, value, path),
        SpaceType::Discrete => structural(contains_discrete(space, value, path)),
        SpaceType::MultiBinary => structural(contains_multibinary(space, value, path)),
        SpaceType::MultiDiscrete => structural(contains_multidiscrete(space, value, path)),
        SpaceType::Text => conform_text(space, value, path),
        SpaceType::Dict => conform_dict(space, value, path),
        SpaceType::Tuple => conform_tuple(space, value, path),
        SpaceType::Unspecified => {
            Conformance::Structural(SpaceError::invalid(path, "space type not specified"))
        }
    }
}

/// Lift an all-structural validator (one that produces no range deviations) into
/// a [`Conformance`].
fn structural(result: Result<(), SpaceError>) -> Conformance {
    match result {
        Ok(()) => Conformance::Ok,
        Err(err) => Conformance::Structural(err),
    }
}

/// The serving side's policy for **range** deviations (Box bounds, Text charset
/// and length). Structural deviations are always rejected and are unaffected by
/// the policy. Observations and actions share one policy; the default is `Warn`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValidationPolicy {
    /// Deliver a range-deviant value and report a conformance warning. The default.
    #[default]
    Warn,
    /// Reject a range-deviant value, as if it were structural.
    Strict,
    /// Skip range checks entirely; structural conformance still applies.
    Off,
}

/// What the serving side should do with a value, given its [`Conformance`] and the
/// active [`ValidationPolicy`].
#[derive(Debug)]
pub enum PolicyOutcome {
    /// Deliver the value with no warning.
    Accept,
    /// Deliver the value but report this range deviation as a conformance warning.
    Warn(SpaceError),
    /// Reject the value: a structural deviation, or a range deviation under `Strict`.
    Reject(SpaceError),
}

impl ValidationPolicy {
    /// Resolve an already-classified [`Conformance`] under this policy.
    #[must_use]
    pub fn evaluate(self, conformance: Conformance) -> PolicyOutcome {
        match conformance {
            Conformance::Ok => PolicyOutcome::Accept,
            // Structural deviations are policy-immune: always rejected.
            Conformance::Structural(err) => PolicyOutcome::Reject(err),
            Conformance::Range(err) => match self {
                ValidationPolicy::Warn => PolicyOutcome::Warn(err),
                ValidationPolicy::Strict => PolicyOutcome::Reject(err),
                ValidationPolicy::Off => PolicyOutcome::Accept,
            },
        }
    }

    /// Classify `value` against `space` and resolve the outcome under this policy.
    #[must_use]
    pub fn check(self, space: &SpaceSpec, value: &SpaceValue) -> PolicyOutcome {
        self.evaluate(conform(space, value))
    }
}
