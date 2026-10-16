use super::*;
use crate::{Error, Result};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
#[derive(Clone, Debug, PartialEq)]
/// A typed adapter vocabulary or tree alternative.
pub enum Normalize {
    /// The `Off` alternative.
    Off,
    /// The `Unit` alternative.
    Unit,
    /// The `Range` alternative.
    Range(f64, f64),
}
impl From<bool> for Normalize {
    fn from(v: bool) -> Self {
        if v { Self::Unit } else { Self::Off }
    }
}
impl From<(f64, f64)> for Normalize {
    fn from(v: (f64, f64)) -> Self {
        Self::Range(v.0, v.1)
    }
}
impl Serialize for Normalize {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Self::Off => false.serialize(s),
            Self::Unit => true.serialize(s),
            Self::Range(a, b) => (*a, *b).serialize(s),
        }
    }
}
impl<'de> Deserialize<'de> for Normalize {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum N {
            Bool(bool),
            Range((f64, f64)),
        }
        Ok(match N::deserialize(d)? {
            N::Bool(b) => b.into(),
            N::Range(r) => r.into(),
        })
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
/// A typed adapter vocabulary or tree alternative.
pub enum AxisValue {
    /// The `Scalar` alternative.
    Scalar(f64),
    /// The `PerAxis` alternative.
    PerAxis(Vec<f64>),
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// A literal rotation, authored as a model state part's `post_rotate`.
pub struct Rotation {
    /// Rotation encoding `value` is written in.
    pub encoding: RotationEncoding,
    /// The rotation's components in that encoding.
    pub value: Vec<f64>,
}
impl Rotation {
    /// Construct a rotation from components in the given encoding.
    pub fn new(encoding: RotationEncoding, value: impl Into<Vec<f64>>) -> Self {
        Self {
            encoding,
            value: value.into(),
        }
    }
    /// Use a matrix’s first two columns as a rot6d literal.
    pub fn from_matrix(rows: [[f64; 3]; 3]) -> Self {
        Self::new(
            RotationEncoding::Rot6d,
            (0..2)
                .flat_map(|c| (0..3).map(move |r| rows[r][c]))
                .collect::<Vec<_>>(),
        )
    }
}
#[derive(Clone, Debug, PartialEq)]
/// A typed adapter vocabulary or tree alternative.
#[allow(
    clippy::large_enum_variant,
    reason = "Python’s ConcatPart stores an authored State value by value"
)]
pub enum ConcatPart {
    /// The `Role` alternative.
    Role(String),
    /// The `State` alternative.
    State(State),
    /// The `Constant` alternative.
    Constant(Constant),
}
impl From<&str> for ConcatPart {
    fn from(s: &str) -> Self {
        Self::Role(s.into())
    }
}
impl From<String> for ConcatPart {
    fn from(s: String) -> Self {
        Self::Role(s)
    }
}
impl From<State> for ConcatPart {
    fn from(s: State) -> Self {
        Self::State(s)
    }
}
impl From<Constant> for ConcatPart {
    fn from(s: Constant) -> Self {
        Self::Constant(s)
    }
}
#[derive(Clone, Debug, Default)]
/// Options for the `Previous` adapter. Build with `..Default::default()`;
/// adding an options field is a minor change.
pub struct PreviousOptions {
    /// The desired body part, or `None` for the whole role (default).
    pub part: Option<String>,
    /// The value before the first action; defaults to zero.
    pub fill: f64,
}
#[allow(non_snake_case)]
/// Select values from a previous step.
pub fn Previous(role: impl Into<String>, options: PreviousOptions) -> State {
    State {
        part: options.part,
        fill: options.fill,
        source: StateSource::Action,
        ..State::new(role)
    }
}
impl Action {
    /// Total declared action width.
    pub fn dim(&self) -> u32 {
        self.components.iter().map(|c| c.dim).sum()
    }
}
impl Image {
    /// Set both image dimensions, refusing an already set dimension.
    pub fn with_size(mut self, size: u32) -> Result<Self> {
        if self.height.is_some() || self.width.is_some() {
            return Err(Error::invalid_argument(
                "Image: pass size, or height/width, not both",
            ));
        }
        self.height = Some(size);
        self.width = Some(size);
        Ok(self)
    }
    /// Expand an evenly spaced frame window from the current stack depth.
    pub fn with_stride(mut self, stride: u32) -> Result<Self> {
        if stride == 0 || self.offsets.is_some() {
            return Err(Error::invalid_argument(
                "Image: stride must be >= 1 and cannot combine with offsets",
            ));
        }
        self.offsets = Some(
            (0..self.stack)
                .map(|i| i64::from(self.stack - 1 - i) * -i64::from(stride))
                .map(|i| {
                    i32::try_from(i)
                        .map_err(|_| Error::invalid_argument("Image stride overflows offsets"))
                })
                .collect::<Result<_>>()?,
        );
        Ok(self)
    }
}
/// Non-fatal information from adapter resolution.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Advisory {
    /// The advisory's severity.
    pub severity: AdvisorySeverity,
    /// The advisory's message.
    pub message: String,
}
impl std::fmt::Display for Advisory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::fmt::Debug for Advisory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Advisory(severity={:?}, message={:?})",
            self.severity.as_str(),
            self.message
        )
    }
}
impl Advisory {
    pub(crate) fn from_core(a: rlmesh_adapters::v1::Advisory) -> Self {
        Self {
            severity: if a.severity == rlmesh_adapters::v1::AdvisorySeverity::Info {
                AdvisorySeverity::Info
            } else {
                AdvisorySeverity::Caution
            },
            message: a.message,
        }
    }
}
/// Adapter-resolution error details. The message lives on
/// `InvalidArgumentError`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct AdapterResolutionError {}
impl AdapterResolutionError {
    /// Construct empty resolution-error details.
    pub fn new() -> Self {
        Self {}
    }
}

impl PreviousOptions {
    /// Set the desired body part.
    pub fn part(mut self, part: impl Into<String>) -> Self {
        self.part = Some(part.into());
        self
    }
    /// Set the value before the first action.
    pub fn fill(mut self, fill: f64) -> Self {
        self.fill = fill;
        self
    }
}
