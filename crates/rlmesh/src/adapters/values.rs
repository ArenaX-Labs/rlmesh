use super::*;
use crate::{Error, Result};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
#[derive(Clone, Debug, PartialEq)]
pub enum Normalize {
    Off,
    Unit,
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
pub enum AxisValue {
    Scalar(f64),
    PerAxis(Vec<f64>),
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rotation {
    pub encoding: RotationEncoding,
    pub value: Vec<f64>,
}
impl Rotation {
    pub fn new(encoding: RotationEncoding, value: impl Into<Vec<f64>>) -> Self {
        Self {
            encoding,
            value: value.into(),
        }
    }
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
pub enum ConcatPart {
    Role(String),
    State(State),
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
pub struct PreviousOptions {
    pub part: Option<String>,
    pub fill: f64,
}
#[allow(non_snake_case)]
pub fn Previous(role: impl Into<String>, options: PreviousOptions) -> State {
    State {
        part: options.part,
        fill: options.fill,
        source: StateSource::Action,
        ..State::new(role)
    }
}
impl Action {
    pub fn dim(&self) -> u32 {
        self.components.iter().map(|c| c.dim).sum()
    }
}
impl Image {
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
    pub severity: AdvisorySeverity,
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
impl From<rlmesh_adapters::v1::Advisory> for Advisory {
    fn from(a: rlmesh_adapters::v1::Advisory) -> Self {
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
/// Python's resolution-error detail. The message lives on `InvalidArgumentError`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct AdapterResolutionError {}
impl AdapterResolutionError {
    pub fn new() -> Self {
        Self {}
    }
}
