use crate::dtype::DType;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct UniformBounds {
    pub low: f64,
    pub high: f64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ElementwiseBounds {
    pub low: Vec<f64>,
    pub high: Vec<f64>,
}

/// A single low/high bound pair carried as raw little-endian bytes in the
/// enclosing space's dtype (one scalar each).
///
/// `low`/`high` must each be exactly `dtype_size(dtype)` bytes. Used so that
/// integer dtypes (notably `int64`/`uint64`) carry exact bounds instead of
/// losing precision through `f64`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TypedUniformBounds {
    pub low: Vec<u8>,
    pub high: Vec<u8>,
}

/// Per-element low/high bounds carried as raw little-endian bytes in the
/// enclosing space's dtype, in row-major (C) order.
///
/// `low`/`high` must each be exactly `numel * dtype_size(dtype)` bytes.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TypedElementwiseBounds {
    pub low: Vec<u8>,
    pub high: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BoxSpec {
    pub bounds: Option<BoxBounds>,
}

/// Bounds for a Box space (the proto `BoxBounds.bounds` oneof).
#[derive(Debug, Clone, PartialEq)]
pub enum BoxBounds {
    Unbounded(bool),
    Uniform(UniformBounds),
    Elementwise(ElementwiseBounds),
    TypedUniform(TypedUniformBounds),
    TypedElementwise(TypedElementwiseBounds),
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DiscreteSpec {
    pub n: i64,
    pub start: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MultiBinarySpec {
    pub n: Option<MultiBinaryDims>,
}

/// Size description for a MultiBinary space (the proto `n` oneof).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultiBinaryDims {
    Size(i64),
    Dims(Vec<i64>),
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MultiDiscreteSpec {
    pub nvec: Option<MultiDiscreteNvec>,
}

/// Count layout for a MultiDiscrete space (the proto `nvec` oneof).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultiDiscreteNvec {
    Flat(Vec<i64>),
    Shaped(Vec<Vec<i64>>),
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TextSpec {
    pub min_length: i64,
    pub max_length: i64,
    pub charset: String,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct DictSpec {
    pub keys: Vec<String>,
    pub spaces: Vec<SpaceSpec>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct TupleSpec {
    pub spaces: Vec<SpaceSpec>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SpaceSpec {
    pub shape: Vec<i64>,
    pub dtype: DType,
    pub spec: Option<SpaceKind>,
}

impl SpaceSpec {
    pub fn space_type(&self) -> SpaceType {
        match self.spec {
            Some(SpaceKind::Box(_)) => SpaceType::Box,
            Some(SpaceKind::Discrete(_)) => SpaceType::Discrete,
            Some(SpaceKind::MultiBinary(_)) => SpaceType::MultiBinary,
            Some(SpaceKind::MultiDiscrete(_)) => SpaceType::MultiDiscrete,
            Some(SpaceKind::Text(_)) => SpaceType::Text,
            Some(SpaceKind::Dict(_)) => SpaceType::Dict,
            Some(SpaceKind::Tuple(_)) => SpaceType::Tuple,
            None => SpaceType::Unspecified,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SpaceKind {
    Box(BoxSpec),
    Discrete(DiscreteSpec),
    MultiBinary(MultiBinarySpec),
    MultiDiscrete(MultiDiscreteSpec),
    Text(TextSpec),
    Dict(DictSpec),
    Tuple(TupleSpec),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum SpaceType {
    #[default]
    Unspecified = 0,
    Box = 1,
    Discrete = 2,
    MultiBinary = 3,
    MultiDiscrete = 4,
    Text = 5,
    Dict = 10,
    Tuple = 11,
}

impl TryFrom<i32> for SpaceType {
    type Error = &'static str;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Unspecified),
            1 => Ok(Self::Box),
            2 => Ok(Self::Discrete),
            3 => Ok(Self::MultiBinary),
            4 => Ok(Self::MultiDiscrete),
            5 => Ok(Self::Text),
            10 => Ok(Self::Dict),
            11 => Ok(Self::Tuple),
            _ => Err("invalid space type"),
        }
    }
}

impl From<SpaceType> for i32 {
    fn from(value: SpaceType) -> Self {
        value as i32
    }
}

/// Per-lane autoreset convention an environment follows (mirrors the proto
/// `AutoresetMode` and gymnasium's `AutoresetMode`). There is intentionally no
/// `Unspecified` variant: the proto `UNSPECIFIED` (0) decodes to `Disabled`, the
/// safe explicit-reset default. An unknown value is NOT folded to a default —
/// [`AutoresetMode::try_from`] rejects it so a newer peer's mode this build does
/// not understand fails loudly instead of silently changing lifecycle semantics.
///
/// The numeric discriminants here MUST stay in sync with the proto
/// `AutoresetMode` (UNSPECIFIED=0, NEXT_STEP=1, SAME_STEP=2, DISABLED=3); the
/// `autoreset_mode_i32_roundtrip` test below locks this so future drift fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum AutoresetMode {
    /// Terminal obs at step `t`; the env internally resets the done lane and
    /// delivers the fresh obs at `t+1`.
    NextStep = 1,
    /// Reset obs delivered in the same step the lane reports done. Reserved;
    /// not honored by the runtime yet.
    SameStep = 2,
    /// No autoreset; a done lane stays inactive until an explicit reset.
    #[default]
    Disabled = 3,
}

/// Error from [`AutoresetMode::try_from`] when an i32 names no known mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnknownAutoresetMode(pub i32);

impl std::fmt::Display for UnknownAutoresetMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unknown autoreset mode {}", self.0)
    }
}

impl std::error::Error for UnknownAutoresetMode {}

impl TryFrom<i32> for AutoresetMode {
    type Error = UnknownAutoresetMode;
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::NextStep),
            2 => Ok(Self::SameStep),
            // proto UNSPECIFIED (0) and DISABLED (3) both decode to the safe
            // explicit-reset default; anything else is rejected loudly.
            0 | 3 => Ok(Self::Disabled),
            other => Err(UnknownAutoresetMode(other)),
        }
    }
}

impl From<AutoresetMode> for i32 {
    fn from(value: AutoresetMode) -> Self {
        value as i32
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct EnvContract {
    pub id: String,
    pub action_space: Option<SpaceSpec>,
    pub observation_space: Option<SpaceSpec>,
    pub metadata: Option<crate::meta::MetaMap>,
    pub render_mode: String,
    pub num_envs: u32,
    /// Per-lane autoreset convention the runtime honors. Derived at construction
    /// from the env's `metadata["autoreset_mode"]`; defaults to `Disabled`.
    pub autoreset_mode: AutoresetMode,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autoreset_mode_i32_roundtrip() {
        // Native -> i32 discriminants, locked to the proto AutoresetMode values
        // (UNSPECIFIED=0, NEXT_STEP=1, SAME_STEP=2, DISABLED=3).
        assert_eq!(i32::from(AutoresetMode::NextStep), 1);
        assert_eq!(i32::from(AutoresetMode::SameStep), 2);
        assert_eq!(i32::from(AutoresetMode::Disabled), 3);

        // i32 -> native: proto UNSPECIFIED (0) and DISABLED (3) decode to the
        // safe Disabled default; 1/2 to their modes.
        assert_eq!(AutoresetMode::try_from(0), Ok(AutoresetMode::Disabled));
        assert_eq!(AutoresetMode::try_from(1), Ok(AutoresetMode::NextStep));
        assert_eq!(AutoresetMode::try_from(2), Ok(AutoresetMode::SameStep));
        assert_eq!(AutoresetMode::try_from(3), Ok(AutoresetMode::Disabled));

        // An unknown value is rejected loudly, never folded to a default.
        assert_eq!(AutoresetMode::try_from(99), Err(UnknownAutoresetMode(99)));
        assert!(AutoresetMode::try_from(-1).is_err());

        // Every native variant round-trips through i32 and back.
        for v in [
            AutoresetMode::NextStep,
            AutoresetMode::SameStep,
            AutoresetMode::Disabled,
        ] {
            assert_eq!(AutoresetMode::try_from(i32::from(v)), Ok(v));
        }
    }
}
