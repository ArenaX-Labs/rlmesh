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
/// `low`/`high` must each be exactly `dtype_size(dtype)` bytes, so integer dtypes
/// (notably `int64`/`uint64`) carry exact bounds instead of passing through `f64`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TypedUniformBounds {
    pub low: Vec<u8>,
    pub high: Vec<u8>,
}

/// Per-element low/high bounds carried as raw little-endian bytes in the
/// enclosing space's dtype, in row-major (C) order.
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

/// Empty marker for a MultiBinary space: its dimensions come entirely from
/// [`SpaceSpec::shape`] (rank-1 `[n]` for a flat vector, rank-N for a tensor of
/// bits).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MultiBinarySpec;

/// Per-element category counts for a MultiDiscrete space, flattened row-major
/// (C order). The logical shape is [`SpaceSpec::shape`] (`[len]` or `[rows, cols]`);
/// `nvec` holds `numel(shape)` entries. Element `i` takes values in
/// `[start[i], start[i] + nvec[i])`; an empty `start` is all zeros.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MultiDiscreteSpec {
    pub nvec: Vec<i64>,
    pub start: Vec<i64>,
}

impl MultiDiscreteSpec {
    /// Element `i`'s lowest value.
    pub fn start(&self, i: usize) -> i64 {
        self.start.get(i).copied().unwrap_or(0)
    }

    /// Whether any element starts away from zero.
    pub fn has_start(&self) -> bool {
        self.start.iter().any(|s| *s != 0)
    }
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
/// `AutoresetMode` and gymnasium's `AutoresetMode`).
///
/// The proto `UNSPECIFIED` (0) decodes to `Disabled`, the explicit-reset default. Unknown
/// values are rejected by [`AutoresetMode::try_from`] rather than folded to a default. The
/// discriminants must match the proto (`UNSPECIFIED=0, NEXT_STEP=1, SAME_STEP=2, DISABLED=3`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum AutoresetMode {
    /// Terminal obs at step `t`; the env internally resets the done lane and
    /// delivers the fresh obs at `t+1`.
    NextStep = 1,
    /// Reset obs delivered in the same step the lane reports done.
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
    /// Per-lane autoreset convention the runtime honors.
    pub autoreset_mode: AutoresetMode,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autoreset_mode_i32_roundtrip() {
        assert_eq!(i32::from(AutoresetMode::NextStep), 1);
        assert_eq!(i32::from(AutoresetMode::SameStep), 2);
        assert_eq!(i32::from(AutoresetMode::Disabled), 3);

        assert_eq!(AutoresetMode::try_from(0), Ok(AutoresetMode::Disabled));
        assert_eq!(AutoresetMode::try_from(1), Ok(AutoresetMode::NextStep));
        assert_eq!(AutoresetMode::try_from(2), Ok(AutoresetMode::SameStep));
        assert_eq!(AutoresetMode::try_from(3), Ok(AutoresetMode::Disabled));

        assert_eq!(AutoresetMode::try_from(99), Err(UnknownAutoresetMode(99)));
        assert!(AutoresetMode::try_from(-1).is_err());

        for v in [
            AutoresetMode::NextStep,
            AutoresetMode::SameStep,
            AutoresetMode::Disabled,
        ] {
            assert_eq!(AutoresetMode::try_from(i32::from(v)), Ok(v));
        }
    }
}
