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

#[derive(Debug, Clone, PartialEq, Default)]
pub struct EnvContract {
    pub id: String,
    pub action_space: Option<SpaceSpec>,
    pub observation_space: Option<SpaceSpec>,
    pub metadata: Option<crate::meta::MetaMap>,
    pub render_mode: String,
    pub num_envs: u32,
}
