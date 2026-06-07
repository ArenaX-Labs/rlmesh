#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum DType {
    #[default]
    Unspecified = 0,
    Bool = 1,
    Uint8 = 2,
    Int32 = 3,
    Int64 = 4,
    Float16 = 5,
    Float32 = 6,
    Float64 = 7,
}

impl TryFrom<i32> for DType {
    type Error = &'static str;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Unspecified),
            1 => Ok(Self::Bool),
            2 => Ok(Self::Uint8),
            3 => Ok(Self::Int32),
            4 => Ok(Self::Int64),
            5 => Ok(Self::Float16),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Float64),
            _ => Err("invalid dtype"),
        }
    }
}

impl From<DType> for i32 {
    fn from(value: DType) -> Self {
        value as i32
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VectorInt {
    pub data: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MatrixInt {
    pub data: Vec<VectorInt>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct UniformBounds {
    pub low: f64,
    pub high: f64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct AxiswiseBounds {
    pub low: Vec<f64>,
    pub high: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ElementwiseBounds {
    pub low: Vec<f64>,
    pub high: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BoxSpec {
    pub bounds: Option<box_spec::Bounds>,
}

pub mod box_spec {
    use crate::v1::{AxiswiseBounds, ElementwiseBounds, UniformBounds};

    #[derive(Debug, Clone, PartialEq)]
    pub enum Bounds {
        Unbounded(bool),
        Uniform(UniformBounds),
        Axiswise(AxiswiseBounds),
        Elementwise(ElementwiseBounds),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DiscreteSpec {
    pub n: i64,
    pub start: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MultiBinarySpec {
    pub n: Option<multi_binary_spec::N>,
}

pub mod multi_binary_spec {
    use crate::v1::VectorInt;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum N {
        Size(i64),
        Dims(VectorInt),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MultiDiscreteSpec {
    pub nvec: Option<multi_discrete_spec::Nvec>,
}

pub mod multi_discrete_spec {
    use crate::v1::{MatrixInt, VectorInt};

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum Nvec {
        Flat(VectorInt),
        Shaped(MatrixInt),
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
    pub spec: Option<space_spec::Spec>,
}

impl SpaceSpec {
    pub fn space_type(&self) -> SpaceType {
        match self.spec {
            Some(space_spec::Spec::Box(_)) => SpaceType::Box,
            Some(space_spec::Spec::Discrete(_)) => SpaceType::Discrete,
            Some(space_spec::Spec::MultiBinary(_)) => SpaceType::MultiBinary,
            Some(space_spec::Spec::MultiDiscrete(_)) => SpaceType::MultiDiscrete,
            Some(space_spec::Spec::Text(_)) => SpaceType::Text,
            Some(space_spec::Spec::Dict(_)) => SpaceType::Dict,
            Some(space_spec::Spec::Tuple(_)) => SpaceType::Tuple,
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

pub mod space_spec {
    pub use crate::v1::SpaceKind as Spec;
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
    pub metadata: Option<crate::v1::meta::MetaMap>,
    pub render_mode: String,
    pub num_envs: u32,
}
