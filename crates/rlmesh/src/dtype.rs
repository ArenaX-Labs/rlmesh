//! Element dtypes and memory devices.

use std::fmt;

use rlmesh_spaces::DType as CoreDType;

/// An element dtype, with its wire spelling (`"float32"`, ...).
///
/// The discriminants are the wire numbers (`Bool = 1` ... `Float64 = 12`). A
/// space with no element dtype of its own (a `Dict`) reads as `Float32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(i32)]
pub enum DType {
    /// `"bool"`.
    Bool = 1,
    /// `"uint8"`.
    Uint8 = 2,
    /// `"uint16"`.
    Uint16 = 3,
    /// `"uint32"`.
    Uint32 = 4,
    /// `"uint64"`.
    Uint64 = 5,
    /// `"int8"`.
    Int8 = 6,
    /// `"int16"`.
    Int16 = 7,
    /// `"int32"`.
    Int32 = 8,
    /// `"int64"`.
    Int64 = 9,
    /// `"float16"`.
    Float16 = 10,
    /// `"float32"`.
    Float32 = 11,
    /// `"float64"`.
    Float64 = 12,
}

impl DType {
    /// Every dtype, in wire order.
    pub(crate) const ALL: [DType; 12] = [
        DType::Bool,
        DType::Uint8,
        DType::Uint16,
        DType::Uint32,
        DType::Uint64,
        DType::Int8,
        DType::Int16,
        DType::Int32,
        DType::Int64,
        DType::Float16,
        DType::Float32,
        DType::Float64,
    ];

    /// The canonical spelling (`"float32"`).
    pub fn as_str(self) -> &'static str {
        self.to_core().name()
    }

    /// The dtype whose canonical spelling is exactly `name` (`"float32"`);
    /// `None` otherwise.
    pub fn parse(name: &str) -> Option<DType> {
        Self::ALL.into_iter().find(|dtype| dtype.as_str() == name)
    }

    pub(crate) fn to_core(self) -> CoreDType {
        match self {
            DType::Bool => CoreDType::Bool,
            DType::Uint8 => CoreDType::Uint8,
            DType::Uint16 => CoreDType::Uint16,
            DType::Uint32 => CoreDType::Uint32,
            DType::Uint64 => CoreDType::Uint64,
            DType::Int8 => CoreDType::Int8,
            DType::Int16 => CoreDType::Int16,
            DType::Int32 => CoreDType::Int32,
            DType::Int64 => CoreDType::Int64,
            DType::Float16 => CoreDType::Float16,
            DType::Float32 => CoreDType::Float32,
            DType::Float64 => CoreDType::Float64,
        }
    }

    /// The facade dtype of a core one; `Unspecified` reads as `Float32`.
    pub(crate) fn from_core(dtype: CoreDType) -> DType {
        match dtype {
            CoreDType::Bool => DType::Bool,
            CoreDType::Uint8 => DType::Uint8,
            CoreDType::Uint16 => DType::Uint16,
            CoreDType::Uint32 => DType::Uint32,
            CoreDType::Uint64 => DType::Uint64,
            CoreDType::Int8 => DType::Int8,
            CoreDType::Int16 => DType::Int16,
            CoreDType::Int32 => DType::Int32,
            CoreDType::Int64 => DType::Int64,
            CoreDType::Float16 => DType::Float16,
            CoreDType::Unspecified | CoreDType::Float32 => DType::Float32,
            CoreDType::Float64 => DType::Float64,
        }
    }

    /// Whether this is an integer or bool dtype (bounds are integers).
    pub(crate) fn is_integral(self) -> bool {
        self == DType::Bool || self.to_core().is_integer()
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Where a tensor's memory lives (`"cpu"`).
pub use rlmesh_spaces::Device;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_spellings_round_trip_and_keep_wire_numbers() {
        for dtype in DType::ALL {
            assert_eq!(DType::parse(dtype.as_str()), Some(dtype));
            assert_eq!(dtype as i32, i32::from(dtype.to_core()));
            assert_eq!(DType::from_core(dtype.to_core()), dtype);
        }
        assert_eq!(DType::parse("Float32"), None);
        assert_eq!(DType::from_core(CoreDType::Unspecified), DType::Float32);
        assert_eq!(DType::Float32.to_string(), "float32");
        assert_eq!(Device::Cpu.as_str(), "cpu");
        assert_eq!(Device::Cpu.to_string(), "cpu");
    }
}
