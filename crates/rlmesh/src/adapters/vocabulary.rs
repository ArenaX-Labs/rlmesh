//! Closed vocabularies and their tolerant string carriers.
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    fmt,
    hash::{Hash, Hasher},
};

pub(crate) mod sealed {
    pub trait Sealed {}
}
/// A library-defined closed string vocabulary. Implementations are sealed.
pub trait Vocabulary:
    Copy + Eq + Hash + fmt::Debug + Send + Sync + 'static + sealed::Sealed
{
    fn as_str(self) -> &'static str;
    fn parse(text: &str) -> Option<Self>;
}
/// A known vocabulary value or a newer peer's spelling, retained verbatim.
#[derive(Clone, Debug)]
pub enum Tolerant<E: Vocabulary> {
    Known(E),
    Other(String),
}
impl<E: Vocabulary> Tolerant<E> {
    pub fn parse(text: &str) -> Self {
        E::parse(text).map_or_else(|| Self::Other(text.into()), Self::Known)
    }
    pub fn as_str(&self) -> &str {
        match self {
            Self::Known(e) => e.as_str(),
            Self::Other(s) => s,
        }
    }
    pub fn known(&self) -> Option<&E> {
        match self {
            Self::Known(e) => Some(e),
            Self::Other(_) => None,
        }
    }
}
impl<E: Vocabulary> From<E> for Tolerant<E> {
    fn from(e: E) -> Self {
        Self::Known(e)
    }
}
impl<E: Vocabulary> PartialEq for Tolerant<E> {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}
impl<E: Vocabulary> Eq for Tolerant<E> {}
impl<E: Vocabulary> Hash for Tolerant<E> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}
impl<E: Vocabulary> fmt::Display for Tolerant<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
impl<E: Vocabulary> Serialize for Tolerant<E> {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}
impl<'de, E: Vocabulary> Deserialize<'de> for Tolerant<E> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        String::deserialize(d).map(|s| Self::parse(&s))
    }
}
macro_rules! vocabulary {
    ($name:ident { $($variant:ident = $text:literal),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[non_exhaustive]
        pub enum $name { $($variant),+ }
        impl $name {
            pub fn as_str(self) -> &'static str { match self { $(Self::$variant => $text),+ } }
            pub fn parse(text: &str) -> Option<Self> { match text { $($text => Some(Self::$variant)),+, _ => None } }
        }
        impl sealed::Sealed for $name {}
        impl Vocabulary for $name { fn as_str(self)-> &'static str { self.as_str() } fn parse(text:&str)->Option<Self> { Self::parse(text) } }
        impl fmt::Display for $name { fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result { f.write_str(self.as_str()) } }
        impl Serialize for $name { fn serialize<S:Serializer>(&self,s:S)->std::result::Result<S::Ok,S::Error>{s.serialize_str(self.as_str())} }
        impl<'de> Deserialize<'de> for $name { fn deserialize<D:Deserializer<'de>>(d:D)->std::result::Result<Self,D::Error>{let s=String::deserialize(d)?; Self::parse(&s).ok_or_else(||serde::de::Error::custom(format!("unknown {}: {s}", stringify!($name))))} }
    };
}
vocabulary!(ImageLayout { Hwc="hwc", Chw="chw" });
vocabulary!(FitMode { Stretch="stretch", Crop="crop", Pad="pad" });
vocabulary!(Frame { World="world", RobotBase="robot_base", Tool="tool" });
vocabulary!(Reference { Current="current", Target="target" });
vocabulary!(Provenance { Sensed="sensed", Estimated="estimated", Privileged="privileged" });
vocabulary!(Resample { Bilinear="bilinear", BilinearAa="bilinear_aa", BicubicAa="bicubic_aa", Lanczos3Aa="lanczos3_aa", Area="area" });
vocabulary!(RotationEncoding { QuatXyzw="quat_xyzw", QuatWxyz="quat_wxyz", AxisAngle="axis_angle", Rot6d="rot6d", Rot6dRowmajor="rot6d_rowmajor", EulerXyz="euler_xyz", GravityXyz="gravity_xyz" });
vocabulary!(StackPad { First="first", Black="black" });
vocabulary!(CropMode { Zoom="zoom", Slice="slice" });
vocabulary!(ChannelOrder { Rgb="rgb", Bgr="bgr" });
vocabulary!(StateContainer { Array="array", List="list" });
vocabulary!(TextContainer { Str="str", List="list" });
vocabulary!(StateSource { Observation="observation", Action="action" });
vocabulary!(AdvisorySeverity { Info="info", Caution="caution" });
impl sealed::Sealed for crate::DType {}
impl Vocabulary for crate::DType {
    fn as_str(self) -> &'static str {
        self.as_str()
    }
    fn parse(text: &str) -> Option<Self> {
        Self::parse(text)
    }
}
impl RotationEncoding {
    pub fn dims(self) -> u32 {
        match self {
            Self::QuatXyzw | Self::QuatWxyz => 4,
            Self::Rot6d | Self::Rot6dRowmajor => 6,
            _ => 3,
        }
    }
}
pub type StackSpec = Vec<i32>;
pub const ROTATION_DIMS: &[(RotationEncoding, u32)] = &[
    (RotationEncoding::AxisAngle, 3),
    (RotationEncoding::EulerXyz, 3),
    (RotationEncoding::GravityXyz, 3),
    (RotationEncoding::QuatWxyz, 4),
    (RotationEncoding::QuatXyzw, 4),
    (RotationEncoding::Rot6d, 6),
    (RotationEncoding::Rot6dRowmajor, 6),
];
