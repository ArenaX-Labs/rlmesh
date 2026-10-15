use super::*;
use crate::{Result, Value};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    fmt,
    hash::{Hash, Hasher},
    sync::Arc,
};

type Transform = dyn Fn(&Value) -> Result<Value> + Send + Sync;
macro_rules! callable {
    ($name:ident) => {
        #[derive(Clone)]
        pub struct $name(Arc<Transform>);
        impl<F: Fn(&Value) -> Result<Value> + Send + Sync + 'static> From<F> for $name {
            fn from(f: F) -> Self {
                Self(Arc::new(f))
            }
        }
        impl $name {
            pub(crate) fn call(&self, value: &Value) -> Result<Value> {
                (self.0)(value)
            }
        }
        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                Arc::ptr_eq(&self.0, &other.0)
            }
        }
        impl Eq for $name {}
        impl Hash for $name {
            fn hash<H: Hasher>(&self, h: &mut H) {
                (Arc::as_ptr(&self.0) as *const ()).hash(h)
            }
        }
        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "<{} {:p}>", stringify!($name), Arc::as_ptr(&self.0))
            }
        }
        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
                s.serialize_str("<local>")
            }
        }
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(_d: D) -> std::result::Result<Self, D::Error> {
                Err(serde::de::Error::custom(
                    "a callable cannot be reconstructed from metadata",
                ))
            }
        }
    };
}
callable!(ObsTransform);
callable!(RotationTransform);
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EncodingArm {
    Entrypoint(String),
    Transform(RotationTransform),
}
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CustomEncoding {
    pub base: RotationEncoding,
    #[serde(default)]
    pub from_base: Option<EncodingArm>,
    #[serde(default)]
    pub to_base: Option<EncodingArm>,
    #[serde(default = "custom_name")]
    pub name: String,
}
fn custom_name() -> String {
    "custom".into()
}
impl CustomEncoding {
    pub fn new(base: RotationEncoding) -> Self {
        Self {
            base,
            from_base: None,
            to_base: None,
            name: custom_name(),
        }
    }
    pub fn is_entrypoint(&self) -> bool {
        matches!(self.from_base, Some(EncodingArm::Entrypoint(_)))
            || matches!(self.to_base, Some(EncodingArm::Entrypoint(_)))
    }
    pub fn width(&self) -> u32 {
        self.base.dims()
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Custom {
    pub transform: Option<ObsTransform>,
    pub entrypoint: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StateEncoding {
    Custom(CustomEncoding),
    Native(#[serde(with = "super::codec::set")] Vec<Tolerant<RotationEncoding>>),
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ActionEncoding {
    Custom(CustomEncoding),
    Native(RotationEncoding),
}
