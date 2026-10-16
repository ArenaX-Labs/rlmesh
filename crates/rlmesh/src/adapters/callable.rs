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
        /// A shared callable; clones and equality retain its construction identity.
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
/// A typed adapter vocabulary or tree alternative.
pub enum EncodingArm {
    /// The `Entrypoint` alternative.
    Entrypoint(String),
    /// The `Transform` alternative.
    Transform(RotationTransform),
}
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// A rotation packing layered on a known native `base` encoding.
pub struct CustomEncoding {
    /// The native encoding the field resolves as.
    pub base: RotationEncoding,
    #[serde(default)]
    /// `base -> custom`, used when the encoding tags an observation (model) state component. An in-
    /// process callable (runs locally, serializes to a `<local>` marker), or a `"module:callable"`
    /// entrypoint string (a describe-only reference; its execution is deferred). None when action-
    /// only.
    pub from_base: Option<EncodingArm>,
    #[serde(default)]
    /// `custom -> base`, used when the encoding tags an action component. A callable or entrypoint
    /// string (see `from_base`). None when obs-only.
    pub to_base: Option<EncodingArm>,
    #[serde(default = "custom_name")]
    /// Display name surfaced in `Adapter.describe`.
    pub name: String,
}
fn custom_name() -> String {
    "custom".into()
}
impl CustomEncoding {
    /// Construct this value with Python’s field defaults.
    pub fn new(base: RotationEncoding) -> Self {
        Self {
            base,
            from_base: None,
            to_base: None,
            name: custom_name(),
        }
    }
    /// Whether either encoding arm holds an entrypoint string.
    pub fn is_entrypoint(&self) -> bool {
        matches!(self.from_base, Some(EncodingArm::Entrypoint(_)))
            || matches!(self.to_base, Some(EncodingArm::Entrypoint(_)))
    }
    /// The base rotation encoding’s element width.
    pub fn width(&self) -> u32 {
        self.base.dims()
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
/// A custom input computed by host-language code.
pub struct Custom {
    /// An in-process callable taking the raw observation mapping.
    pub transform: Option<ObsTransform>,
    /// A `module:callable` string.
    pub entrypoint: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
/// A typed adapter vocabulary or tree alternative.
pub enum StateEncoding {
    /// The `Custom` alternative.
    Custom(CustomEncoding),
    /// The `Native` alternative.
    Native(#[serde(with = "super::codec::set")] Vec<Tolerant<RotationEncoding>>),
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
/// A typed adapter vocabulary or tree alternative.
pub enum ActionEncoding {
    /// The `Custom` alternative.
    Custom(CustomEncoding),
    /// The `Native` alternative.
    Native(RotationEncoding),
}
