//! Host-side custom rotation encodings on the wire.
//!
//! A [`CustomEncoding`] is an object in the `encoding` slot, `{base, name?, from_base?,
//! to_base?}`, next to the bare-string and accept-set forms. [`StateEncoding`] (observation)
//! and [`ActionEncoding`] (action) hold either form; the native forms keep their wire bytes.

use std::fmt;
use std::marker::PhantomData;

use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::AcceptSet;
use super::rotations::RotationEncoding;

fn default_name() -> String {
    "custom".to_owned()
}

fn is_default_name(name: &str) -> bool {
    name == "custom"
}

/// A rotation re-packing layered on a native `base` encoding.
///
/// The arms are `module:callable` references, never executed here: a custom encoding is a
/// describe/validate schema, and its transform runs only in the process that defined the callable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CustomEncoding {
    /// The native encoding this field resolves as. Role matching, range mapping and the
    /// env conversion run on `base`; the custom arms repack it at the same width.
    pub base: RotationEncoding,
    /// Display name surfaced by `describe`.
    #[serde(default = "default_name", skip_serializing_if = "is_default_name")]
    pub name: String,
    /// `base -> custom`, applied host-side on the observation side.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from_base: Option<String>,
    /// `custom -> base`, applied host-side on the action side.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to_base: Option<String>,
}

impl CustomEncoding {
    /// The native base encoding this field resolves as.
    pub fn base(&self) -> RotationEncoding {
        self.base
    }

    /// Structural invariant checked at parse: a custom encoding is meaningless
    /// with neither arm.
    pub fn validate(&self) -> Result<(), String> {
        if self.from_base.is_none() && self.to_base.is_none() {
            return Err(format!(
                "custom encoding {:?} needs at least one of from_base (observation \
                 side) or to_base (action side)",
                self.name
            ));
        }
        Ok(())
    }
}

/// A rotation declaration in the `encoding` slot: a native form (`N` -- an
/// accept-set on the observation side, a single encoding on the rigid action
/// side) or a host-side [`CustomEncoding`] object that shadows to its `base`.
///
/// A JSON object is a [`CustomEncoding`]; anything else is read by `N`, so the
/// observation side ([`StateEncoding`]) accepts an accept-set list and the action side
/// ([`ActionEncoding`]) rejects one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Encoding<N> {
    Native(N),
    Custom(CustomEncoding),
}

/// Model observation side: an accept-set of native encodings (preference order),
/// or a single host-side custom re-packing.
pub type StateEncoding = Encoding<AcceptSet<RotationEncoding>>;

/// Model action side: a single native encoding, or a custom re-packing.
pub type ActionEncoding = Encoding<RotationEncoding>;

impl<N> Encoding<N> {
    /// The custom re-packing, if this declaration is one.
    pub fn custom(&self) -> Option<&CustomEncoding> {
        match self {
            Encoding::Custom(custom) => Some(custom),
            Encoding::Native(_) => None,
        }
    }
}

impl StateEncoding {
    /// The declaration as an accept-set for the resolver's negotiation: the
    /// native set as-is, or the single `base` a custom encoding shadows to.
    pub fn accept_set(&self) -> AcceptSet<RotationEncoding> {
        match self {
            Encoding::Native(set) => set.clone(),
            Encoding::Custom(custom) => AcceptSet::single(custom.base),
        }
    }

    /// The effective native encoding (first recognized, or a custom's `base`),
    /// used to size an optional part's zero fill.
    pub fn base(&self) -> Option<RotationEncoding> {
        match self {
            Encoding::Native(set) => set.first_known(),
            Encoding::Custom(custom) => Some(custom.base),
        }
    }
}

impl ActionEncoding {
    /// The native base encoding this actuator resolves as (a custom shadows to
    /// its `base`).
    pub fn base(&self) -> RotationEncoding {
        match self {
            Encoding::Native(encoding) => *encoding,
            Encoding::Custom(custom) => custom.base,
        }
    }
}

impl<N: Serialize> Serialize for Encoding<N> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Encoding::Native(native) => native.serialize(serializer),
            Encoding::Custom(custom) => custom.serialize(serializer),
        }
    }
}

impl<'de, N: Deserialize<'de>> Deserialize<'de> for Encoding<N> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct EncodingVisitor<N>(PhantomData<N>);

        impl<'de, N: Deserialize<'de>> Visitor<'de> for EncodingVisitor<N> {
            type Value = Encoding<N>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str(
                    "an encoding name, a list of encoding names, or a \
                     custom-encoding object",
                )
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<Encoding<N>, E> {
                N::deserialize(de::value::StrDeserializer::new(value)).map(Encoding::Native)
            }

            fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<Encoding<N>, A::Error> {
                N::deserialize(de::value::SeqAccessDeserializer::new(seq)).map(Encoding::Native)
            }

            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Encoding<N>, A::Error> {
                let custom =
                    CustomEncoding::deserialize(de::value::MapAccessDeserializer::new(map))?;
                custom.validate().map_err(de::Error::custom)?;
                Ok(Encoding::Custom(custom))
            }
        }

        deserializer.deserialize_any(EncodingVisitor(PhantomData))
    }
}

#[cfg(test)]
mod tests {
    use super::{ActionEncoding, CustomEncoding, StateEncoding};
    use crate::spec::rotations::RotationEncoding;

    #[test]
    fn custom_encoding_object_round_trips() {
        let json = r#"{"base":"euler_xyz","name":"default_rot","from_base":"m:f","to_base":"m:g"}"#;
        let custom: CustomEncoding = serde_json::from_str(json).unwrap();
        assert_eq!(custom.base, RotationEncoding::EulerXyz);
        assert_eq!(custom.name, "default_rot");
        assert_eq!(custom.from_base.as_deref(), Some("m:f"));
        assert_eq!(custom.to_base.as_deref(), Some("m:g"));
        assert_eq!(serde_json::to_string(&custom).unwrap(), json);
    }

    #[test]
    fn default_name_is_elided_on_the_wire() {
        let custom: CustomEncoding =
            serde_json::from_str(r#"{"base":"rot6d","from_base":"m:f"}"#).unwrap();
        assert_eq!(custom.name, "custom");
        assert_eq!(
            serde_json::to_string(&custom).unwrap(),
            r#"{"base":"rot6d","from_base":"m:f"}"#
        );
    }

    #[test]
    fn custom_needs_at_least_one_arm() {
        let err = serde_json::from_str::<StateEncoding>(r#"{"base":"euler_xyz"}"#).unwrap_err();
        assert!(
            err.to_string().contains("at least one of from_base"),
            "{err}"
        );
    }

    #[test]
    fn custom_rejects_unknown_field() {
        let err = serde_json::from_str::<StateEncoding>(
            r#"{"base":"euler_xyz","from_base":"m:f","extra_field":"nope"}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("unknown field"), "{err}");
    }

    #[test]
    fn state_encoding_native_forms_round_trip() {
        let bare: StateEncoding = serde_json::from_str(r#""quat_xyzw""#).unwrap();
        assert!(matches!(bare, StateEncoding::Native(_)));
        assert_eq!(serde_json::to_string(&bare).unwrap(), r#""quat_xyzw""#);
        assert_eq!(bare.base(), Some(RotationEncoding::QuatXyzw));

        let list: StateEncoding = serde_json::from_str(r#"["rot6d","quat_xyzw"]"#).unwrap();
        assert_eq!(
            serde_json::to_string(&list).unwrap(),
            r#"["rot6d","quat_xyzw"]"#
        );
        assert_eq!(
            list.accept_set().first_known(),
            Some(RotationEncoding::Rot6d)
        );
    }

    #[test]
    fn state_encoding_custom_shadows_to_base() {
        let custom: StateEncoding =
            serde_json::from_str(r#"{"base":"euler_xyz","from_base":"m:f"}"#).unwrap();
        assert_eq!(custom.base(), Some(RotationEncoding::EulerXyz));
        assert_eq!(
            custom.accept_set().first_known(),
            Some(RotationEncoding::EulerXyz)
        );
        assert!(custom.custom().is_some());
    }

    #[test]
    fn action_encoding_native_and_custom_but_not_list() {
        let native: ActionEncoding = serde_json::from_str(r#""rot6d""#).unwrap();
        assert_eq!(native.base(), RotationEncoding::Rot6d);
        assert_eq!(serde_json::to_string(&native).unwrap(), r#""rot6d""#);

        let custom: ActionEncoding =
            serde_json::from_str(r#"{"base":"euler_xyz","to_base":"m:g"}"#).unwrap();
        assert_eq!(custom.base(), RotationEncoding::EulerXyz);
        assert!(custom.custom().is_some());

        assert!(serde_json::from_str::<ActionEncoding>(r#"["rot6d","quat_wxyz"]"#).is_err());
        assert!(serde_json::from_str::<ActionEncoding>(r#""rot10d""#).is_err());
    }
}
