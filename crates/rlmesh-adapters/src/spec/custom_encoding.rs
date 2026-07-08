//! Host-side custom rotation encodings on the wire.
//!
//! A [`CustomEncoding`] is a rotation re-packing layered on a native `base`
//! encoding: one or two `module:callable` entrypoint arms that convert
//! `base <-> custom` at the field boundary (`from_base` on the observation side,
//! `to_base` on the action side). On the wire it is an object in the `encoding`
//! slot -- `{base, name?, from_base?, to_base?}` -- next to the bare-string and
//! accept-set forms.
//!
//! **The arms are opaque references, never executed here.** Resolution shadows a
//! custom encoding to its `base` for the structural negotiation (width, role,
//! env conversion all run on `base`); the platform validates the base against an
//! env and describes the field, but never runs the arm. A custom encoding is a
//! describe/validate *schema*: the transform executes only in the process that
//! defined its in-process callable (via the host-side
//! [`EncodingTransform`](crate::v1::EncodingTransform) shim the Python layer
//! builds). A serialized arm -- an entrypoint reference, or the host-local
//! marker a callable serializes to -- is not run; execution stays pinned to the
//! model layer.
//!
//! [`StateEncoding`] (obs) and [`ActionEncoding`] (action) are the "native or
//! custom" wrappers the two carrier fields hold; both keep the bare-string /
//! accept-set forms byte-identical for the common native case.

use std::fmt;
use std::marker::PhantomData;

use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::AcceptSet;
use super::rotations::RotationEncoding;

fn default_name() -> String {
    "custom".to_owned()
}

/// The default display name is elided on the wire (byte-parity for the common
/// unnamed case), like every other defaulted field.
fn is_default_name(name: &str) -> bool {
    name == "custom"
}

/// A rotation re-packing layered on a native `base` encoding.
///
/// See the module docs: the arms are `module:callable` references the platform
/// never imports; only a trusted in-process resolve does.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CustomEncoding {
    /// The native encoding this field resolves as. Role matching, range mapping,
    /// and the env->base conversion all run on `base`; the custom arm is an
    /// additional host-side repack that preserves the base width.
    pub base: RotationEncoding,
    /// Display name surfaced by `describe`. Elided on the wire when the default
    /// (`"custom"`).
    #[serde(default = "default_name", skip_serializing_if = "is_default_name")]
    pub name: String,
    /// `base -> custom`, applied host-side on the observation side. A
    /// `module:callable` entrypoint string. Absent when the encoding is
    /// action-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from_base: Option<String>,
    /// `custom -> base`, applied host-side on the action side. Absent when the
    /// encoding is observation-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to_base: Option<String>,
}

impl CustomEncoding {
    /// The native base encoding this field resolves as.
    pub fn base(&self) -> RotationEncoding {
        self.base
    }

    /// Structural invariant checked at parse: a custom encoding is meaningless
    /// with neither arm. The *side-appropriate* arm (obs needs `from_base`,
    /// action needs `to_base`) is checked at resolve, where the side is known.
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
/// The native forms round-trip byte-identically to `N`'s own wire form; a JSON
/// object is a [`CustomEncoding`]. `N`'s own reader decides list tolerance, so
/// the observation side ([`StateEncoding`]) accepts an accept-set list while the
/// rigid action side ([`ActionEncoding`]) rejects one, preserving the
/// `no_action_accept_set` contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Encoding<N> {
    Native(N),
    Custom(CustomEncoding),
}

/// Model observation side: an accept-set of native encodings (preference order),
/// or a single host-side custom re-packing.
pub type StateEncoding = Encoding<AcceptSet<RotationEncoding>>;

/// Model action side: a single native encoding, or a custom re-packing. A JSON
/// array is rejected (a producer and consumer are both fixed).
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
        // name==default and to_base==None are both omitted.
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
        // Bare string and list both stay the plain AcceptSet wire form.
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

        // Rigid action side: a list is rejected (the no-accept-set contract).
        assert!(serde_json::from_str::<ActionEncoding>(r#"["rot6d","quat_wxyz"]"#).is_err());
        // And an unknown bare encoding is rejected at parse.
        assert!(serde_json::from_str::<ActionEncoding>(r#""rot10d""#).is_err());
    }
}
