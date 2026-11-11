//! Declaration parsing shared by every part: an order-keeping map, field
//! and limit declarations, and the keys each kind allows.

use crate::prelude::*;

use core::fmt;

use serde::Deserialize;
use serde::de::value::{MapAccessDeserializer, SeqAccessDeserializer};
use serde::de::{MapAccess, SeqAccess, Visitor};

use crate::spec::kinds::Kind;

/// A JSON object kept in authored order, refusing duplicate keys.
pub(crate) struct Ordered<T>(pub(crate) Vec<(String, T)>);

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Ordered<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct OrderedVisitor<T>(core::marker::PhantomData<T>);

        impl<'de, T: Deserialize<'de>> Visitor<'de> for OrderedVisitor<T> {
            type Value = Ordered<T>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("an object")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut entries: Vec<(String, T)> = Vec::new();
                while let Some((key, value)) = map.next_entry::<String, T>()? {
                    if entries.iter().any(|(k, _)| *k == key) {
                        return Err(serde::de::Error::custom(format!("duplicate key {key:?}")));
                    }
                    entries.push((key, value));
                }
                Ok(Ordered(entries))
            }
        }

        deserializer.deserialize_map(OrderedVisitor(core::marker::PhantomData))
    }
}

/// `[dtype, shape]`: one field of a custom kind.
#[derive(Deserialize)]
pub(crate) struct FieldDecl(pub(crate) String, pub(crate) Vec<u32>);

/// A part's `fields`: a list of optional field names (joints, hand, imu),
/// or a custom kind's schema `{name: [dtype, shape]}` in payload order.
pub(crate) enum FieldsDecl {
    /// Core kinds: which optional fields are published.
    Names(Vec<String>),
    /// Custom kinds: the schema.
    Schema(Ordered<FieldDecl>),
}

impl<'de> Deserialize<'de> for FieldsDecl {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct FieldsVisitor;

        impl<'de> Visitor<'de> for FieldsVisitor {
            type Value = FieldsDecl;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a list of field names, or {name: [dtype, shape]}")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut names = Vec::new();
                while let Some(name) = seq.next_element::<String>()? {
                    names.push(name);
                }
                Ok(FieldsDecl::Names(names))
            }

            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
                Ordered::deserialize(MapAccessDeserializer::new(map)).map(FieldsDecl::Schema)
            }
        }

        deserializer.deserialize_any(FieldsVisitor)
    }
}

/// A part's `range`: `[low, high]` (a `value` part), or
/// `{field: [low, high]}` (any other kind).
pub(super) enum RangeDecl {
    /// The bounds of the part's one field.
    Bounds([f64; 2]),
    /// The bounds of each named field.
    Fields(Ordered<[f64; 2]>),
}

impl<'de> Deserialize<'de> for RangeDecl {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct RangeVisitor;

        impl<'de> Visitor<'de> for RangeVisitor {
            type Value = RangeDecl;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("[low, high], or {field: [low, high]}")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<Self::Value, A::Error> {
                <[f64; 2]>::deserialize(SeqAccessDeserializer::new(seq)).map(RangeDecl::Bounds)
            }

            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
                Ordered::deserialize(MapAccessDeserializer::new(map)).map(RangeDecl::Fields)
            }
        }

        deserializer.deserialize_any(RangeVisitor)
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
pub(super) enum LimitDecl {
    Pair([f64; 2]),
    Pairs(Vec<[f64; 2]>),
    Norm { max_norm: f64 },
}

/// Keys every streaming kind takes.
const STREAMING: &[&str] = &["kind", "hz", "send", "priority", "doc", "max_age_ms"];

/// Keys of a commandable part, less the kind-specific ones.
pub(super) const COMMAND: &[&str] = &[
    "command",
    "control_hz",
    "safe",
    "hold_max_ms",
    "max_action_age_ms",
    "limits",
    "merge",
    "group",
];

/// The keys `kind` takes (beyond `x_*`).
pub(super) fn allowed(kind: Kind) -> Vec<&'static str> {
    let mut keys: Vec<&'static str> = match kind {
        Kind::Text | Kind::Reward | Kind::Success => return vec!["kind", "priority", "doc"],
        _ => STREAMING.to_vec(),
    };
    let commandable = !kind.command_modes().is_empty() || kind == Kind::Custom;
    if commandable {
        keys.extend(COMMAND);
    }
    keys.extend(match kind {
        Kind::Camera => &[
            "frame",
            "width",
            "height",
            "encoding",
            "quality",
            "format",
            "fov_deg",
            "intrinsics",
            "mount",
        ][..],
        Kind::Depth => &[
            "frame",
            "width",
            "height",
            "dtype",
            "range",
            "fov_deg",
            "intrinsics",
        ],
        Kind::Joints | Kind::Hand => &[
            "dtype",
            "units",
            "range",
            "fields",
            "names",
            "count",
            "prismatic",
            "gains",
        ],
        Kind::Wrench => &["frame", "point", "dtype", "units", "range"],
        Kind::Imu => &["frame", "dtype", "units", "range", "fields", "rotation"],
        Kind::Pose => &[
            "frame",
            "child",
            "rotation",
            "dtype",
            "units",
            "range",
            "reference",
        ],
        Kind::Gripper => &["dtype", "units", "range", "width_m", "binary"],
        Kind::Switch => &[],
        Kind::Value => &["unit", "range"],
        Kind::Velocity => &["frame", "dof", "dtype", "units", "range"],
        Kind::Tactile => &["shape", "layout", "dtype", "units", "range"],
        Kind::Custom => &["frame", "units", "range", "fields", "command_fields"],
        Kind::Text | Kind::Reward | Kind::Success => &[],
    });
    keys
}
