//! The single home for the tolerant-reader leaf serde codec.
//!
//! Both spec trees carry a tolerant *leaf* enum — [`ObsLeaf`](super::env_tags::ObsLeaf)
//! and [`ModelLeaf`](super::model::ModelLeaf) — with the identical shape: a set
//! of internally-tagged *known* variants plus a catch-all `Unknown` arm that
//! retains an unrecognized `type` verbatim for byte-faithful round-trip. The
//! serde machinery driving that — the owned `*Known` enum, the borrowed
//! `*KnownRef<'a>` enum used for zero-copy serialize, the `From` lift, and the
//! two hand-written [`Serialize`]/[`Deserialize`] impls — is fragile in exactly
//! the same way for both (the internally-tagged `type` discriminant must be
//! stripped before each variant's `#[serde(flatten)]` capture sees it, and a
//! non-string `type` is a reserved-key error). [`leaf_codec!`] generates all of
//! it from one site so that fragile logic lives exactly once.

/// Generate the tolerant-reader serde codec for a leaf enum.
///
/// Given the public leaf enum (already defined, with its `Unknown` arm) the
/// macro emits: the owned internally-tagged `$known` enum, the borrowed
/// `$known_ref<'a>` enum, the `From<$known>` lift, the leaf-`type` vocabulary
/// constant, and the hand-written [`Serialize`]/[`Deserialize`] impls. A known
/// `type` deserializes into the strict variant (a malformed payload of a
/// recognized kind still hard-errors); any other string `type` becomes the
/// `Unknown` arm, retained verbatim.
///
/// `lift_role: yes` makes the `Unknown` arm lift a top-level `role` string from
/// the raw object — `Unknown { kind, role, raw }`; `lift_role: no` builds
/// `Unknown { kind, raw }`.
macro_rules! leaf_codec {
    // Internal: build the `Unknown` arm, lifting `role` from the raw object.
    // `$kind` is the owned kind string; `$value` is the raw object it was
    // borrowed from, so `role` must be read before `value` is moved into `raw`.
    (@unknown yes, $leaf:ident, $kind:expr, $value:expr) => {{
        let kind = $kind;
        let value = $value;
        let role = value
            .get("role")
            .and_then(|role| role.as_str())
            .map(str::to_owned);
        $leaf::Unknown { kind, role, raw: value }
    }};
    // Internal: build the `Unknown` arm without a `role`.
    (@unknown no, $leaf:ident, $kind:expr, $value:expr) => {
        $leaf::Unknown { kind: $kind, raw: $value }
    };
    (
        leaf: $leaf:ident,
        known: $known:ident,
        known_ref: $known_ref:ident,
        vocab: $vocab:ident = $vocab_doc:literal,
        missing_type_msg: $missing_type_msg:literal,
        lift_role: $lift_role:tt,
        variants: { $( $variant:ident($payload:ty) = $tag:literal ),+ $(,)? }
    ) => {
        #[doc = $vocab_doc]
        pub const $vocab: &[&str] = &[ $( $tag ),+ ];

        /// Owned mirror of the *known* leaf variants. Reuses serde's
        /// internally-tagged derive (which strips `type` before the variant's
        /// flatten capture sees it) so the fragile tagged+flatten interaction
        /// lives in one derive, not hand-rolled dispatch.
        #[derive(serde::Deserialize)]
        #[serde(tag = "type", rename_all = "lowercase")]
        enum $known {
            $( $variant($payload) ),+
        }

        impl From<$known> for $leaf {
            fn from(known: $known) -> Self {
                match known {
                    $( $known::$variant(inner) => $leaf::$variant(inner) ),+
                }
            }
        }

        /// Borrowed mirror for Serialize (re-emits the internally-tagged known form).
        #[derive(serde::Serialize)]
        #[serde(tag = "type", rename_all = "lowercase")]
        enum $known_ref<'a> {
            $( $variant(&'a $payload) ),+
        }

        impl serde::Serialize for $leaf {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                match self {
                    $( $leaf::$variant(inner) => $known_ref::$variant(inner).serialize(serializer), )+
                    // The raw object already embeds `type`; emit it verbatim.
                    $leaf::Unknown { raw, .. } => raw.serialize(serializer),
                }
            }
        }

        impl<'de> serde::Deserialize<'de> for $leaf {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                use serde::de::Error as _;
                let value = serde_json::Value::deserialize(deserializer)?;
                let kind = value
                    .get("type")
                    .and_then(|tag| tag.as_str())
                    .ok_or_else(|| D::Error::custom($missing_type_msg))?;
                if $vocab.contains(&kind) {
                    // Malformed payload of a recognized kind still hard-errors here.
                    $known::deserialize(value)
                        .map($leaf::from)
                        .map_err(D::Error::custom)
                } else {
                    Ok($crate::spec::leaf_codec::leaf_codec!(
                        @unknown $lift_role, $leaf, kind.to_owned(), value
                    ))
                }
            }
        }
    };
}

pub(crate) use leaf_codec;
