//! The tolerant-reader serde codec shared by [`ObsLeaf`](super::env_tags::ObsLeaf)
//! and [`ModelLeaf`](super::model::ModelLeaf): internally-tagged known variants
//! plus an `Unknown` arm that keeps an unrecognized `type` verbatim for
//! round-trip. The `type` discriminant must be stripped before each variant's
//! `#[serde(flatten)]` capture sees it, and a non-string `type` is a
//! reserved-key error.

macro_rules! leaf_codec {
    (@unknown yes, $leaf:ident, $kind:expr, $value:expr) => {{
        let kind = $kind;
        let value = $value;
        let role = value
            .get("role")
            .and_then(|role| role.as_str())
            .map(str::to_owned);
        $leaf::Unknown { kind, role, raw: value }
    }};
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

        #[derive(serde::Deserialize)]
        #[serde(tag = "type", rename_all = "lowercase")]
        #[allow(
            clippy::large_enum_variant,
            reason = "a transient decode mirror of the leaf enum, which carries the same \
                      allowance; it lives only until the `From` lift below"
        )]
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

        #[derive(serde::Serialize)]
        #[serde(tag = "type", rename_all = "lowercase")]
        enum $known_ref<'a> {
            $( $variant(&'a $payload) ),+
        }

        impl serde::Serialize for $leaf {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                match self {
                    $( $leaf::$variant(inner) => $known_ref::$variant(inner).serialize(serializer), )+
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
