//! An ordered preference list of a frozen vocabulary value, tolerant of unknown
//! (future) entries.
//!
//! A spec field that once named a single encoding (`"encoding": "quat_xyzw"`)
//! becomes an [`AcceptSet`]: still a bare string on the wire when it holds one
//! value (byte-identical to before), but optionally a list when a side declares
//! more than one acceptable encoding (`["rot6d", "quat_xyzw"]`). The resolver
//! reads these as a *preference order* and picks the encoding the two sides can
//! bridge — see `resolver`. Unknown (future) strings parse and round-trip but
//! are skipped by [`AcceptSet::known`], so an old reader degrades gracefully
//! instead of failing at parse: the recognized subset still negotiates, and a
//! wholly-unrecognized declaration surfaces at *resolve* (a typed error), never
//! here. The vocabulary enums themselves ([`RotationEncoding`](super::RotationEncoding),
//! [`ImageLayout`](super::ImageLayout)) stay closed and `Copy`; the unknown arm
//! lives only at this boundary so the apply/geometry layers never see it.

use std::collections::BTreeSet;
use std::fmt;
use std::marker::PhantomData;

use serde::de::{self, SeqAccess, Visitor};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A closed wire vocabulary: a value with a canonical wire string, parseable
/// back from one. Implemented by [`RotationEncoding`](super::RotationEncoding)
/// and [`ImageLayout`](super::ImageLayout).
pub trait WireVocab: Copy + Eq + Sized {
    /// Parse a wire string into a known value, or `None` if unrecognized.
    fn from_wire(name: &str) -> Option<Self>;
    /// The canonical wire string for this value.
    fn as_wire(self) -> &'static str;
}

/// One entry of an [`AcceptSet`]: a recognized vocabulary value, or an unknown
/// (future) wire string retained verbatim for round-trip but never selected by
/// a reader that does not recognize it.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Entry<T> {
    Known(T),
    Unknown(String),
}

impl<T: WireVocab> Entry<T> {
    fn parse(raw: String) -> Self {
        match T::from_wire(&raw) {
            Some(value) => Entry::Known(value),
            None => Entry::Unknown(raw),
        }
    }

    fn wire(&self) -> &str {
        match self {
            Entry::Known(value) => value.as_wire(),
            Entry::Unknown(raw) => raw,
        }
    }
}

/// An ordered, non-empty preference list of a frozen vocabulary value.
///
/// Entries are de-duplicated by wire string, preserving first-seen order. See
/// the module docs for the wire form and the graceful-degradation contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptSet<T> {
    // Invariant: non-empty. Enforced by the deserializer (a list must name at
    // least one encoding) and by `single`.
    entries: Vec<Entry<T>>,
}

impl<T: WireVocab> AcceptSet<T> {
    /// A single recognized value — the common (pre-accept-set) declaration.
    pub fn single(value: T) -> Self {
        AcceptSet {
            entries: vec![Entry::Known(value)],
        }
    }

    /// The recognized values, in declared preference order (unknown entries
    /// skipped).
    pub fn known(&self) -> impl Iterator<Item = T> + '_ {
        self.entries.iter().filter_map(|entry| match entry {
            Entry::Known(value) => Some(*value),
            Entry::Unknown(_) => None,
        })
    }

    /// The first recognized value, if any. For a producer this is its native
    /// (raw) encoding; `None` means the declaration named only unknown values.
    pub fn first_known(&self) -> Option<T> {
        self.known().next()
    }

    /// Whether `value` appears among the recognized entries.
    pub fn accepts(&self, value: T) -> bool {
        self.known().any(|known| known == value)
    }

    /// The declared entries as wire strings (known and unknown), for messages.
    pub fn wire_names(&self) -> Vec<&str> {
        self.entries.iter().map(Entry::wire).collect()
    }
}

impl<T: WireVocab> Serialize for AcceptSet<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Byte-parity: a single entry serializes as a bare string, exactly like
        // the pre-accept-set single-value wire form, so existing frozen specs
        // round-trip unchanged. A list only appears when a side genuinely
        // declares more than one acceptable encoding.
        if let [only] = self.entries.as_slice() {
            return serializer.serialize_str(only.wire());
        }
        let mut seq = serializer.serialize_seq(Some(self.entries.len()))?;
        for entry in &self.entries {
            seq.serialize_element(entry.wire())?;
        }
        seq.end()
    }
}

impl<'de, T: WireVocab> Deserialize<'de> for AcceptSet<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct AcceptSetVisitor<T>(PhantomData<T>);

        impl<'de, T: WireVocab> Visitor<'de> for AcceptSetVisitor<T> {
            type Value = AcceptSet<T>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("an encoding name or a non-empty list of encoding names")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<AcceptSet<T>, E> {
                Ok(AcceptSet {
                    entries: vec![Entry::parse(value.to_owned())],
                })
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<AcceptSet<T>, A::Error> {
                let mut entries: Vec<Entry<T>> = Vec::new();
                let mut seen: BTreeSet<String> = BTreeSet::new();
                while let Some(raw) = seq.next_element::<String>()? {
                    // De-dup by wire string, preserving first-seen order: a
                    // repeated preference is harmless noise, not an error.
                    if seen.insert(raw.clone()) {
                        entries.push(Entry::parse(raw));
                    }
                }
                if entries.is_empty() {
                    return Err(de::Error::custom(
                        "an encoding preference list must name at least one encoding",
                    ));
                }
                Ok(AcceptSet { entries })
            }
        }

        deserializer.deserialize_any(AcceptSetVisitor(PhantomData))
    }
}

#[cfg(test)]
mod tests {
    use super::{AcceptSet, WireVocab};

    // A tiny two-value vocabulary so the AcceptSet tests do not depend on the
    // real encoding enums.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Vocab {
        A,
        B,
    }

    impl WireVocab for Vocab {
        fn from_wire(name: &str) -> Option<Self> {
            match name {
                "a" => Some(Vocab::A),
                "b" => Some(Vocab::B),
                _ => None,
            }
        }
        fn as_wire(self) -> &'static str {
            match self {
                Vocab::A => "a",
                Vocab::B => "b",
            }
        }
    }

    fn parse(json: &str) -> AcceptSet<Vocab> {
        serde_json::from_str(json).expect("parse")
    }

    #[test]
    fn singleton_round_trips_as_a_bare_string() {
        let set = parse(r#""a""#);
        assert_eq!(set.known().collect::<Vec<_>>(), vec![Vocab::A]);
        // Byte parity: a single entry serializes back to a bare string.
        assert_eq!(serde_json::to_string(&set).unwrap(), r#""a""#);
        assert_eq!(AcceptSet::single(Vocab::A), set);
    }

    #[test]
    fn list_preserves_order_and_serializes_as_a_list() {
        let set = parse(r#"["b", "a"]"#);
        assert_eq!(set.known().collect::<Vec<_>>(), vec![Vocab::B, Vocab::A]);
        assert_eq!(set.first_known(), Some(Vocab::B));
        assert!(set.accepts(Vocab::A));
        assert_eq!(serde_json::to_string(&set).unwrap(), r#"["b","a"]"#);
    }

    #[test]
    fn unknown_entries_are_tolerated_and_skipped() {
        // A future encoding in a list parses, round-trips verbatim, but is not a
        // selectable known value — graceful forward-compatible degradation.
        let set = parse(r#"["c", "a"]"#);
        assert_eq!(set.known().collect::<Vec<_>>(), vec![Vocab::A]);
        assert_eq!(set.first_known(), Some(Vocab::A));
        assert_eq!(set.wire_names(), vec!["c", "a"]);
        assert_eq!(serde_json::to_string(&set).unwrap(), r#"["c","a"]"#);
    }

    #[test]
    fn a_wholly_unknown_declaration_has_no_known_value() {
        // Parses (no hard fail), but first_known is None: the resolver turns
        // this into a typed resolve-time error, not a parse error.
        let set = parse(r#""rot10d""#);
        assert_eq!(set.first_known(), None);
        assert_eq!(set.known().count(), 0);
        assert_eq!(serde_json::to_string(&set).unwrap(), r#""rot10d""#);
    }

    #[test]
    fn duplicate_entries_are_de_duplicated_in_order() {
        let set = parse(r#"["a", "b", "a"]"#);
        assert_eq!(set.wire_names(), vec!["a", "b"]);
    }

    #[test]
    fn empty_list_is_rejected_at_parse() {
        let err = serde_json::from_str::<AcceptSet<Vocab>>("[]").unwrap_err();
        assert!(
            err.to_string().contains("at least one encoding"),
            "got: {err}"
        );
    }
}
