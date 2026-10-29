//! Gym keys to path segments (§12.5).
//!
//! A `Dict` key becomes one segment: lowercased, every character outside
//! `[a-z0-9_]` replaced by `_`, a leading digit given an `n` prefix, and a
//! segment over 32 characters cut to its first 23 characters + `_` + 8 hex
//! characters of the BLAKE3 hash of the original key. A `Tuple` item `i` is
//! the segment `item{i}`. Two readings the table leaves open, taken here as
//! the smallest that keeps every path valid (§2.1): a leading `_` (a reserved
//! segment) also gets the `n` prefix, and an empty key is `n`. Keys that
//! escape alike (`Pos` and `pos`) are not refused: the later leaf's last
//! segment is cut the same way, with the hash of its original keys
//! ([`distinct`]).

/// The longest segment (§2.1).
const MAX_SEGMENT: usize = 32;

/// One step into a Gym space: a `Dict` key or a `Tuple` index.
#[derive(
    Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(untagged)]
pub enum KeySeg {
    /// A `Dict` key, verbatim.
    Key(String),
    /// A `Tuple` index.
    Item(usize),
}

impl std::fmt::Display for KeySeg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Key(key) => write!(f, "[{key:?}]"),
            Self::Item(i) => write!(f, "[{i}]"),
        }
    }
}

/// The original keys of a leaf, for messages: `["obs"][0]["x"]`.
pub fn key_text(keys: &[KeySeg]) -> String {
    if keys.is_empty() {
        return "the root".into();
    }
    keys.iter().map(ToString::to_string).collect()
}

/// One escaped path segment.
pub fn segment(key: &KeySeg) -> String {
    let key = match key {
        KeySeg::Item(i) => return format!("item{i}"),
        KeySeg::Key(key) => key,
    };
    let mut out: String = key
        .chars()
        .map(|c| {
            let c = c.to_ascii_lowercase();
            if c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if out.is_empty() || !out.starts_with(|c: char| c.is_ascii_lowercase()) {
        out.insert(0, 'n');
    }
    if out.len() > MAX_SEGMENT {
        out.truncate(23);
        out.push('_');
        out.push_str(&hash8(key));
    }
    out
}

/// `path` with its last segment made distinct for the leaf `origin` (its
/// original keys): the segment's first 23 characters + `_` + 8 hex
/// characters of the BLAKE3 hash of `origin`, as an over-long key is cut.
pub fn distinct(path: &str, origin: &str) -> String {
    let (head, last) = path.rsplit_once('/').map_or(("", path), |(h, l)| (h, l));
    let mut segment: String = last.chars().take(23).collect();
    segment.push('_');
    segment.push_str(&hash8(origin));
    if head.is_empty() {
        segment
    } else {
        format!("{head}/{segment}")
    }
}

fn hash8(text: &str) -> String {
    blake3::hash(text.as_bytes()).as_bytes()[..4]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The escaped path of a key path below `prefix` (empty: the root).
pub fn path(prefix: &[&str], keys: &[KeySeg]) -> String {
    prefix
        .iter()
        .map(|s| (*s).to_string())
        .chain(keys.iter().map(segment))
        .collect::<Vec<_>>()
        .join("/")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(k: &str) -> KeySeg {
        KeySeg::Key(k.into())
    }

    #[test]
    fn keys_escape_to_valid_segments() {
        assert_eq!(segment(&key("Image")), "image");
        assert_eq!(segment(&key("agent-pos.x")), "agent_pos_x");
        assert_eq!(segment(&key("0rgb")), "n0rgb");
        assert_eq!(segment(&key("_hidden")), "n_hidden");
        assert_eq!(segment(&key("")), "n");
        assert_eq!(segment(&KeySeg::Item(3)), "item3");
        let long = "a_very_long_observation_key_that_goes_on";
        let s = segment(&key(long));
        assert_eq!(s.len(), 32);
        assert!(s.starts_with("a_very_long_observation_"));
        let hex = &s[24..];
        let expected = blake3::hash(long.as_bytes());
        assert_eq!(
            hex,
            format!(
                "{:02x}{:02x}{:02x}{:02x}",
                expected.as_bytes()[0],
                expected.as_bytes()[1],
                expected.as_bytes()[2],
                expected.as_bytes()[3]
            )
        );
        for s in [segment(&key("Ünïcode")), s] {
            rlmesh_session::wire::PartPath::new(&s).expect("valid");
        }
        assert_eq!(
            path(&["action"], &[key("Arm"), KeySeg::Item(0)]),
            "action/arm/item0"
        );
        assert_eq!(key_text(&[key("a"), KeySeg::Item(1)]), "[\"a\"][1]");
    }
}
