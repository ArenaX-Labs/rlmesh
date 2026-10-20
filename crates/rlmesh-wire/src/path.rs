//! Part paths (§2): the only address of a part.
//!
//! A path is 1-6 `/`-separated segments of `[a-z][a-z0-9_]*`, each 1-32
//! characters, at most 96 bytes in all, with no leading `/`. Every path is
//! therefore a run of valid Zenoh key chunks (no wildcards, no `@`, no `$`)
//! and every segment is an identifier in Python, Rust and C#. Segments
//! starting with `_` are reserved: the edge's derived `{path}/_applied`
//! stream ([`APPLIED`]) and the host's `_session/...` streams are never
//! declared.

use std::fmt;
use std::str::FromStr;

use crate::ids::IdError;

/// The most segments a path has.
pub const MAX_DEPTH: usize = 6;
/// The longest segment.
pub const MAX_SEGMENT_LEN: usize = 32;
/// The longest path, in bytes.
pub const MAX_PATH_LEN: usize = 96;
/// The segment the edge's applied stream adds to a commandable part's path
/// (§9.4).
pub const APPLIED: &str = "_applied";

/// Whether `segment` is `[a-z][a-z0-9_]*` of 1 to `max` characters.
pub fn is_identifier(segment: &str, max: usize) -> bool {
    let bytes = segment.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= max
        && bytes[0].is_ascii_lowercase()
        && bytes
            .iter()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || *b == b'_')
}

/// A validated part path.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PartPath(String);

impl PartPath {
    /// Validates `path` against §2.1.
    pub fn new(path: impl Into<String>) -> Result<Self, IdError> {
        let path = path.into();
        let refuse = |reason: &str| Err(IdError::new("path", &path, reason));
        if path.is_empty() || path.len() > MAX_PATH_LEN {
            return refuse("must be 1-96 bytes");
        }
        let mut depth = 0;
        for segment in path.split('/') {
            depth += 1;
            if segment.starts_with('_') {
                return refuse("segments starting with `_` are reserved");
            }
            if !is_identifier(segment, MAX_SEGMENT_LEN) {
                return refuse("segments must be 1-32 characters of [a-z][a-z0-9_]*");
            }
        }
        if depth > MAX_DEPTH {
            return refuse("must have at most 6 segments");
        }
        Ok(Self(path))
    }

    /// The path as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The path of this part's derived applied stream.
    pub fn applied(&self) -> String {
        format!("{}/{APPLIED}", self.0)
    }
}

impl fmt::Display for PartPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl FromStr for PartPath {
    type Err = IdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

impl AsRef<str> for PartPath {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_follow_the_syntax() {
        for good in [
            "arm",
            "arm/tcp",
            "left/wrist/cam",
            "a/b/c/d/e/f",
            "x9_y",
            &"a".repeat(32),
        ] {
            assert!(PartPath::new(good).is_ok(), "{good:?} refused");
        }
        let long = format!("{}/{}/{}", "a".repeat(32), "b".repeat(32), "c".repeat(32));
        assert_eq!(long.len(), 98);
        for bad in [
            "",
            "/arm",
            "arm/",
            "arm//tcp",
            "Arm",
            "wrist-cam",
            "wrist.cam",
            "9arm",
            "_session/episode",
            "arm/_applied",
            "a/b/c/d/e/f/g",
            "arm/*",
            "arm/**",
            "@adv",
            "a b",
            &"a".repeat(33),
            &long,
        ] {
            assert!(PartPath::new(bad).is_err(), "{bad:?} accepted");
        }
        let path = PartPath::new("arm/tcp").expect("path");
        assert_eq!(path.applied(), "arm/tcp/_applied");
        assert!(is_identifier("rlmesh", 32) && !is_identifier("", 32));
    }
}
