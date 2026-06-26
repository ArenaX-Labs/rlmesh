//! Structured paths into a (possibly nested) observation or payload tree.
//!
//! Replaces the old dotted-string observation keys and the reserved `"."` root
//! sentinel. A path is an ordered list of [`PathSeg`] steps; an **empty** path
//! is the root (a single-leaf space, or a bare payload). A `Key` segment is a
//! `Dict` descent and an `Index` segment is a `Tuple` descent, so the two are
//! never ambiguous — a literal Dict key that contains a dot is just one `Key`
//! segment, and a Tuple element is addressable by position.
//!
//! The same type serves two roles, distinguished only by the field name that
//! holds it: a *source* (env side — where a feature is read from the raw
//! observation) and a *placement* (model side — where a produced tensor lands
//! in the payload). They are never mixed within one value.

use std::fmt;

/// One step into a nested tree: a `Dict` key or a `Tuple` index.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PathSeg {
    Key(String),
    Index(usize),
}

/// An ordered path into an observation or payload tree. Empty = the root.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct NodePath(pub Vec<PathSeg>);

impl NodePath {
    /// The root path (empty) — the whole single-leaf space / bare payload.
    pub fn root() -> Self {
        NodePath(Vec::new())
    }

    /// Whether this is the root (empty) path.
    pub fn is_root(&self) -> bool {
        self.0.is_empty()
    }

    /// This path extended by a `Dict` key.
    pub fn push_key(&self, key: impl Into<String>) -> Self {
        let mut segments = self.0.clone();
        segments.push(PathSeg::Key(key.into()));
        NodePath(segments)
    }

    /// This path extended by a `Tuple` index.
    pub fn push_index(&self, index: usize) -> Self {
        let mut segments = self.0.clone();
        segments.push(PathSeg::Index(index));
        NodePath(segments)
    }

    /// The first segment, if any — the top-level observation entry a source
    /// lives under (replaces the old `top_level_key`).
    pub fn first(&self) -> Option<&PathSeg> {
        self.0.first()
    }

    /// The segments after the first, as a borrowed slice.
    pub fn rest(&self) -> &[PathSeg] {
        self.0.get(1..).unwrap_or(&[])
    }
}

/// Renders as a readable dotted/bracketed path for error messages and for the
/// canonical string keys frame-stacking uses (`robot.eef_pos`, `sensors[0]`,
/// `<root>` for the empty path).
impl fmt::Display for NodePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return f.write_str("<root>");
        }
        for (position, segment) in self.0.iter().enumerate() {
            match segment {
                PathSeg::Key(key) => {
                    if position > 0 {
                        f.write_str(".")?;
                    }
                    f.write_str(key)?;
                }
                PathSeg::Index(index) => write!(f, "[{index}]")?,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_and_renders_paths() {
        let root = NodePath::root();
        assert!(root.is_root());
        assert_eq!(root.to_string(), "<root>");

        let nested = root.push_key("robot").push_key("eef_pos");
        assert_eq!(nested.to_string(), "robot.eef_pos");
        assert_eq!(nested.first(), Some(&PathSeg::Key("robot".to_owned())));
        assert_eq!(nested.rest(), &[PathSeg::Key("eef_pos".to_owned())]);

        let tuple = root.push_key("sensors").push_index(0);
        assert_eq!(tuple.to_string(), "sensors[0]");

        // A literal dot in a key is one segment, never split.
        let dotted = root.push_key("weird.key");
        assert_eq!(dotted.0.len(), 1);
    }
}
