//! Selectors (§6.4): which parts an entry's `at` names.
//!
//! | Selector | Meaning |
//! |---|---|
//! | `robot:...` | Resolved in that robot, ignoring the scope. |
//! | `/a/b` | Absolute: from the robot root, ignoring the scope. |
//! | `a/b` | Relative: from the scope root (`under`, default the robot root). |
//! | segment `*` | Exactly one segment, any value. |
//! | segment `{name}` | Exactly one segment, captured; one name takes one value across entries. |
//! | no `at` | Every part of the kind at any depth below the scope root. |
//!
//! A run-time `at=` override is a literal path from the robot root.

use std::fmt;

use rlmesh_wire::ParticipantId;
use rlmesh_wire::path::{MAX_DEPTH, MAX_SEGMENT_LEN, is_identifier};

/// Run-time option names, which cannot be capture names.
pub const RUN_OPTIONS: [&str; 7] = ["robot", "under", "at", "names", "allow", "session", "clock"];

/// One selector segment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Segment {
    /// This segment exactly.
    Literal(String),
    /// Any one segment.
    Any,
    /// Any one segment, captured under this name.
    Capture(String),
}

/// A parsed selector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Selector {
    /// `robot:` qualification.
    pub robot: Option<ParticipantId>,
    /// From the robot root rather than the scope root.
    pub absolute: bool,
    /// Its segments.
    pub segments: Vec<Segment>,
    text: String,
}

impl Selector {
    /// Parses an `at` selector.
    pub fn parse(text: &str) -> Result<Self, String> {
        let refuse = |why: &str| Err(format!("at {text:?}: {why}"));
        let (robot, rest) = match text.split_once(':') {
            Some((robot, rest)) => (
                Some(ParticipantId::new(robot).map_err(|err| format!("at {text:?}: {err}"))?),
                rest,
            ),
            None => (None, text),
        };
        let (absolute, rest) = match rest.strip_prefix('/') {
            Some(rest) => (true, rest),
            None => (robot.is_some(), rest),
        };
        if rest.is_empty() {
            return refuse("names no path");
        }
        let mut segments = Vec::new();
        for segment in rest.split('/') {
            segments.push(if segment == "*" {
                Segment::Any
            } else if let Some(name) = segment.strip_prefix('{').and_then(|s| s.strip_suffix('}')) {
                if !is_identifier(name, MAX_SEGMENT_LEN) {
                    return refuse("a capture is {name} with name [a-z][a-z0-9_]*");
                }
                if RUN_OPTIONS.contains(&name) {
                    return refuse("a run-time option name cannot be a capture");
                }
                Segment::Capture(name.to_string())
            } else if is_identifier(segment, MAX_SEGMENT_LEN) {
                Segment::Literal(segment.to_string())
            } else {
                return refuse("segments are [a-z][a-z0-9_]*, `*` or `{name}`");
            });
        }
        if segments.len() > MAX_DEPTH {
            return refuse("more than 6 segments");
        }
        Ok(Self {
            robot,
            absolute,
            segments,
            text: text.to_string(),
        })
    }

    /// A literal path from the robot root (a run-time override).
    pub fn literal(path: &rlmesh_wire::PartPath) -> Self {
        Self {
            robot: None,
            absolute: true,
            segments: path
                .as_str()
                .split('/')
                .map(|s| Segment::Literal(s.to_string()))
                .collect(),
            text: path.to_string(),
        }
    }

    /// The capture names, in order of first appearance.
    pub fn captures(&self) -> Vec<&str> {
        let mut names: Vec<&str> = Vec::new();
        for segment in &self.segments {
            if let Segment::Capture(name) = segment
                && !names.contains(&name.as_str())
            {
                names.push(name);
            }
        }
        names
    }

    /// Whether `path` matches under `scope` (segments of the scope root,
    /// ignored for absolute selectors). On a match, the captured values in
    /// [`Self::captures`] order.
    pub fn matches(&self, path: &str, scope: &[&str]) -> Option<Vec<(String, String)>> {
        let segments: Vec<&str> = path.split('/').collect();
        let rest = if self.absolute {
            segments.as_slice()
        } else {
            segments.strip_prefix(scope)?
        };
        if rest.len() != self.segments.len() {
            return None;
        }
        let mut captured: Vec<(String, String)> = Vec::new();
        for (want, have) in self.segments.iter().zip(rest) {
            match want {
                Segment::Literal(lit) if lit != have => return None,
                Segment::Literal(_) | Segment::Any => {}
                Segment::Capture(name) => match captured.iter().find(|(n, _)| n == name) {
                    Some((_, value)) if value != have => return None,
                    Some(_) => {}
                    None => captured.push((name.clone(), (*have).to_string())),
                },
            }
        }
        Some(captured)
    }
}

/// Whether `path` is below the scope root (any depth): the no-`at` search.
pub fn below(path: &str, scope: &[&str]) -> bool {
    let segments: Vec<&str> = path.split('/').collect();
    segments.len() > scope.len() && segments.starts_with(scope)
}

impl fmt::Display for Selector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(selector: &str, path: &str, scope: &[&str]) -> Option<Vec<(String, String)>> {
        Selector::parse(selector)
            .expect("selector")
            .matches(path, scope)
    }

    #[test]
    fn selectors_resolve_from_the_scope_or_the_root() {
        assert_eq!(m("arm", "arm", &[]), Some(vec![]));
        assert_eq!(m("arm", "right/arm", &["right"]), Some(vec![]));
        assert_eq!(m("arm", "arm", &["right"]), None);
        assert_eq!(m("/scene/cam", "scene/cam", &["right"]), Some(vec![]));
        assert_eq!(m("*/cam", "wrist/cam", &[]), Some(vec![]));
        assert_eq!(m("*/cam", "left/wrist/cam", &[]), None, "one segment");
        assert_eq!(
            m("{side}/hand/tactile", "left/hand/tactile", &[]),
            Some(vec![("side".into(), "left".into())])
        );
        assert_eq!(m("{s}/x/{s}", "a/x/b", &[]), None, "one value per name");
        let qualified = Selector::parse("teleop:goal/cmd_vel").expect("qualified");
        assert_eq!(qualified.robot.as_ref().map(|r| r.as_str()), Some("teleop"));
        assert!(qualified.absolute && qualified.matches("goal/cmd_vel", &["right"]).is_some());
        assert!(below("right/wrist/cam", &["right"]) && !below("right", &["right"]));
        assert!(below("task", &[]));
    }

    #[test]
    fn malformed_selectors_are_refused() {
        for bad in [
            "",
            "/",
            "Arm",
            "a//b",
            "a/**",
            "{Side}/x",
            "{at}/x",
            "a/b/c/d/e/f/g",
            "x:",
        ] {
            assert!(Selector::parse(bad).is_err(), "{bad:?}");
        }
        assert_eq!(
            Selector::parse("{side}/{arm}/{side}")
                .expect("ok")
                .captures(),
            ["side", "arm"]
        );
    }
}
