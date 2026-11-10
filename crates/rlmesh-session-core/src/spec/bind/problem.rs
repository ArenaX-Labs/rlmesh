//! Refusals: their codes and problems, the re-run call they suggest, and
//! how the binder records them.

use super::table::Dir;
use super::{Binder, Slot};
use crate::prelude::*;

use core::fmt;

use crate::wire::ParticipantId;
use crate::wire::part::PartSpec;

use crate::spec::kinds::Kind;
use crate::spec::policy::{Allow, Allows, Entry, RunOptions};

/// A binder diagnostic code (lowercase; wire statuses are uppercase and
/// never share a name).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Code {
    /// A required entry has no candidate.
    Unbound,
    /// An entry has several candidates.
    Ambiguous,
    /// Several capture assignments bind every required entry.
    AmbiguousCapture,
    /// A conversion exists but is not allowed.
    NotAllowed,
    /// No conversion exists.
    Incompatible,
    /// Two actions target one part or one command group.
    DuplicateTarget,
    /// The captures allow more than 4096 assignments.
    PatternTooBroad,
}

impl Code {
    /// Its name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Unbound => "unbound",
            Self::Ambiguous => "ambiguous",
            Self::AmbiguousCapture => "ambiguous_capture",
            Self::NotAllowed => "not_allowed",
            Self::Incompatible => "incompatible",
            Self::DuplicateTarget => "duplicate_target",
            Self::PatternTooBroad => "pattern_too_broad",
        }
    }
}

impl fmt::Display for Code {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Model and robot names that share nothing, on one part: rendered as one
/// block for every entry they affect.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct NamesClash {
    pub(crate) part: String,
    pub(crate) model: Vec<String>,
    pub(crate) robot: Vec<String>,
}

/// One refusal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Problem {
    /// Its code.
    pub code: Code,
    /// The entry's direction.
    pub dir: Dir,
    /// The entry.
    pub entry: String,
    /// The first line, after `[code] `.
    pub(crate) head: String,
    /// The lines under it, indented.
    pub(crate) body: Vec<String>,
    /// A names clash (grouped across entries when rendered).
    pub(crate) names: Option<NamesClash>,
}

/// A refused binding: every problem found.
#[derive(Clone, Debug, PartialEq)]
pub struct Refusal {
    /// The policy.
    pub policy: String,
    /// The robot.
    pub robot: ParticipantId,
    /// The scope.
    pub under: Option<String>,
    /// The captures given.
    pub captures: Vec<(String, String)>,
    /// The problems, in entry order.
    pub problems: Vec<Problem>,
    pub(crate) run: RunCall,
}

/// Why nothing was bound.
#[derive(Clone, Debug, PartialEq)]
pub enum BindError {
    /// The run options do not fit the session or the policy.
    Options(String),
    /// Bound and refused, with every problem.
    Refused(Box<Refusal>),
}

impl fmt::Display for BindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Options(why) => write!(f, "binding refused: {why}"),
            Self::Refused(refusal) => refusal.fmt(f),
        }
    }
}

impl core::error::Error for BindError {}

/// The options a refusal's fix repeats, plus what it adds.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct RunCall {
    robot: Option<String>,
    under: Option<String>,
    captures: Vec<(String, String)>,
    at: Vec<(String, String)>,
    names: Vec<String>,
    allow: Vec<String>,
}

impl RunCall {
    pub(super) fn of(options: &RunOptions) -> Self {
        Self {
            robot: options.robot.as_ref().map(ToString::to_string),
            under: options.under.as_ref().map(ToString::to_string),
            captures: options
                .captures
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            at: options
                .at
                .iter()
                .map(|(k, v)| (k.clone(), v.to_string()))
                .collect(),
            names: options.positional.clone(),
            allow: options.allow.iter().map(|a| a.name().to_string()).collect(),
        }
    }

    pub(crate) fn with_at(&self, entry: &str, path: &str) -> String {
        let mut call = self.clone();
        call.at.retain(|(e, _)| e != entry);
        call.at.push((entry.into(), path.into()));
        call.render()
    }

    pub(crate) fn with_positional<'a>(&self, entries: impl IntoIterator<Item = &'a str>) -> String {
        let mut call = self.clone();
        for entry in entries {
            if !call.names.iter().any(|e| e == entry) {
                call.names.push(entry.into());
            }
        }
        call.render()
    }

    pub(crate) fn with_captures(&self, captures: &[(String, String)]) -> String {
        let mut call = self.clone();
        for (k, v) in captures {
            call.captures.retain(|(n, _)| n != k);
            call.captures.push((k.clone(), v.clone()));
        }
        call.render()
    }

    fn render(&self) -> String {
        let quote = |s: &str| format!("{s:?}");
        let mut args = vec!["...".to_string()];
        if let Some(robot) = &self.robot {
            args.push(format!("robot={}", quote(robot)));
        }
        if let Some(under) = &self.under {
            args.push(format!("under={}", quote(under)));
        }
        for (k, v) in &self.captures {
            args.push(format!("{k}={}", quote(v)));
        }
        if !self.at.is_empty() {
            let items: Vec<String> = self
                .at
                .iter()
                .map(|(e, p)| format!("{}: {}", quote(e), quote(p)))
                .collect();
            args.push(format!("at={{{}}}", items.join(", ")));
        }
        if !self.names.is_empty() {
            let items: Vec<String> = self
                .names
                .iter()
                .map(|e| format!("{}: \"positional\"", quote(e)))
                .collect();
            args.push(format!("names={{{}}}", items.join(", ")));
        }
        if !self.allow.is_empty() {
            let items: Vec<String> = self.allow.iter().map(|a| quote(a)).collect();
            args.push(format!("allow=[{}]", items.join(", ")));
        }
        format!("rlmesh.run({})", args.join(", "))
    }
}

pub(super) fn problem(code: Code, slot: &Slot<'_>, head: String, body: Vec<String>) -> Problem {
    Problem {
        code,
        dir: slot.dir,
        entry: slot.entry.name.clone(),
        head,
        body,
        names: None,
    }
}

impl<'a> Binder<'a> {
    pub(super) fn refusal(&mut self) -> BindError {
        BindError::Refused(Box::new(Refusal {
            policy: self.options.name.clone(),
            robot: self.robot.clone(),
            under: self.options.under.as_ref().map(ToString::to_string),
            captures: self
                .options
                .captures
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            problems: core::mem::take(&mut self.problems),
            run: self.run.clone(),
        }))
    }

    pub(super) fn allowed(&self, entry: &Entry, allow: Allow) -> bool {
        entry.allow.has(allow) || self.options.allow.contains(&allow)
    }

    pub(super) fn allows(&self, entry: &Entry) -> Allows {
        self.options
            .allow
            .iter()
            .fold(entry.allow, |set, a| set.with(*a))
    }

    pub(super) fn unbound(&mut self, slot: &Slot<'a>, considered: &[&'a PartSpec]) {
        let kind = &slot.entry.kind_name;
        let (head, body) = if considered.is_empty() {
            let need = if slot.entry.kind == Kind::Custom {
                format!("a {kind} part")
            } else {
                slot.entry.kind.need().to_string()
            };
            (
                format!("{}: {} has no {kind} part.", slot.subject(), slot.robot),
                vec![format!(
                    "    fix: this policy needs {need}; it cannot run on {}.",
                    slot.robot
                )],
            )
        } else {
            let paths: Vec<String> = considered.iter().map(|p| p.path.to_string()).collect();
            (
                format!(
                    "{}: no part matches; {}'s {kind} parts: {}.",
                    slot.subject(),
                    slot.robot,
                    paths.join(", ")
                ),
                vec![format!(
                    "    fix: if one of them is meant, {}",
                    self.run.with_at(&slot.entry.name, &paths[0])
                )],
            )
        };
        self.problems.push(problem(Code::Unbound, slot, head, body));
    }

    pub(super) fn not_allowed(&mut self, slot: &Slot<'_>, what: String, allow: Allow) {
        self.problems.push(problem(
            Code::NotAllowed,
            slot,
            format!("{}: {what}", slot.name()),
            vec![
                format!("    needs: {allow}"),
                format!("    fix: add {:?} to allow", allow.name()),
            ],
        ));
    }

    pub(super) fn incompatible(&mut self, slot: &Slot<'_>, head: String, body: Vec<String>) {
        self.problems
            .push(problem(Code::Incompatible, slot, head, body));
    }
}
