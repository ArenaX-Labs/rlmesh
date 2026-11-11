//! Binding: a policy map against the robot maps in a session, to a
//! binding table or the full list of refusals.
//!
//! 1. **Normalise** the policy ([`Policy::parse`]).
//! 2. **Selectors**, with run-time `at=` overrides.
//! 3. **Structural candidates**: parts whose path matches and whose kind
//!    equals the entry's (custom kinds by exact name); other parts of the
//!    kind are kept for diagnostics.
//! 4. **Captures**: domains from the required entries' candidates,
//!    intersected; the assignments (at most 4096) under which every required
//!    entry has exactly one candidate survive. One binds; none reports each
//!    entry with no candidate (`unbound`) or several (`ambiguous`) under the
//!    assignment that satisfies most; several is `ambiguous_capture`.
//! 5. **Optional entries** under the chosen assignment: absent binds `blank`
//!    (inputs) or `drop` (actions) when allowed.
//! 6. **Compatibility** against the one candidate: each difference is
//!    automatic, allowed, or refused (`not_allowed`, naming the
//!    conversion, or `incompatible`).
//! 7. **Duplicate targets**: two actions on one part, or on two members of
//!    one command group.
//! 8. **Emit** the table, its delivery contracts, its action grids and
//!    its fingerprint (BLAKE3 over the canonical table).
//!
//! Lease availability is not part of binding.
//!
//! In a scene, a run that names no robot binds against the
//! view ([`crate::spec::view`]): selectors resolve from the view root, a
//! qualified selector (`eye-l:gaze`) in that robot's own map, and actions may
//! target any placed robot, aligned by `t0` only. Lockstep sessions keep
//! every action on one robot.

#[allow(unused_imports)]
use crate::prelude::*;

use crate::wire::part::{DType, FieldSpec, PartSpec};
use crate::wire::{AuthoritySnapshot, ParticipantId, Place};

use crate::spec::explain::hz;
use crate::spec::kinds::Kind;
use crate::spec::normalise::attrs;
use crate::spec::policy::{Action, Entry, Input, Policy, RunOptions};
use crate::spec::selector::Selector;
use crate::spec::view::{self, VIEW};

mod action;
mod image;
mod input;
mod problem;
mod resolve;
mod steps;
mod table;
pub use problem::{BindError, Code, Problem, Refusal};
pub(crate) use problem::{NamesClash, RunCall};
pub use table::{ActionRow, Binding, Dir, InputRow, Status};

/// The most capture assignments the binder enumerates.
pub const MAX_ASSIGNMENTS: usize = 4096;

/// One policy entry, uniformly.
struct Slot<'a> {
    dir: Dir,
    entry: &'a Entry,
    input: Option<&'a Input>,
    action: Option<&'a Action>,
    selector: Option<Selector>,
    overridden: bool,
    robot: ParticipantId,
}

impl Slot<'_> {
    fn required(&self) -> bool {
        !self.entry.optional
    }

    fn subject(&self) -> String {
        let kind = &self.entry.kind_name;
        match &self.selector {
            Some(sel) => format!(
                "{} {:?} (kind {kind}, at {:?})",
                self.dir.word(),
                self.entry.name,
                sel.to_string()
            ),
            None => format!("{} {:?} (kind {kind})", self.dir.word(), self.entry.name),
        }
    }

    fn name(&self) -> String {
        format!("{} {:?}", self.dir.word(), self.entry.name)
    }
}

/// A part a slot's selector matched, with what its captures took.
#[derive(Clone)]
struct Candidate<'a> {
    part: &'a PartSpec,
    captures: Vec<(String, String)>,
}

impl Candidate<'_> {
    fn fits(&self, assignment: &[(String, String)]) -> bool {
        self.captures
            .iter()
            .all(|(n, v)| assignment.iter().any(|(an, av)| an == n && av == v))
    }
}

/// A part's one-line summary in refusals: `camera 640x480 @30 Hz`.
fn summary(part: &PartSpec) -> String {
    let kind = Kind::parse(&part.kind).unwrap_or(Kind::Custom);
    let rate = format!("@{} Hz", hz(part.hz));
    match kind {
        Kind::Camera | Kind::Depth => {
            let shape = &part.fields[0].shape;
            format!("{} {}x{} {rate}", part.kind, shape[1], shape[0])
        }
        Kind::Joints | Kind::Hand => {
            format!("{} {} joints {rate}", part.kind, joint_count(&part.fields))
        }
        Kind::Wrench | Kind::Imu | Kind::Velocity => {
            format!("{} in {} {rate}", part.kind, part.frame)
        }
        Kind::Pose => format!(
            "pose {} in {} {rate}",
            part.attrs.get(attrs::CHILD).map_or("", String::as_str),
            part.frame
        ),
        Kind::Text | Kind::Reward | Kind::Success => part.kind.clone(),
        _ => format!("{} {rate}", part.kind),
    }
}

fn joint_count(fields: &[FieldSpec]) -> u32 {
    fields
        .iter()
        .find_map(|f| f.shape.first().copied())
        .unwrap_or(0)
}

impl Binder<'_> {
    /// `part` as its robot declared it (a view renames it).
    fn declared<'p>(&'p self, part: &'p PartSpec) -> &'p PartSpec {
        self.parts
            .iter()
            .find(|p| p.channel == part.channel)
            .unwrap_or(part)
    }

    /// How a refusal names `part`: `robot:path`.
    fn part_name(&self, part: &PartSpec) -> String {
        let own = self.declared(part);
        own.qualified()
    }

    /// How a refusal names `part`'s applied stream.
    fn applied_name(&self, part: &PartSpec) -> String {
        let own = self.declared(part);
        format!("{}:{}", own.owner, own.path.applied())
    }
}

/// Pads the first column to its widest value plus `gap`.
fn two_columns(rows: &[(String, String)], gap: usize) -> Vec<String> {
    let width = rows
        .iter()
        .map(|(a, _)| a.chars().count())
        .max()
        .unwrap_or(0)
        + gap;
    rows.iter()
        .map(|(a, b)| format!("{a}{}{b}", " ".repeat(width - a.chars().count())))
        .collect()
}

/// Whether `from → to` is an exact cast: `Some(true)` exact, `Some(false)`
/// lossy (`allow: dtype`), `None` refused (bool ↔ float).
fn cast(from: DType, to: DType) -> Option<bool> {
    use DType::*;
    if from == to {
        return Some(true);
    }
    let int_bits = |d: DType| match d {
        U8 | I8 => Some(8),
        U16 | I16 => Some(16),
        U32 | I32 => Some(32),
        U64 | I64 => Some(64),
        _ => None,
    };
    let signed = |d: DType| matches!(d, I8 | I16 | I32 | I64);
    let mantissa = |d: DType| match d {
        F16 => Some(11),
        F32 => Some(24),
        F64 => Some(53),
        _ => None,
    };
    match (from, to) {
        (Bool, t) if t.is_float() => None,
        (f, Bool) if f.is_float() => None,
        (Bool, _) => Some(true),
        (_, Bool) => Some(false),
        (Utf8, _) | (_, Utf8) => None,
        (f, t) if f.is_float() && t.is_float() => Some(mantissa(t) >= mantissa(f)),
        (f, t) if f.is_float() => {
            let _ = t;
            Some(false)
        }
        (f, t) if t.is_float() => Some(int_bits(f).zip(mantissa(t)).is_some_and(|(b, m)| b <= m)),
        (f, t) => {
            let (fb, tb) = (int_bits(f).unwrap_or(64), int_bits(t).unwrap_or(0));
            Some(match (signed(f), signed(t)) {
                (false, false) | (true, true) => tb >= fb,
                (false, true) => tb > fb,
                (true, false) => false,
            })
        }
    }
}

/// The binder's working state for one call.
struct Binder<'a> {
    policy: &'a Policy,
    options: &'a RunOptions,
    robot: ParticipantId,
    parts: &'a [PartSpec],
    /// The view's parts under their view paths, when binding a view.
    view: Option<&'a [PartSpec]>,
    places: &'a [Place],
    scope: Vec<String>,
    run: RunCall,
    problems: Vec<Problem>,
    warnings: Vec<String>,
}

/// What a policy binds against: every robot's parts and, when the session
/// runs a scene, its place table.
#[derive(Clone, Copy, Debug, Default)]
pub struct Target<'a> {
    /// Every declared part.
    pub parts: &'a [PartSpec],
    /// Where each placed robot's parts appear in the view.
    pub places: &'a [Place],
}

impl<'a> Target<'a> {
    /// What `state` offers a binder.
    pub fn of(state: &'a AuthoritySnapshot) -> Self {
        Self {
            parts: &state.parts,
            places: &state.places,
        }
    }
}

/// Binds `policy` against the session's `parts` (every robot's), with the
/// run-time `options`.
pub fn bind(
    policy: &Policy,
    parts: &[PartSpec],
    options: &RunOptions,
) -> Result<Binding, BindError> {
    bind_in(
        policy,
        &Target {
            parts,
            ..Target::default()
        },
        options,
    )
}

/// Binds `policy` against `target` with the run-time `options`: against
/// its view when it has places and `options` names no robot, else against
/// one robot.
pub fn bind_in(
    policy: &Policy,
    target: &Target<'_>,
    options: &RunOptions,
) -> Result<Binding, BindError> {
    let parts = target.parts;
    let mut owners: Vec<&ParticipantId> = Vec::new();
    for part in parts {
        if !owners.contains(&&part.owner) {
            owners.push(&part.owner);
        }
    }
    // A plain session's several robots with no robot named: each in the
    // view under its own name, so one policy drives them all (its actions
    // name their robots: `eye-1:gaze`).
    let implicit;
    let places: &[Place] =
        if target.places.is_empty() && options.robot.is_none() && owners.len() > 1 {
            implicit = view::each_at_its_name(&owners);
            &implicit
        } else {
            target.places
        };
    let view = match options.robot {
        None if !places.is_empty() => Some(view::parts(parts, places).map_err(BindError::Options)?),
        _ => None,
    };
    let robot = match (&options.robot, owners.as_slice()) {
        (None, _) if view.is_some() => {
            ParticipantId::new(VIEW).map_err(|e| BindError::Options(e.to_string()))?
        }
        (Some(robot), _) if owners.contains(&robot) => robot.clone(),
        (Some(robot), _) => {
            return Err(BindError::Options(format!("{robot} has declared no parts")));
        }
        (None, [one]) => (*one).clone(),
        (None, []) => return Err(BindError::Options("no robot has declared parts".into())),
        (None, many) => {
            let names: Vec<String> = many.iter().map(ToString::to_string).collect();
            return Err(BindError::Options(format!(
                "{} robots ({}): pass robot=",
                many.len(),
                names.join(", ")
            )));
        }
    };
    let names: Vec<&str> = policy
        .inputs
        .iter()
        .map(|i| i.entry.name.as_str())
        .chain(policy.actions.iter().map(|a| a.entry.name.as_str()))
        .collect();
    for entry in options.at.keys().chain(&options.positional) {
        if !names.contains(&entry.as_str()) {
            return Err(BindError::Options(format!(
                "an override names {entry:?}, which is not an entry of {}",
                options.name
            )));
        }
    }
    let binder = Binder {
        policy,
        options,
        robot,
        parts,
        view: view.as_deref(),
        places: if view.is_some() { places } else { &[] },
        scope: options
            .under
            .as_ref()
            .map(|u| u.as_str().split('/').map(String::from).collect())
            .unwrap_or_default(),
        run: RunCall::of(options),
        problems: Vec::new(),
        warnings: Vec::new(),
    };
    binder.bind()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_casts_are_automatic() {
        assert_eq!(cast(DType::U8, DType::F32), Some(true));
        assert_eq!(cast(DType::I16, DType::F32), Some(true));
        assert_eq!(cast(DType::I32, DType::F32), Some(false));
        assert_eq!(cast(DType::F32, DType::F64), Some(true));
        assert_eq!(cast(DType::F64, DType::F32), Some(false));
        assert_eq!(cast(DType::U16, DType::I16), Some(false));
        assert_eq!(cast(DType::U16, DType::I32), Some(true));
        assert_eq!(cast(DType::Bool, DType::F32), None);
        assert_eq!(cast(DType::Bool, DType::U8), Some(true));
    }
}
