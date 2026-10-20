//! Binding (§6.5): a policy map against the robot maps in a session, to a
//! binding table or the full list of refusals.
//!
//! 1. **Normalise** the policy ([`Policy::parse`]).
//! 2. **Selectors** (§6.4), with run-time `at=` overrides.
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
//!    automatic (§7.1), allowed (§7.2), or refused (`not_allowed`, naming the
//!    conversion, or `incompatible`).
//! 7. **Duplicate targets**: two actions on one part, or on two members of
//!    one command group.
//! 8. **Emit** the table, its delivery contracts (§8.2), its action grids and
//!    its fingerprint (BLAKE3 over the canonical table).
//!
//! Lease availability is not part of binding.

use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

use rlmesh_wire::part::{DType, Encoding, FieldSpec, MAX_CHUNK_BYTES, PartSpec};
use rlmesh_wire::{ActionGrid, ChannelId, Fingerprint, ParticipantId, StreamContract};
use serde_json::{Value, json};

use crate::canonical;
use crate::contract;
use crate::explain::{hz, ms, num};
use crate::kinds::{Kind, quantity};
use crate::normalise::attrs;
use crate::plan::{Delivered, Delivery, Fit, Order, Plan, Size, Step};
use crate::policy::{
    Action, Allow, Allows, ChunkStart, Entry, ImageFormat, ImageLayout, Input, Names, Policy,
    Resample, RunOptions, Source,
};
use crate::rotation::{self, Rotation, RotationChange};
use crate::selector::{Selector, below};
use crate::units::{self, Quantity, UnitChange};

/// The most capture assignments the binder enumerates.
pub const MAX_ASSIGNMENTS: usize = 4096;
/// How far `history_hz` may sit from the source rate.
const HISTORY_HZ_TOLERANCE: f64 = 0.01;

/// An entry's direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dir {
    /// An input.
    In,
    /// An action.
    Out,
}

impl Dir {
    fn word(self) -> &'static str {
        match self {
            Self::In => "input",
            Self::Out => "action",
        }
    }

    /// `in` or `out`, as tables show it.
    pub fn short_name(self) -> &'static str {
        match self {
            Self::In => "in",
            Self::Out => "out",
        }
    }
}

/// A row's status.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Status {
    /// Bound by the policy's own selector.
    Bound,
    /// Bound through a run-time override (`at=`, `names=`).
    Override,
    /// An absent optional input bound to a constant.
    Blank,
    /// An absent optional action, discarded.
    Dropped,
}

impl Status {
    /// Its name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Bound => "bound",
            Self::Override => "override",
            Self::Blank => "blank",
            Self::Dropped => "dropped",
        }
    }
}

/// A binder diagnostic code (lowercase; wire statuses are uppercase and
/// never share a name, §6.5).
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

/// One refusal (§6.7).
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

impl std::error::Error for BindError {}

/// A bound input.
#[derive(Clone, Debug, PartialEq)]
pub struct InputRow {
    /// The entry.
    pub entry: String,
    /// Its kind.
    pub kind: String,
    /// The part it reads (`None`: blank).
    pub part: Option<PartSpec>,
    /// The stream: the part's, or its applied stream's.
    pub channel: Option<ChannelId>,
    /// Reading the applied stream.
    pub applied: bool,
    /// Its status.
    pub status: Status,
    /// What is done to each sample.
    pub plan: Plan,
    /// What `predict` receives.
    pub delivers: Delivery,
    /// The window (1: none).
    pub history: u32,
    /// The source rate S.
    pub source_hz: f64,
    /// The freshness bound at the policy tick (`None`: events and blanks).
    pub max_age: Option<Duration>,
    /// Whether it was declared rather than derived.
    pub max_age_explicit: bool,
    /// A required input (not `optional`).
    pub required: bool,
    /// `allow: hold`.
    pub hold: bool,
    /// `allow: warmup`.
    pub warmup: bool,
}

impl InputRow {
    fn path(&self) -> String {
        match (&self.part, self.applied) {
            (Some(p), true) => p.path.applied(),
            (Some(p), false) => p.path.to_string(),
            (None, _) => "none".into(),
        }
    }
}

/// A bound action.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionRow {
    /// The entry.
    pub entry: String,
    /// Its kind.
    pub kind: String,
    /// The part it commands (`None`: dropped).
    pub part: Option<PartSpec>,
    /// Its status.
    pub status: Status,
    /// What is done to each action.
    pub plan: Plan,
    /// What `predict` returns.
    pub delivers: Delivery,
    /// The action grid rate A.
    pub grid_hz: f64,
    /// Whether A was declared (else the replan rate).
    pub grid_declared: bool,
    /// `1 / A`.
    pub dt: Duration,
    /// Actions per chunk.
    pub horizon: u32,
    /// The chunk's life: `min(horizon × dt, ttl)`.
    pub valid: Duration,
    /// When a chunk starts.
    pub t0: ChunkStart,
}

/// A binding table (§6.8).
#[derive(Clone, Debug, PartialEq)]
pub struct Binding {
    /// The policy.
    pub policy: String,
    /// The robot its actions target.
    pub robot: ParticipantId,
    /// The scope.
    pub under: Option<String>,
    /// Capture values, given or inferred.
    pub captures: Vec<(String, String)>,
    /// The replan rate R.
    pub replan_hz: f64,
    /// Inputs, in policy order.
    pub inputs: Vec<InputRow>,
    /// Actions, in policy order.
    pub actions: Vec<ActionRow>,
    /// One delivery contract per bound input stream.
    pub contracts: Vec<StreamContract>,
    /// Accepted with a warning (horizon past the edge's action-age cap).
    pub warnings: Vec<String>,
    /// The fingerprint of [`Self::table_json`].
    pub fingerprint: Fingerprint,
}

impl Binding {
    /// The commandable parts it leases, in action order.
    pub fn channels(&self) -> Vec<ChannelId> {
        self.actions
            .iter()
            .filter_map(|a| a.part.as_ref().map(|p| p.channel))
            .collect()
    }

    /// The action grid per commanded part (§9.2).
    pub fn grids(&self) -> Vec<ActionGrid> {
        self.actions
            .iter()
            .filter_map(|a| {
                a.part.as_ref().map(|p| ActionGrid {
                    channel: p.channel,
                    dt: a.dt,
                })
            })
            .collect()
    }

    /// The input bound to `entry`.
    pub fn input(&self, entry: &str) -> Option<&InputRow> {
        self.inputs.iter().find(|r| r.entry == entry)
    }

    /// The action bound to `entry`.
    pub fn action(&self, entry: &str) -> Option<&ActionRow> {
        self.actions.iter().find(|r| r.entry == entry)
    }

    /// The machine-readable table: what `DeclareBinding.table_json` carries
    /// and the fingerprint is taken over (canonical JSON).
    pub fn table_json(&self) -> Value {
        table_value(self)
    }

    /// [`Self::table_json`] as RFC 8785 text.
    pub fn table_text(&self) -> String {
        canonical::to_string(&self.table_json())
    }
}

fn table_value(b: &Binding) -> Value {
    let inputs: Vec<Value> = b
        .inputs
        .iter()
        .map(|r| {
            json!({
                "entry": r.entry,
                "kind": r.kind,
                "part": r.part.as_ref().map(|p| p.path.to_string()),
                "channel": r.channel.map(|c| c.0),
                "source": if r.applied { "applied" } else { "published" },
                "status": r.status.name(),
                "plan": r.plan.steps.iter().map(ToString::to_string).collect::<Vec<_>>(),
                "notes": r.plan.notes,
                "delivers": r.delivers.to_string(),
                "history": r.history,
                "max_age_ms": r.max_age.map(|d| d.as_secs_f64() * 1e3),
                "max_age_explicit": r.max_age_explicit,
                "required": r.required,
                "hold": r.hold,
                "warmup": r.warmup,
            })
        })
        .collect();
    let actions: Vec<Value> = b
        .actions
        .iter()
        .map(|r| {
            json!({
                "entry": r.entry,
                "kind": r.kind,
                "part": r.part.as_ref().map(|p| p.path.to_string()),
                "channel": r.part.as_ref().map(|p| p.channel.0),
                "status": r.status.name(),
                "plan": r.plan.steps.iter().map(ToString::to_string).collect::<Vec<_>>(),
                "delivers": r.delivers.to_string(),
                "dt_ns": u64::try_from(r.dt.as_nanos()).unwrap_or(u64::MAX),
                "horizon": r.horizon,
                "valid_ns": u64::try_from(r.valid.as_nanos()).unwrap_or(u64::MAX),
                "t0": match r.t0 { ChunkStart::Snapshot => "snapshot", ChunkStart::Send => "send" },
            })
        })
        .collect();
    json!({
        "policy": b.policy,
        "robot": b.robot.as_str(),
        "under": b.under,
        "captures": b.captures.iter().map(|(k, v)| (k.clone(), Value::String(v.clone()))).collect::<serde_json::Map<_, _>>(),
        "replan_hz": b.replan_hz,
        "inputs": inputs,
        "actions": actions,
    })
}

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
    fn of(options: &RunOptions) -> Self {
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

fn problem(code: Code, slot: &Slot<'_>, head: String, body: Vec<String>) -> Problem {
    Problem {
        code,
        dir: slot.dir,
        entry: slot.entry.name.clone(),
        head,
        body,
        names: None,
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
    scope: Vec<String>,
    run: RunCall,
    problems: Vec<Problem>,
    warnings: Vec<String>,
}

/// Binds `policy` against the session's `parts` (every robot's), with the
/// run-time `options`.
pub fn bind(
    policy: &Policy,
    parts: &[PartSpec],
    options: &RunOptions,
) -> Result<Binding, BindError> {
    let mut owners: Vec<&ParticipantId> = Vec::new();
    for part in parts {
        if !owners.contains(&&part.owner) {
            owners.push(&part.owner);
        }
    }
    let robot = match (&options.robot, owners.as_slice()) {
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

impl<'a> Binder<'a> {
    fn slots(&self) -> Vec<Slot<'a>> {
        let mut slots = Vec::new();
        let make = |dir, entry: &'a Entry, input, action| {
            let override_at = self.options.at.get(&entry.name);
            let positional = self.options.positional.contains(&entry.name);
            let selector = match override_at {
                Some(path) => Some(Selector::literal(path)),
                None => entry.at.clone(),
            };
            let robot = selector
                .as_ref()
                .and_then(|s| s.robot.clone())
                .unwrap_or_else(|| self.robot.clone());
            Slot {
                dir,
                entry,
                input,
                action,
                selector,
                overridden: override_at.is_some() || positional,
                robot,
            }
        };
        for input in &self.policy.inputs {
            slots.push(make(Dir::In, &input.entry, Some(input), None));
        }
        for action in &self.policy.actions {
            slots.push(make(Dir::Out, &action.entry, None, Some(action)));
        }
        slots
    }

    fn kind_matches(entry: &Entry, part: &PartSpec) -> bool {
        if entry.kind == Kind::Custom {
            part.kind == entry.kind_name
        } else {
            part.kind == entry.kind.name()
        }
    }

    /// (candidates, considered) of a slot.
    fn candidates(&self, slot: &Slot<'a>) -> (Vec<Candidate<'a>>, Vec<&'a PartSpec>) {
        let scope: Vec<&str> = self.scope.iter().map(String::as_str).collect();
        let mut candidates = Vec::new();
        let mut considered = Vec::new();
        for part in self.parts.iter().filter(|p| p.owner == slot.robot) {
            if !Self::kind_matches(slot.entry, part) {
                continue;
            }
            let matched = match &slot.selector {
                Some(sel) => sel.matches(part.path.as_str(), &scope),
                None => below(part.path.as_str(), &scope).then(Vec::new),
            };
            match matched {
                Some(captures) => candidates.push(Candidate { part, captures }),
                None => considered.push(part),
            }
        }
        (candidates, considered)
    }

    fn bind(mut self) -> Result<Binding, BindError> {
        let slots = self.slots();
        let found: Vec<(Vec<Candidate<'a>>, Vec<&'a PartSpec>)> =
            slots.iter().map(|s| self.candidates(s)).collect();
        let given: Vec<(String, String)> = self
            .options
            .captures
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let all_captures: Vec<String> = {
            let mut names: Vec<String> = Vec::new();
            for slot in &slots {
                for name in slot.selector.iter().flat_map(|s| s.captures()) {
                    if !names.iter().any(|n| n == name) {
                        names.push(name.to_string());
                    }
                }
            }
            names
        };
        if let Some((name, _)) = given.iter().find(|(n, _)| !all_captures.contains(n)) {
            return Err(BindError::Options(format!(
                "capture {name:?} is not in any selector of {}",
                self.options.name
            )));
        }

        // Step 4: captures, from the required entries.
        let required: Vec<usize> = (0..slots.len()).filter(|i| slots[*i].required()).collect();
        let mut free: Vec<String> = Vec::new();
        for i in &required {
            for name in slots[*i].selector.iter().flat_map(|s| s.captures()) {
                if !given.iter().any(|(n, _)| n == name) && !free.iter().any(|f| f == name) {
                    free.push(name.to_string());
                }
            }
        }
        let fits_given = |c: &Candidate<'_>| {
            c.captures
                .iter()
                .all(|(n, v)| given.iter().all(|(gn, gv)| gn != n || gv == v))
        };
        let domains: Vec<Vec<String>> = free
            .iter()
            .map(|name| {
                let mut domain: Option<Vec<String>> = None;
                for i in &required {
                    if !slots[*i]
                        .selector
                        .iter()
                        .any(|s| s.captures().contains(&name.as_str()))
                    {
                        continue;
                    }
                    let mut values: Vec<String> = Vec::new();
                    for c in found[*i].0.iter().filter(|c| fits_given(c)) {
                        if let Some((_, v)) = c.captures.iter().find(|(n, _)| n == name)
                            && !values.contains(v)
                        {
                            values.push(v.clone());
                        }
                    }
                    domain = Some(match domain {
                        None => values,
                        Some(d) => d.into_iter().filter(|v| values.contains(v)).collect(),
                    });
                }
                domain.unwrap_or_default()
            })
            .collect();
        let count: usize = domains.iter().map(Vec::len).product();
        if count > MAX_ASSIGNMENTS {
            let first = &slots[required.first().copied().unwrap_or(0)];
            let names: Vec<String> = free.iter().map(|n| format!("{{{n}}}")).collect();
            self.problems.push(problem(
                Code::PatternTooBroad,
                first,
                format!(
                    "captures {} allow {count} assignments (at most {MAX_ASSIGNMENTS})",
                    names.join(", ")
                ),
                vec![format!(
                    "    fix: give a capture explicitly, e.g. {}",
                    self.run
                        .with_captures(&[(free[0].clone(), domains[0][0].clone())])
                )],
            ));
            return Err(self.refusal());
        }
        let mut assignments: Vec<Vec<(String, String)>> = vec![given.clone()];
        for (name, domain) in free.iter().zip(&domains) {
            assignments = assignments
                .into_iter()
                .flat_map(|a| {
                    domain.iter().map(move |v| {
                        let mut a = a.clone();
                        a.push((name.clone(), v.clone()));
                        a
                    })
                })
                .collect();
        }
        let tally = |assignment: &[(String, String)]| -> Vec<usize> {
            required
                .iter()
                .map(|i| found[*i].0.iter().filter(|c| c.fits(assignment)).count())
                .collect()
        };
        let survivors: Vec<&Vec<(String, String)>> = assignments
            .iter()
            .filter(|a| tally(a).iter().all(|n| *n == 1))
            .collect();
        let chosen: Vec<(String, String)> = match survivors.as_slice() {
            [one] => (*one).clone(),
            [] => assignments
                .iter()
                .max_by_key(|a| {
                    let ones = tally(a).iter().filter(|n| **n == 1).count();
                    // max_by_key keeps the last maximum; prefer the first.
                    (
                        ones,
                        std::cmp::Reverse(assignments.iter().position(|b| b == *a)),
                    )
                })
                .cloned()
                .unwrap_or_else(|| given.clone()),
            many => {
                let first = &slots[required[0]];
                let names: Vec<String> = free.iter().map(|n| format!("{{{n}}}")).collect();
                let rows: Vec<(String, String)> = many
                    .iter()
                    .map(|a| {
                        let label: Vec<String> = a
                            .iter()
                            .filter(|(n, _)| free.contains(n))
                            .map(|(n, v)| format!("{n}={v:?}"))
                            .collect();
                        let mut paths: Vec<String> = Vec::new();
                        let captured = required.iter().filter(|i| {
                            slots[**i].selector.iter().any(|s| !s.captures().is_empty())
                        });
                        for i in captured {
                            if let Some(c) = found[*i].0.iter().find(|c| c.fits(a)) {
                                let path = c.part.path.to_string();
                                if !paths.contains(&path) {
                                    paths.push(path);
                                }
                            }
                        }
                        (label.join(", "), format!("-> {}", paths.join(", ")))
                    })
                    .collect();
                let head = if free.len() == 1 {
                    format!(
                        "{} has {} values that bind every required entry:",
                        names[0],
                        many.len()
                    )
                } else {
                    format!(
                        "{} have {} assignments that bind every required entry:",
                        names.join(", "),
                        many.len()
                    )
                };
                let mut body: Vec<String> = two_columns(&rows, 2)
                    .into_iter()
                    .map(|l| format!("      {l}"))
                    .collect();
                let pick: Vec<(String, String)> = many[0]
                    .iter()
                    .filter(|(n, _)| free.contains(n))
                    .cloned()
                    .collect();
                body.push(format!("    fix: {}", self.run.with_captures(&pick)));
                self.problems
                    .push(problem(Code::AmbiguousCapture, first, head, body));
                return Err(self.refusal());
            }
        };

        // Steps 3-5: each slot's one candidate under the chosen assignment.
        let mut bound: Vec<Option<&'a PartSpec>> = vec![None; slots.len()];
        let mut absent: Vec<bool> = vec![false; slots.len()];
        let unique: Vec<Option<&'a PartSpec>> = slots
            .iter()
            .zip(&found)
            .map(|(_, (cands, _))| {
                let fitting: Vec<&Candidate<'a>> =
                    cands.iter().filter(|c| c.fits(&chosen)).collect();
                match fitting.as_slice() {
                    [one] => Some(one.part),
                    _ => None,
                }
            })
            .collect();
        for (i, slot) in slots.iter().enumerate() {
            let (cands, considered) = &found[i];
            let unassigned: Vec<&str> = slot
                .selector
                .iter()
                .flat_map(|s| s.captures())
                .filter(|n| !chosen.iter().any(|(c, _)| c == n))
                .collect();
            if !slot.required() && !unassigned.is_empty() {
                self.problems.push(problem(
                    Code::Unbound,
                    slot,
                    format!(
                        "{}: {{{}}} is used only by optional entries, so it is not inferred.",
                        slot.subject(),
                        unassigned[0]
                    ),
                    vec![format!(
                        "    fix: {}",
                        self.run
                            .with_captures(&[(unassigned[0].to_string(), "...".to_string())])
                    )],
                ));
                continue;
            }
            let fitting: Vec<&Candidate<'a>> = cands.iter().filter(|c| c.fits(&chosen)).collect();
            match fitting.as_slice() {
                [one] => bound[i] = Some(one.part),
                [] if !slot.required() => {
                    let allow = match slot.dir {
                        Dir::In => Allow::Blank,
                        Dir::Out => Allow::Drop,
                    };
                    if self.allowed(slot.entry, allow) {
                        absent[i] = true;
                    } else {
                        self.problems.push(problem(
                            Code::NotAllowed,
                            slot,
                            format!(
                                "{}: {} has no {} part {}",
                                slot.name(),
                                slot.robot,
                                slot.entry.kind_name,
                                slot.selector.as_ref().map_or_else(String::new, |s| format!(
                                    "at {:?}",
                                    s.to_string()
                                ))
                            )
                            .trim_end()
                            .to_string(),
                            vec![
                                format!("    needs: {allow} (it is optional)"),
                                format!("    fix: add {:?} to allow", allow.name()),
                            ],
                        ));
                    }
                }
                [] => self.unbound(slot, considered),
                many => {
                    let rows: Vec<(String, String)> = many
                        .iter()
                        .map(|c| (c.part.path.to_string(), summary(c.part)))
                        .collect();
                    let taken: Vec<&PartSpec> = unique
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .filter_map(|(_, p)| *p)
                        .collect();
                    let suggest = many
                        .iter()
                        .find(|c| !taken.iter().any(|t| std::ptr::eq(*t, c.part)))
                        .unwrap_or(&many[0]);
                    let mut body: Vec<String> = two_columns(&rows, 3)
                        .into_iter()
                        .map(|l| format!("      {l}"))
                        .collect();
                    body.push(format!(
                        "    fix: {}",
                        self.run
                            .with_at(&slot.entry.name, suggest.part.path.as_str())
                    ));
                    self.problems.push(problem(
                        Code::Ambiguous,
                        slot,
                        format!("{} matches {} parts:", slot.subject(), many.len()),
                        body,
                    ));
                }
            }
        }

        // Step 6: compatibility.
        let replan = self.policy.hz;
        let mut inputs = Vec::new();
        let mut actions = Vec::new();
        for (i, slot) in slots.iter().enumerate() {
            let status = if absent[i] {
                if slot.dir == Dir::In {
                    Status::Blank
                } else {
                    Status::Dropped
                }
            } else if slot.overridden {
                Status::Override
            } else {
                Status::Bound
            };
            match (slot.input, slot.action) {
                (Some(input), _) => {
                    if absent[i] {
                        if let Some(row) = self.blank(slot, input) {
                            inputs.push(row);
                        }
                    } else if let Some(part) = bound[i]
                        && let Some(row) = self.input(slot, input, part, status, replan)
                    {
                        inputs.push(row);
                    }
                }
                (_, Some(action)) => {
                    if absent[i] {
                        actions.push(self.dropped(slot, action, replan));
                    } else if let Some(part) = bound[i]
                        && let Some(row) = self.action(slot, action, part, status, replan)
                    {
                        actions.push(row);
                    }
                }
                _ => {}
            }
        }

        // Step 7: duplicate targets.
        self.duplicates(&slots, &actions);

        // Contracts, one per bound input stream.
        let contracts = self.contracts(&slots, &inputs, replan);

        // Problems are reported in entry order.
        let order = |p: &Problem| {
            slots
                .iter()
                .position(|s| s.dir == p.dir && s.entry.name == p.entry)
                .unwrap_or(usize::MAX)
        };
        self.problems.sort_by_key(|p| order(p));
        if !self.problems.is_empty() {
            return Err(self.refusal());
        }
        let mut binding = Binding {
            policy: self.options.name.clone(),
            robot: self.robot.clone(),
            under: self.options.under.as_ref().map(ToString::to_string),
            captures: chosen,
            replan_hz: replan,
            inputs,
            actions,
            contracts,
            warnings: self.warnings,
            fingerprint: Fingerprint::default(),
        };
        binding.fingerprint = canonical::fingerprint(&binding.table_json());
        Ok(binding)
    }

    fn refusal(&mut self) -> BindError {
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
            problems: std::mem::take(&mut self.problems),
            run: self.run.clone(),
        }))
    }

    fn allowed(&self, entry: &Entry, allow: Allow) -> bool {
        entry.allow.has(allow) || self.options.allow.contains(&allow)
    }

    fn allows(&self, entry: &Entry) -> Allows {
        self.options
            .allow
            .iter()
            .fold(entry.allow, |set, a| set.with(*a))
    }

    fn unbound(&mut self, slot: &Slot<'a>, considered: &[&'a PartSpec]) {
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

    fn not_allowed(&mut self, slot: &Slot<'_>, what: String, allow: Allow) {
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

    fn incompatible(&mut self, slot: &Slot<'_>, head: String, body: Vec<String>) {
        self.problems
            .push(problem(Code::Incompatible, slot, head, body));
    }

    /// Frame identities (§5.2): `frame`, pose `child`, wrench `point`.
    fn frames(&mut self, slot: &Slot<'_>, part: &PartSpec, point: Option<&String>) -> bool {
        let kind = Kind::parse(&part.kind).unwrap_or(Kind::Custom);
        let path = part.path.as_str();
        let checks = [
            (
                "frame",
                slot.entry.frame.as_ref(),
                Some(part.frame.as_str()),
            ),
            (
                "child",
                slot.entry.child.as_ref(),
                part.attrs.get(attrs::CHILD).map(String::as_str),
            ),
            (
                "point",
                point,
                Some(
                    part.attrs
                        .get(attrs::POINT)
                        .map_or(part.frame.as_str(), String::as_str),
                ),
            ),
        ];
        let mut ok = true;
        for (key, want, have) in checks {
            let Some(want) = want else { continue };
            let have = have.filter(|h| !h.is_empty());
            if have == Some(want.as_str()) {
                continue;
            }
            ok = false;
            let found = match (key, have) {
                (_, None) => format!("{path} has no {key}."),
                ("frame", Some(h)) => format!("{path} is in frame {h}."),
                (_, Some(h)) => format!("{path} has {key} {h}."),
            };
            let have = have.unwrap_or("its own");
            self.incompatible(
                slot,
                format!("{} wants {key} {want}; {found}", slot.name()),
                vec![
                    format!(
                        "    RLMesh does not transform frames. Publish the {} in {want} (as {path:?},",
                        kind.value_noun()
                    ),
                    format!(
                        "    {key} {want:?}), or use a policy trained on {have}-{key} {}.",
                        kind.values_noun()
                    ),
                ],
            );
        }
        ok
    }

    /// Joint names (§6.2): the order the model's elements take from the
    /// robot's, or a refusal.
    fn names(&mut self, slot: &Slot<'_>, part: &PartSpec, n: u32, path: &str) -> Option<Order> {
        let entry = slot.entry;
        let positional = self.options.positional.contains(&entry.name);
        let model_count = |names: &Names| match names {
            Names::List(list) => Some(list.len() as u32),
            _ => entry.count,
        };
        let count_mismatch = |this: &mut Self, count: u32| {
            this.incompatible(
                slot,
                format!(
                    "{}: the model has {count} joints; {path} has {n}.",
                    slot.name()
                ),
                Vec::new(),
            );
        };
        if positional || entry.names == Names::Positional {
            let Some(count) = model_count(&entry.names) else {
                self.incompatible(
                    slot,
                    format!("{}: positional binding needs a count.", slot.name()),
                    Vec::new(),
                );
                return None;
            };
            if count != n {
                count_mismatch(self, count);
                return None;
            }
            return Some(Order::Positional);
        }
        match &entry.names {
            Names::Positional => Some(Order::Positional),
            Names::Robot => match entry.count {
                Some(count) if count != n => {
                    count_mismatch(self, count);
                    None
                }
                Some(_) => Some(Order::Positional),
                None => Some(Order::Robot),
            },
            Names::List(model) => {
                if part.names.is_empty() {
                    self.incompatible(
                        slot,
                        format!(
                            "{}: {path} declares a count, not joint names, so the model's names cannot be matched.",
                            slot.name()
                        ),
                        vec![format!(
                            "    fix: if index i means the same joint on both, bind it positionally:\n         {}",
                            self.run.with_positional([entry.name.as_str()])
                        )],
                    );
                    return None;
                }
                let index: Vec<Option<usize>> = model
                    .iter()
                    .map(|m| part.names.iter().position(|r| r == m))
                    .collect();
                if index.iter().all(Option::is_none) {
                    self.problems.push(Problem {
                        code: Code::Incompatible,
                        dir: slot.dir,
                        entry: entry.name.clone(),
                        head: String::new(),
                        body: Vec::new(),
                        names: Some(NamesClash {
                            part: path.to_string(),
                            model: model.clone(),
                            robot: part.names.clone(),
                        }),
                    });
                    return None;
                }
                let missing: Vec<&String> = model
                    .iter()
                    .zip(&index)
                    .filter(|(_, i)| i.is_none())
                    .map(|(m, _)| m)
                    .collect();
                if !missing.is_empty() {
                    let missing: Vec<&str> = missing.iter().map(|s| s.as_str()).collect();
                    self.incompatible(
                        slot,
                        format!(
                            "{}: {path} has no joint named {}.",
                            slot.name(),
                            missing.join(", ")
                        ),
                        Vec::new(),
                    );
                    return None;
                }
                let index: Vec<usize> = index.into_iter().flatten().collect();
                if index.len() < part.names.len() {
                    if slot.dir == Dir::Out {
                        self.incompatible(
                            slot,
                            format!(
                                "{}: the model commands {} of {path}'s {} joints.",
                                slot.name(),
                                index.len(),
                                part.names.len()
                            ),
                            vec!["    Commanding a subset of a part (scatter) is Later (§19); declare the subset as its own part.".into()],
                        );
                        return None;
                    }
                    if !self.allowed(entry, Allow::Select) {
                        self.not_allowed(
                            slot,
                            format!(
                                "{path} has {} joints, the model names {} of them",
                                part.names.len(),
                                index.len()
                            ),
                            Allow::Select,
                        );
                        return None;
                    }
                    return Some(Order::Select(index));
                }
                if index.iter().enumerate().all(|(i, j)| i == *j) {
                    Some(Order::Identical)
                } else {
                    Some(Order::Reorder(index))
                }
            }
        }
    }

    /// Units for one field: the step, or a refusal. `to` is the target unit,
    /// `from` the source's.
    #[allow(clippy::too_many_arguments)]
    fn unit_step(
        &mut self,
        slot: &Slot<'_>,
        part: &PartSpec,
        kind: Kind,
        field: &str,
        from: &str,
        to: &str,
        path: &str,
    ) -> Result<Option<Step>, ()> {
        let q = quantity(kind, field, from);
        if from == to {
            return Ok(None);
        }
        if !q.accepts(to) {
            self.incompatible(
                slot,
                format!(
                    "{}: {to:?} is not a {q} unit ({path} publishes {field} in {from}).",
                    slot.name()
                ),
                Vec::new(),
            );
            return Err(());
        }
        match units::change(q, from, to) {
            UnitChange::Same => Ok(None),
            UnitChange::Auto(factor) => Ok(Some(Step::Units {
                field: field.into(),
                from: from.into(),
                to: to.into(),
                factor,
                auto: true,
            })),
            UnitChange::Allowed(factor) => {
                if !self.allowed(slot.entry, Allow::Units) {
                    self.not_allowed(
                        slot,
                        format!("{path} publishes {field} in {from}, the model wants {to}"),
                        Allow::Units,
                    );
                    return Err(());
                }
                if kind == Kind::Depth && from == "mm" {
                    return Ok(Some(Step::DepthMetres));
                }
                Ok(Some(Step::Units {
                    field: field.into(),
                    from: from.into(),
                    to: to.into(),
                    factor,
                    auto: false,
                }))
            }
            UnitChange::GripperWidth { to_normalized } => {
                let Some((low, high)) = width_m(part) else {
                    self.incompatible(
                        slot,
                        format!(
                            "{}: {path} declares no width_m, so metres cannot be mapped.",
                            slot.name()
                        ),
                        Vec::new(),
                    );
                    return Err(());
                };
                if !self.allowed(slot.entry, Allow::Units) {
                    self.not_allowed(
                        slot,
                        format!("{path} publishes pos in {from}, the model wants {to}"),
                        Allow::Units,
                    );
                    return Err(());
                }
                Ok(Some(Step::GripperWidth {
                    to_normalized,
                    low,
                    high,
                }))
            }
            UnitChange::Incompatible => {
                self.incompatible(
                    slot,
                    format!(
                        "{}: {path} publishes {field} in {from}; the model wants {to}, and {q} units compare by equality.",
                        slot.name()
                    ),
                    Vec::new(),
                );
                Err(())
            }
        }
    }

    /// A rotation re-encoding for one field.
    fn rotation_step(
        &mut self,
        slot: &Slot<'_>,
        field: &str,
        from: Rotation,
        to: Rotation,
        path: &str,
    ) -> Result<Option<Step>, ()> {
        match rotation::change(from, to) {
            RotationChange::Same => Ok(None),
            RotationChange::Auto => Ok(Some(Step::Rotation {
                field: field.into(),
                from,
                to,
                auto: true,
            })),
            RotationChange::Allowed => {
                if !self.allowed(slot.entry, Allow::Rotation) {
                    let what = match slot.dir {
                        Dir::In => {
                            format!("{path} publishes {field} ({from}), the model wants {to}")
                        }
                        Dir::Out => format!("the model emits {field} as {from}, {path} takes {to}"),
                    };
                    self.not_allowed(slot, what, Allow::Rotation);
                    return Err(());
                }
                Ok(Some(Step::Rotation {
                    field: field.into(),
                    from,
                    to,
                    auto: false,
                }))
            }
            RotationChange::Incompatible => {
                self.incompatible(
                    slot,
                    format!(
                        "{}: nothing converts from {from} to {to} ({from} discards yaw).",
                        slot.name()
                    ),
                    Vec::new(),
                );
                Err(())
            }
        }
    }

    /// A cast to `to`, automatic when exact.
    fn cast_step(
        &mut self,
        slot: &Slot<'_>,
        from: DType,
        to: DType,
        path: &str,
    ) -> Result<Option<Step>, ()> {
        match cast(from, to) {
            Some(true) if from == to => Ok(None),
            Some(true) => Ok(Some(Step::Cast {
                from,
                to,
                exact: true,
            })),
            Some(false) => {
                if !self.allowed(slot.entry, Allow::Dtype) {
                    self.not_allowed(
                        slot,
                        format!("{path} publishes {from}, the model wants {to} (not exact)"),
                        Allow::Dtype,
                    );
                    return Err(());
                }
                Ok(Some(Step::Cast {
                    from,
                    to,
                    exact: false,
                }))
            }
            None => {
                self.incompatible(
                    slot,
                    format!(
                        "{}: {path} publishes {from}; no cast to {to} exists.",
                        slot.name()
                    ),
                    Vec::new(),
                );
                Err(())
            }
        }
    }

    /// The affine's arity: one value, or one per element.
    fn affine(
        &mut self,
        slot: &Slot<'_>,
        elements: Option<u64>,
        action: bool,
    ) -> Result<Option<Step>, ()> {
        let (scale, offset) = (&slot.entry.scale, &slot.entry.offset);
        if scale.is_none() && offset.is_none() {
            return Ok(None);
        }
        for (key, values) in [("scale", scale), ("offset", offset)] {
            if let Some(values) = values
                && values.len() > 1
                && Some(values.len() as u64) != elements
            {
                self.incompatible(
                    slot,
                    format!(
                        "{}: {key} has {} values for {} elements.",
                        slot.name(),
                        values.len(),
                        elements.map_or_else(|| "per-field".to_string(), |e| e.to_string())
                    ),
                    Vec::new(),
                );
                return Err(());
            }
        }
        Ok(Some(Step::Affine {
            scale: scale.clone().unwrap_or_default(),
            offset: offset.clone().unwrap_or_default(),
            action,
        }))
    }

    /// Step 6 for one input.
    fn input(
        &mut self,
        slot: &Slot<'a>,
        input: &Input,
        part: &'a PartSpec,
        status: Status,
        replan: f64,
    ) -> Option<InputRow> {
        let kind = Kind::parse(&part.kind).unwrap_or(Kind::Custom);
        let applied = input.source == Source::Applied;
        let path = if applied {
            part.path.applied()
        } else {
            part.path.to_string()
        };
        let before = self.problems.len();
        let source_fields: Vec<FieldSpec> = if applied {
            match &part.command {
                Some(command) => command.fields.clone(),
                None => {
                    self.incompatible(
                        slot,
                        format!(
                            "{}: {} has no command, so no applied stream.",
                            slot.name(),
                            part.path
                        ),
                        Vec::new(),
                    );
                    return None;
                }
            }
        } else {
            part.fields.clone()
        };
        self.frames(slot, part, input.point.as_ref());
        let mut plan = Plan::default();
        if part.encoding != Encoding::Raw {
            plan.steps.push(Step::Decode);
        }
        let mut delivered: Vec<Delivered>;
        let mut elements: Option<u64> = None;
        if kind.is_image() {
            delivered = self.image(slot, input, part, kind, &source_fields[0], &path, &mut plan)?;
        } else {
            // Fields, in the model's order.
            let wanted: Vec<String> = match (&input.fields, &slot.entry.schema) {
                (_, Some(schema)) => schema.iter().map(|f| f.name.clone()).collect(),
                (Some(fields), None) => fields.clone(),
                (None, None) => source_fields.iter().map(|f| f.name.clone()).collect(),
            };
            let mut fields: Vec<FieldSpec> = Vec::new();
            for name in &wanted {
                match source_fields.iter().find(|f| &f.name == name) {
                    Some(f) => fields.push(f.clone()),
                    None => {
                        let have: Vec<&str> =
                            source_fields.iter().map(|f| f.name.as_str()).collect();
                        self.incompatible(
                            slot,
                            format!(
                                "{}: {path} publishes {}; the model wants {name}.",
                                slot.name(),
                                have.join(", ")
                            ),
                            Vec::new(),
                        );
                        return None;
                    }
                }
            }
            if let Some(schema) = &slot.entry.schema {
                let matches = schema.len() == source_fields.len()
                    && schema
                        .iter()
                        .zip(&source_fields)
                        .all(|(a, b)| a.name == b.name && a.shape == b.shape);
                if !matches {
                    self.incompatible(
                        slot,
                        format!(
                            "{}: {path}'s fields do not match the declared schema of {}.",
                            slot.name(),
                            part.kind
                        ),
                        vec!["    A custom kind binds only by exact name and a matching fields schema (§4.2).".into()],
                    );
                    return None;
                }
            }
            if source_fields.len() > 1 {
                plan.steps.push(Step::Fields(wanted.clone()));
            }
            // Names.
            let mut count = None;
            if kind.is_joint_set() {
                let n = joint_count(&source_fields);
                let order = self.names(slot, part, n, &path)?;
                if let Order::Select(picked) = &order {
                    count = Some(picked.len() as u32);
                }
                plan.steps.push(Step::Names(order));
            }
            // Units.
            for (field, _) in &slot.entry.units {
                if !fields.iter().any(|f| &f.name == field) {
                    self.incompatible(
                        slot,
                        format!(
                            "{}: units on {field}, which it does not deliver.",
                            slot.name()
                        ),
                        Vec::new(),
                    );
                    return None;
                }
            }
            for f in &mut fields {
                let q = quantity(kind, &f.name, &f.unit);
                let target = slot
                    .entry
                    .units
                    .iter()
                    .find(|(n, _)| n == &f.name)
                    .map(|(_, u)| u.clone())
                    .unwrap_or_else(|| {
                        if q == Quantity::Custom {
                            f.unit.clone()
                        } else {
                            q.si().to_string()
                        }
                    });
                if q == Quantity::None {
                    continue;
                }
                match self.unit_step(slot, part, kind, &f.name, &f.unit, &target, &path) {
                    Ok(Some(step)) => {
                        if !f.dtype.is_float() {
                            f.dtype = DType::F32;
                        }
                        f.unit = target;
                        plan.steps.push(step);
                    }
                    Ok(None) => {}
                    Err(()) => return None,
                }
            }
            // Rotation.
            let rotation_fields: Vec<usize> = fields
                .iter()
                .enumerate()
                .filter(|(_, f)| !f.rotation.is_empty())
                .map(|(i, _)| i)
                .collect();
            if let Some(to) = slot.entry.rotation {
                if rotation_fields.is_empty() {
                    self.incompatible(
                        slot,
                        format!(
                            "{}: rotation {to}, but it delivers no rotation field.",
                            slot.name()
                        ),
                        Vec::new(),
                    );
                    return None;
                }
                for i in rotation_fields {
                    let from =
                        Rotation::from_name(&fields[i].rotation).unwrap_or(Rotation::QuatXyzw);
                    match self.rotation_step(slot, &fields[i].name.clone(), from, to, &path) {
                        Ok(Some(step)) => {
                            fields[i].rotation = to.name().into();
                            fields[i].shape = vec![to.width()];
                            plan.steps.push(step);
                        }
                        Ok(None) => {}
                        Err(()) => return None,
                    }
                }
            }
            // Tactile and custom shapes.
            if let Some(shape) = &input.shape
                && fields.iter().any(|f| &f.shape != shape)
            {
                self.incompatible(
                    slot,
                    format!(
                        "{}: {path} has shape {:?}; the model wants {shape:?}.",
                        slot.name(),
                        fields[0].shape
                    ),
                    vec!["    RLMesh does not reshape.".into()],
                );
                return None;
            }
            // dtype (from the schema, for custom kinds).
            let targets: Vec<Option<DType>> = match &slot.entry.schema {
                Some(schema) => schema.iter().map(|s| Some(s.dtype)).collect(),
                None => vec![input.dtype; fields.len()],
            };
            for (f, target) in fields.iter_mut().zip(targets) {
                let Some(to) = target else { continue };
                if f.dtype == DType::Utf8 {
                    continue;
                }
                match self.cast_step(slot, f.dtype, to, &path) {
                    Ok(Some(step)) => {
                        f.dtype = to;
                        plan.steps.push(step);
                    }
                    Ok(None) => {}
                    Err(()) => return None,
                }
            }
            // A declared range maps through rescale.
            if let Some(to) = slot.entry.range {
                let from = fields
                    .iter()
                    .map(|f| field_range(part, kind, &f.name, &f.unit))
                    .collect::<Option<Vec<_>>>()
                    .and_then(|r| {
                        r.first()
                            .copied()
                            .filter(|first| r.iter().all(|x| x == first))
                    });
                let Some(from) = from else {
                    self.incompatible(
                        slot,
                        format!(
                            "{}: the model wants range {}…{}, but {path} declares no range for {}.",
                            slot.name(),
                            num(to.0),
                            num(to.1),
                            wanted.join(", ")
                        ),
                        Vec::new(),
                    );
                    return None;
                };
                if from != to {
                    if !self.allowed(slot.entry, Allow::Rescale) {
                        self.not_allowed(
                            slot,
                            format!(
                                "{path} ranges {}…{}, the model wants {}…{}",
                                num(from.0),
                                num(from.1),
                                num(to.0),
                                num(to.1)
                            ),
                            Allow::Rescale,
                        );
                        return None;
                    }
                    for f in &mut fields {
                        if !f.dtype.is_float() {
                            f.dtype = DType::F32;
                        }
                    }
                    plan.steps.push(Step::Rescale { from, to });
                }
            }
            if let Some(n) = count {
                for f in &mut fields {
                    if !f.shape.is_empty() {
                        f.shape[0] = n;
                    }
                }
            }
            delivered = fields
                .iter()
                .map(|f| Delivered {
                    name: f.name.clone(),
                    dtype: f.dtype,
                    shape: f.shape.clone(),
                })
                .collect();
            elements = if delivered.len() == 1
                || delivered
                    .iter()
                    .all(|d| d.shape.len() == 1 && d.dtype == delivered[0].dtype)
            {
                Some(
                    delivered
                        .iter()
                        .map(|d| d.shape.iter().map(|x| u64::from(*x)).product::<u64>())
                        .sum(),
                )
            } else {
                None
            };
        }
        if kind.is_image() {
            // Per-channel affines are the common image normalisation.
            elements = delivered
                .first()
                .and_then(|d| match input.layout {
                    ImageLayout::Hwc => d.shape.last(),
                    ImageLayout::Chw => d.shape.first(),
                })
                .map(|c| u64::from(*c));
        }
        match self.affine(slot, elements, false) {
            Ok(Some(step)) => {
                for d in &mut delivered {
                    if !d.dtype.is_float() {
                        d.dtype = DType::F32;
                    }
                }
                plan.steps.push(step);
            }
            Ok(None) => {}
            Err(()) => return None,
        }
        // History.
        let source_hz = contract::source_hz(part, applied);
        if input.history > 1 {
            if let Some(trained) = input.history_hz
                && (trained - source_hz).abs() > HISTORY_HZ_TOLERANCE * source_hz
            {
                self.incompatible(
                    slot,
                    format!(
                        "{}: the window was trained at {} Hz; {path} publishes at {} Hz.",
                        slot.name(),
                        hz(trained),
                        hz(source_hz)
                    ),
                    vec!["    Resampling a window to another rate is Later (§19).".into()],
                );
                return None;
            }
            plan.steps.push(Step::Window(input.history));
        }
        let allows = self.allows(slot.entry);
        if input.history > 1 && allows.has(Allow::Warmup) {
            plan.notes.push("warmup at episode start".into());
        }
        if allows.has(Allow::Hold) {
            plan.notes.push("hold when stale".into());
        }
        if self.problems.len() > before {
            return None;
        }
        let (max_age, explicit) = contract::freshness(part, applied, replan, input.max_age);
        let delivers = Delivery::of(delivered, (input.history > 1).then_some(input.history));
        Some(InputRow {
            entry: slot.entry.name.clone(),
            kind: slot.entry.kind_name.clone(),
            part: Some(part.clone()),
            channel: if applied {
                part.applied_channel
            } else {
                Some(part.channel)
            },
            applied,
            status,
            plan,
            delivers,
            history: input.history,
            source_hz,
            max_age,
            max_age_explicit: explicit,
            required: slot.required(),
            hold: allows.has(Allow::Hold),
            warmup: allows.has(Allow::Warmup),
        })
    }

    /// The image pipeline (§7.4): decode → rotate180 → declared crop →
    /// aspect fit + resize → bgr → layout → dtype or range.
    #[allow(clippy::too_many_arguments)]
    fn image(
        &mut self,
        slot: &Slot<'_>,
        input: &Input,
        part: &PartSpec,
        kind: Kind,
        field: &FieldSpec,
        path: &str,
        plan: &mut Plan,
    ) -> Option<Vec<Delivered>> {
        if let Some(fields) = &input.fields
            && fields.iter().any(|f| f != &field.name)
        {
            self.incompatible(
                slot,
                format!(
                    "{}: {path} publishes {}; fields are not selectable on an image.",
                    slot.name(),
                    field.name
                ),
                Vec::new(),
            );
            return None;
        }
        let (height, width) = (field.shape[0], field.shape[1]);
        let channels = field.shape.get(2).copied();
        let mut dtype = field.dtype;
        // Units (depth mm → m).
        if kind == Kind::Depth {
            let target = slot
                .entry
                .units
                .iter()
                .find(|(n, _)| n == "depth")
                .map_or("m", |(_, u)| u.as_str())
                .to_string();
            match self.unit_step(slot, part, kind, "depth", &field.unit, &target, path) {
                Ok(Some(step)) => {
                    dtype = if dtype.is_float() { dtype } else { DType::F32 };
                    plan.steps.push(step);
                }
                Ok(None) => {}
                Err(()) => return None,
            }
        }
        if kind == Kind::Camera {
            let source_format = if channels == Some(1) {
                ImageFormat::Mono8
            } else {
                ImageFormat::Rgb8
            };
            if (input.format == ImageFormat::Mono8) != (source_format == ImageFormat::Mono8) {
                self.incompatible(
                    slot,
                    format!(
                        "{}: {path} is {}, the model wants {}.",
                        slot.name(),
                        source_format.name(),
                        input.format.name()
                    ),
                    vec!["    Mono ↔ RGB conversion is Later (§19).".into()],
                );
                return None;
            }
        } else if input.layout == ImageLayout::Chw {
            self.incompatible(
                slot,
                format!("{}: a depth image has no channels to lay out.", slot.name()),
                Vec::new(),
            );
            return None;
        }
        // rotate180.
        let upside_down = part
            .attrs
            .get(attrs::MOUNT)
            .is_some_and(|m| m == "upside_down");
        if upside_down != input.upside_down {
            if !self.allowed(slot.entry, Allow::Rotate180) {
                let what = if upside_down {
                    format!("{path} is mounted upside down, the model was trained upright")
                } else {
                    format!("{path} is upright, the model was trained on frames rotated 180°")
                };
                self.not_allowed(slot, what, Allow::Rotate180);
                return None;
            }
            plan.steps.push(Step::Rotate180);
        }
        // Declared crop.
        let mut size = Size { width, height };
        if let Some(fraction) = input.crop {
            let cut = |side: u32| ((f64::from(side) * fraction).round() as u32).clamp(1, side);
            let to = Size {
                width: cut(size.width),
                height: cut(size.height),
            };
            plan.steps.push(Step::CenterCrop {
                fraction,
                from: size,
                to,
            });
            size = to;
        }
        // Aspect fit and resize, one step (v1 `fit` semantics).
        let target = input.size.map_or(size, |(h, w)| Size {
            width: w,
            height: h,
        });
        if target != size {
            let allows = self.allows(slot.entry);
            let fit_needed = !size.same_aspect(target);
            let fit = [Fit::Crop, Fit::Pad, Fit::Stretch]
                .into_iter()
                .find(|f| fit_needed && allows.has(f.allow()));
            let (sx, sy) = (
                f64::from(target.width) / f64::from(size.width),
                f64::from(target.height) / f64::from(size.height),
            );
            let enlarges = match fit {
                Some(Fit::Crop) => sx.max(sy) > 1.0,
                Some(Fit::Pad) => sx.min(sy) > 1.0,
                _ => sx > 1.0 || sy > 1.0,
            };
            let missing_fit = fit_needed && fit.is_none();
            let missing_resize = !allows.has(Allow::Resize);
            let what = format!(
                "{path} is {size} ({}), the model wants {target} ({})",
                size.aspect(),
                target.aspect()
            );
            if missing_fit || missing_resize {
                let (needs, fix) = match (missing_fit, missing_resize) {
                    (true, true) => (
                        "resize, plus one of crop | pad | stretch to change the aspect ratio",
                        "add \"resize\" and \"pad\" (or \"crop\") to allow",
                    ),
                    (true, false) => (
                        "one of crop | pad | stretch to change the aspect ratio",
                        "add \"pad\" (or \"crop\") to allow",
                    ),
                    _ => ("resize", "add \"resize\" to allow"),
                };
                self.problems.push(problem(
                    Code::NotAllowed,
                    slot,
                    format!("{}: {what}", slot.name()),
                    vec![format!("    needs: {needs}"), format!("    fix: {fix}")],
                ));
                return None;
            }
            if enlarges && !allows.has(Allow::Upscale) {
                self.problems.push(problem(
                    Code::NotAllowed,
                    slot,
                    format!("{}: {what}, which enlarges it", slot.name()),
                    vec![
                        "    needs: upscale".into(),
                        "    fix: add \"upscale\" to allow".into(),
                    ],
                ));
                return None;
            }
            plan.steps.push(Step::Resize {
                kernel: if kind == Kind::Depth {
                    Resample::Nearest
                } else {
                    input.resample
                },
                from: size,
                to: target,
                fit,
            });
            size = target;
        }
        // bgr.
        if input.format == ImageFormat::Bgr8 {
            if !self.allowed(slot.entry, Allow::Bgr) {
                self.not_allowed(
                    slot,
                    format!("{path} is rgb8, the model wants bgr8"),
                    Allow::Bgr,
                );
                return None;
            }
            plan.steps.push(Step::Bgr);
        }
        if input.layout == ImageLayout::Chw {
            plan.steps.push(Step::Chw);
        }
        // Range normalisation, or a cast.
        if let Some((low, high)) = slot.entry.range {
            if kind == Kind::Depth {
                self.incompatible(
                    slot,
                    format!(
                        "{}: range normalises u8 images; depth has no such range.",
                        slot.name()
                    ),
                    Vec::new(),
                );
                return None;
            }
            let to = input.dtype.unwrap_or(DType::F32);
            if !to.is_float() {
                self.incompatible(
                    slot,
                    format!(
                        "{}: range normalisation produces floats, not {to}.",
                        slot.name()
                    ),
                    Vec::new(),
                );
                return None;
            }
            plan.steps.push(Step::Normalise {
                low,
                high,
                dtype: to,
            });
            dtype = to;
        } else if let Some(to) = input.dtype {
            match self.cast_step(slot, dtype, to, path) {
                Ok(Some(step)) => {
                    plan.steps.push(step);
                    dtype = to;
                }
                Ok(None) => {}
                Err(()) => return None,
            }
        }
        let shape = match (kind, channels, input.layout) {
            (Kind::Depth, _, _) => vec![size.height, size.width],
            (_, Some(c), ImageLayout::Hwc) => vec![size.height, size.width, c],
            (_, Some(c), ImageLayout::Chw) => vec![c, size.height, size.width],
            _ => vec![size.height, size.width],
        };
        Some(vec![Delivered {
            name: field.name.clone(),
            dtype,
            shape,
        }])
    }

    /// An absent optional input bound to a constant, when its target is
    /// fully declared.
    fn blank(&mut self, slot: &Slot<'_>, input: &Input) -> Option<InputRow> {
        let kind = slot.entry.kind;
        let undeclared = |what: &str| format!("{}: cannot blank it without {what}.", slot.name());
        // A black frame through the declared preprocessing: range, then the
        // affine (per channel when it has one value per channel).
        let mut fill = vec![match (kind, slot.entry.range) {
            (Kind::Camera, Some((low, _))) => low,
            _ => 0.0,
        }];
        if kind == Kind::Camera && (slot.entry.scale.is_some() || slot.entry.offset.is_some()) {
            let n = [&slot.entry.scale, &slot.entry.offset]
                .iter()
                .filter_map(|v| v.as_ref().map(Vec::len))
                .max()
                .unwrap_or(1);
            let pick = |v: &Option<Vec<f64>>, i: usize, d: f64| {
                v.as_ref().map_or(d, |v| {
                    if v.len() == 1 {
                        v[0]
                    } else {
                        v.get(i).copied().unwrap_or(d)
                    }
                })
            };
            fill = (0..n)
                .map(|i| {
                    (fill[0] - pick(&slot.entry.offset, i, 0.0)) * pick(&slot.entry.scale, i, 1.0)
                })
                .collect();
        }
        let (what, delivered) = match kind {
            Kind::Camera => {
                let Some((h, w)) = input.size else {
                    self.incompatible(slot, undeclared("a size"), Vec::new());
                    return None;
                };
                let c = if input.format == ImageFormat::Mono8 {
                    1
                } else {
                    3
                };
                let shape = match input.layout {
                    ImageLayout::Hwc => vec![h, w, c],
                    ImageLayout::Chw => vec![c, h, w],
                };
                let dtype = if slot.entry.range.is_some() {
                    input.dtype.unwrap_or(DType::F32)
                } else {
                    input.dtype.unwrap_or(DType::U8)
                };
                (
                    format!("u8 {w}x{h}x{c} black, then declared preprocessing"),
                    vec![Delivered {
                        name: "image".into(),
                        dtype,
                        shape,
                    }],
                )
            }
            Kind::Text => (
                "\"\"".to_string(),
                vec![Delivered {
                    name: "text".into(),
                    dtype: DType::Utf8,
                    shape: Vec::new(),
                }],
            ),
            Kind::Joints | Kind::Hand => {
                let count = match &slot.entry.names {
                    Names::List(l) => Some(l.len() as u32),
                    _ => slot.entry.count,
                };
                let (Some(n), Some(fields)) = (count, &input.fields) else {
                    self.incompatible(slot, undeclared("names or count, and fields"), Vec::new());
                    return None;
                };
                let dtype = input.dtype.unwrap_or(DType::F32);
                (
                    format!("zeros {} [{n}] per field", crate::plan::short(dtype)),
                    fields
                        .iter()
                        .map(|f| Delivered {
                            name: f.clone(),
                            dtype,
                            shape: vec![n],
                        })
                        .collect(),
                )
            }
            _ => {
                let (Some(shape), Some(dtype)) = (&input.shape, input.dtype) else {
                    self.incompatible(slot, undeclared("a shape and a dtype"), Vec::new());
                    return None;
                };
                let names = input
                    .fields
                    .clone()
                    .unwrap_or_else(|| vec![slot.entry.kind_name.replace('/', "_")]);
                (
                    format!("zeros {} {shape:?}", crate::plan::short(dtype)),
                    names
                        .into_iter()
                        .map(|name| Delivered {
                            name,
                            dtype,
                            shape: shape.clone(),
                        })
                        .collect(),
                )
            }
        };
        Some(InputRow {
            entry: slot.entry.name.clone(),
            kind: slot.entry.kind_name.clone(),
            part: None,
            channel: None,
            applied: false,
            status: Status::Blank,
            plan: Plan {
                steps: vec![Step::Blank { what, fill }],
                notes: Vec::new(),
            },
            delivers: Delivery::of(delivered, (input.history > 1).then_some(input.history)),
            history: input.history,
            source_hz: 0.0,
            max_age: None,
            max_age_explicit: false,
            required: false,
            hold: false,
            warmup: false,
        })
    }

    fn grid(&self, action: &Action, replan: f64) -> (f64, bool, Duration) {
        let grid = action.hz.unwrap_or(replan);
        let dt = Duration::from_nanos((1e9 / grid).round() as u64);
        (grid, action.hz.is_some(), dt)
    }

    fn dropped(&mut self, slot: &Slot<'_>, action: &Action, replan: f64) -> ActionRow {
        let (grid_hz, grid_declared, dt) = self.grid(action, replan);
        ActionRow {
            entry: slot.entry.name.clone(),
            kind: slot.entry.kind_name.clone(),
            part: None,
            status: Status::Dropped,
            plan: Plan::default(),
            delivers: Delivery::Array {
                dtype: DType::F32,
                shape: vec![action.horizon],
                concat: Vec::new(),
            },
            grid_hz,
            grid_declared,
            dt,
            horizon: action.horizon,
            valid: Duration::ZERO,
            t0: action.t0,
        }
    }

    /// Step 6 for one action.
    fn action(
        &mut self,
        slot: &Slot<'a>,
        action: &Action,
        part: &'a PartSpec,
        status: Status,
        replan: f64,
    ) -> Option<ActionRow> {
        let kind = Kind::parse(&part.kind).unwrap_or(Kind::Custom);
        let path = part.path.to_string();
        if slot.robot != self.robot {
            self.incompatible(
                slot,
                format!(
                    "{}: targets {}, but the policy's actions target {}.",
                    slot.name(),
                    slot.robot,
                    self.robot
                ),
                vec![
                    "    All of a policy's action entries target one robot in v0.2 (§6.6).".into(),
                ],
            );
            return None;
        }
        let Some(command) = &part.command else {
            let mode = action.command.clone().unwrap_or_else(|| {
                kind.command_modes()
                    .first()
                    .map_or("value", |m| *m)
                    .to_string()
            });
            let reference = action
                .reference
                .as_ref()
                .map_or_else(String::new, |r| format!(" (reference {r})"));
            let mut body = vec![format!(
                "      {path} is a {} part without a command.",
                part.kind
            )];
            if kind == Kind::Pose {
                body.push(format!(
                    "    RLMesh does not do IK (scope rule). Declare {path:?} with \"command\": {mode:?}"
                ));
                body.push("    and serve it from your robot or controller.".into());
            } else {
                body.push(format!(
                    "    Declare {path:?} with \"command\": {mode:?} if the robot accepts commands there."
                ));
            }
            self.incompatible(
                slot,
                format!(
                    "{} wants {} command {mode:?}{reference} at {path:?}:",
                    slot.name(),
                    part.kind
                ),
                body,
            );
            return None;
        };
        let before = self.problems.len();
        if let Some(mode) = &action.command
            && mode != &command.mode
        {
            self.incompatible(
                slot,
                format!(
                    "{} wants {} command {mode:?} at {path:?}:",
                    slot.name(),
                    part.kind
                ),
                vec![
                    format!("      {path} accepts {:?}.", command.mode),
                    "    RLMesh does not convert between command modes.".into(),
                ],
            );
            return None;
        }
        if let Some(reference) = &action.reference
            && reference != &command.reference
        {
            self.incompatible(
                slot,
                format!(
                    "{}: the model's deltas are relative to the {reference} pose; {path} composes them on the {}.",
                    slot.name(),
                    if command.reference.is_empty() { "absolute" } else { &command.reference }
                ),
                Vec::new(),
            );
            return None;
        }
        self.frames(slot, part, None);
        let mut plan = Plan::default();
        let mut fields = command.fields.clone();
        // Declared affine first (actions apply it before the conversions).
        let elements = if fields
            .iter()
            .all(|f| f.shape.len() == 1 && f.dtype == fields[0].dtype)
        {
            Some(fields.iter().map(|f| u64::from(f.shape[0])).sum::<u64>())
        } else {
            None
        };
        match self.affine(slot, elements, true) {
            Ok(Some(step)) => plan.steps.push(step),
            Ok(None) => {}
            Err(()) => return None,
        }
        // Rescale: the gripper's open/closed, or a declared range.
        if kind == Kind::Gripper {
            let (open, closed) = (action.open.unwrap_or(1.0), action.closed.unwrap_or(0.0));
            if (open, closed) != (1.0, 0.0) {
                if !self.allowed(slot.entry, Allow::Rescale) {
                    self.not_allowed(
                        slot,
                        format!(
                            "the model's gripper is closed at {} and open at {}; {path} is 0 closed … 1 open",
                            num(closed),
                            num(open)
                        ),
                        Allow::Rescale,
                    );
                    return None;
                }
                plan.steps.push(Step::Rescale {
                    from: (closed, open),
                    to: (0.0, 1.0),
                });
            }
        }
        if let Some(from) = slot.entry.range {
            let to = fields
                .iter()
                .map(|f| field_range(part, kind, &f.name, &f.unit))
                .collect::<Option<Vec<_>>>()
                .and_then(|r| {
                    r.first()
                        .copied()
                        .filter(|first| r.iter().all(|x| x == first))
                });
            let Some(to) = to else {
                self.incompatible(
                    slot,
                    format!(
                        "{}: the model emits range {}…{}, but {path} declares no range.",
                        slot.name(),
                        num(from.0),
                        num(from.1)
                    ),
                    Vec::new(),
                );
                return None;
            };
            if from != to {
                if !self.allowed(slot.entry, Allow::Rescale) {
                    self.not_allowed(
                        slot,
                        format!(
                            "the model emits {}…{}, {path} takes {}…{}",
                            num(from.0),
                            num(from.1),
                            num(to.0),
                            num(to.1)
                        ),
                        Allow::Rescale,
                    );
                    return None;
                }
                plan.steps.push(Step::Rescale { from, to });
            }
        }
        // Rotation: the model's encoding to the part's.
        if let Some(from) = slot.entry.rotation {
            let Some(i) = fields.iter().position(|f| !f.rotation.is_empty()) else {
                self.incompatible(
                    slot,
                    format!(
                        "{}: rotation {from}, but {path} takes no rotation.",
                        slot.name()
                    ),
                    Vec::new(),
                );
                return None;
            };
            let to = Rotation::from_name(&fields[i].rotation).unwrap_or(Rotation::QuatXyzw);
            match self.rotation_step(slot, &fields[i].name.clone(), from, to, &path) {
                Ok(Some(step)) => {
                    fields[i].shape = vec![from.width()];
                    plan.steps.push(step);
                }
                Ok(None) => {}
                Err(()) => return None,
            }
        }
        // Units: the model's (SI unless declared) to the part's.
        for (field, _) in &slot.entry.units {
            if !fields.iter().any(|f| &f.name == field) {
                self.incompatible(
                    slot,
                    format!(
                        "{}: units on {field}, which {path} does not take.",
                        slot.name()
                    ),
                    Vec::new(),
                );
                return None;
            }
        }
        for f in &fields {
            let q = quantity(kind, &f.name, &f.unit);
            if q == Quantity::None {
                continue;
            }
            let from = slot
                .entry
                .units
                .iter()
                .find(|(n, _)| n == &f.name)
                .map(|(_, u)| u.clone())
                .unwrap_or_else(|| {
                    if q == Quantity::Custom {
                        f.unit.clone()
                    } else {
                        q.si().to_string()
                    }
                });
            match self.unit_step(slot, part, kind, &f.name, &from, &f.unit, &path) {
                Ok(Some(step)) => plan.steps.push(step),
                Ok(None) => {}
                Err(()) => return None,
            }
        }
        // Names.
        let mut count = None;
        if kind.is_joint_set() {
            let n = joint_count(&fields);
            let order = self.names(slot, part, n, &path)?;
            if let Order::Select(picked) = &order {
                count = Some(picked.len() as u32);
            }
            plan.steps.push(Step::Names(order));
        }
        if let Some(schema) = &slot.entry.schema {
            let matches = schema.len() == fields.len()
                && schema
                    .iter()
                    .zip(&fields)
                    .all(|(a, b)| a.name == b.name && a.shape == b.shape);
            if !matches {
                self.incompatible(
                    slot,
                    format!("{}: {path}'s command fields do not match the declared schema.", slot.name()),
                    vec!["    A custom kind binds only by exact name and a matching fields schema (§4.2).".into()],
                );
                return None;
            }
        }
        // Timing.
        let (grid_hz, grid_declared, dt) = self.grid(action, replan);
        if command.is_delta() && grid_hz > part.control_hz {
            self.incompatible(
                slot,
                format!(
                    "{}: a delta every {} ms ({} Hz) is faster than {path}'s control rate of {} Hz.",
                    slot.name(),
                    ms(dt),
                    hz(grid_hz),
                    hz(part.control_hz)
                ),
                vec!["    At most one delta is dispatched per control tick (§9.3).".into()],
            );
            return None;
        }
        let span = dt * action.horizon;
        let valid = action.ttl.map_or(span, |ttl| ttl.min(span));
        let stride = command.layout().ok().and_then(|l| l.stride()).unwrap_or(0);
        if (action.horizon as usize).saturating_mul(stride) > MAX_CHUNK_BYTES {
            self.incompatible(
                slot,
                format!(
                    "{}: {} actions of {stride} bytes is over the 4 MiB chunk cap.",
                    slot.name(),
                    action.horizon
                ),
                Vec::new(),
            );
            return None;
        }
        if valid > command.max_action_age {
            self.warnings.push(format!(
                "{}: a chunk covers {} ms, but {path}'s edge plays at most {} ms of it (max_action_age_ms); the rest is never dispatched.",
                slot.name(),
                ms(valid),
                ms(command.max_action_age)
            ));
        }
        if self.problems.len() > before {
            return None;
        }
        let delivered: Vec<Delivered> = fields
            .iter()
            .map(|f| Delivered {
                name: f.name.clone(),
                dtype: f.dtype,
                shape: match count {
                    Some(n) => vec![n],
                    None => f.shape.clone(),
                },
            })
            .collect();
        Some(ActionRow {
            entry: slot.entry.name.clone(),
            kind: slot.entry.kind_name.clone(),
            part: Some(part.clone()),
            status,
            plan,
            delivers: Delivery::of(delivered, Some(action.horizon)),
            grid_hz,
            grid_declared,
            dt,
            horizon: action.horizon,
            valid,
            t0: action.t0,
        })
    }

    fn duplicates(&mut self, slots: &[Slot<'a>], actions: &[ActionRow]) {
        for (i, a) in actions.iter().enumerate() {
            let Some(pa) = &a.part else { continue };
            for b in &actions[..i] {
                let Some(pb) = &b.part else { continue };
                let group = |p: &PartSpec| p.command.as_ref().and_then(|c| c.group.clone());
                let same_group =
                    pa.owner == pb.owner && group(pa).is_some() && group(pa) == group(pb);
                if pa.channel != pb.channel && !same_group {
                    continue;
                }
                let Some(slot) = slots
                    .iter()
                    .find(|s| s.dir == Dir::Out && s.entry.name == a.entry)
                else {
                    continue;
                };
                let head = if pa.channel == pb.channel {
                    format!(
                        "actions {:?} and {:?} both command {}.",
                        b.entry, a.entry, pa.path
                    )
                } else {
                    format!(
                        "actions {:?} and {:?} command {} and {}, members of one command group {:?}.",
                        b.entry,
                        a.entry,
                        pb.path,
                        pa.path,
                        group(pa).unwrap_or_default()
                    )
                };
                self.problems.push(problem(
                    Code::DuplicateTarget,
                    slot,
                    head,
                    vec![
                        "    fix: keep one of them; a part (or a group) has one commanding entry."
                            .into(),
                    ],
                ));
            }
        }
    }

    fn contracts(
        &mut self,
        slots: &[Slot<'a>],
        inputs: &[InputRow],
        replan: f64,
    ) -> Vec<StreamContract> {
        let mut by_stream: BTreeMap<ChannelId, Vec<&InputRow>> = BTreeMap::new();
        let mut order: Vec<ChannelId> = Vec::new();
        for row in inputs {
            if let Some(channel) = row.channel {
                if !by_stream.contains_key(&channel) {
                    order.push(channel);
                }
                by_stream.entry(channel).or_default().push(row);
            }
        }
        let mut contracts = Vec::new();
        let mut rings = Vec::new();
        for channel in order {
            let rows = &by_stream[&channel];
            let Some(part) = rows[0].part.as_ref() else {
                continue;
            };
            let applied = rows[0].applied;
            let history = rows.iter().map(|r| r.history).max().unwrap_or(1);
            let mode = contract::mode(part, rows[0].source_hz, replan, history);
            let bounds: Vec<(Duration, bool)> = rows
                .iter()
                .filter_map(|r| r.max_age.map(|m| (m, r.max_age_explicit)))
                .collect();
            let explicit = bounds.iter().any(|(_, e)| *e);
            let max_age = bounds
                .iter()
                .filter(|(_, e)| *e == explicit)
                .map(|(m, _)| *m)
                .min();
            let c = StreamContract {
                channel,
                mode,
                delivery_hz: replan,
                history,
                max_age,
                max_age_explicit: explicit,
            };
            if let Err(why) = contract::admit(&c, part, applied) {
                let row = rows
                    .iter()
                    .find(|r| r.history == history && (r.max_age_explicit || !explicit))
                    .unwrap_or(&rows[0]);
                if let Some(slot) = slots
                    .iter()
                    .find(|s| s.dir == Dir::In && s.entry.name == row.entry)
                {
                    let why = why.split_once(": ").map_or(why.as_str(), |(_, w)| w);
                    let head = format!(
                        "{}: {}.",
                        slot.name(),
                        why.replace("it publishes", &format!("{} publishes", row.path()))
                    );
                    let robots_bound = why.contains("max_age_ms")
                        && slots
                            .iter()
                            .filter_map(|s| s.input)
                            .filter(|i| rows.iter().any(|r| r.entry == i.entry.name))
                            .all(|i| i.max_age.is_none());
                    let body = if robots_bound {
                        vec![format!(
                            "    The bound is the robot's own max_age_ms on {}; a slower delivery cannot meet it.",
                            row.path()
                        )]
                    } else {
                        Vec::new()
                    };
                    self.problems
                        .push(problem(Code::Incompatible, slot, head, body));
                }
                continue;
            }
            rings.push((c.clone(), part.clone(), applied));
            contracts.push(c);
        }
        let bytes = contract::ring_bytes(rings.iter().map(|(c, p, a)| (c, p, *a)));
        if bytes > contract::MAX_RING_BYTES
            && let Some(slot) = slots.iter().find(|s| s.dir == Dir::In)
        {
            self.problems.push(problem(
                Code::Incompatible,
                slot,
                format!(
                    "the windows of {} hold {} MiB of samples (at most 512 MiB per controller).",
                    self.options.name,
                    bytes >> 20
                ),
                Vec::new(),
            ));
        }
        contracts
    }
}

/// `width_m`, when the part declares it.
fn width_m(part: &PartSpec) -> Option<(f64, f64)> {
    let value: Value = serde_json::from_str(part.attrs.get(attrs::WIDTH_M)?).ok()?;
    Some((value.get(0)?.as_f64()?, value.get(1)?.as_f64()?))
}

/// A field's declared range: a gripper's 0…1 (or `width_m` in metres), or
/// the part's `range` for it.
fn field_range(part: &PartSpec, kind: Kind, field: &str, unit: &str) -> Option<(f64, f64)> {
    if kind == Kind::Gripper {
        return if unit == "m" {
            width_m(part)
        } else {
            Some((0.0, 1.0))
        };
    }
    let ranges: Value = serde_json::from_str(part.attrs.get(attrs::RANGE)?).ok()?;
    let r = ranges.get(field)?;
    Some((r.get(0)?.as_f64()?, r.get(1)?.as_f64()?))
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
