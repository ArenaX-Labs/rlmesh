//! The binding table: its input and action rows, and its canonical JSON.

use crate::prelude::*;

use core::time::Duration;

use crate::wire::part::PartSpec;
use crate::wire::{ActionGrid, ChannelId, Fingerprint, ParticipantId, Place, StreamContract};
use serde_json::{Value, json};

use crate::spec::canonical;
use crate::spec::plan::{Delivery, Plan};
use crate::spec::policy::ChunkStart;

/// An entry's direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dir {
    /// An input.
    In,
    /// An action.
    Out,
}

impl Dir {
    pub(super) fn word(self) -> &'static str {
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
    pub chunk: u32,
    /// The chunk's life: `min(chunk × dt, ttl)`.
    pub valid: Duration,
    /// When a chunk starts.
    pub t0: ChunkStart,
}

/// A binding table.
#[derive(Clone, Debug, PartialEq)]
pub struct Binding {
    /// The policy.
    pub policy: String,
    /// The robot its actions target, or the view (`scene`).
    pub robot: ParticipantId,
    /// The place table of the view it bound against (empty: one robot).
    pub places: Vec<Place>,
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
    /// Accepted with a warning (a chunk past the edge's action-age cap).
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

    /// The action grid per commanded part.
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
    // A view's rows name their robot, so its fingerprint covers them.
    let robot = |row: &mut Value, part: &Option<PartSpec>| {
        if !b.places.is_empty() {
            row["robot"] = json!(part.as_ref().map(|p| p.owner.as_str()));
        }
    };
    let inputs: Vec<Value> = b
        .inputs
        .iter()
        .map(|r| {
            let mut row = json!({
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
            });
            robot(&mut row, &r.part);
            row
        })
        .collect();
    let actions: Vec<Value> = b
        .actions
        .iter()
        .map(|r| {
            let mut row = json!({
                "entry": r.entry,
                "kind": r.kind,
                "part": r.part.as_ref().map(|p| p.path.to_string()),
                "channel": r.part.as_ref().map(|p| p.channel.0),
                "status": r.status.name(),
                "plan": r.plan.steps.iter().map(ToString::to_string).collect::<Vec<_>>(),
                "delivers": r.delivers.to_string(),
                "dt_ns": u64::try_from(r.dt.as_nanos()).unwrap_or(u64::MAX),
                "chunk": r.chunk,
                "valid_ns": u64::try_from(r.valid.as_nanos()).unwrap_or(u64::MAX),
                "t0": match r.t0 { ChunkStart::Snapshot => "snapshot", ChunkStart::Send => "send" },
            });
            robot(&mut row, &r.part);
            row
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
