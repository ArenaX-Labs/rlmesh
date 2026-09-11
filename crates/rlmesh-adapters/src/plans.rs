//! Plan types: the resolved, already-validated adapter instructions.

mod action;
mod custom;
mod image;
mod state;
mod text;

use std::collections::{BTreeMap, BTreeSet};

use super::advisory::Advisory;
use super::describe;
use super::path::{NodePath, PathSeg};
use super::spec::StackPad;

pub use action::{ActionPlan, ActionSegment};

/// The reserved raw-obs envelope key for a root/empty or Tuple-rooted source —
/// the single top-level entry that holds the whole observation `Value`.
pub(crate) const OBS_ROOT_KEY: &str = "<obs>";

/// The top-level raw-obs envelope entry a `source` path lives under: the first
/// segment's key for a Dict-rooted source, else the reserved root key (an empty
/// or Tuple-rooted source whose whole `Value` is the single envelope entry).
pub(crate) fn envelope_key(source: &NodePath) -> String {
    match source.first() {
        Some(PathSeg::Key(key)) => key.clone(),
        _ => OBS_ROOT_KEY.to_owned(),
    }
}
pub use custom::CustomPlan;
pub use image::{CropPlan, ImagePlan};
pub use state::{StatePiece, StatePlan};
pub use text::TextPlan;

/// Resolved instructions for one model input.
#[derive(Debug, Clone, PartialEq)]
#[allow(
    clippy::large_enum_variant,
    reason = "one plan per model input, built once at resolve and read per step; \
              boxing the image arm would add an indirection to the hot path to save \
              a few hundred bytes that are allocated a handful of times per route"
)]
pub enum ObsPlan {
    Image(ImagePlan),
    State(StatePlan),
    Text(TextPlan),
    Custom(CustomPlan),
}

/// One frame-stacked placement, precomputed once at resolve time.
///
/// The single source of truth for stacking: the assemble path walks these to
/// stack in place (using `placement` for the payload-tree slot and `key` for
/// the per-episode window), and the observe path walks the same list to tick a
/// window on a step that predicts nothing. `key` is `placement.to_string()`,
/// precomputed so the per-step path never re-renders the canonical name.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StackedPlacement {
    pub placement: NodePath,
    pub key: String,
    /// How many frames the stacked tensor carries (`offsets.len()`, else
    /// `stack`).
    pub depth: u32,
    /// Index of this input's plan in `obs_plans`, so the observe path can apply
    /// exactly the stacked images without re-walking (or scattering) the rest.
    pub plan_index: usize,
    /// The window's gather offsets, or `None` for a contiguous stack.
    pub offsets: Option<Vec<i32>>,
    /// Consecutive frames the window holds to reach its oldest offset:
    /// `1 - offsets[0]`, and `depth` for a contiguous stack.
    pub span: u32,
    /// What fills the window at the start of an episode.
    pub stack_pad: StackPad,
    /// Bytes of one processed frame, from the plan (`0` when unknown).
    pub frame_bytes: u64,
}

/// One frame window this adapter needs held per live episode: what a caller
/// sizes an admission-control budget from before a route ever runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryWindow {
    /// The canonical placement string the window is keyed by.
    pub key: String,
    /// Consecutive frames held (`1 - offsets[0]`, or the stack depth).
    pub span: u32,
    /// Bytes of one processed frame (`0` when the env's resolution was not
    /// derivable, so a budget cannot be computed from it).
    pub frame_bytes: u64,
}

/// A resolved env-to-model adapter: build instances with [`resolve`](crate::resolver::resolve).
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedAdapter {
    pub obs_plans: Vec<ObsPlan>,
    pub action_plan: ActionPlan,
    /// Frame-stacked placements (depth `> 1`), precomputed once from
    /// `obs_plans` so both the stacking presence check and the per-step assemble
    /// path read one canonical list — see [`StackedPlacement`]. Private and only
    /// ever built by [`ResolvedAdapter::new`] (the resolver), which derives it
    /// from `obs_plans` in the same step; do not mutate `obs_plans` afterward or
    /// this summary goes stale.
    stacked: Vec<StackedPlacement>,
    /// Resolve-time advisories the plans cannot reconstruct: one per
    /// *unreferenced* unknown observation kind the env declared (an old core
    /// ignored it). Surfaced through [`advisories`](Self::advisories) alongside
    /// the per-apply data-loss notes. Empty on a fully-understood spec.
    resolve_advisories: Vec<Advisory>,
    /// Advisories surfaced through [`advisories`](Self::advisories) but kept out
    /// of [`describe`](Self::describe): the hints the env's own declaration raised
    /// at join (e.g. an image layout that looks mis-declared given its shape),
    /// and the geometry cautions from resolve (a model-declared `frame` or
    /// `reference` the env does not confirm). None of these is a *dropped* env
    /// modality, so none may land under that header (which vectors pin).
    quiet_advisories: Vec<Advisory>,
}

impl ResolvedAdapter {
    /// Assemble a resolved adapter, precomputing the frame-stacked placements
    /// ([`stacked`](Self::stacked_placements)) once from `obs_plans`. The single
    /// entry point so the precomputed stacking list can never drift from the
    /// plans it summarizes. `resolve_advisories` carries notes derived at resolve
    /// (unreferenced unknown kinds) that the plans no longer encode.
    pub(crate) fn new(
        obs_plans: Vec<ObsPlan>,
        action_plan: ActionPlan,
        resolve_advisories: Vec<Advisory>,
        quiet_advisories: Vec<Advisory>,
    ) -> Self {
        let stacked = obs_plans
            .iter()
            .enumerate()
            .filter_map(|(plan_index, plan)| match plan {
                ObsPlan::Image(image) if image.stack > 1 => Some(StackedPlacement {
                    placement: image.placement.clone(),
                    key: image.placement.to_string(),
                    depth: image.stack,
                    plan_index,
                    offsets: image.offsets.clone(),
                    // A contiguous window spans exactly its depth; a strided one
                    // reaches back to its oldest offset. Resolve already proved
                    // the offsets law, so the arithmetic cannot go negative.
                    span: image
                        .offsets
                        .as_ref()
                        .and_then(|offsets| offsets.first())
                        .map_or(image.stack, |&oldest| (1 - oldest) as u32),
                    stack_pad: image.stack_pad,
                    frame_bytes: image.frame_bytes,
                }),
                _ => None,
            })
            .collect();
        Self {
            obs_plans,
            action_plan,
            stacked,
            resolve_advisories,
            quiet_advisories,
        }
    }

    /// Human-readable summary of the resolved transformations.
    ///
    /// A *reference snapshot*: the conformance vectors pin the exact text so
    /// implementations stay consistent, but the wording is not a stable
    /// cross-language contract. Ends with a `dropped:` section listing any env
    /// modality this core ignored (an unrecognized kind no model input needed) —
    /// a genuine data-loss event the per-plan transform list cannot show, since a
    /// dropped leaf produces no plan.
    pub fn describe(&self) -> String {
        let mut summary = describe::describe_adapter(self);
        if !self.resolve_advisories.is_empty() {
            summary.push_str("\ndropped:");
            for note in &self.resolve_advisories {
                summary.push_str("\n  ");
                summary.push_str(&note.message);
            }
        }
        summary
    }

    /// Per-env data-loss / fabrication notes (a zero-filled camera, an aspect
    /// crop or letterbox, a dropped unknown-kind modality): the "warn" subset of
    /// [`describe`](Self::describe), for a caller to surface (e.g. log) without
    /// failing. Empty when nothing noteworthy happened.
    pub fn advisories(&self) -> Vec<Advisory> {
        // Resolve-time advisories (unreferenced unknown kinds) first, then the
        // env-declaration hints raised at join, then the per-apply data-loss
        // notes derived from the plans.
        let mut all = self.resolve_advisories.clone();
        all.extend(self.quiet_advisories.iter().cloned());
        all.extend(describe::adapter_advisories(self));
        all
    }

    /// The observation keys this adapter actually reads.
    ///
    /// Lets a caller decode only the referenced observation leaves instead of
    /// the whole observation — so an unused (possibly unencodable) env key
    /// never aborts a step. Custom inputs are excluded: they resolve against
    /// the raw host observation, not the decoded payload.
    pub fn referenced_obs_keys(&self) -> BTreeSet<String> {
        let mut keys = BTreeSet::new();
        for plan in &self.obs_plans {
            match plan {
                ObsPlan::Image(image) => {
                    // A zero-filled image has no env source.
                    if image.zero_fill.is_none() {
                        keys.insert(envelope_key(&image.source));
                    }
                }
                ObsPlan::State(state) => {
                    for piece in &state.pieces {
                        if piece.fill.is_none() {
                            keys.insert(envelope_key(&piece.source));
                        }
                    }
                }
                ObsPlan::Text(text) => {
                    // A default-only text input has `source = None` (it never
                    // looks one up); reporting nothing avoids encoding a
                    // non-existent top-level key.
                    if let Some(source) = &text.source {
                        keys.insert(envelope_key(source));
                    }
                }
                ObsPlan::Custom(_) => {}
            }
        }
        keys
    }

    /// Frame-stack depths the model wants, keyed by canonical placement string.
    ///
    /// Only entries with depth `> 1` (actual stacking) appear — a `stack == 1`
    /// image needs no per-episode buffer and is omitted. The episode-keyed
    /// frame-stack engine ([`crate::v1::FrameBuffers`]) buffers exactly these
    /// placements. This is the single source of truth for stacking depth
    /// (replacing the old Python `stacks` dict). The key is the placement's
    /// [`Display`](std::fmt::Display) form so a nested/positioned stacked input
    /// has a stable canonical name. Built from the precomputed per-episode
    /// `stacked_placements` list.
    pub fn stacks(&self) -> BTreeMap<String, u32> {
        self.stacked
            .iter()
            .map(|entry| (entry.key.clone(), entry.depth))
            .collect()
    }

    /// Canonical placement strings of the inputs that hold a frame window.
    ///
    /// The set a host binding drives [`observe`](crate::v1::observe_obs) for:
    /// empty means an env step that predicts nothing has no state to advance,
    /// so the tick can be skipped entirely.
    pub fn history_keys(&self) -> Vec<String> {
        self.stacked.iter().map(|entry| entry.key.clone()).collect()
    }

    /// The frame windows this adapter holds per live episode, for an
    /// admission-control budget: `num_envs x sum(span x frame_bytes)`.
    pub fn history_windows(&self) -> Vec<HistoryWindow> {
        self.stacked
            .iter()
            .map(|entry| HistoryWindow {
                key: entry.key.clone(),
                span: entry.span,
                frame_bytes: entry.frame_bytes,
            })
            .collect()
    }

    /// Apply only the frame-stacked image plans, in `history_keys` order.
    ///
    /// The observe half of [`transform_obs`](Self::transform_obs): one processed
    /// frame per stacked input, with nothing scattered into a payload tree — what
    /// a step that replays a queued action pushes into its windows so the history
    /// stays the same whatever the execution horizon is.
    pub fn history_frames(
        &self,
        raw_obs: &BTreeMap<String, crate::apply::Value>,
    ) -> Result<Vec<rlmesh_spaces::Tensor>, crate::error::ApplyError> {
        crate::apply::obs::observe_obs(&self.obs_plans, &self.stacked, raw_obs)
    }

    /// The precomputed frame-stacked placements: the single iterator the assemble
    /// path consumes so it never re-derives the `stack > 1` filter (or re-renders
    /// each placement string) on the per-step hot path.
    pub(crate) fn stacked_placements(&self) -> &[StackedPlacement] {
        &self.stacked
    }
}
