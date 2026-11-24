//! What a Gym binding does: the resolved, already-validated plan of every
//! model input and of the action vector.

#[allow(unused_imports)]
use crate::prelude::*;

mod action;
mod custom;
mod image;
mod state;
mod text;

use alloc::collections::{BTreeMap, BTreeSet};

use crate::spec::projection::Projection;
use crate::spec::role::describe;
use crate::spec::role::error::Advisory;
use crate::spec::role::input::StackPad;
use crate::spec::role::path::{NodePath, PathSeg};

pub use action::{ActionPlan, ActionSegment};

/// The reserved raw-obs envelope key for a root/empty or Tuple-rooted source —
/// the single top-level entry that holds the whole observation `Value`.
pub const OBS_ROOT_KEY: &str = "<obs>";

/// The top-level raw-obs envelope entry a `source` path lives under: the first
/// segment's key for a Dict-rooted source, else the reserved root key (an empty
/// or Tuple-rooted source whose whole `Value` is the single envelope entry).
pub fn envelope_key(source: &NodePath) -> String {
    match source.first() {
        Some(PathSeg::Key(key)) => key.clone(),
        _ => OBS_ROOT_KEY.to_owned(),
    }
}
pub use custom::CustomPlan;
pub use image::{CropPlan, ImagePlan};
pub use state::{PreviousAction, StatePiece, StatePlan};
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
    /// A camera input.
    Image(ImagePlan),
    /// A state vector.
    State(StatePlan),
    /// A text input.
    Text(TextPlan),
    /// A host-language hole.
    Custom(CustomPlan),
}

/// One frame-stacked placement, precomputed at resolve time.
///
/// `key` is `placement.to_string()`.
#[derive(Debug, Clone, PartialEq)]
pub struct StackedPlacement {
    /// Where the stacked tensor lands.
    pub placement: NodePath,
    /// `placement.to_string()`, the window's key.
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

/// A model bound to a Gym env by role: build one with [`bind`](super::bind).
#[derive(Debug, Clone, PartialEq)]
pub struct Binding {
    /// One plan per model input leaf, in tree order.
    pub obs_plans: Vec<ObsPlan>,
    /// The action vector's plan.
    pub action_plan: ActionPlan,
    /// Derived from `obs_plans` by [`Binding::new`]; stale if `obs_plans` is mutated afterward.
    stacked: Vec<StackedPlacement>,
    previous: Vec<String>,
    resolve_advisories: Vec<Advisory>,
    quiet_advisories: Vec<Advisory>,
    projections: Vec<AppliedProjection>,
}

/// A named projection a resolve applied, and what it applied to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedProjection {
    /// The projection.
    pub projection: Projection,
    /// What it discarded or fabricated, for describe and the run record.
    pub target: String,
}

impl Binding {
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
        let previous = obs_plans
            .iter()
            .filter_map(|plan| match plan {
                ObsPlan::State(state)
                    if state.pieces.iter().any(|piece| piece.previous.is_some()) =>
                {
                    Some(state.placement.to_string())
                }
                _ => None,
            })
            .collect();
        Self {
            obs_plans,
            action_plan,
            stacked,
            previous,
            resolve_advisories,
            quiet_advisories,
            projections: Vec::new(),
        }
    }

    pub(crate) fn with_projections(mut self, projections: Vec<AppliedProjection>) -> Self {
        self.projections = projections;
        self
    }

    /// The named projections the resolve applied (empty for a lossless one).
    pub fn projections(&self) -> &[AppliedProjection] {
        &self.projections
    }

    /// Human-readable summary of the resolved transformations, ending with a `dropped:` section for
    /// any env modality this core ignored.
    ///
    /// The wording is not a stable cross-language contract.
    pub fn describe(&self) -> String {
        let mut summary = describe::describe_binding(self);
        if !self.projections.is_empty() {
            summary.push_str("\nprojections:");
            for applied in &self.projections {
                summary.push_str(&format!("\n  {}: {}", applied.projection, applied.target));
            }
        }
        if !self.resolve_advisories.is_empty() {
            summary.push_str("\ndropped:");
            for note in &self.resolve_advisories {
                summary.push_str("\n  ");
                summary.push_str(&note.message);
            }
        }
        summary
    }

    /// Per-env data-loss and fabrication notes (a zero-filled camera, an aspect crop or letterbox, a
    /// dropped modality): the warning subset of [`describe`](Self::describe).
    pub fn advisories(&self) -> Vec<Advisory> {
        let mut all = self.resolve_advisories.clone();
        all.extend(self.projections.iter().map(|applied| {
            Advisory::caution(format!(
                "{} (allowed): {}",
                applied.projection, applied.target
            ))
        }));
        all.extend(self.quiet_advisories.iter().cloned());
        all.extend(describe::binding_advisories(self));
        all
    }

    /// The observation keys this adapter reads, so a caller can decode only those leaves. Custom
    /// inputs are excluded: they resolve against the raw host observation.
    pub fn referenced_obs_keys(&self) -> BTreeSet<String> {
        let mut keys = BTreeSet::new();
        for plan in &self.obs_plans {
            match plan {
                ObsPlan::Image(image) => {
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
                    if let Some(source) = &text.source {
                        keys.insert(envelope_key(source));
                    }
                }
                ObsPlan::Custom(_) => {}
            }
        }
        keys
    }

    /// Frame-stack depths the model wants, keyed by canonical placement string; only depths above 1 appear.
    pub fn stacks(&self) -> BTreeMap<String, u32> {
        self.stacked
            .iter()
            .map(|entry| (entry.key.clone(), entry.depth))
            .collect()
    }

    /// Canonical placement strings of the inputs that hold per-episode history: a frame window
    /// (`stack > 1`) or a previous-action part.
    ///
    /// Empty means a step that predicts nothing has no state to advance. A runtime that
    /// replays chunks must deliver every step to a route with any of these.
    pub fn history_keys(&self) -> Vec<String> {
        self.stacked
            .iter()
            .map(|entry| entry.key.clone())
            .chain(self.previous.iter().cloned())
            .collect()
    }

    /// Whether any input holds per-episode history (see
    /// [`history_keys`](Self::history_keys)), without allocating the list.
    pub fn holds_history(&self) -> bool {
        !self.stacked.is_empty() || !self.previous.is_empty()
    }

    /// Whether any state input reads the model's own previous action, so the
    /// stateful seam has to record every executed action under its step.
    pub fn reads_previous_action(&self) -> bool {
        !self.previous.is_empty()
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

    /// The precomputed frame-stacked placements.
    pub fn stacked_placements(&self) -> &[StackedPlacement] {
        &self.stacked
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use crate::prelude::*;

    use alloc::collections::{BTreeMap, BTreeSet};

    use super::{
        ActionPlan, Binding, HistoryWindow, ImagePlan, ObsPlan, PreviousAction, StatePiece,
        StatePlan, TextPlan,
    };
    use crate::spec::plan::Fit;
    use crate::spec::policy::{ImageLayout, Resample};
    use crate::spec::role::input::StackPad;
    use crate::spec::role::path::NodePath;
    use crate::wire::part::DType;

    fn key(name: &str) -> NodePath {
        NodePath::root().push_key(name)
    }

    fn image(placement: &str, source: &str, stack: u32, offsets: Option<Vec<i32>>) -> ImagePlan {
        ImagePlan {
            placement: key(placement),
            source: key(source),
            src_layout: ImageLayout::Hwc,
            dst_layout: ImageLayout::Hwc,
            flip: false,
            size: None,
            fit: Fit::Stretch,
            resample: Resample::Bilinear,
            dtype: DType::U8,
            normalize: None,
            lead_dims: 0,
            src_range: None,
            stack,
            zero_fill: None,
            fill: 0,
            crop: None,
            jpeg_quality: None,
            swap_rb: false,
            render: None,
            offsets,
            stack_pad: StackPad::First,
            frame_bytes: 12,
            role_rebound: None,
            part: None,
        }
    }

    fn piece(source: &str) -> StatePiece {
        StatePiece {
            source: key(source),
            src_offset: None,
            src_dim: None,
            src_encoding: None,
            dst_encoding: None,
            post_rotate: None,
            dim: None,
            index: None,
            src_range: None,
            dst_range: None,
            scale: None,
            offset: None,
            axis_scale: None,
            axis_offset: None,
            gather: None,
            labels: None,
            fill: None,
            absent_role: None,
            previous: None,
            width: Some(1),
            frame: None,
            provenance: None,
            part: None,
            flip: None,
        }
    }

    fn state(placement: &str, pieces: Vec<StatePiece>) -> ObsPlan {
        ObsPlan::State(StatePlan {
            placement: key(placement),
            pieces,
            pad_to: None,
            clip: None,
            native_width: None,
            dtype: DType::F32,
            reshape: None,
            list: false,
        })
    }

    fn text(placement: &str, source: Option<&str>) -> ObsPlan {
        ObsPlan::Text(TextPlan {
            placement: key(placement),
            source: source.map(key),
            list: false,
            fill: None,
        })
    }

    fn binding(plans: Vec<ObsPlan>) -> Binding {
        let action = ActionPlan {
            segments: Vec::new(),
            clip: None,
            in_dim: 0,
        };
        Binding::new(plans, action, Vec::new(), Vec::new())
    }

    fn previous() -> StatePiece {
        StatePiece {
            source: NodePath::root(),
            fill: Some(0.0),
            previous: Some(PreviousAction {
                role: "action/gripper".into(),
                start: 0,
                stop: 1,
            }),
            ..piece("")
        }
    }

    #[test]
    fn a_binding_reports_what_it_reads_stacks_and_holds() {
        let zeroed = ImagePlan {
            source: NodePath::root(),
            zero_fill: Some((2, 2, 3)),
            ..image("blank", "", 1, None)
        };
        let constant = StatePiece {
            source: NodePath::root(),
            fill: Some(1.0),
            ..piece("")
        };
        type Row = (
            &'static str,
            Vec<ObsPlan>,
            &'static [&'static str],
            &'static [(&'static str, u32)],
            bool,
            &'static [(&'static str, u32, u64)],
        );
        let rows: Vec<Row> = vec![
            ("nothing", Vec::new(), &[], &[], false, &[]),
            (
                "one unstacked camera",
                vec![ObsPlan::Image(image("image", "cam", 1, None))],
                &["cam"],
                &[],
                false,
                &[],
            ),
            (
                "a contiguous stack of three",
                vec![ObsPlan::Image(image("image", "cam", 3, None))],
                &["cam"],
                &[("image", 3)],
                true,
                &[("image", 3, 12)],
            ),
            (
                "a strided stack",
                vec![ObsPlan::Image(image("image", "cam", 2, Some(vec![-4, 0])))],
                &["cam"],
                &[("image", 2)],
                true,
                &[("image", 5, 12)],
            ),
            (
                "a zero-filled camera reads nothing",
                vec![ObsPlan::Image(zeroed)],
                &[],
                &[],
                false,
                &[],
            ),
            (
                "state pieces, a constant and a previous action",
                vec![state(
                    "state",
                    vec![piece("joints"), constant, previous(), piece("gripper")],
                )],
                &["gripper", "joints"],
                &[],
                true,
                &[],
            ),
            (
                "an env state alone holds nothing",
                vec![state("state", vec![piece("joints")])],
                &["joints"],
                &[],
                false,
                &[],
            ),
            (
                "text, read and default-only",
                vec![text("prompt", Some("task")), text("hint", None)],
                &["task"],
                &[],
                false,
                &[],
            ),
            (
                "a nested source reads its top-level key",
                vec![ObsPlan::Image(ImagePlan {
                    source: key("cams").push_key("wrist"),
                    ..image("image", "", 1, None)
                })],
                &["cams"],
                &[],
                false,
                &[],
            ),
        ];
        for (what, plans, keys, stacks, holds, windows) in rows {
            let bound = binding(plans);
            let want: BTreeSet<String> = keys.iter().map(|k| (*k).to_owned()).collect();
            assert_eq!(bound.referenced_obs_keys(), want, "{what}: keys");
            let want: BTreeMap<String, u32> =
                stacks.iter().map(|(k, d)| ((*k).to_owned(), *d)).collect();
            assert_eq!(bound.stacks(), want, "{what}: stacks");
            assert_eq!(bound.holds_history(), holds, "{what}: holds");
            let want: Vec<HistoryWindow> = windows
                .iter()
                .map(|(k, span, bytes)| HistoryWindow {
                    key: (*k).to_owned(),
                    span: *span,
                    frame_bytes: *bytes,
                })
                .collect();
            assert_eq!(bound.history_windows(), want, "{what}: windows");
        }
        let both = binding(vec![
            ObsPlan::Image(image("image", "cam", 2, None)),
            state("state", vec![previous()]),
        ]);
        assert!(both.holds_history());
    }
}
