//! Render resolved plans as human-readable summaries.

use std::fmt::Write as _;

use crate::fmt::{quoted, quoted_range};
use crate::plans::{ActionSegment, ImagePlan, ObsPlan, ResolvedAdapter, StatePlan, TextPlan};
use crate::spec::{FitMode, ImageLayout};

/// Summarize how one model input is derived from the observation.
fn describe_obs_plan(plan: &ObsPlan) -> String {
    match plan {
        ObsPlan::Image(image) => describe_image(image),
        ObsPlan::State(state) => describe_state(state),
        ObsPlan::Text(text) => describe_text(text),
        ObsPlan::Custom(custom) => {
            format!(
                "{} <- custom transform",
                quoted(&custom.placement.to_string())
            )
        }
    }
}

/// Notable, potentially-surprising outcomes of resolving against *this* env:
/// per-env data loss or fabrication a caller may want to surface (a zero-filled
/// camera, an aspect crop/letterbox). Lossless or explicitly-requested steps
/// (layout, dtype, normalize, stretch) are omitted -- this is the "warn" subset
/// of [`describe_adapter`], not the full transform list.
pub(crate) fn adapter_advisories(adapter: &ResolvedAdapter) -> Vec<String> {
    let mut notes: Vec<String> = Vec::new();
    for plan in &adapter.obs_plans {
        match plan {
            ObsPlan::Image(image) if image.zero_fill.is_some() => notes.push(format!(
                "image {}: the env provides no source camera; using a blank (zero) frame",
                quoted(&image.placement.to_string())
            )),
            ObsPlan::Image(image) if image.size.is_some() => match image.fit {
                FitMode::Crop => notes.push(format!(
                    "image {}: aspect crop drops edge pixels",
                    quoted(&image.placement.to_string())
                )),
                FitMode::Pad => notes.push(format!(
                    "image {}: aspect pad adds letterbox borders",
                    quoted(&image.placement.to_string())
                )),
                FitMode::Stretch => {}
            },
            ObsPlan::State(state) => {
                let zeros = state.pieces.iter().filter(|piece| piece.zero_fill).count();
                if zeros > 0 {
                    notes.push(format!(
                        "state {}: {zeros} component(s) zero-filled for an absent env role",
                        quoted(&state.placement.to_string())
                    ));
                }
            }
            _ => {}
        }
    }
    notes
}

pub(crate) fn describe_adapter(adapter: &ResolvedAdapter) -> String {
    let mut lines: Vec<String> = vec!["observation:".to_owned()];
    for plan in &adapter.obs_plans {
        lines.push(format!("  {}", describe_obs_plan(plan)));
    }
    lines.push("action:".to_owned());
    for segment in &adapter.action_plan.segments {
        lines.push(format!("  {}", describe_segment(segment)));
    }
    if let Some(clip) = adapter.action_plan.clip {
        lines.push(format!("  clip to {}", quoted_range(clip)));
    }
    lines.join("\n")
}

/// Summarize how one env action component is derived from the model output.
fn describe_segment(segment: &ActionSegment) -> String {
    let mut note = format!(
        "{} <- model[{}:{}]",
        quoted(&segment.role),
        segment.start,
        segment.stop
    );
    if segment.src_encoding != segment.dst_encoding {
        let src = segment
            .src_encoding
            .expect("conversions require both encodings");
        let dst = segment
            .dst_encoding
            .expect("conversions require both encodings");
        let _ = write!(note, " ({}->{})", src.as_str(), dst.as_str());
    }
    if let (Some(src_range), Some(dst_range)) = (segment.src_range, segment.dst_range) {
        let _ = write!(
            note,
            " (range {}->{})",
            quoted_range(src_range),
            quoted_range(dst_range)
        );
    }
    if let Some(scale) = segment.scale {
        let _ = write!(note, " (*{scale})");
    }
    if segment.invert {
        note.push_str(" (invert)");
    }
    if let Some(threshold) = segment.threshold {
        let _ = write!(note, " (-{threshold})");
    }
    if segment.binarize {
        note.push_str(" (sign)");
    }
    note
}

fn describe_image(plan: &ImagePlan) -> String {
    if let Some((height, width, channels)) = plan.zero_fill {
        return format!(
            "{} <- zeros({height}x{width}x{channels})",
            quoted(&plan.placement.to_string())
        );
    }
    let mut steps: Vec<String> = Vec::new();
    if plan.src_layout != ImageLayout::Hwc {
        steps.push(format!("{}->hwc", plan.src_layout.as_str()));
    }
    if plan.flip {
        steps.push("flip 180".to_owned());
    }
    if let Some((height, width)) = plan.size {
        steps.push(format!("resize {height}x{width} ({})", plan.resample));
    }
    if let Some((low, high)) = plan.normalize {
        if (low, high) == (0.0, 1.0) {
            steps.push("normalize /255".to_owned());
        } else {
            steps.push(format!("normalize [{low}, {high}]"));
        }
    }
    steps.push(plan.dtype.clone());
    if plan.dst_layout != ImageLayout::Hwc {
        steps.push(format!("hwc->{}", plan.dst_layout.as_str()));
    }
    if plan.lead_dims > 0 {
        steps.push(format!("+{} lead dims", plan.lead_dims));
    }
    format!(
        "{} <- image {} ({})",
        quoted(&plan.placement.to_string()),
        quoted(&plan.source.to_string()),
        steps.join(", ")
    )
}

fn describe_state(plan: &StatePlan) -> String {
    let mut parts: Vec<String> = Vec::new();
    for piece in &plan.pieces {
        if piece.zero_fill {
            parts.push(format!(
                "zeros({})",
                piece.dim.expect("zero-fill pieces always carry a width")
            ));
            continue;
        }
        let mut note = piece.source.to_string();
        // A SplitLayout field reads a fixed `[offset, offset+width)` slice of a
        // flat leaf; show it so the split is visible. A whole-leaf state leaves
        // src_offset None and reads the entire value.
        if let Some(offset) = piece.src_offset {
            let width = piece.src_dim.expect("layout fields carry src_dim");
            let _ = write!(note, "[{offset}:{}]", offset + width);
        }
        if piece.src_encoding != piece.dst_encoding {
            let src = piece
                .src_encoding
                .expect("conversions require both encodings");
            let dst = piece
                .dst_encoding
                .expect("conversions require both encodings");
            let _ = write!(note, " ({}->{})", src.as_str(), dst.as_str());
        }
        if let Some(index) = piece.index {
            let _ = write!(note, "[{index}]");
        } else if let Some(dim) = piece.dim {
            // The env slice above already states a layout field's width; only
            // note a model-side truncation when it narrows that slice further.
            if piece.src_dim != Some(dim) {
                let _ = write!(note, "[:{dim}]");
            }
        }
        if let (Some(src), Some(dst)) = (piece.src_range, piece.dst_range) {
            let _ = write!(
                note,
                " (range {}->{})",
                quoted_range(src),
                quoted_range(dst)
            );
        }
        parts.push(note);
    }
    let suffix = match plan.pad_to {
        Some(pad_to) => format!(", pad to {pad_to}"),
        None => String::new(),
    };
    format!(
        "{} <- concat({}){}",
        quoted(&plan.placement.to_string()),
        parts.join(", "),
        suffix
    )
}

fn describe_text(plan: &TextPlan) -> String {
    let source = match &plan.source {
        None => format!(
            "default {}",
            match &plan.default {
                Some(default) => quoted(default),
                None => "None".to_owned(),
            }
        ),
        Some(source) => quoted(&source.to_string()),
    };
    format!("{} <- text {}", quoted(&plan.placement.to_string()), source)
}
