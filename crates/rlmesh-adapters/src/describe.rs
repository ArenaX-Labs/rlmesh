//! Render resolved plans as human-readable summaries.

use std::fmt::Write as _;

use crate::fmt::{quoted, quoted_range};
use crate::plans::{ActionSegment, ImagePlan, ObsPlan, ResolvedAdapter, StatePlan, TextPlan};
use crate::spec::ImageLayout;

/// Summarize how one model input is derived from the observation.
fn describe_obs_plan(plan: &ObsPlan) -> String {
    match plan {
        ObsPlan::Image(image) => describe_image(image),
        ObsPlan::State(state) => describe_state(state),
        ObsPlan::Text(text) => describe_text(text),
        ObsPlan::Custom(custom) => {
            format!("{} <- custom transform", quoted(&custom.model_key))
        }
    }
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
    if segment.binarize {
        note.push_str(" (sign)");
    }
    note
}

fn describe_image(plan: &ImagePlan) -> String {
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
    if plan.normalize {
        steps.push("normalize /255".to_owned());
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
        quoted(&plan.model_key),
        quoted(&plan.env_key),
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
        let mut note = piece.env_key.clone();
        // A StateLayout field reads a fixed `[offset, offset+width)` slice of a
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
        quoted(&plan.model_key),
        parts.join(", "),
        suffix
    )
}

fn describe_text(plan: &TextPlan) -> String {
    let source = if plan.env_key.is_empty() {
        format!(
            "default {}",
            match &plan.default {
                Some(default) => quoted(default),
                None => "None".to_owned(),
            }
        )
    } else {
        quoted(&plan.env_key)
    };
    format!("{} <- text {}", quoted(&plan.model_key), source)
}
