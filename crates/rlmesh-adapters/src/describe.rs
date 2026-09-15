//! Render resolved plans as human-readable summaries.

use std::fmt::Write as _;

use crate::advisory::Advisory;
use crate::fmt::{number, quoted, quoted_range};
use crate::plans::{ActionSegment, ImagePlan, ObsPlan, ResolvedAdapter, StatePlan, TextPlan};
use crate::spec::{Attr, FitMode, FrameRef, ImageLayout, StackPad};

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
pub(crate) fn adapter_advisories(adapter: &ResolvedAdapter) -> Vec<Advisory> {
    let mut notes: Vec<Advisory> = Vec::new();
    for plan in &adapter.obs_plans {
        match plan {
            ObsPlan::Image(image) if image.zero_fill.is_some() => {
                notes.push(Advisory::caution(format!(
                    "image {}: the env provides no source camera; using a blank (zero) frame",
                    quoted(&image.placement.to_string())
                )))
            }
            ObsPlan::Image(image) if image.size.is_some() => match image.fit {
                FitMode::Crop => notes.push(Advisory::info(format!(
                    "image {}: aspect crop drops edge pixels",
                    quoted(&image.placement.to_string())
                ))),
                FitMode::Pad => notes.push(Advisory::info(format!(
                    "image {}: aspect pad adds letterbox borders",
                    quoted(&image.placement.to_string())
                ))),
                FitMode::Stretch => {}
            },
            ObsPlan::State(state) => {
                // A declared constant part is authored data, not fabricated:
                // only an absent role counts.
                let zeros = state
                    .pieces
                    .iter()
                    .filter(|piece| piece.absent_role)
                    .count();
                if zeros > 0 {
                    notes.push(Advisory::caution(format!(
                        "state {}: {zeros} component(s) zero-filled for an absent env role",
                        quoted(&state.placement.to_string())
                    )));
                }
            }
            _ => {}
        }
    }
    // An optional roled actuator the model does not output: the env receives a
    // fabricated constant where a real command belongs -- the action-side twin
    // of a zero-filled camera. A role-less fill is the opaque control dim the
    // env always meant to set itself, so it stays silent.
    for segment in &adapter.action_plan.segments {
        if let (Some(role), Some((width, value))) = (&segment.role, segment.fill) {
            notes.push(Advisory::caution(format!(
                "action {}: the model does not output this optional role; \
                 fabricating {width} dim(s) of {value}",
                quoted(role)
            )));
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

/// Append a geometry qualifier: `@robot_base` for a frame, `~target` for a
/// delta's reference. Renders only when a side declared one, so every
/// pre-geometry summary is byte-identical.
fn write_geometry(note: &mut String, attr: Attr, value: Option<&FrameRef>) {
    if let Some(value) = value {
        let _ = write!(note, "{}{value}", attr.sigil());
    }
}

/// Append the part a leaf was bound under, as `#left_arm`. Renders only when
/// a side declared one, so every pre-`part` summary is byte-identical.
fn write_part(note: &mut String, part: Option<&str>) {
    if let Some(part) = part {
        let _ = write!(note, "#{part}");
    }
}

/// A per-axis vector as `[FR_hip:0.125,FR_thigh:0.25]`, or bare values when
/// no labels name the axes.
fn axis_list(values: &[f64], labels: Option<&[String]>) -> String {
    let entries: Vec<String> = values
        .iter()
        .enumerate()
        .map(
            |(index, value)| match labels.and_then(|labels| labels.get(index)) {
                Some(label) => format!("{label}:{}", number(*value)),
                None => number(*value),
            },
        )
        .collect();
    format!("[{}]", entries.join(","))
}

/// The per-axis affine tokens `*[..]` and `+[..]`, rendered only when a side
/// declared a per-axis vector, so every scalar-only summary is unchanged.
fn axis_affine_tokens(
    axis_scale: Option<&[f64]>,
    axis_offset: Option<&[f64]>,
    labels: Option<&[String]>,
) -> Vec<String> {
    let mut tokens = Vec::new();
    if let Some(scale) = axis_scale {
        tokens.push(format!("*{}", axis_list(scale, labels)));
    }
    if let Some(offset) = axis_offset {
        tokens.push(format!("+{}", axis_list(offset, labels)));
    }
    tokens
}

/// A comma-joined index list, `-` marking a slot nothing drives.
fn index_list(indices: &[Option<u32>]) -> String {
    indices
        .iter()
        .map(|index| index.map_or("-".to_owned(), |index| index.to_string()))
        .collect::<Vec<_>>()
        .join(",")
}

/// Summarize how one env action component is derived from the model output.
fn describe_segment(segment: &ActionSegment) -> String {
    if let Some((width, value)) = segment.fill {
        // A roled fill segment is an *optional* role the model did not output:
        // surface it as a fabrication (like a zero-filled camera) rather than a
        // plain opaque dim. A role-less fill is the opaque control-dim case.
        return match &segment.role {
            Some(role) => {
                let mut note = match &segment.axis_fill {
                    Some(axis) => format!(
                        "{} <- fill {} ({width}d; model did not output this optional role)",
                        quoted(role),
                        axis_list(axis, segment.labels.as_deref())
                    ),
                    None => format!(
                        "{} <- fill {value} ({width}d; model did not output this optional role)",
                        quoted(role)
                    ),
                };
                write_part(&mut note, segment.part.as_deref());
                note
            }
            None => format!("(opaque {width}d) <- fill {value}"),
        };
    }
    let role = segment.role.as_deref().unwrap_or("?");
    let mut note = format!(
        "{} <- model[{}:{}]",
        quoted(role),
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
    // The model's own affine, in the model's axis order; new tokens, so a
    // scalar-only model side stays silent as it always was.
    let mut model_affine = axis_affine_tokens(
        segment.model_axis_scale.as_deref(),
        segment.model_axis_offset.as_deref(),
        segment.model_labels.as_deref(),
    );
    if let Some(offset) = segment.model_offset {
        model_affine.push(format!("+{}", number(offset)));
    }
    if !model_affine.is_empty() {
        let _ = write!(note, " (model {})", model_affine.join(" "));
    }
    // The scatter onto the env's axes: a permutation when every env axis is
    // driven, a selection (with `-` for a filled axis) when the model drives
    // a subset.
    if let Some(scatter) = &segment.scatter {
        let word = if scatter.iter().all(Option::is_some) {
            "perm"
        } else {
            "select"
        };
        let _ = write!(note, " {word}[{}]", index_list(scatter));
        if let Some(axis) = &segment.axis_fill {
            let filled: Vec<f64> = scatter
                .iter()
                .zip(axis)
                .filter(|(slot, _)| slot.is_none())
                .map(|(_, value)| *value)
                .collect();
            let labels: Option<Vec<String>> = segment.labels.as_ref().map(|labels| {
                scatter
                    .iter()
                    .zip(labels)
                    .filter(|(slot, _)| slot.is_none())
                    .map(|(_, label)| label.clone())
                    .collect()
            });
            let _ = write!(note, " (fill {})", axis_list(&filled, labels.as_deref()));
        }
    }
    if let Some(scale) = segment.scale {
        let _ = write!(note, " (*{scale})");
    }
    for token in axis_affine_tokens(
        segment.axis_scale.as_deref(),
        segment.axis_offset.as_deref(),
        segment.labels.as_deref(),
    ) {
        let _ = write!(note, " ({token})");
    }
    if let Some(offset) = segment.offset {
        let _ = write!(note, " (+{})", number(offset));
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
    write_geometry(&mut note, Attr::Frame, segment.frame.as_ref());
    write_geometry(&mut note, Attr::Reference, segment.reference.as_ref());
    write_part(&mut note, segment.part.as_deref());
    note
}

fn describe_image(plan: &ImagePlan) -> String {
    if let Some((height, width, channels)) = plan.zero_fill {
        let mut note = format!(
            "{} <- zeros({height}x{width}x{channels})",
            quoted(&plan.placement.to_string())
        );
        write_part(&mut note, plan.part.as_deref());
        return note;
    }
    let mut steps: Vec<String> = Vec::new();
    // The asserted camera size leads: it describes the frame arriving, not a
    // step taken on it.
    if let Some((height, width)) = plan.render {
        steps.push(format!("render {height}x{width}"));
    }
    if plan.src_layout != ImageLayout::Hwc {
        steps.push(format!("{}->hwc", plan.src_layout.as_str()));
    }
    if plan.flip {
        steps.push("flip 180".to_owned());
    }
    if let Some(quality) = plan.jpeg_quality {
        steps.push(format!("jpeg q{quality}"));
    }
    if let Some(crop) = &plan.crop {
        let mut step = if crop.slice {
            format!("crop {:.3} (slice)", crop.fraction)
        } else {
            format!("zoom {:.3}", crop.fraction)
        };
        if let Some(area) = crop.area {
            let _ = write!(step, " (crop {:.1}% area)", area * 100.0);
        }
        if let Some((height, width)) = crop.cut {
            let _ = write!(step, " -> {height}x{width}");
        }
        steps.push(step);
    }
    if let Some((height, width)) = plan.size {
        steps.push(format!("resize {height}x{width} ({})", plan.resample));
    }
    if plan.swap_rb {
        steps.push("bgr".to_owned());
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
    // Stacking is the last step: it assembles frames the rest of the pipeline
    // already produced. A contiguous window says nothing here (the `stack` field
    // is the whole story, and every spec written before offsets existed prints
    // exactly as it always did); a declared window shows what it gathers.
    if let Some(offsets) = &plan.offsets {
        let listed: Vec<String> = offsets.iter().map(i32::to_string).collect();
        steps.push(format!("stack {} @[{}]", plan.stack, listed.join(",")));
    }
    if plan.stack_pad != StackPad::First {
        steps.push("pad black".to_owned());
    }
    let mut source = quoted(&plan.source.to_string());
    write_part(&mut source, plan.part.as_deref());
    format!(
        "{} <- image {} ({})",
        quoted(&plan.placement.to_string()),
        source,
        steps.join(", ")
    )
}

fn describe_state(plan: &StatePlan) -> String {
    let mut parts: Vec<String> = Vec::new();
    for piece in &plan.pieces {
        if let Some(fill) = piece.fill {
            let width = piece.dim.expect("fill pieces always carry a width");
            // An absent optional role filled with zeros keeps the original
            // `zeros(n)` wording; the new tokens render only when the new
            // features are used.
            let mut note = match (piece.absent_role, fill == 0.0) {
                (true, true) => format!("zeros({width})"),
                (true, false) => format!("fill({width})={}", number(fill)),
                (false, _) => format!("const({width})={}", number(fill)),
            };
            let affine = axis_affine_tokens(
                piece.axis_scale.as_deref(),
                piece.axis_offset.as_deref(),
                piece.labels.as_deref(),
            );
            if !affine.is_empty() {
                let _ = write!(note, " ({})", affine.join(" "));
            }
            write_part(&mut note, piece.part.as_deref());
            parts.push(note);
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
        // The gather by name: `perm` when the model reads every env axis in its
        // own order, `select` when it reads a subset. The index list states the
        // width, so the `[:dim]` suffix below stays quiet.
        if let Some(gather) = &piece.gather {
            let word = match (&piece.labels, &piece.src_labels) {
                (Some(labels), Some(src)) if labels.len() == src.len() => "perm",
                _ => "select",
            };
            let listed: Vec<String> = gather.iter().map(u32::to_string).collect();
            let _ = write!(note, " {word}[{}]", listed.join(","));
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
            if piece.src_dim != Some(dim) && piece.gather.is_none() {
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
        if piece.post_rotate.is_some() {
            note.push_str(" (post_rotate)");
        }
        let mut affine: Vec<String> = Vec::new();
        if let Some(scale) = piece.scale {
            affine.push(format!("*{}", number(scale)));
        }
        if let Some(offset) = piece.offset {
            let sign = if offset.is_sign_negative() { "" } else { "+" };
            affine.push(format!("{sign}{}", number(offset)));
        }
        affine.extend(axis_affine_tokens(
            piece.axis_scale.as_deref(),
            piece.axis_offset.as_deref(),
            piece.labels.as_deref(),
        ));
        if !affine.is_empty() {
            let _ = write!(note, " ({})", affine.join(" "));
        }
        write_geometry(&mut note, Attr::Frame, piece.frame.as_ref());
        write_part(&mut note, piece.part.as_deref());
        write_geometry(&mut note, Attr::Provenance, piece.provenance.as_ref());
        parts.push(note);
    }
    let mut suffix = String::new();
    // The container steps in their order: the clamp, then the pad.
    if let Some((low, high)) = plan.clip {
        let _ = write!(suffix, " clip[{},{}]", number(low), number(high));
    }
    if let Some(pad_to) = plan.pad_to {
        let _ = write!(suffix, ", pad to {pad_to}");
    }
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
            "fill {}",
            match &plan.fill {
                Some(fill) => quoted(fill),
                None => "None".to_owned(),
            }
        ),
        Some(source) => quoted(&source.to_string()),
    };
    format!("{} <- text {}", quoted(&plan.placement.to_string()), source)
}
