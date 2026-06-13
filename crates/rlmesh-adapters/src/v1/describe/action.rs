//! Summarize a resolved action segment.

use std::fmt::Write as _;

use super::super::fmt::{quoted, quoted_range};
use super::super::plans::ActionSegment;

/// Summarize how one env action component is derived from the model output.
pub fn describe_segment(segment: &ActionSegment) -> String {
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
