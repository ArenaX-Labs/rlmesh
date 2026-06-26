//! Apply a resolved action plan to a model action output.

use rlmesh_spaces::Tensor;

use super::geometry::convert_rotation;
use super::lookup::{map_range, numeric_vector};
use super::value::{self, Value};
use crate::error::ApplyError;
use crate::plans::ActionPlan;

/// Snap a corrected value to a definite binary side. A value exactly on the
/// boundary (e.g. the model output equals the threshold, giving 0.0 here) opens
/// (`>= 0`) rather than emitting an undefined 0.0, which is neither open nor close.
fn binary_snap(value: f32) -> f32 {
    if value >= 0.0 { 1.0 } else { -1.0 }
}

/// Convert a model action vector into the env action vector (float32).
pub fn transform_action(plan: &ActionPlan, raw_action: &Value) -> Result<Tensor, ApplyError> {
    let action = numeric_vector(raw_action)?;
    if action.len() != plan.in_dim as usize {
        return Err(ApplyError::new(format!(
            "expected {}-dim model action, got shape ({},)",
            plan.in_dim,
            action.len()
        )));
    }
    let mut pieces: Vec<f32> = Vec::new();
    for segment in &plan.segments {
        let mut piece: Vec<f32> = action[segment.start as usize..segment.stop as usize].to_vec();
        if let (Some(src), Some(dst)) = (segment.src_encoding, segment.dst_encoding)
            && src != dst
        {
            piece = convert_rotation(&piece, src, dst)?;
        }
        if let (Some(src_range), Some(dst_range)) = (segment.src_range, segment.dst_range) {
            map_range(&mut piece, src_range, dst_range)?;
        }
        // Env-side scalar corrections, after the declared formats are bridged
        // and before the binary snap, in the order scale, invert, threshold.
        if let Some(scale) = segment.scale {
            let scale = scale as f32;
            for entry in &mut piece {
                *entry *= scale;
            }
        }
        if segment.invert {
            for entry in &mut piece {
                *entry = -*entry;
            }
        }
        if let Some(threshold) = segment.threshold {
            let threshold = threshold as f32;
            for entry in &mut piece {
                *entry -= threshold;
            }
        }
        if segment.binarize {
            for entry in &mut piece {
                // TODO: verify intended binary-gripper behavior at raw 0.0.
                // threshold=None keeps a raw 0.0 neutral; a thresholded boundary opens.
                if segment.threshold.is_some() || *entry != 0.0 {
                    *entry = binary_snap(*entry);
                }
            }
        }
        pieces.extend(piece);
    }
    if let Some((low, high)) = plan.clip {
        let (low, high) = (low as f32, high as f32);
        for entry in &mut pieces {
            *entry = entry.clamp(low, high);
        }
    }
    let len = pieces.len();
    Ok(value::tensor_from_f32(vec![len as i64], &pieces))
}

#[cfg(test)]
mod tests {
    use super::super::super::plans::{ActionPlan, ActionSegment};
    use super::super::value::{Value, to_f32_vec};
    use super::transform_action;

    /// A single-component plan carrying the env-side scalar corrections.
    fn one_segment(
        scale: Option<f64>,
        invert: bool,
        threshold: Option<f64>,
        binarize: bool,
    ) -> ActionPlan {
        ActionPlan {
            segments: vec![ActionSegment {
                role: "action/gripper".to_owned(),
                start: 0,
                stop: 1,
                src_encoding: None,
                dst_encoding: None,
                src_range: None,
                dst_range: None,
                scale,
                invert,
                threshold,
                binarize,
            }],
            clip: None,
            in_dim: 1,
            execute_horizon: 1,
        }
    }

    fn apply_one(plan: &ActionPlan, value: f32) -> f32 {
        let out = transform_action(plan, &Value::List(vec![Value::Number(value as f64)])).unwrap();
        to_f32_vec(&out)[0]
    }

    #[test]
    fn invert_flips_the_binary_decision() {
        let plan = one_segment(None, true, None, true);
        assert_eq!(apply_one(&plan, 0.8), -1.0);
        assert_eq!(apply_one(&plan, -0.8), 1.0);
    }

    #[test]
    fn scale_multiplies_the_value() {
        let plan = one_segment(Some(2.0), false, None, false);
        assert_eq!(apply_one(&plan, 0.3), 0.6);
    }

    #[test]
    fn threshold_recenters_the_binary_split() {
        // Subtract 0.5, then snap: values above 0.5 open, below close.
        let plan = one_segment(None, false, Some(0.5), true);
        assert_eq!(apply_one(&plan, 0.8), 1.0);
        assert_eq!(apply_one(&plan, 0.3), -1.0);
    }

    #[test]
    fn binary_boundary_opens_rather_than_emitting_zero() {
        // model output == threshold lands exactly on 0.0 after the subtract; it
        // must snap to a definite side (open), never an undefined 0.0.
        let plan = one_segment(None, false, Some(0.5), true);
        assert_eq!(apply_one(&plan, 0.5), 1.0);
    }

    #[test]
    fn raw_zero_without_threshold_stays_neutral() {
        let plan = one_segment(None, false, None, true);
        assert_eq!(apply_one(&plan, 0.0), 0.0);
        assert_eq!(apply_one(&plan, 0.2), 1.0);
        assert_eq!(apply_one(&plan, -0.2), -1.0);
    }

    #[test]
    fn corrections_apply_in_scale_invert_threshold_order() {
        // 0.3 -> *2 = 0.6 -> invert = -0.6 -> -0.5 = -1.1 (no binarize).
        let plan = one_segment(Some(2.0), true, Some(0.5), false);
        assert!((apply_one(&plan, 0.3) - (-1.1)).abs() < 1e-6);
    }
}
