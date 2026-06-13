//! Apply a resolved action plan to a model action output.

use rlmesh_spaces::Tensor;

use super::super::plans::ActionPlan;
use super::error::ApplyError;
use super::geometry::convert_rotation;
use super::lookup::{map_range, numeric_vector};
use super::value::{self, Value};

fn sign(value: f32) -> f32 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
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
        if segment.binarize {
            for entry in &mut piece {
                *entry = sign(*entry);
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
