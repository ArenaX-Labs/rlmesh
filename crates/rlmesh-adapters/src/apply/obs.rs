//! Apply resolved observation plans: dispatch each model input to its
//! per-kind application (image / state / text / custom), scattering each
//! produced value into the assembled payload `Value` tree at its placement.

use std::collections::BTreeMap;

use super::CustomTransform;
use super::image::apply_image;
use super::state::apply_state;
use super::text::apply_text;
use super::value::Value;
use crate::error::ApplyError;
use crate::path::{NodePath, PathSeg};
use crate::plans::ObsPlan;

/// Convert a raw env observation into the model input payload `Value` tree.
///
/// Each plan produces one leaf [`Value`]; the leaf is scattered to its
/// `placement` in the tree (materializing `Value::Map`/`Value::List` containers
/// en route). A root (empty) placement means the whole payload IS that one
/// value (a bare-tensor model). Role fan-out is automatic: two plans with the
/// same role but different placements each read the same env source and place
/// into different positions — no special machinery.
pub fn transform_obs(
    plans: &[ObsPlan],
    raw_obs: &BTreeMap<String, Value>,
    customs: &dyn CustomTransform,
) -> Result<Value, ApplyError> {
    let mut builder = NodeBuilder::new();
    for plan in plans {
        match plan {
            ObsPlan::Image(image_plan) => {
                builder.place(&image_plan.placement, apply_image(image_plan, raw_obs)?)?;
            }
            ObsPlan::State(state_plan) => {
                builder.place(&state_plan.placement, apply_state(state_plan, raw_obs)?)?;
            }
            ObsPlan::Text(text_plan) => {
                if let Some(value) = apply_text(text_plan, raw_obs)? {
                    builder.place(&text_plan.placement, value)?;
                }
            }
            ObsPlan::Custom(custom_plan) => {
                if let Some(value) =
                    customs.apply(&custom_plan.placement_key, &custom_plan.transform, raw_obs)?
                {
                    builder.place(&custom_plan.placement, value)?;
                }
            }
        }
    }
    Ok(builder.finish())
}

/// Scatters produced leaves into a payload `Value` tree by placement path.
///
/// A `Key` segment materializes a [`Value::Map`]; an `Index` segment a
/// [`Value::List`] (grown with placeholders as needed). A root (empty)
/// placement sets the whole payload to that one value (a bare-tensor model).
struct NodeBuilder {
    /// `None` until the first placement; the bare-tensor (root) case sets it
    /// directly, every keyed/indexed case grows a Map/List from an empty Map.
    root: Option<Value>,
}

impl NodeBuilder {
    fn new() -> Self {
        Self { root: None }
    }

    fn place(&mut self, placement: &NodePath, value: Value) -> Result<(), ApplyError> {
        if placement.is_root() {
            if self.root.is_some() {
                return Err(ApplyError::new(
                    "two model inputs both target the root payload placement".to_owned(),
                ));
            }
            self.root = Some(value);
            return Ok(());
        }
        // Non-root placements grow from a Map root (a dict payload) or a List
        // root (a tuple payload), chosen by the first segment's kind.
        let root = self.root.get_or_insert_with(|| match placement.0.first() {
            Some(PathSeg::Index(_)) => Value::List(Vec::new()),
            _ => Value::Map(BTreeMap::new()),
        });
        insert_at(root, &placement.0, value)
    }

    fn finish(self) -> Value {
        self.root.unwrap_or_else(|| Value::Map(BTreeMap::new()))
    }
}

/// Insert `value` at the `segments` path within `node`, materializing
/// `Map`/`List` containers en route.
fn insert_at(node: &mut Value, segments: &[PathSeg], value: Value) -> Result<(), ApplyError> {
    let Some((head, rest)) = segments.split_first() else {
        *node = value;
        return Ok(());
    };
    match head {
        PathSeg::Key(key) => {
            let Value::Map(map) = node else {
                return Err(ApplyError::new(format!(
                    "payload placement conflict: expected a Map to place key '{key}'"
                )));
            };
            let child = map
                .entry(key.clone())
                .or_insert(Value::Map(BTreeMap::new()));
            insert_at(child, rest, value)
        }
        PathSeg::Index(index) => {
            let Value::List(items) = node else {
                return Err(ApplyError::new(format!(
                    "payload placement conflict: expected a List to place index [{index}]"
                )));
            };
            if items.len() <= *index {
                items.resize(*index + 1, Value::Map(BTreeMap::new()));
            }
            insert_at(&mut items[*index], rest, value)
        }
    }
}
