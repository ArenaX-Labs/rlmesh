//! Apply resolved observation plans: dispatch each model input to its
//! per-kind application (image / state / text / custom), scattering each
//! produced value into the assembled payload `Value` tree at its placement.

use std::collections::BTreeMap;

use rlmesh_session_core::spec::role::apply::Array;
use rlmesh_spaces::{DType, Tensor};

use super::lookup::{numeric_vector, resolve_in_obs};
use super::text::apply_text;
use super::value::{self, Value};
use super::{CustomTransform, Jpeg, from_array, from_core, to_core};
use crate::error::ApplyError;
use crate::path::{NodePath, PathSeg};
use crate::plans::{ImagePlan, ObsPlan, StackedPlacement, StatePlan};

/// One model image input from a raw observation.
pub(crate) fn apply_image(
    plan: &ImagePlan,
    raw_obs: &BTreeMap<String, Value>,
) -> Result<Value, ApplyError> {
    if plan.zero_fill.is_some() {
        let image = plan.apply(None, &Jpeg).map_err(from_core)?;
        return Ok(Value::Tensor(from_array(image)?));
    }
    let frame = decode_frame(resolve_in_obs(raw_obs, &plan.source)?)?;
    let shape = value::shape_usize(&frame);
    let bytes = frame.to_contiguous_bytes();
    let dtype = rlmesh_session_core::wire::DType::from_code(frame.dtype() as u32)
        .ok_or_else(|| ApplyError::new(format!("an image of dtype {:?}", frame.dtype())))?;
    let array = Array {
        dtype,
        shape: &shape,
        data: &bytes,
    };
    let image = plan.apply(Some(array), &Jpeg).map_err(from_core)?;
    Ok(Value::Tensor(from_array(image)?))
}

/// A camera value as a tensor: an array as is, encoded image bytes decoded
/// with their own channel count (grayscale → 1, luma+alpha → 2, RGB → 3,
/// RGBA → 4), so a declared-channel check sees what the env sent.
fn decode_frame(value: &Value) -> Result<Tensor, ApplyError> {
    match value {
        Value::Tensor(tensor) => Ok(tensor.clone()),
        Value::Bytes(raw) => {
            let decoded = image::load_from_memory(raw)
                .map_err(|err| ApplyError::new(format!("could not decode image bytes: {err}")))?;
            let (width, height) = (decoded.width(), decoded.height());
            let (pixels, channels) = match decoded.color().channel_count() {
                1 => (decoded.to_luma8().into_raw(), 1i64),
                2 => (decoded.to_luma_alpha8().into_raw(), 2i64),
                4 => (decoded.to_rgba8().into_raw(), 4i64),
                _ => (decoded.to_rgb8().into_raw(), 3i64),
            };
            Tensor::from_vec(
                pixels,
                vec![i64::from(height), i64::from(width), channels],
                DType::Uint8,
            )
            .map_err(|err| ApplyError::new(err.to_string()))
        }
        _ => Err(ApplyError::new(
            "expected an image array observation value or encoded image bytes".to_owned(),
        )),
    }
}

/// One model state input from a raw observation; `previous` is the raw
/// action executed at the previous step, for an action-source part.
pub(crate) fn apply_state(
    plan: &StatePlan,
    raw_obs: &BTreeMap<String, Value>,
    previous: Option<&[f32]>,
) -> Result<Value, ApplyError> {
    let mut read = |piece: &rlmesh_session_core::spec::role::plan::StatePiece| {
        resolve_in_obs(raw_obs, &piece.source)
            .and_then(numeric_vector)
            .map_err(to_core)
    };
    let state = from_array(plan.apply(&mut read, previous).map_err(from_core)?)?;
    if plan.list {
        return Ok(Value::List(
            value::to_f64_vec(&state)
                .into_iter()
                .map(Value::Number)
                .collect(),
        ));
    }
    Ok(Value::Tensor(state))
}

/// Convert a raw env observation into the model input payload `Value` tree.
///
/// Each plan produces one leaf [`Value`]; the leaf is scattered to its
/// `placement` in the tree (materializing `Value::Map`/`Value::List` containers
/// en route). A root (empty) placement means the whole payload IS that one
/// value (a bare-tensor model). Role fan-out is automatic: two plans with the
/// same role but different placements each read the same env source and place
/// into different positions — no special machinery. `previous` is the raw
/// action executed at the previous step for any action-source state part
/// (`None` reads its fill; the stateful seam supplies the recorded row).
pub fn transform_obs(
    plans: &[ObsPlan],
    raw_obs: &BTreeMap<String, Value>,
    customs: &dyn CustomTransform,
    previous: Option<&[f32]>,
) -> Result<Value, ApplyError> {
    let mut builder = NodeBuilder::new();
    for plan in plans {
        match plan {
            ObsPlan::Image(image_plan) => {
                builder.place(&image_plan.placement, apply_image(image_plan, raw_obs)?)?;
            }
            ObsPlan::State(state_plan) => {
                builder.place(
                    &state_plan.placement,
                    apply_state(state_plan, raw_obs, previous)?,
                )?;
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

/// Apply only the frame-stacked image plans and return their processed frames,
/// in `stacked` order.
///
/// The observe half of [`transform_obs`]: an env step that replays a queued
/// action still has to advance every frame window, but it assembles no payload
/// and calls no model. `plan_index` addresses each stacked input's plan
/// directly, so nothing else in the spec is applied — a custom input's host
/// transform in particular never runs on a step that predicts nothing.
pub(crate) fn observe_obs(
    plans: &[ObsPlan],
    stacked: &[StackedPlacement],
    raw_obs: &BTreeMap<String, Value>,
) -> Result<Vec<Tensor>, ApplyError> {
    stacked
        .iter()
        .map(|entry| {
            let ObsPlan::Image(image_plan) = &plans[entry.plan_index] else {
                return Err(ApplyError::new(format!(
                    "frame-stacked input '{}' is not an image",
                    entry.key
                )));
            };
            match apply_image(image_plan, raw_obs)? {
                Value::Tensor(frame) => Ok(frame),
                _ => Err(ApplyError::new(format!(
                    "frame-stacked input '{}' must be a tensor",
                    entry.key
                ))),
            }
        })
        .collect()
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
    // An empty placeholder (a freshly traversed child or a grown list slot)
    // takes the container kind its next segment needs.
    match (head, &*node) {
        (PathSeg::Key(_), Value::List(items)) if items.is_empty() => {
            *node = Value::Map(BTreeMap::new());
        }
        (PathSeg::Index(_), Value::Map(map)) if map.is_empty() => {
            *node = Value::List(Vec::new());
        }
        _ => {}
    }
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use image::ImageEncoder;

    use super::{NodeBuilder, decode_frame, transform_obs};
    use crate::apply::{CustomTransform, Value};
    use crate::error::ApplyError;
    use crate::path::NodePath;
    use crate::plans::{CustomPlan, ObsPlan};

    fn built(placements: &[NodePath]) -> Value {
        let mut builder = NodeBuilder::new();
        for (leaf, placement) in placements.iter().enumerate() {
            builder
                .place(placement, Value::Number(leaf as f64))
                .unwrap();
        }
        builder.finish()
    }

    #[test]
    fn a_tuple_nested_under_a_dict_key_places_as_a_list() {
        let payload = built(&[NodePath::root().push_key("nested").push_index(0)]);
        let expected = Value::Map(BTreeMap::from([(
            "nested".to_owned(),
            Value::List(vec![Value::Number(0.0)]),
        )]));
        assert_eq!(payload, expected);
    }

    #[test]
    fn a_tuple_nested_in_a_tuple_places_as_a_list() {
        let payload = built(&[
            NodePath::root().push_index(1).push_index(0),
            NodePath::root().push_index(0).push_index(0),
        ]);
        let expected = Value::List(vec![
            Value::List(vec![Value::Number(1.0)]),
            Value::List(vec![Value::Number(0.0)]),
        ]);
        assert_eq!(payload, expected);
    }

    struct Constant;

    impl CustomTransform for Constant {
        fn apply(
            &self,
            _model_key: &str,
            _entrypoint: &str,
            _raw_obs: &BTreeMap<String, Value>,
        ) -> Result<Option<Value>, ApplyError> {
            Ok(Some(Value::Number(7.0)))
        }
    }

    #[test]
    fn an_all_custom_root_tuple_places_as_a_list() {
        let placement = NodePath::root().push_index(0);
        let plan = ObsPlan::Custom(CustomPlan {
            placement_key: placement.to_string(),
            placement,
            transform: "host:x".to_owned(),
        });
        let payload = transform_obs(&[plan], &BTreeMap::new(), &Constant, None).unwrap();
        assert_eq!(payload, Value::List(vec![Value::Number(7.0)]));
    }

    #[test]
    fn encoded_image_bytes_decode_with_their_own_channel_count() {
        // A grayscale or RGBA image must not be silently expanded or cut to 3
        // channels: a declared-channel check has to see what the env sent.
        for (pixels, color, channels) in [
            (vec![10_u8, 20, 30], image::ColorType::Rgb8, 3),
            (vec![42], image::ColorType::L8, 1),
            (vec![10, 20, 30, 128], image::ColorType::Rgba8, 4),
        ] {
            let mut encoded = Vec::new();
            image::codecs::png::PngEncoder::new(&mut encoded)
                .write_image(&pixels, 1, 1, color.into())
                .expect("encode png");
            let decoded = decode_frame(&Value::Bytes(encoded)).expect("decode");
            assert_eq!(decoded.shape(), &[1, 1, channels]);
            assert_eq!(decoded.to_contiguous_bytes().as_ref(), pixels.as_slice());
        }
        let error = decode_frame(&Value::Bytes(b"not an image".to_vec())).unwrap_err();
        assert!(error.message.contains("could not decode image bytes"));
    }
}
