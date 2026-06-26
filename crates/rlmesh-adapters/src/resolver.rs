//! Resolve env and model IO specs into a concrete adapter plan.

mod action;
mod custom;
mod image;
mod state;
mod text;

use std::collections::BTreeMap;

use super::error::{AdapterResolutionError, ErrorCode};
use super::fmt::quoted;
use super::join::join;
use super::path::NodePath;
use super::plans::{ObsPlan, ResolvedAdapter};
use super::space_view::SpaceView;
use super::spec::{
    EnvFeature, EnvImage, EnvState, EnvTags, EnvText, InputNode, ModelLeaf, ModelSpec,
};

type Result<T> = std::result::Result<T, AdapterResolutionError>;

fn err(code: ErrorCode, message: String) -> AdapterResolutionError {
    AdapterResolutionError::new(code, message)
}

fn index_by_role<'spec, T>(
    features: impl Iterator<Item = (&'spec String, T)>,
    label: &str,
) -> Result<BTreeMap<String, T>> {
    let mut by_role: BTreeMap<String, T> = BTreeMap::new();
    for (role, feature) in features {
        if by_role.contains_key(role) {
            return Err(err(
                ErrorCode::Duplicate,
                format!("duplicate {label} role {}", quoted(role)),
            ));
        }
        by_role.insert(role.clone(), feature);
    }
    Ok(by_role)
}

/// One model input leaf paired with its placement (tree position) in the
/// payload, produced by walking the [`InputNode`] tree.
struct PlacedLeaf<'spec> {
    leaf: &'spec ModelLeaf,
    placement: NodePath,
}

/// Flatten the model input tree into a list of leaves, each carrying the
/// [`NodePath`] placement of where its produced tensor lands in the payload. A
/// `Dict` node recurses each key, a `Tuple` each index, a `Leaf` emits itself.
fn collect_leaves<'spec>(
    node: &'spec InputNode,
    placement: NodePath,
    out: &mut Vec<PlacedLeaf<'spec>>,
) {
    match node {
        InputNode::Leaf(leaf) => out.push(PlacedLeaf { leaf, placement }),
        InputNode::Dict(map) => {
            for (key, child) in map {
                collect_leaves(child, placement.push_key(key.clone()), out);
            }
        }
        InputNode::Tuple(items) => {
            for (index, item) in items.iter().enumerate() {
                collect_leaves(item, placement.push_index(index), out);
            }
        }
    }
}

/// Derive a [`ResolvedAdapter`] for an env/model pair.
///
/// The env side is given as the [`EnvTags`] observation/action tree over its
/// observation and action spaces; [`join`] derives the keyed env features (with
/// widths and ranges) those plus the spaces imply, then each model input leaf is
/// matched to an env feature by role and placed at its tree position. A model
/// role may be reused across leaves (one camera → several input slots).
pub fn resolve(
    env_tags: &EnvTags,
    observation_space: &SpaceView,
    action_space: &SpaceView,
    model_spec: &ModelSpec,
    trust_entrypoints: bool,
) -> Result<ResolvedAdapter> {
    let env_spec = join(env_tags, observation_space, action_space)
        .map_err(|error| err(ErrorCode::InvalidTag, error.to_string()))?;
    let images = env_spec
        .observation
        .iter()
        .filter_map(|feature| match feature {
            EnvFeature::Image(image) => Some((&image.role, image)),
            _ => None,
        });
    let states = env_spec
        .observation
        .iter()
        .filter_map(|feature| match feature {
            EnvFeature::State(state) => Some((&state.role, state)),
            _ => None,
        });
    let texts = env_spec
        .observation
        .iter()
        .filter_map(|feature| match feature {
            EnvFeature::Text(text) => Some((&text.role, text)),
            _ => None,
        });
    let images_by_role: BTreeMap<String, &EnvImage> = index_by_role(images, "env image")?;
    let states_by_role: BTreeMap<String, &EnvState> = index_by_role(states, "env state")?;
    let texts_by_role: BTreeMap<String, &EnvText> = index_by_role(texts, "env text")?;

    let mut leaves: Vec<PlacedLeaf> = Vec::new();
    collect_leaves(&model_spec.input, NodePath::root(), &mut leaves);

    let mut obs_plans: Vec<ObsPlan> = Vec::with_capacity(leaves.len());
    for PlacedLeaf { leaf, placement } in leaves {
        obs_plans.push(match leaf {
            ModelLeaf::Image(input) => {
                ObsPlan::Image(image::plan_image(input, placement, &images_by_role)?)
            }
            ModelLeaf::State(input) => {
                ObsPlan::State(state::plan_state(input, placement, &states_by_role)?)
            }
            ModelLeaf::Text(input) => {
                ObsPlan::Text(text::plan_text(input, placement, &texts_by_role)?)
            }
            ModelLeaf::Custom(input) => {
                ObsPlan::Custom(custom::plan_custom(input, placement, trust_entrypoints)?)
            }
        });
    }

    let action_plan = action::plan_action(&model_spec.output, &env_spec.action)?;
    let resolved = ResolvedAdapter {
        obs_plans,
        action_plan,
    };
    // Frame-stacking and action-chunk replay are mutually exclusive. During chunk
    // replay the engine skips observation assembly (it is replaying a buffered
    // action), so a stacked input would only ever observe decision-point frames
    // spaced `execute_horizon` apart -- not the consecutive history a frame-stacked
    // policy was trained on. Reject the combination here, at the one seam both the
    // served engine and the run(env) loop resolve through, rather than silently
    // feeding temporally-aliased frames.
    if resolved.action_plan.execute_horizon > 1
        && let Some((key, depth)) = resolved.stacks().into_iter().next()
    {
        return Err(err(
            ErrorCode::Unsupported,
            format!(
                "frame-stacking (input {} stack={depth}) cannot be combined with action-chunk \
                 replay (execute_horizon={}): during replay the engine skips observation \
                 assembly, so the frame window would hold only decision-point frames. Use \
                 stack=1 or execute_horizon=1.",
                quoted(&key),
                resolved.action_plan.execute_horizon,
            ),
        ));
    }
    Ok(resolved)
}
