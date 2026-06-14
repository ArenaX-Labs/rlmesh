//! Resolve env and model IO specs into a concrete adapter plan.

mod action;
mod custom;
mod image;
mod state;
mod text;

use std::collections::{BTreeMap, BTreeSet};

use super::error::{AdapterResolutionError, ErrorCode};
use super::fmt::quoted;
use super::join::join;
use super::plans::{ObsPlan, ResolvedAdapter};
use super::space_view::SpaceView;
use super::spec::{EnvFeature, EnvTags, ModelInput, ModelSpec};

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

/// Derive a [`ResolvedAdapter`] for an env/model pair.
///
/// The env side is given as sparse [`EnvTags`] over its observation and
/// action spaces; [`join`] derives the keyed env features (with widths and
/// ranges) those plus the spaces imply, then each model input is matched to an
/// env feature by role.
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
    let images_by_role = index_by_role(images, "env image")?;
    let states_by_role = index_by_role(states, "env state")?;
    let texts_by_role = index_by_role(texts, "env text")?;

    let mut obs_plans: Vec<ObsPlan> = Vec::with_capacity(model_spec.inputs.len());
    let mut seen_keys: BTreeSet<&str> = BTreeSet::new();
    for model_input in &model_spec.inputs {
        let key = match model_input {
            ModelInput::Image(input) => input.key.as_str(),
            ModelInput::State(input) => input.key.as_str(),
            ModelInput::Text(input) => input.key.as_str(),
            ModelInput::Custom(input) => input.key.as_str(),
        };
        if !seen_keys.insert(key) {
            return Err(err(
                ErrorCode::Duplicate,
                format!("duplicate model input key {}", quoted(key)),
            ));
        }
        obs_plans.push(match model_input {
            ModelInput::Image(input) => ObsPlan::Image(image::plan_image(input, &images_by_role)?),
            ModelInput::State(input) => ObsPlan::State(state::plan_state(input, &states_by_role)?),
            ModelInput::Text(input) => ObsPlan::Text(text::plan_text(input, &texts_by_role)?),
            ModelInput::Custom(input) => {
                ObsPlan::Custom(custom::plan_custom(input, trust_entrypoints)?)
            }
        });
    }

    let action_plan = action::plan_action(&model_spec.action, &env_spec.action)?;
    Ok(ResolvedAdapter {
        obs_plans,
        action_plan,
    })
}
