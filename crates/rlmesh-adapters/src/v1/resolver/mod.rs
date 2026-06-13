//! Resolve env and model IO specs into a concrete adapter plan.

mod action;
mod custom;
mod image;
mod state;
mod text;

use std::collections::{BTreeMap, BTreeSet};

use super::error::AdapterResolutionError;
use super::plans::{ObsPlan, ResolvedAdapter};
use super::pyfmt::py_repr;
use super::spec::{EnvFeature, EnvFeatures, ModelInput, ModelIoSpec};

type Result<T> = std::result::Result<T, AdapterResolutionError>;

fn err(message: String) -> AdapterResolutionError {
    AdapterResolutionError::new(message)
}

fn index_by_role<'spec, T>(
    features: impl Iterator<Item = (&'spec String, T)>,
    label: &str,
) -> Result<BTreeMap<String, T>> {
    let mut by_role: BTreeMap<String, T> = BTreeMap::new();
    for (role, feature) in features {
        if by_role.contains_key(role) {
            return Err(err(format!("duplicate {label} role {}", py_repr(role))));
        }
        by_role.insert(role.clone(), feature);
    }
    Ok(by_role)
}

/// Derive a [`ResolvedAdapter`] for an env/model pair from their specs.
pub fn resolve(
    env_spec: &EnvFeatures,
    model_spec: &ModelIoSpec,
    trust_entrypoints: bool,
) -> Result<ResolvedAdapter> {
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
            return Err(err(format!("duplicate model input key {}", py_repr(key))));
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
