//! Pair each model state component with an env feature and derive the plan.

use std::collections::BTreeMap;

use super::super::plans::{StatePiece, StatePlan};
use super::super::pyfmt::{py_repr, py_repr_encoding, py_repr_sorted_keys};
use super::super::spec::{EnvState, StateComponent, StateInput};
use super::{Result, err};

/// Width of an optional component's zero fill when the env lacks it.
fn zero_fill_width(component: &StateComponent, model_key: &str) -> Result<u32> {
    if component.index.is_some() {
        return Ok(1);
    }
    if let Some(dim) = component.dim {
        return Ok(dim);
    }
    if let Some(encoding) = component.encoding {
        return Ok(encoding.dims());
    }
    Err(err(format!(
        "model input {}: optional state role {} needs dim, index, or encoding \
         to size its zero fill",
        py_repr(model_key),
        py_repr(&component.role)
    )))
}

pub(super) fn plan_state(
    model_input: &StateInput,
    states_by_role: &BTreeMap<String, &EnvState>,
) -> Result<StatePlan> {
    let mut pieces: Vec<StatePiece> = Vec::with_capacity(model_input.components.len());
    for component in &model_input.components {
        let Some(env_state) = states_by_role.get(&component.role).copied() else {
            if component.optional {
                pieces.push(StatePiece {
                    env_key: String::new(),
                    src_encoding: None,
                    dst_encoding: None,
                    dim: Some(zero_fill_width(component, &model_input.key)?),
                    index: None,
                    zero_fill: true,
                });
                continue;
            }
            return Err(err(format!(
                "model input {} needs state role {} but the env offers {}",
                py_repr(&model_input.key),
                py_repr(&component.role),
                py_repr_sorted_keys(states_by_role)
            )));
        };
        if component.encoding != env_state.encoding
            && (component.encoding.is_none() || env_state.encoding.is_none())
        {
            return Err(err(format!(
                "state role {}: cannot convert encoding {} to {}; both sides \
                 must declare a rotation encoding",
                py_repr(&component.role),
                py_repr_encoding(env_state.encoding),
                py_repr_encoding(component.encoding)
            )));
        }
        if let (Some(component_encoding), Some(env_encoding)) =
            (component.encoding, env_state.encoding)
            && component_encoding != env_encoding
            && let Some(env_dim) = env_state.dim
            && env_dim != env_encoding.dims()
        {
            return Err(err(format!(
                "state role {}: env feature {} declares {env_dim} dims but \
                 encoding {} has {}",
                py_repr(&component.role),
                py_repr(&env_state.key),
                py_repr_encoding(Some(env_encoding)),
                env_encoding.dims()
            )));
        }
        // Bounds-check the requested slice against the source width. The
        // width is the env feature's, unless a rotation conversion reshapes it
        // first (in which case the converted width applies). Without this an
        // out-of-range index or dim silently yields fewer values.
        let converts = matches!(
            (env_state.encoding, component.encoding),
            (Some(src), Some(dst)) if src != dst
        );
        let source_width = if converts {
            component.encoding.map(|encoding| encoding.dims())
        } else {
            env_state.dim
        };
        if let Some(width) = source_width {
            if let Some(index) = component.index {
                if index >= width {
                    return Err(err(format!(
                        "state role {}: index {index} is out of range for the \
                         width-{width} source feature {}",
                        py_repr(&component.role),
                        py_repr(&env_state.key)
                    )));
                }
            } else if let Some(dim) = component.dim
                && dim > width
            {
                return Err(err(format!(
                    "state role {}: requested {dim} dims but the source feature \
                     {} has width {width}",
                    py_repr(&component.role),
                    py_repr(&env_state.key)
                )));
            }
        }
        pieces.push(StatePiece {
            env_key: env_state.key.clone(),
            src_encoding: env_state.encoding,
            dst_encoding: component.encoding,
            dim: component.dim,
            index: component.index,
            zero_fill: false,
        });
    }
    Ok(StatePlan {
        model_key: model_input.key.clone(),
        pieces,
        pad_to: model_input.pad_to,
        dtype: model_input.dtype.clone(),
        reshape: model_input.reshape.clone(),
        container: model_input.container,
    })
}
