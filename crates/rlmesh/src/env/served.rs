//! [`ServedEnv`]: the env Core's `describe_json(served_env=)` reads instead
//! of building a representative one.

use std::fmt;

use super::factory::MadeEnv;
use super::{Env, VectorEnv};
use crate::{Error, Result};

/// An env already built from the described factory: an env, a vector env, or
/// a list of lane envs, of which only lane 0 is read. Describe reads its
/// spaces and never closes it.
#[non_exhaustive]
pub enum ServedEnv<'a> {
    /// One env.
    Env(&'a dyn Env),
    /// A vector env (its `single_*` spaces and `num_envs`).
    Vector(&'a dyn VectorEnv),
    /// Lane envs; lane 0 is read, and an empty list is the ValueError class.
    Lanes(Vec<&'a dyn Env>),
}

impl<'a> ServedEnv<'a> {
    /// One env.
    pub fn env(env: &'a impl Env) -> ServedEnv<'a> {
        ServedEnv::Env(env)
    }

    /// A vector env.
    pub fn vector(env: &'a impl VectorEnv) -> ServedEnv<'a> {
        ServedEnv::Vector(env)
    }

    /// Lane envs (lane 0 is read).
    pub fn lanes<E: Env>(lanes: &'a [E]) -> ServedEnv<'a> {
        ServedEnv::Lanes(lanes.iter().map(|lane| lane as &dyn Env).collect())
    }

    /// The env whose spaces describe reads, or the ValueError class for an
    /// empty lane list.
    pub(crate) fn read(&self) -> Result<ServedRead<'a>> {
        match self {
            ServedEnv::Env(env) => Ok(ServedRead::Env(*env)),
            ServedEnv::Vector(env) => Ok(ServedRead::Vector(*env)),
            ServedEnv::Lanes(lanes) => {
                lanes
                    .first()
                    .map(|env| ServedRead::Env(*env))
                    .ok_or_else(|| {
                        Error::invalid_argument(
                            "served_env must hold at least one lane env, got []",
                        )
                    })
            }
        }
    }
}

impl<'a> From<&'a MadeEnv> for ServedEnv<'a> {
    fn from(env: &'a MadeEnv) -> ServedEnv<'a> {
        match env {
            MadeEnv::Env(env) => ServedEnv::Env(env.as_ref()),
            MadeEnv::Vector(env) => ServedEnv::Vector(env.as_ref()),
        }
    }
}

impl fmt::Debug for ServedEnv<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ServedEnv::Env(_) => f.write_str("ServedEnv::Env(..)"),
            ServedEnv::Vector(_) => f.write_str("ServedEnv::Vector(..)"),
            ServedEnv::Lanes(lanes) => write!(f, "ServedEnv::Lanes({} lanes)", lanes.len()),
        }
    }
}

/// The one env a served-env describe reads.
pub(crate) enum ServedRead<'a> {
    Env(&'a dyn Env),
    Vector(&'a dyn VectorEnv),
}
