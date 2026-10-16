//! A borrowed local env behind a `'static` stand-in.
//!
//! The native driver and an env server need `'static` envs, and a run holds
//! only `&'a mut` of the caller's. The stand-in keeps the env's members read
//! once at construction and relays every `reset`, `step`, `render` and
//! `close` to the caller through a [`CallerPort`], where the borrowed env
//! runs them one at a time, between driver segments.

use super::drive::CallerPort;
use crate::spaces::Space;
use crate::{Env, InfoDict, Metadata, Result, Value, VectorEnv, VectorSeed};

/// An env's members, read once (the relay's contract source).
struct Members {
    observation_space: Space,
    action_space: Space,
    render_mode: Option<String>,
    metadata: Option<Metadata>,
    id: Option<String>,
}

/// A scalar env lent through a port.
pub(crate) struct LentEnv {
    members: Members,
    port: CallerPort,
}

impl LentEnv {
    pub(crate) fn new(env: &dyn Env, port: CallerPort) -> LentEnv {
        LentEnv {
            members: Members {
                observation_space: env.observation_space(),
                action_space: env.action_space(),
                render_mode: env.render_mode(),
                metadata: env.metadata(),
                id: env.id(),
            },
            port,
        }
    }
}

impl Env for LentEnv {
    fn observation_space(&self) -> Space {
        self.members.observation_space.clone()
    }
    fn action_space(&self) -> Space {
        self.members.action_space.clone()
    }
    fn render_mode(&self) -> Option<String> {
        self.members.render_mode.clone()
    }
    fn metadata(&self) -> Option<Metadata> {
        self.members.metadata.clone()
    }
    fn id(&self) -> Option<String> {
        self.members.id.clone()
    }

    fn reset(
        &mut self,
        seed: Option<i64>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        let options = options.cloned();
        self.port
            .call_blocking(move |side| side.env()?.reset(seed, options.as_ref()))
    }

    fn step(&mut self, action: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
        self.port
            .call_blocking(move |side| side.env()?.step(action))
    }

    fn render(&mut self) -> Result<Option<Value>> {
        self.port.call_blocking(|side| side.env()?.render())
    }

    fn close(&mut self) -> Result<()> {
        self.port.call_blocking(|side| side.env()?.close())
    }
}

/// A vector env lent through a port.
pub(crate) struct LentVectorEnv {
    members: Members,
    num_envs: u64,
    port: CallerPort,
}

impl LentVectorEnv {
    pub(crate) fn new(env: &dyn VectorEnv, port: CallerPort) -> LentVectorEnv {
        LentVectorEnv {
            members: Members {
                observation_space: env.single_observation_space(),
                action_space: env.single_action_space(),
                render_mode: env.render_mode(),
                metadata: env.metadata(),
                id: env.id(),
            },
            num_envs: env.num_envs(),
            port,
        }
    }
}

impl VectorEnv for LentVectorEnv {
    fn num_envs(&self) -> u64 {
        self.num_envs
    }
    fn single_observation_space(&self) -> Space {
        self.members.observation_space.clone()
    }
    fn single_action_space(&self) -> Space {
        self.members.action_space.clone()
    }
    fn render_mode(&self) -> Option<String> {
        self.members.render_mode.clone()
    }
    fn metadata(&self) -> Option<Metadata> {
        self.members.metadata.clone()
    }
    fn id(&self) -> Option<String> {
        self.members.id.clone()
    }

    fn reset(
        &mut self,
        seed: Option<VectorSeed>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        let options = options.cloned();
        self.port
            .call_blocking(move |side| side.vector()?.reset(seed, options.as_ref()))
    }

    fn step(
        &mut self,
        actions: Value,
    ) -> Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)> {
        self.port
            .call_blocking(move |side| side.vector()?.step(actions))
    }

    fn render(&mut self) -> Result<Option<Value>> {
        self.port.call_blocking(|side| side.vector()?.render())
    }

    fn close(&mut self) -> Result<()> {
        self.port.call_blocking(|side| side.vector()?.close())
    }
}
